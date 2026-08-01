#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tri-dialect LLM Schema expander

Expand SQLite original Schema + allocated difference requirements -> MySQL + PostgreSQL + Oracle tri-dialect equivalent DDL

Workflow:
  1. Load difference allocation results (output/schema_expansion/allocation_{split}.json)
  2. For each database:
     a. Read original SQLite DDL
     b. Get allocated difference requirements (including schema_requirements details)
     c. Call LLM to generate tri-dialect DDL in one step
     d. Call LLM to verify tri-dialect equivalence
     e. If not equivalent, reflect and correct (max max_retries attempts)
     f. Save results to database directory

Usage:
  # Process all train databases (default)
  .venv/bin/python -m schema_synthesis.schema_expander

  # Process dev split
  .venv/bin/python -m schema_synthesis.schema_expander --split dev

  # Process test split
  .venv/bin/python -m schema_synthesis.schema_expander --split test

  # Process only specified databases
  .venv/bin/python -m schema_synthesis.schema_expander --split dev --databases db1 db2

  # Process only first N
  .venv/bin/python -m schema_synthesis.schema_expander --split test --limit 10
"""

import asyncio
import json
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm.client import LLMClient
from utils.schema_sanitizer import sanitize_self_ref_fk_nullable
from utils.oracle_schema_normalizer import normalize_oracle_schema
from schema_synthesis.prompts import (
    SCHEMA_EXPANSION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    build_expansion_prompt,
    build_verification_prompt,
    build_reflection_prompt,
    format_diffs_text,
)
from schema_synthesis.db_tester import run_db_tests
from utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ExpansionResult:
    """Schema expansion result for a single database."""
    db_id: str
    original_ddl: str
    mysql_ddl: str
    pg_ddl: str
    oracle_ddl: str
    diffs_assigned: List[Dict]
    verification_passed: bool
    verification_feedback: Optional[str] = None
    db_test_passed: bool = False
    db_test_feedback: Optional[str] = None
    db_test_retries: int = 0
    retry_count: int = 0
    analysis: str = ""
    changes_summary: str = ""
    error: Optional[str] = None
    latency_seconds: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Core class
# =============================================================================

class SchemaExpander:
    """Tri-dialect Schema expander."""

    def __init__(
        self,
        config: Dict,
        allocation: Dict,
        differences_detail: Dict[str, Dict],
        split: str = "train",
    ):
        """
        Args:
            config: schema_synthesis.yaml configuration
            allocation: "allocation" field in allocation.json
            differences_detail: diff_id -> full difference details (including test_requirements)
            split: Dataset split name ("train" / "dev" / "test")
        """
        self.config = config
        self.allocation = allocation
        self.differences_detail = differences_detail
        self.split = split
        
        exp_cfg = config["expansion"]
        self.max_retries = exp_cfg.get("max_retries", 3)
        self.db_test_max_retries = exp_cfg.get("db_test_max_retries", 2)
        self.temperature = exp_cfg.get("temperature", 0.6)
        self.max_tokens = exp_cfg.get("max_tokens", 16384)
        self.concurrent_batch_size = exp_cfg.get("concurrent_batch_size", 6)
        self.output_dir = PROJECT_ROOT / exp_cfg.get("output_dir", "output/schema_expansion")
        self.database_dir = PROJECT_ROOT / config.get("database_dir", "database")
        
        # Initialize LLM client
        llm_provider = exp_cfg.get("provider", "aliyun")
        llm_model = exp_cfg.get("model", "qwen3.5-flash")
        self.llm = LLMClient(provider=llm_provider, model=llm_model)
        
        # Result collection
        self.results: List[ExpansionResult] = []
        
        logger.info(f"SchemaExpander initialized: split={self.split}, max_retries={self.max_retries}, "
                     f"concurrent_batch_size={self.concurrent_batch_size}, llm={llm_provider}/{llm_model}")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _read_original_ddl(self, db_id: str) -> str:
        """Read original SQLite DDL."""
        ddl_path = self.database_dir / self.split / db_id / f"{db_id}_original.sql"
        if not ddl_path.exists():
            raise FileNotFoundError(f"Original DDL not found: {ddl_path}")
        return ddl_path.read_text(encoding="utf-8")

    def _get_diffs_for_db(self, db_id: str) -> List[Dict]:
        """
        Get allocated difference requirements for the specified database (full details, including test_requirements)
        """
        db_alloc = self.allocation.get(db_id)
        if not db_alloc:
            return []
        
        diff_ids = db_alloc.get("diff_ids", [])
        result = []
        for diff_id in diff_ids:
            detail = self.differences_detail.get(diff_id)
            if detail:
                result.append(detail)
            else:
                # Fall back to brief info in allocation
                for d in db_alloc.get("diffs", []):
                    if d["id"] == diff_id:
                        result.append(d)
                        break
        return result

    def _save_result(self, result: ExpansionResult):
        """Save results to database directory."""
        db_dir = self.database_dir / self.split / result.db_id
        db_dir.mkdir(parents=True, exist_ok=True)

        # Post-sanitize 1: self-referencing FK columns should not be NOT NULL (root node must be NULLABLE).
        if result.mysql_ddl:
            result.mysql_ddl = sanitize_self_ref_fk_nullable(result.mysql_ddl, "mysql")
        if result.pg_ddl:
            result.pg_ddl = sanitize_self_ref_fk_nullable(result.pg_ddl, "pg")
        if result.oracle_ddl:
            result.oracle_ddl = sanitize_self_ref_fk_nullable(result.oracle_ddl, "oracle")

        # Post-sanitize 2: Oracle schema normalization
        #   - Force double-quoted identifiers to lowercase (prevent LLM outputting uppercase identifiers)
        #   - Verify tri-engine table/column name consistency and auto-fix
        #   - NUMBER(p,s) preserved as-is, overflow handled automatically during data import
        if result.oracle_ddl:
            result.oracle_ddl = normalize_oracle_schema(
                result.oracle_ddl, result.mysql_ddl, result.pg_ddl
            )

        # Save tri-dialect DDL
        if result.mysql_ddl:
            (db_dir / f"{result.db_id}_schema_mysql.sql").write_text(
                result.mysql_ddl, encoding="utf-8"
            )
        if result.pg_ddl:
            (db_dir / f"{result.db_id}_schema_pg.sql").write_text(
                result.pg_ddl, encoding="utf-8"
            )
        if result.oracle_ddl:
            (db_dir / f"{result.db_id}_schema_oracle.sql").write_text(
                result.oracle_ddl, encoding="utf-8"
            )
        
        # Save expansion metadata
        meta = {
            "db_id": result.db_id,
            "diffs_assigned": [d.get("id", "") for d in result.diffs_assigned],
            "verification_passed": result.verification_passed,
            "db_test_passed": result.db_test_passed,
            "db_test_retries": result.db_test_retries,
            "db_test_feedback": result.db_test_feedback,
            "retry_count": result.retry_count,
            "analysis": result.analysis,
            "changes_summary": result.changes_summary,
            "error": result.error,
            "latency_seconds": result.latency_seconds,
            "total_tokens": result.total_tokens,
        }
        (db_dir / f"{result.db_id}_expansion_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        
        logger.info(f"  💾 Results saved: {db_dir}/")

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response (enhanced version)
        
        Handles common LLM output issues:
        - Wrapped in ```json ... ``` code blocks
        - Thinking/explanation text mixed before/after JSON
        - Unescaped control characters in SQL strings
        - trailing commas
        - Backtick interference in nested code blocks
        """
        if not text or not text.strip():
            return None
        
        # Collect all candidate JSON texts, try in order
        candidates = []
        
        # Candidate 1: raw text
        candidates.append(text.strip())
        
        # Candidate 2: extract from ```json ... ``` code block (greedy match outermost)
        # Use more robust pattern: match ```json to last ```
        code_block_pattern = r'```(?:json|JSON)?\s*\n?([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)
        for m in matches:
            stripped = m.strip()
            if stripped:
                candidates.append(stripped)
        
        # Candidate 3: if multiple code blocks, try to find the one containing target keys
        # (LLM may embed ``` in SQL values in JSON, breaking regex)
        if '```' in text:
            # Find content between first ```json and last ```
            first_start = re.search(r'```(?:json|JSON)?\s*\n?', text)
            if first_start:
                last_end = text.rfind('```')
                if last_end > first_start.end():
                    inner = text[first_start.end():last_end].strip()
                    if inner:
                        candidates.append(inner)
        
        # Candidate 4: extract from first { to last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
        
        # Try parsing each candidate in order
        for candidate in candidates:
            result = self._try_parse_json_text(candidate)
            if result is not None:
                return result
        
        # All candidates failed, log details for debugging
        logger.warning(f"All JSON parsing failed，first 200 chars of text: {text[:200]!r}")
        return None
    
    def _try_parse_json_text(self, text: str) -> Optional[Dict]:
        """
        Try parsing a single JSON text candidate, with multi-level repair strategies
        """
        # Strategy 1: direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 2: fix unescaped control characters in JSON string values
        # LLM-generated SQL DDL often has raw newlines/tabs, which are illegal in JSON strings
        fixed = self._fix_json_control_chars(text)
        if fixed != text:
            try:
                result = json.loads(fixed)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Strategy 3: fix trailing commas (e.g., {"a":1, "b":2,})
        fixed_trailing = self._fix_trailing_commas(fixed or text)
        try:
            result = json.loads(fixed_trailing)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 4: line-by-line rebuild - find JSON top-level { }, handle internals manually
        result = self._extract_json_by_brace_matching(text)
        if result is not None:
            return result
        
        # Strategy 5: regex extraction based on known keys (ultimate fallback)
        # When SQL DDL has unescaped double quotes (e.g., PG/Oracle "table_name"),
        # all above quote-tracking methods will fail.
        # This method uses our known JSON key name list to extract.
        result = self._extract_by_known_keys(text)
        if result is not None:
            return result
        
        return None
    
    def _fix_json_control_chars(self, text: str) -> str:
        """
        Fix unescaped control characters in JSON string values
        
        LLM often includes raw newlines, tabs, etc. in JSON string values (like SQL DDL),
        which must be escaped as \\n, \\t, etc. in the JSON spec.
        
        Strategy: inside JSON strings (between quotes), replace raw control characters with escape sequences.
        """
        result = []
        in_string = False
        escape_next = False
        i = 0
        
        while i < len(text):
            ch = text[i]
            
            if escape_next:
                result.append(ch)
                escape_next = False
                i += 1
                continue
            
            if ch == '\\' and in_string:
                escape_next = True
                result.append(ch)
                i += 1
                continue
            
            if ch == '"' and not escape_next:
                in_string = not in_string
                result.append(ch)
                i += 1
                continue
            
            if in_string:
                # Inside strings, replace unescaped control characters
                if ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                elif ch == '\x08':
                    result.append('\\b')
                elif ch == '\x0c':
                    result.append('\\f')
                elif ord(ch) < 0x20:
                    # Other control characters use unicode escape
                    result.append(f'\\u{ord(ch):04x}')
                else:
                    result.append(ch)
            else:
                result.append(ch)
            
            i += 1
        
        return ''.join(result)
    
    def _fix_trailing_commas(self, text: str) -> str:
        """Fix trailing commas in JSON."""
        # Remove trailing comma in objects: ,} -> }
        text = re.sub(r',\s*}', '}', text)
        # Remove trailing comma in arrays: ,] -> ]
        text = re.sub(r',\s*]', ']', text)
        return text
    
    def _extract_json_by_brace_matching(self, text: str) -> Optional[Dict]:
        """
        Extract JSON object by brace matching
        
        Handles cases where LLM output has extra text before/after JSON,
        and nested braces in JSON values.
        """
        # Find first {
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                start = i
                break
        
        if start == -1:
            return None
        
        # From start, do brace/bracket/string-aware matching
        depth = 0
        in_string = False
        escape_next = False
        end = -1
        
        for i in range(start, len(text)):
            ch = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if ch == '\\' and in_string:
                escape_next = True
                continue
            
            if ch == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if ch == '{' or ch == '[':
                depth += 1
            elif ch == '}' or ch == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end == -1:
            return None
        
        json_text = text[start:end + 1]
        
        # Try directly first
        try:
            result = json.loads(json_text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try again after fixing control characters
        fixed = self._fix_json_control_chars(json_text)
        fixed = self._fix_trailing_commas(fixed)
        try:
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None

    def _extract_by_known_keys(self, text: str) -> Optional[Dict]:
        """
        Regex extraction based on known keys (ultimate fallback)
        
        When LLM includes unescaped double quotes in JSON values (e.g., PG/Oracle SQL identifier quotes)
        causing all regular parsing methods to fail, use known key list to extract via regex.
        
        Known key set (expansion/verify/reflect three response types):
        - expansion: mysql_ddl, pg_ddl, oracle_ddl, analysis, changes_summary
        - verify: equivalent, confidence, reflection, issues
        - reflect: mysql_ddl, pg_ddl, oracle_ddl, changes_summary
        """
        # All possible top-level keys
        known_keys = [
            "mysql_ddl", "pg_ddl", "oracle_ddl",
            "analysis", "changes_summary",
            "equivalent", "confidence", "reflection", "issues",
        ]
        
        result = {}
        
        # Build match pattern: find value after "key":
        # For each key, find its start position, then determine value range (to next key start or text end)
        key_positions = []
        for key in known_keys:
            # Match "key" : or "key": (allow spaces around key)
            pattern = rf'"{re.escape(key)}"\s*:\s*'
            for match in re.finditer(pattern, text):
                key_positions.append((match.start(), match.end(), key))
        
        if not key_positions:
            return None
        
        # Sort by position
        key_positions.sort(key=lambda x: x[0])
        
        for idx, (key_start, value_start, key) in enumerate(key_positions):
            # Value end position: next key start (back up to comma/whitespace), or text } end
            if idx + 1 < len(key_positions):
                next_key_start = key_positions[idx + 1][0]
                # From next_key_start, look backward, skip commas and whitespace
                raw_value = text[value_start:next_key_start].rstrip()
                # Remove trailing comma
                if raw_value.endswith(','):
                    raw_value = raw_value[:-1].rstrip()
            else:
                # Last key, value up to last }
                end_pos = text.rfind('}')
                if end_pos > value_start:
                    raw_value = text[value_start:end_pos].rstrip()
                    if raw_value.endswith(','):
                        raw_value = raw_value[:-1].rstrip()
                else:
                    raw_value = text[value_start:].rstrip()
            
            # Parse value
            parsed_value = self._parse_raw_value(raw_value, key)
            if parsed_value is not None:
                result[key] = parsed_value
        
        # Must have at least core keys to succeed
        if result:
            # expansion response needs at least mysql_ddl
            if "mysql_ddl" in result:
                return result
            # verify response needs equivalent
            if "equivalent" in result:
                return result
        
        return None
    
    def _parse_raw_value(self, raw_value: str, key: str) -> Any:
        """
        Parse raw value string extracted by position from JSON text
        """
        raw_value = raw_value.strip()
        
        if not raw_value:
            return ""
        
        # Try direct JSON parse (for boolean, number, array, etc.)
        try:
            return json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # For string values: remove surrounding quotes, process content
        if raw_value.startswith('"'):
            # Find real closing quote - search backward from end
            if raw_value.endswith('"') and len(raw_value) > 1:
                inner = raw_value[1:-1]
                # For DDL fields, return content directly (no unescape needed, may have unescaped quotes)
                if key in ("mysql_ddl", "pg_ddl", "oracle_ddl", "analysis",
                           "changes_summary", "reflection"):
                    return inner
                # Other fields try standard unescape
                try:
                    return json.loads(raw_value)
                except (json.JSONDecodeError, ValueError):
                    return inner
        
        # boolean / number quick check
        lower = raw_value.lower()
        if lower == 'true':
            return True
        if lower == 'false':
            return False
        if lower == 'null':
            return None
        try:
            if '.' in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            pass
        
        # Fallback: return as string
        return raw_value

    def _call_expand(self, ddl: str, diffs: List[Dict]) -> Dict:
        """Call LLM to generate tri-dialect DDL (with JSON parse retries)."""
        user_prompt = build_expansion_prompt(ddl, diffs)
        
        total_tokens = 0
        max_parse_retries = 2  # Max 2 retries on JSON parse failure
        
        for parse_attempt in range(max_parse_retries + 1):
            response = self.llm.complete(
                system_prompt=SCHEMA_EXPANSION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            total_tokens += response.total_tokens
            
            result = self._parse_json(response.content)
            if result:
                return result, total_tokens
            
            if parse_attempt < max_parse_retries:
                logger.warning(
                    f"  ⚠️ Expansion result JSON parse failed (attempt {parse_attempt + 1} times)，re-calling LLM..."
                )
                # Emphasize output format on next call
                user_prompt = (
                    f"{build_expansion_prompt(ddl, diffs)}\n\n"
                    "[Important] Please output pure JSON strictly, do not wrap in ```json ``` code blocks,"
                    "do not add any explanation text before/after JSON."
                    "Ensure newlines in JSON string values are represented as \\n, not raw newlines."
                )
        
        # All attempts failed
        content_preview = response.content[:500] if response.content else "(empty)"
        raise ValueError(
            f"Unable to parse expansion result (retried {max_parse_retries} times)。"
            f"first 500 chars of response: {content_preview}"
        )

    def _call_verify(self, mysql_ddl: str, pg_ddl: str, oracle_ddl: str) -> Dict:
        """Call LLM to verify tri-dialect equivalence (with JSON parse retries)."""
        user_prompt = build_verification_prompt(mysql_ddl, pg_ddl, oracle_ddl)
        
        total_tokens = 0
        max_parse_retries = 2
        
        for parse_attempt in range(max_parse_retries + 1):
            response = self.llm.complete(
                system_prompt=VERIFICATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=4096,
            )
            total_tokens += response.total_tokens
            
            result = self._parse_json(response.content)
            if result:
                return result, total_tokens
            
            if parse_attempt < max_parse_retries:
                logger.warning(
                    f"  ⚠️ Verification result JSON parse failed (attempt {parse_attempt + 1} times)，re-calling LLM..."
                )
                user_prompt = (
                    f"{build_verification_prompt(mysql_ddl, pg_ddl, oracle_ddl)}\n\n"
                    "[Important] Please output pure JSON strictly, do not wrap in ```json ``` code blocks."
                )
        
        content_preview = response.content[:500] if response.content else "(empty)"
        raise ValueError(
            f"Unable to parse verification result (retried {max_parse_retries} times)。"
            f"first 500 chars of response: {content_preview}"
        )

    def _call_reflect(
        self, original_ddl: str, diffs: List[Dict],
        mysql_ddl: str, pg_ddl: str, oracle_ddl: str,
        reflection: str,
    ) -> Dict:
        """Call LLM to reflect and correct (with JSON parse retries)."""
        user_prompt = build_reflection_prompt(
            original_ddl, diffs, mysql_ddl, pg_ddl, oracle_ddl, reflection
        )
        
        total_tokens = 0
        max_parse_retries = 2
        
        for parse_attempt in range(max_parse_retries + 1):
            response = self.llm.complete(
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            total_tokens += response.total_tokens
            
            result = self._parse_json(response.content)
            if result:
                return result, total_tokens
            
            if parse_attempt < max_parse_retries:
                logger.warning(
                    f"  ⚠️ Reflection result JSON parse failed (attempt {parse_attempt + 1} times)，re-calling LLM..."
                )
                user_prompt = (
                    f"{build_reflection_prompt(original_ddl, diffs, mysql_ddl, pg_ddl, oracle_ddl, reflection)}\n\n"
                    "[Important] Please output pure JSON strictly, do not wrap in ```json ``` code blocks."
                )
        
        content_preview = response.content[:500] if response.content else "(empty)"
        raise ValueError(
            f"Unable to parse reflection result (retried {max_parse_retries} times)。"
            f"first 500 chars of response: {content_preview}"
        )

    # ------------------------------------------------------------------
    # Single database processing
    # ------------------------------------------------------------------

    def expand_one(self, db_id: str) -> ExpansionResult:
        """
        Process a single database: expand + verify + (optional) reflect and correct
        """
        start_time = time.time()
        total_tokens = 0
        
        logger.info(f"🔧 Processing: {db_id}")
        
        # Read original DDL
        original_ddl = self._read_original_ddl(db_id)
        
        # Get difference requirements
        diffs = self._get_diffs_for_db(db_id)
        if not diffs:
            logger.warning(f"  ⚠️ {db_id}: no allocated differences, skipping")
            return ExpansionResult(
                db_id=db_id, original_ddl=original_ddl,
                mysql_ddl="", pg_ddl="", oracle_ddl="",
                diffs_assigned=[], verification_passed=False,
                error="no_diffs_assigned",
            )
        
        logger.info(f"  📋 Allocated differences: {[d.get('id', '?') for d in diffs]}")
        
        # Phase 1: Expansion
        try:
            expansion, tokens = self._call_expand(original_ddl, diffs)
            total_tokens += tokens
        except Exception as e:
            logger.error(f"  ❌ Expansion failed: {e}")
            return ExpansionResult(
                db_id=db_id, original_ddl=original_ddl,
                mysql_ddl="", pg_ddl="", oracle_ddl="",
                diffs_assigned=diffs, verification_passed=False,
                error=str(e), latency_seconds=time.time() - start_time,
                total_tokens=total_tokens,
            )
        
        mysql_ddl = expansion.get("mysql_ddl", "")
        pg_ddl = expansion.get("pg_ddl", "")
        oracle_ddl = expansion.get("oracle_ddl", "")
        analysis = expansion.get("analysis", "")
        changes = expansion.get("changes_summary", "")
        
        logger.info(f"  ✅ Expansion complete (MySQL: {len(mysql_ddl)} chars, "
                     f"PG: {len(pg_ddl)} chars, Oracle: {len(oracle_ddl)} chars)")
        
        # Phase 2: Verification + reflection loop
        verification_passed = False
        verification_feedback = None
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):  # 0 = first verify, 1..N = retries
            try:
                verify_result, tokens = self._call_verify(mysql_ddl, pg_ddl, oracle_ddl)
                total_tokens += tokens
            except Exception as e:
                logger.warning(f"  ⚠️ Verification call failed: {e}")
                reflection_text = f"Verification call exception: {e}。Please check if DDL syntax is correct."
                if attempt >= self.max_retries:
                    verification_feedback = reflection_text
                    break
                # Do not break, let reflection correction handle it
                retry_count += 1
                logger.info(f"  🔄 Reflection correction (attempt {retry_count} times)...")
                try:
                    reflect_result, tokens = self._call_reflect(
                        original_ddl, diffs,
                        mysql_ddl, pg_ddl, oracle_ddl,
                        reflection_text,
                    )
                    total_tokens += tokens
                    mysql_ddl = reflect_result.get("mysql_ddl", mysql_ddl)
                    pg_ddl = reflect_result.get("pg_ddl", pg_ddl)
                    oracle_ddl = reflect_result.get("oracle_ddl", oracle_ddl)
                    changes = reflect_result.get("changes_summary", changes)
                    if reflect_result.get("analysis"):
                        analysis = reflect_result["analysis"]
                except Exception as e2:
                    logger.warning(f"  ⚠️ Reflection call failed: {e2}")
                    verification_feedback = str(e2)
                    # Do not break immediately, continue if retries remaining
                    if attempt >= self.max_retries - 1:
                        break
                continue
            
            equivalent = verify_result.get("equivalent", False)
            confidence = verify_result.get("confidence", 0)
            reflection_text = verify_result.get("reflection", "")
            issues = verify_result.get("issues", [])
            
            if equivalent and confidence >= 80:
                verification_passed = True
                logger.info(f"  ✅ Verification passed (confidence={confidence})")
                break
            
            # Only retry on error-level issues (compatible with LLM returning string list instead of dict list)
            error_issues = [
                i for i in issues
                if isinstance(i, dict) and i.get("severity") == "error"
            ]
            if not error_issues and confidence >= 70:
                verification_passed = True
                logger.info(f"  ✅ Verification basically passed (confidence={confidence}, no error-level issues)")
                break
            
            if attempt >= self.max_retries:
                verification_feedback = reflection_text
                logger.warning(f"  ⚠️ Verification failed, max retries reached ({self.max_retries})")
                break
            
            # Reflection correction
            retry_count += 1
            logger.info(f"  🔄 Reflection correction (attempt {retry_count} times)...")
            
            try:
                reflect_result, tokens = self._call_reflect(
                    original_ddl, diffs,
                    mysql_ddl, pg_ddl, oracle_ddl,
                    reflection_text,
                )
                total_tokens += tokens
                
                mysql_ddl = reflect_result.get("mysql_ddl", mysql_ddl)
                pg_ddl = reflect_result.get("pg_ddl", pg_ddl)
                oracle_ddl = reflect_result.get("oracle_ddl", oracle_ddl)
                changes = reflect_result.get("changes_summary", changes)
                # If reflection result contains analysis, sync update for consistency
                if reflect_result.get("analysis"):
                    analysis = reflect_result["analysis"]
                
            except Exception as e:
                logger.warning(f"  ⚠️ Reflection call failed: {e}")
                verification_feedback = str(e)
                break
        
        # Phase 3: Real database table creation test (executed regardless of LLM verification, uses real errors to aid correction)
        db_test_passed = False
        db_test_feedback = None
        db_test_retries = 0
        
        if mysql_ddl and pg_ddl and oracle_ddl:
            if verification_passed:
                logger.info(f"  🗄️ Phase 3: Real database table creation test")
            else:
                logger.info(f"  🗄️ Phase 3: Real database table creation test（LLM verification not passed, trying to get real errors）")
            
            for db_attempt in range(self.db_test_max_retries + 1):
                db_test_result = run_db_tests(
                    db_name=db_id,
                    mysql_ddl=mysql_ddl,
                    pg_ddl=pg_ddl,
                    oracle_ddl=oracle_ddl,
                )
                
                if db_test_result.all_passed:
                    db_test_passed = True
                    logger.info(f"  ✅ All DB tests passed")
                    break
                
                # If failed, generate feedback
                feedback = db_test_result.to_feedback()
                failed = db_test_result.failed_dialects
                logger.warning(f"  ⚠️ DB test failed: {failed}")
                
                if db_attempt >= self.db_test_max_retries:
                    db_test_feedback = feedback
                    logger.warning(f"  ⚠️ DB test repair limit reached ({self.db_test_max_retries} times)")
                    break
                
                # Reflection correction: feed real database errors back to LLM
                db_test_retries += 1
                logger.info(f"  🔄 DB test repair (attempt {db_test_retries} times)...")
                
                reflection_text = (
                    "Real database table creation test failed, below are actual execution errors per dialect：\n\n"
                    f"{feedback}\n\n"
                    "Please correct DDL based on the above error info. Note:\n"
                    "- Errors are from real databases, fix strictly according to errors\n"
                    "- Note: foreign key referenced tables must be created first\n"
                    "- Oracle identifier length limit 30 characters\n"
                    "- PostgreSQL does not support MySQL COMMENT 'xxx' inline comment syntax\n"
                    "- Oracle does not support BOOLEAN type, use NUMBER(1)\n"
                    "- Oracle 11g does not support IDENTITY columns"
                )
                
                try:
                    reflect_result, tokens = self._call_reflect(
                        original_ddl, diffs,
                        mysql_ddl, pg_ddl, oracle_ddl,
                        reflection_text,
                    )
                    total_tokens += tokens
                    mysql_ddl = reflect_result.get("mysql_ddl", mysql_ddl)
                    pg_ddl = reflect_result.get("pg_ddl", pg_ddl)
                    oracle_ddl = reflect_result.get("oracle_ddl", oracle_ddl)
                    changes = reflect_result.get("changes_summary", changes)
                    if reflect_result.get("analysis"):
                        analysis = reflect_result["analysis"]
                except Exception as e:
                    logger.warning(f"  ⚠️ DB test repair call failed: {e}")
                    db_test_feedback = str(e)
                    break
        
        latency = time.time() - start_time
        
        result = ExpansionResult(
            db_id=db_id,
            original_ddl=original_ddl,
            mysql_ddl=mysql_ddl,
            pg_ddl=pg_ddl,
            oracle_ddl=oracle_ddl,
            diffs_assigned=diffs,
            verification_passed=verification_passed,
            verification_feedback=verification_feedback,
            db_test_passed=db_test_passed,
            db_test_feedback=db_test_feedback,
            db_test_retries=db_test_retries,
            retry_count=retry_count,
            analysis=analysis,
            changes_summary=changes,
            latency_seconds=round(latency, 1),
            total_tokens=total_tokens,
        )
        
        # Save
        self._save_result(result)
        
        status = "✅" if (verification_passed and db_test_passed) else "⚠️"
        logger.info(f"  {status} complete: {latency:.1f}s, {total_tokens} tokens, "
                     f"LLM retries {retry_count} times, DB repairs {db_test_retries} times")
        
        return result

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_databases(
        self,
        db_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        skip_existing: bool = True,
    ):
        """
        Batch process databases
        
        Args:
            db_ids: specified database list, None means all train databases
            limit: process only first N
            skip_existing: skip databases that already have tri-dialect DDL
        """
        # Determine databases to process
        if db_ids is None:
            db_ids = sorted(self.allocation.keys())
        
        if limit:
            db_ids = db_ids[:limit]
        
        # Skip already processed (only skip those with both verification and DB test passed)
        if skip_existing:
            todo = []
            skipped = 0
            for db_id in db_ids:
                db_dir = self.database_dir / self.split / db_id
                mysql_exists = (db_dir / f"{db_id}_schema_mysql.sql").exists()
                pg_exists = (db_dir / f"{db_id}_schema_pg.sql").exists()
                oracle_exists = (db_dir / f"{db_id}_schema_oracle.sql").exists()
                
                if mysql_exists and pg_exists and oracle_exists:
                    # Also need to check if verification and DB test both passed in metadata
                    meta_path = db_dir / f"{db_id}_expansion_meta.json"
                    should_skip = False
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            if meta.get("verification_passed") and meta.get("db_test_passed"):
                                should_skip = True
                        except (json.JSONDecodeError, IOError):
                            pass  # metadata corrupted, do not skip
                    
                    if should_skip:
                        logger.debug(f"Skip already processed (all passed): {db_id}")
                        skipped += 1
                    else:
                        logger.debug(f"Reprocess (not all passed before): {db_id}")
                        todo.append(db_id)
                else:
                    todo.append(db_id)
            logger.info(f"Skipped already processed: {skipped} , to process: {len(todo)} ")
            db_ids = todo
        
        logger.info(f"=" * 60)
        logger.info(f"Start processing {len(db_ids)} databases (concurrency={self.concurrent_batch_size})")
        logger.info(f"=" * 60)
        
        # Thread-safe counter
        completed_count = [0]  # Use list to avoid nonlocal
        results_lock = threading.Lock()
        
        def _process_one(db_id: str) -> ExpansionResult:
            """Process single database within thread"""
            try:
                result = self.expand_one(db_id)
            except Exception as e:
                logger.error(f"  ❌ Processing exception: {db_id}: {e}", exc_info=True)
                result = ExpansionResult(
                    db_id=db_id, original_ddl="",
                    mysql_ddl="", pg_ddl="", oracle_ddl="",
                    diffs_assigned=[], verification_passed=False,
                    error=str(e),
                )
            
            with results_lock:
                self.results.append(result)
                completed_count[0] += 1
                cnt = completed_count[0]
            
            logger.info(f"[completed {cnt}/{len(db_ids)}] {db_id}")
            return result
        
        # Batch concurrent processing
        batch_size = self.concurrent_batch_size
        total = len(db_ids)
        
        for batch_start in range(0, total, batch_size):
            batch = db_ids[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, total)
            logger.info(f"\n{'='*40}")
            logger.info(f"📦 Batch {batch_start//batch_size + 1}: "
                         f"[{batch_start+1}-{batch_end}/{total}] ({len(batch)} )")
            logger.info(f"{'='*40}")
            
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(_process_one, db_id): db_id
                    for db_id in batch
                }
                
                for future in as_completed(futures):
                    db_id = futures[future]
                    try:
                        future.result()  # Exception already caught in _process_one
                    except Exception as e:
                        logger.error(f"  ❌ Unexpected exception: {db_id}: {e}", exc_info=True)
            
            # Save progress after each batch
            self._save_progress()
        
        # Final report
        self._save_progress()
        self._print_summary()

    def _save_progress(self):
        """Save processing progress."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        progress = {
            "total_processed": len(self.results),
            "llm_passed": sum(1 for r in self.results if r.verification_passed),
            "db_test_passed": sum(1 for r in self.results if r.db_test_passed),
            "both_passed": sum(1 for r in self.results if r.verification_passed and r.db_test_passed),
            "failed": sum(1 for r in self.results if not (r.verification_passed and r.db_test_passed)),
            "errors": sum(1 for r in self.results if r.error),
            "total_tokens": sum(r.total_tokens for r in self.results),
            "databases": {
                r.db_id: {
                    "llm_passed": r.verification_passed,
                    "db_test_passed": r.db_test_passed,
                    "retries": r.retry_count,
                    "db_test_retries": r.db_test_retries,
                    "error": r.error,
                    "tokens": r.total_tokens,
                }
                for r in self.results
            },
        }
        
        progress_path = self.output_dir / "progress.json"
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _print_summary(self):
        """Print processing summary."""
        total = len(self.results)
        llm_passed = sum(1 for r in self.results if r.verification_passed)
        db_passed = sum(1 for r in self.results if r.db_test_passed)
        both_passed = sum(1 for r in self.results if r.verification_passed and r.db_test_passed)
        errors = sum(1 for r in self.results if r.error)
        total_tokens = sum(r.total_tokens for r in self.results)
        total_time = sum(r.latency_seconds for r in self.results)
        avg_retries = (
            sum(r.retry_count for r in self.results) / total if total > 0 else 0
        )
        avg_db_retries = (
            sum(r.db_test_retries for r in self.results) / total if total > 0 else 0
        )
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Schema expansion summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Total processed: {total}")
        if total > 0:
            logger.info(f"  ✅ LLM Verification passed: {llm_passed} ({llm_passed/total*100:.1f}%)")
            logger.info(f"  ✅ DB test passed: {db_passed} ({db_passed/total*100:.1f}%)")
            logger.info(f"  ✅ All passed: {both_passed} ({both_passed/total*100:.1f}%)")
        logger.info(f"  ❌ Errors: {errors}")
        logger.info(f"  Avg LLM retries: {avg_retries:.2f} times")
        logger.info(f"  Avg DB repairs: {avg_db_retries:.2f} times")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
        if total > 0:
            logger.info(f"  Avg time: {total_time/total:.1f}s/db")
        logger.info(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Tri-dialect Schema expander - SQLite -> MySQL + PG + Oracle"
    )
    parser.add_argument(
        "--split", "-s", type=str, default="train",
        choices=["train", "dev", "test"],
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--databases", "-d", nargs="+", default=None,
        help="Process only specified databases (space-separated)"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Process only first N databases"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Do not skip already processed databases (skip by default)"
    )
    parser.add_argument(
        "--allocate-first", action="store_true",
        help="Run difference allocator first before expansion"
    )
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / "config" / "schema_synthesis.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Determine allocation file path
    output_dir = PROJECT_ROOT / config["expansion"]["output_dir"]
    allocation_path = output_dir / f"allocation_{args.split}.json"
    
    # Optional: run allocator first
    if args.allocate_first or not allocation_path.exists():
        logger.info(f"Running difference allocator (split={args.split})...")
        from schema_synthesis.diff_allocator import (
            load_differences, load_split_databases, allocate_differences, save_allocation
        )
        alloc_cfg = config["allocation"]
        differences, _ = load_differences(alloc_cfg["differences_file"])
        split_dbs = load_split_databases(alloc_cfg["split_info_file"], split=args.split)
        
        # Different splits use different random seeds
        base_seed = alloc_cfg["random_seed"]
        split_seed_offsets = {"train": 0, "dev": 1000, "test": 2000}
        random_seed = base_seed + split_seed_offsets.get(args.split, 0)
        
        alloc_result = allocate_differences(
            differences=differences,
            train_dbs=split_dbs,
            diffs_per_db=alloc_cfg["diffs_per_db"],
            min_coverage=alloc_cfg["min_coverage"],
            random_seed=random_seed,
        )
        save_allocation(alloc_result, config["expansion"]["output_dir"], split=args.split)
    
    # Load allocation results
    logger.info(f"Load allocation results: {allocation_path}")
    with open(allocation_path, "r", encoding="utf-8") as f:
        alloc_data = json.load(f)
    
    allocation = alloc_data["allocation"]
    
    # Load full difference details (including test_requirements)
    diff_file = PROJECT_ROOT / config["allocation"]["differences_file"]
    with open(diff_file, "r", encoding="utf-8") as f:
        diff_data = json.load(f)
    differences_detail = {d["id"]: d for d in diff_data.get("differences", [])}
    
    # Create expander
    expander = SchemaExpander(
        config=config,
        allocation=allocation,
        differences_detail=differences_detail,
        split=args.split,
    )
    
    # Execute
    expander.process_databases(
        db_ids=args.databases,
        limit=args.limit,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
