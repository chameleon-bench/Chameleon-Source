"""
LLM Data Generator - Business operator mode.

Core idea:
  Let LLM play a "database business operator" role. Given the full DDL, LLM analyzes the schema itself,
  simulates K independent business operation scenarios, and decides which tables each operation involves and what data to insert.

Flow (new version - single call pattern):
  1. Pass in DDL + SQLite sample data + entity pool + data_requirements
  2. Single API call (M=1, N=1), LLM generates K=50 business operations
  3. Parse JSON + lightweight post-processing (FK cleanup + UNIQUE dedup)

Data sources:
  - DDL: database CREATE TABLE statements
  - Sample data: 2 rows per table from SQLite file (from SynSQL-2.5M)
  - Entity pool: EntityHarvester searches and collects real entities
  - data_requirements: extracted from allocated diffs
"""

import json
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_synthesis.schema_parser import (
    SchemaInfo, DualSchemaInfo, TableInfo, ColumnInfo,
)
from llm.client import LLMClient
from utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# System Prompt — 「business operator」role
# =============================================================================

BUSINESS_OPERATOR_SYSTEM_PROMPT = """You are an experienced database business operator. You have a real database system in front of you.

## Your Job

You will execute {K} independent business operations. Each operation simulates a real business scenario, inserting real data.

## Working Method

1. First read the entire DDL, understand this database's business domain and table relationships
2. Analyze data requirements, reasonably set up business operation logic, satisfy data requirements across multiple business operations
2. Then sequentially execute {K} business operations, each time:
   - Imagine a concrete business scenario
   - Decide which tables need data insertion
   - Generate real data that satisfies foreign key constraints
   - Keep consistent data style across the same column
3. Operations can reference each other — later operations can reference documents created by earlier operations

## Output Format

Output a JSON array of length {K}. Each element represents one business operation, is an object, key is table name, value is the rows to insert for that table:

[
  {{
    "_scenario": "describe this business operation",
    "table_A": [{{"id": 101, "col1": "value1", "col2": "value2"}}],
    "table_B": [{{"id": 301, "ref_id": 101, "date": "2024-03-15"}}]
  }},
  {{
    "_scenario": "another business operation",
    "table_C": [
      {{"id": 50, "name": "xxx", "ref_id": 3}},
      {{"id": 51, "name": "yyy", "ref_id": 3}}
    ]
  }}
]

The "_scenario" field uses one sentence to describe what this business operation is (ignored when parsing, but helps you keep your thinking clear).

## Key Rules
1. **Rule constraints**: When simulating business operations, refer to the given data requirements, arrange business operations according to data requirements, satisfy data requirements in the final data
2. **Semantic authenticity**: Data must look like it comes from a real system. Names should mimic real names, book titles should mimic real book titles, prices should be in reasonable ranges.
   **Placeholder data strictly prohibited!!!** The following patterns must never appear:
   - ❌ Email: @example.com, @test.com, @dummy.com → ✅ should use @gmail.com, @outlook.com, @company.com etc. real domains
   - ❌ URL: https://example.com, http://test.com → ✅ should use https://cdn.unsplash.com/photo-xxx, https://images.pexels.com/photos/xxx etc. real format
   - ❌ Hash: hash1, hash2, hash3 → ✅ should use 32/64-bit hex like a3f2b7c9d1e4f5a6b7c8d9e0f1a2b3c4
   - ❌ Phone: 000-0000, 123-4567 → ✅ should use (555) 123-4567, +1-555-987-6543 etc. real format
   - ❌ IP: 0.0.0.0, 1.1.1.1 → ✅ should use 192.168.1.105, 10.0.3.42 etc. real format
   **Note**: The sample data may contain the above placeholders (like @example.com), that is a raw data defect, you must generate real data, no need to imitate placeholder style!
3. **Foreign key consistency**: Referenced foreign key values must exist — either in "existing data" or created by earlier operations in this batch of K operations.
4. **Each operation is an independent scenario**: Each operation simulates a different business event, involving different table combinations.
5. **Coverage**: Each operation should cover as many tables as possible to make synthesized data more comprehensive. K operations combined should cover all tables in the database.
6. **Diversity**:
   - Names, organization names, meeting names, locations etc. entities should be diverse
   - Dates should cover different years and months, numeric ranges should be diverse
   - Numeric type data should be diverse and conform to real business specifications
   - **Carefully read the "existing data" section, avoid generating duplicate names and entities**
7. **ID rules**: New inserted records should start from each table's current max ID + 1 incrementally, no need to conflict with existing data.
8. **Data completeness**: Core identifier fields of each record **must have values**, never allow null or empty strings. Auxiliary fields (like description, note, URL etc.) can occasionally be null to simulate real data.
9. **Only output JSON array**, do not output any comments, explanations, or markdown wrapping."""


# =============================================================================
# LLM Data Generator - business operator mode
# =============================================================================

class LLMDataGenerator:
    """
    LLM-driven data generation engine - business operator mode.

    Pass in DDL + sample data + entity pool + data_requirements,
    LLM plays the business operator role, one call simulates K business operations.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "aliyun",
        model: str = "qwen3.5-flash",
        seed: int = 42,
        scenarios_per_completion: int = 50,
        temperature: float = 0.9,
        max_tokens: int = 20480,
        # The following parameter is kept but has no actual meaning, only for compatible synthesizer arguments
        constrained_multiplier: int = 1,
        natural_multiplier: int = 0,
        num_completions: int = 1,
        concurrent_batch_size: int = 1,
    ):
        """
        Args:
            llm_client: LLM client (optional, reuse if passed in)
            provider: LLM provider
            model: LLM model
            seed: Random seed
            scenarios_per_completion: Number of business operations per call (K)
            temperature: LLM temperature
            max_tokens: Max token count
        """
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(provider=provider, model=model)

        self.seed = seed
        self.scenarios_per_completion = scenarios_per_completion
        self.temperature = temperature
        self.max_tokens = max_tokens

        random.seed(seed)

        # alreadygenerate datastore: {table_name: [row_dict, ...]}
        self.generated_data: Dict[str, List[Dict]] = {}

        # Global ID tracker: {table_name: current_max_id}
        self.max_id_tracker: Dict[str, int] = {}

        # primary keydeduplicateindex: {table_name: set_of_pk_tuples}
        self.pk_index: Dict[str, set] = {}

        # contentdeduplicateindex: {table_name: {field_name: set_of_values}}
        self.content_index: Dict[str, Dict[str, set]] = {}

        # statistics info
        self.stats: Dict[str, Any] = {}

    # =========================================================================
    # main entry
    # =========================================================================

    def generate_all(
        self,
        schema: SchemaInfo,
        row_counts: Optional[Dict[str, int]] = None,
        selected_constraints: Optional[List[Dict]] = None,
        constraint_map: Optional[Dict[str, Dict]] = None,
        entity_pool: Optional[Dict[str, List[Dict]]] = None,
        seed_data: Optional[Dict[str, List[Dict]]] = None,
        data_requirements_text: Optional[str] = None,
        pg_schema: Optional['SchemaInfo'] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Generate data for all tables in entire database — single LLM call

        Args:
            schema: database schema information (MySQL)
            entity_pool: entity pool {table_name: [entity_dict, ...]}
            seed_data: SQLite sample data {table_name: [row_dict, ...]}
            data_requirements_text: from allocation diffs extractdatarequirementtext
            pg_schema: PG schema info, for cross-dialect type annotation (like JSON→TEXT[] mapping hint)
            (remaining parameters are legacy interface compatibility, can ignore)

        Returns:
            {table_name: [row_dict, ...]}
        """
        db_name = schema.database_name
        K = self.scenarios_per_completion

        logger.info(f"LLM datagenerate: {db_name} (K={K})")
        logger.info(f"  Table count: {len(schema.tables)}")

        self.generated_data = {t.name: [] for t in schema.tables}
        self.max_id_tracker = {t.name: 0 for t in schema.tables}
        self.pk_index = {t.name: set() for t in schema.tables}
        self.content_index = {t.name: {} for t in schema.tables}
        self.entity_pool = entity_pool or {}
        self.seed_data = seed_data or {}
        self.data_requirements_text = data_requirements_text or ""
        self.pg_schema = pg_schema
        self.stats = {"total_api_calls": 0, "total_tokens": 0, "total_cost": 0.0}

        if self.entity_pool:
            pool_summary = {k: len(v) for k, v in self.entity_pool.items()}
            logger.info(f"  entity pool: {pool_summary}")

        if self.seed_data:
            seed_summary = {k: len(v) for k, v in self.seed_data.items()}
            logger.info(f"  Sample data: {seed_summary}")

        # build prompt
        ddl_text = self._build_ddl_text(schema)

        if self.data_requirements_text:
            constraint_text = self.data_requirements_text
        else:
            constraint_text = self._build_constraint_text(
                selected_constraints or [], constraint_map or {}
            )

        system_prompt = BUSINESS_OPERATOR_SYSTEM_PROMPT.format(K=K)
        user_prompt = self._build_user_prompt(
            schema=schema,
            ddl_text=ddl_text,
            constraint_text=constraint_text if constraint_text else None,
        )

        # === Single LLM call ===
        start_time = time.time()
        logger.info(f"  Calling LLM (K={K} business operations)...")

        try:
            responses = self.llm_client.complete_n(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"  LLM call failed: {e}")
            return self.generated_data

        self.stats["total_api_calls"] = 1

        # parseresponse
        total_operations = 0
        total_rows_added = 0

        for i, response in enumerate(responses):
            operations = self._parse_operations_response(response.content)
            if not operations:
                logger.warning(f"  completion {i+1}: parse failedoris empty")
                continue

            rows_added = self._merge_operations(operations, schema)
            total_operations += len(operations)
            total_rows_added += rows_added

            logger.info(
                f"  completion {i+1}: {len(operations)} operations, "
                f"+{rows_added} rows, {response.total_tokens} tokens"
            )

        call_tokens = sum(r.total_tokens for r in responses)
        call_cost = sum(r.cost for r in responses)
        self.stats["total_tokens"] = call_tokens
        self.stats["total_cost"] = call_cost

        elapsed = time.time() - start_time
        logger.info(
            f"  LLM call complete: {total_operations} operations, +{total_rows_added} rows, "
            f"{elapsed:.1f}s, ${call_cost:.4f}"
        )

        # === backprocessing: FK clean + UNIQUE deduplicate ===
        self._fix_all_foreign_keys(schema)
        self._fix_unique_columns(schema)
        self._fix_all_foreign_keys(schema)

        # === Final statistics ===
        total_rows = sum(len(rows) for rows in self.generated_data.values())
        logger.info(
            f"LLM data generation complete: {db_name}, total {total_rows} rows, "
            f"${self.stats['total_cost']:.4f}"
        )

        return self.generated_data

    # =========================================================================
    # Prompt build
    # =========================================================================

    def _build_ddl_text(self, schema: SchemaInfo) -> str:
        """Rebuild DDL text from SchemaInfo (simplified version).

        When self.pg_schema exists, for PG-side ARRAY type columns attach cross-dialect mapping comment,
        hint LLM to generate JSON array instead of JSON object (like MySQL JSON → PG TEXT[]).
        """
        # Build PG column name→type mapping (for cross-dialect type annotation)
        pg_col_type_map: Dict[str, Dict[str, str]] = {}  # {table_name_lower: {col_name_lower: pg_data_type}}
        if getattr(self, 'pg_schema', None):
            for pg_table in self.pg_schema.tables:
                table_map = {}
                for col in pg_table.columns:
                    table_map[col.name.lower()] = col.data_type
                pg_col_type_map[pg_table.name.lower()] = table_map

        lines = []
        for table in schema.tables:
            lines.append(f"CREATE TABLE {table.name} (")
            col_defs = []
            for col in table.columns:
                flags = []
                if col.is_primary_key:
                    flags.append("PRIMARY KEY")
                if col.is_auto_increment:
                    flags.append("AUTO_INCREMENT")
                if not col.is_nullable and not col.is_primary_key:
                    flags.append("NOT NULL")
                if col.default_value:
                    flags.append(f"DEFAULT {col.default_value}")
                flag_str = " " + " ".join(flags) if flags else ""
                # Wrap column name with backticks when it contains spaces or special chars
                col_name = f"`{col.name}`" if ' ' in col.name or not col.name.replace('_', '').isalnum() else col.name

                # Display ENUM info: if column has allowed_values, attach comment to hint LLM
                enum_hint = ""
                if col.allowed_values:
                    vals = ", ".join(f"'{v}'" for v in col.allowed_values)
                    enum_hint = f"  /* ENUM: only allowed values: ({vals}) */"

                # === Cross-dialect type annotation: PG ARRAY mapping ===
                # When MySQL JSON/VARCHAR column should be ARRAY type on PG side,
                # hint LLM must generate JSON array (instead of JSON object)
                cross_dialect_hint = ""
                pg_table_map = pg_col_type_map.get(table.name.lower(), {})
                pg_type = pg_table_map.get(col.name.lower())
                if pg_type and "[]" in pg_type.upper():
                    array_base = re.sub(r'\[\]$', '', pg_type.upper())
                    if col.is_json_type():
                        # MySQL JSON → PG ARRAY: must generate JSON array
                        cross_dialect_hint = (
                            f"  /* PGmapping={pg_type}, "
                            f"must generate JSON array like [\"a\",\"b\",\"c\"], "
                            f"do not generate JSON objects */"
                        )
                    else:
                        cross_dialect_hint = f"  /* PG mapping={pg_type}, please generate array format data */"

                col_defs.append(f"  {col_name} {col.data_type}{flag_str}{enum_hint}{cross_dialect_hint}")

            for fk in table.foreign_keys:
                col_defs.append(
                    f"  FOREIGN KEY ({', '.join(fk.columns)}) "
                    f"REFERENCES {fk.ref_table}({', '.join(fk.ref_columns)})"
                )

            lines.append(",\n".join(col_defs))
            lines.append(");\n")

        return "\n".join(lines)

    def _build_existing_data_summary(self, schema: SchemaInfo) -> str:
        """Build existing data primary key summary + existing name samples (prevent duplicate generation)."""
        lines = []
        for table in schema.tables:
            rows = self.generated_data.get(table.name, [])
            if not rows:
                tracked_max = self.max_id_tracker.get(table.name, 0)
                if tracked_max > 0:
                    pk_cols = table.primary_key
                    pk_col = pk_cols[0] if pk_cols else "Id"
                    lines.append(
                        f"  {table.name}: {len(rows)} rows, "
                        f"max({pk_col})={tracked_max}, new data starts from {tracked_max + 1}"
                    )
                else:
                    lines.append(f"  {table.name}: nodata (from ID=1 start)")
                continue

            # Use max_id_tracker instead of scanning rows to get max_id
            pk_cols = table.primary_key
            if pk_cols:
                pk_col = pk_cols[0]
                tracked_max = self.max_id_tracker.get(table.name, 0)
                if tracked_max > 0:
                    lines.append(
                        f"  {table.name}: {len(rows)} rows, "
                        f"max({pk_col})={tracked_max}, new data starts from {tracked_max + 1}"
                    )
                else:
                    pk_values = [r.get(pk_col) for r in rows if r.get(pk_col) is not None]
                    if pk_values:
                        max_id = max(pk_values) if all(isinstance(v, (int, float)) for v in pk_values) else len(rows)
                        lines.append(
                            f"  {table.name}: {len(rows)} rows, "
                            f"max({pk_col})={max_id}, new data starts from {max_id + 1}"
                        )
                    else:
                        lines.append(f"  {table.name}: {len(rows)} rows")
            else:
                lines.append(f"  {table.name}: {len(rows)} rows")

            # Attach existing name/value samples — let LLM see which values already exist, avoid duplicates
            name_sample = self._sample_existing_values(table, rows)
            if name_sample:
                lines.append(f"    Existing values (do not duplicate): {name_sample}")

        return "\n".join(lines)

    def _sample_existing_values(self, table: TableInfo, rows: List[Dict], max_show: int = 30) -> str:
        """
        Sample key text field values from existing rows, to tell LLM not to duplicate

        prioritysample Name、ShortName、Title etc.semantickeywordsegment
        """
        # Define field names to sample (priority high to low)
        key_field_names = [
            'Name', 'name', 'ShortName', 'short_name', 'shortname',
            'Title', 'title', 'FullName', 'full_name', 'fullname',
        ]

        # Find existing key fields in this table
        col_names = {c.name for c in table.columns}
        target_fields = [f for f in key_field_names if f in col_names]

        if not target_fields:
            return ""

        # Collect unique values for each key field
        parts = []
        for field_name in target_fields[:2]:  # Show at most 2 fields
            values = set()
            for row in rows:
                v = row.get(field_name)
                if v is not None and isinstance(v, str) and v.strip():
                    values.add(v.strip())

            if values:
                sorted_vals = sorted(values)
                if len(sorted_vals) > max_show:
                    shown = sorted_vals[:max_show]
                    parts.append(
                        f"{field_name}: [{', '.join(repr(v) for v in shown)}...total {len(sorted_vals)}]"
                    )
                else:
                    parts.append(
                        f"{field_name}: [{', '.join(repr(v) for v in sorted_vals)}]"
                    )

        return "; ".join(parts)

    def _build_constraint_text(
        self,
        selected_constraints: List[Dict],
        constraint_map: Dict[str, Dict],
    ) -> str:
        """buildconstraintdescribetext"""
        if not selected_constraints:
            return ""

        lines = []
        for sel in selected_constraints:
            cid = sel.get("constraint_id", "")
            c_def = constraint_map.get(cid, {})
            constraint_desc = c_def.get("constraint", cid)
            examples = c_def.get("examples", [])

            lines.append(f"- [{cid}] {constraint_desc}")

            if examples:
                if isinstance(examples, dict):
                    for k, v in examples.items():
                        if isinstance(v, list):
                            lines.append(
                                f"  {k} example: {json.dumps(v[:4], ensure_ascii=False, default=str)}"
                            )
                elif isinstance(examples, list):
                    lines.append(
                        f"  example: {json.dumps(examples[:6], ensure_ascii=False, default=str)}"
                    )

        return "\n".join(lines)

    def _build_entity_pool_text(self, schema: SchemaInfo) -> str:
        """buildentity pool Prompt text

        Each call randomly samples a different batch, ensuring each API call sees different style references.
        This is the diversity key — 16 API calls × different batch each time → LLM imitates different styles.
        """
        if not self.entity_pool:
            return ""

        lines = []
        for table in schema.tables:
            entities = self.entity_pool.get(table.name, [])
            if not entities:
                continue

            # Each time randomly sample 10 (not first 30, but random 10)
            # Fewer but different each time → much better than more but same each time
            sample_size = min(15, len(entities))
            sample = random.sample(entities, sample_size)

            lines.append(f"### {table.name} (random sample {len(sample)}/{len(entities)} as style reference)")
            for entity in sample:
                compact = {k: v for k, v in entity.items() if v is not None}
                lines.append(f"  {json.dumps(compact, ensure_ascii=False)}")

        return "\n".join(lines) if lines else ""

    def _build_user_prompt(
        self,
        schema: SchemaInfo,
        ddl_text: str,
        constraint_text: Optional[str],
        focus_tables: Optional[List[str]] = None,
    ) -> str:
        """build user prompt"""
        K = self.scenarios_per_completion

        parts = []

        # 1. DDL
        parts.append("## database DDL\n")
        parts.append(ddl_text)

        # 2. SQLite sample data (if available)
        seed_data = getattr(self, 'seed_data', {})
        if seed_data:
            parts.append("\n## Data format examples (2 rows per table, reference format and value style)\n")
            for table_name, rows in seed_data.items():
                if rows:
                    parts.append(f"### {table_name} ({len(rows)} rows)")
                    for row in rows:
                        compact = {k: v for k, v in row.items() if v is not None}
                        parts.append(f"  {json.dumps(compact, ensure_ascii=False, default=str)}")
            parts.append(
                "\nNote: The above is sample data, can be used as data style and content reference. Your new data can reference and use content from sample data. Some columns in sample data may be missing, but your generated new data should cover all columns in the table."
            )

        # 3. Existing data summary
        parts.append("\n## Existing data (current status per table)\n")
        parts.append(self._build_existing_data_summary(schema))
        parts.append("\n(New data should start from max ID + 1 per table, no need to conflict with existing data)")

        # 4. Entity style reference (if entity pool available)
        entity_pool_text = self._build_entity_pool_text(schema)
        if entity_pool_text:
            parts.append(f"\n## Real data style reference (for naming style reference only, lower priority than data requirements)\n")
            parts.append(
                "The following are real data samples searched from the internet, please reference their **naming style and format** to generate data, and can also use these real data for synthesis."
                "- attention：ifdownsurface「datarequirement」conflicts with style reference，**datarequirementpriority**\n"
            )
            parts.append(entity_pool_text)

        # 5. Data requirements (data_requirements / constraints) — highest priority
        if constraint_text:
            parts.append(f"\n## Data requirements (!!!highest priority, must satisfy!!!)\n")
            parts.append(
                f"The following data requirements support subsequent SQL dialect translation testing."
                f"Please naturally distribute and satisfy these requirements across {K} business operations:\n"
            )
            parts.append(constraint_text)

        # 6. Focus tables (supplementary rounds)
        if focus_tables:
            parts.append(f"\n## Focus\n")
            parts.append(
                f"The following tables still have insufficient data, please focus on involving these tables in business operations:\n"
                f"{', '.join(focus_tables)}"
            )

        # 7. Execution instructions
        parts.append(f"\n## Please execute {K} business operations\n")
        parts.append(
            f"Please simulate {K} different business scenarios, output JSON array (length {K}).\n"
            f"Each operation can involve 1 to multiple tables, you decide based on business scenario.\n"
            f"Ensure cross-table foreign key references are correct. Only output JSON array, no other content.\n\n"
            f"**Diversity reminder**: Carefully read the existing values list in the existing data section above, "
            f"every new name/title/abbreviation you generate must be different from existing values."
        )

        return "\n".join(parts)

    # =========================================================================
    # responseparse
    # =========================================================================

    def _parse_operations_response(self, content: str) -> List[Dict]:
        """
        Parse LLM output business operation JSON array

        Each element is one business operation:
        {
            "_scenario": "...",
            "table_name": [row_dict, ...],
            ...
        }

        Enhanced: support truncated JSON fix (complete unclosed brackets)
        """
        # Quick return for null content (DeepSeek thinking mode may exhaust tokens leaving content empty)
        if not content or not content.strip():
            logger.warning("    LLM returned null content (may be thinking mode exhausted token budget, no actual output)")
            return []

        json_str = content.strip()

        # Remove DeepSeek <think>...</think> chain-of-thought content
        if '<think>' in json_str:
            think_end = json_str.find('</think>')
            if think_end != -1:
                json_str = json_str[think_end + len('</think>'):].strip()
            else:
                # <think> not closed, output may be truncated inside chain-of-thought
                # Try to find JSON after chain-of-thought
                think_start = json_str.find('<think>')
                after_think = json_str[think_start:]
                # Find first [ starting position
                bracket_pos = after_think.find('[')
                if bracket_pos != -1:
                    json_str = after_think[bracket_pos:]
                else:
                    logger.error("    LLM output truncated inside <think> chain-of-thought, no actual JSON content")
                    return []

        # Remove markdown code block
        if '```json' in json_str:
            json_str = json_str.split('```json')[1]
            json_str = json_str.split('```')[0]
        elif '```' in json_str:
            parts = json_str.split('```')
            if len(parts) >= 3:
                json_str = parts[1]
            elif len(parts) >= 2:
                json_str = parts[1]

        json_str = json_str.strip()

        # Try direct parse
        data = self._try_parse_json(json_str, "direct parse")
        if data is not None:
            return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Try cleaning invalid control chars (LLM sometimes outputs \x00-\x1f etc. invalid chars)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)
        if cleaned != json_str:
            data = self._try_parse_json(cleaned, "clean control chars")
            if data is not None:
                return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Fix leading zero numbers (LLM sometimes generates 060000 such invalid JSON numbers)
        # Match: colon/comma/[ followed by whitespace + 0-leading multi-digit number (not inside quotes)
        cleaned = re.sub(r'(?<=[\s:,\[])0+(\d+)(?=[,\s\]\}])', r'\1', cleaned)
        data = self._try_parse_json(cleaned, "fix leading zero numbers")
        if data is not None:
            return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Try cleaning invalid escapes then parse
        sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
        data = self._try_parse_json(sanitized, "fix invalid escapes")
        if data is not None:
            return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Try extracting JSON array
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            extracted = match.group()
            cleaned_ext = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', extracted)
            sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned_ext)
            data = self._try_parse_json(sanitized, "extract array")
            if data is not None:
                return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Try truncated fix: complete unclosed brackets
        data = self._try_repair_truncated_json(json_str)
        if data is not None:
            return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Try per-object extraction: when overall parse fails, extract top-level {...} objects one by one
        # Suitable for when LLM generates invalid JSON in some objects (like unescaped quotes)
        recovered = self._try_extract_objects(json_str)
        if recovered:
            logger.info(f"    Per-object extraction recovered {len(recovered)} operations (overall parse failed)")
            return recovered

        # Last attempt: clean control chars + truncated fix
        if cleaned != json_str:
            data = self._try_repair_truncated_json(cleaned)
            if data is not None:
                return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])

        # Print specific error to help debug
        try:
            json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(
                f"    JSON parse completely failed, content length={len(content)}, "
                f"Specific error: {e.msg} (position {e.pos}), "
                f"Last 50 chars: {repr(content[-50:])}, "
                f"First 100 chars: {repr(content[:100])}"
            )
        else:
            logger.error(
                f"    JSON parse completely failed, content length={len(content)}, "
                f"Last 50 chars: {repr(content[-50:])}, "
                f"First 100 chars: {repr(content[:100])}"
            )
        return []

    def _try_parse_json(self, text: str, label: str = "") -> Optional[Any]:
        """Safely try to parse JSON, return None on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            if label:
                logger.debug(f"    JSON parse failed [{label}]: {e}")
            return None

    def _try_repair_truncated_json(self, text: str) -> Optional[List]:
        """
        Fix truncated JSON (frequently occurs when LLM output exceeds max_tokens)
        Strategy: start from the end, progressively delete incomplete content, complete brackets
        """
        text = text.strip()
        if not text.startswith('['):
            return None

        # Count unclosed brackets
        for attempt in range(20):
            # Find the last complete } or ] from the end
            truncated = text
            for _ in range(attempt):
                # Delete trailing incomplete chars
                last_brace = max(truncated.rfind('}'), truncated.rfind(']'))
                if last_brace <= 0:
                    break
                truncated = truncated[:last_brace + 1]

            # Complete unclosed brackets
            open_brackets = 0
            open_braces = 0
            in_string = False
            escape = False

            for ch in truncated:
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '[':
                    open_brackets += 1
                elif ch == ']':
                    open_brackets -= 1
                elif ch == '{':
                    open_braces += 1
                elif ch == '}':
                    open_braces -= 1

            # Complete
            repair = truncated
            repair += '}' * max(0, open_braces)
            repair += ']' * max(0, open_brackets)

            try:
                data = json.loads(repair)
                if isinstance(data, list) and len(data) > 0:
                    logger.info(
                        f"    truncate JSON repair succeeded: resume {len(data)} operations "
                        f"(deleted {attempt} segments, not full content)"
                    )
                    return data
            except json.JSONDecodeError:
                continue

        return None

    def _try_extract_objects(self, text: str) -> Optional[List]:
        """
        Per-object extraction: when overall JSON array parse fails,
        try extracting top-level {...} objects one by one and parse individually.
        
        Typical scenario: LLM writes unescaped quotes in the 30th object,
        causing the entire group parse to fail, but the first 29 and later objects are all valid.
        """
        results = []
        depth = 0
        in_string = False
        escape = False
        obj_start = None
        
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    obj_str = text[obj_start:i + 1]
                    try:
                        obj = json.loads(obj_str)
                        if isinstance(obj, dict) and '_scenario' in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        # Try fixing leading zero numbers then retry
                        try:
                            fixed = re.sub(r'(?<=[\s:,\[])0+(\d+)(?=[,\s\]\}])', r'\1', obj_str)
                            obj = json.loads(fixed)
                            if isinstance(obj, dict) and '_scenario' in obj:
                                results.append(obj)
                        except json.JSONDecodeError:
                            # This object indeed has issues, skip
                            pass
                    obj_start = None
        
        return results if results else None

    # =========================================================================
    # datamerging
    # =========================================================================

    def _merge_operations(
        self,
        operations: List[Dict],
        schema: SchemaInfo,
    ) -> int:
        """
        Merge parsed business operation data into generated_data

        Core improvement: ID remapping + FK cascade + PK dedup
        -----------------------------------------------
        Multiple operations within one completion share a local ID space。
        Multiple completions from the same API call may each start from ID=1 ("parallel universe" issue).
        This method:
          1. Scan all table→rows within completion, for each single-column integer PK table establish
             old_id → new_id mapping (based on global max_id_tracker)
          2. Remap all PK and FK (FK references within same completion updated synchronously)
          3. PK dedup: if still PK collision after remapping (theoretically won't happen), skip this row
          4. Type correction (_fix_row) executed after remapping is complete

        Returns:
            Number of new rows added in this merge
        """
        # ---- Step 1: Collect all rows within completion (preserve raw order) ----
        # structure: [(table_name, row_dict), ...]
        all_rows: List[Tuple[str, Dict]] = []
        for op in operations:
            if not isinstance(op, dict):
                continue
            for key, rows in op.items():
                if key.startswith("_"):
                    continue
                if not isinstance(rows, list):
                    continue
                table_info = schema.get_table(key)
                if table_info is None:
                    logger.debug(f"    Skipping unknown table: {key}")
                    continue
                for row in rows:
                    if isinstance(row, dict):
                        all_rows.append((table_info.name, row))

        if not all_rows:
            return 0

        # ---- Step 2: For single-column integer PK establish old→new mapping ----
        # id_remap[table_name][old_id] = new_id
        id_remap: Dict[str, Dict[int, int]] = {}

        # First scan to find PK column info for each table
        pk_col_cache: Dict[str, Optional[str]] = {}  # table_name → pk_col_name or None
        for table in schema.tables:
            pk_cols = table.primary_key
            if len(pk_cols) == 1:
                pk_col_obj = table.get_column(pk_cols[0])
                if pk_col_obj and pk_col_obj.is_integer_type():
                    pk_col_cache[table.name] = pk_cols[0]
                else:
                    pk_col_cache[table.name] = None
            else:
                pk_col_cache[table.name] = None  # Composite primary key, no remapping

        # Scan all rows within completion, establish mapping
        for table_name, row in all_rows:
            pk_col = pk_col_cache.get(table_name)
            if pk_col is None:
                continue
            old_id = row.get(pk_col)
            if old_id is None or not isinstance(old_id, (int, float)):
                continue
            old_id = int(old_id)

            if table_name not in id_remap:
                id_remap[table_name] = {}

            if old_id not in id_remap[table_name]:
                self.max_id_tracker[table_name] = self.max_id_tracker.get(table_name, 0) + 1
                id_remap[table_name][old_id] = self.max_id_tracker[table_name]

        # ---- Step 3: build FK→(ref_table, ref_col) cache ----
        # fk_lookup[(table_name, col_name)] = (ref_table, ref_col)
        fk_lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for table in schema.tables:
            for fk in table.foreign_keys:
                for i, col in enumerate(fk.columns):
                    fk_lookup[(table.name, col)] = (fk.ref_table, fk.ref_columns[i])

        # ---- Step 4: Remap and merge ----
        total_added = 0

        for table_name, row in all_rows:
            table_info = schema.get_table(table_name)
            if table_info is None:
                continue

            # 4a. Remap PK
            pk_col = pk_col_cache.get(table_name)
            if pk_col and pk_col in row:
                old_val = row.get(pk_col)
                if old_val is not None and isinstance(old_val, (int, float)):
                    old_val = int(old_val)
                    remap = id_remap.get(table_name, {})
                    if old_val in remap:
                        row[pk_col] = remap[old_val]

            # 4b. Remap FK (references within same completion)
            for fk in table_info.foreign_keys:
                for i, fk_col in enumerate(fk.columns):
                    ref_table = fk.ref_table
                    old_fk_val = row.get(fk_col)
                    if old_fk_val is None or not isinstance(old_fk_val, (int, float)):
                        continue
                    old_fk_val = int(old_fk_val)
                    remap = id_remap.get(ref_table, {})
                    if old_fk_val in remap:
                        row[fk_col] = remap[old_fk_val]

            # 4c. _fix_row: typecorrect + foreign keyvaliditycorrect
            fixed_row = self._fix_row(row, table_info)
            if not fixed_row:
                continue

            # 4d. PK deduplicatechecking
            pk_cols = table_info.primary_key
            if pk_cols:
                pk_tuple = tuple(fixed_row.get(c) for c in pk_cols)
                if pk_tuple in self.pk_index.get(table_name, set()):
                    logger.debug(
                        f"    Skipping duplicate PK: {table_name} {pk_cols}={pk_tuple}"
                    )
                    continue
                if table_name not in self.pk_index:
                    self.pk_index[table_name] = set()
                self.pk_index[table_name].add(pk_tuple)

            # 4e. Content-level dedup: for non-bridge tables, check if Name/ShortName/Title is duplicated
            if self._is_content_duplicate(table_name, table_info, fixed_row):
                continue

            # 4f. Append data
            if table_name not in self.generated_data:
                self.generated_data[table_name] = []
            self.generated_data[table_name].append(fixed_row)
            total_added += 1

        return total_added

    def _fix_row(self, row: Dict, table_info: TableInfo) -> Optional[Dict]:
        """Validate and patch a data row."""
        fixed_row = {}

        for col in table_info.columns:
            value = row.get(col.name)

            # foreign keycorrect：ensurereferencevaluevalid
            fk_ref = self._get_fk_ref(col.name, table_info)
            if fk_ref:
                ref_table, ref_col = fk_ref
                # Self-referencing foreign key
                if ref_table.lower() == table_info.name.lower():
                    if value is not None:
                        parent_rows = self.generated_data.get(ref_table, [])
                        # Use set dedup to avoid bias from duplicate IDs
                        valid_values = list(set(
                            r.get(ref_col) for r in parent_rows
                            if r.get(ref_col) is not None
                        ))
                        if value not in valid_values:
                            value = random.choice(valid_values) if valid_values else None
                    fixed_row[col.name] = value
                else:
                    # Regular foreign key
                    parent_rows = self.generated_data.get(ref_table, [])
                    if parent_rows:
                        valid_values = list(set(
                            r.get(ref_col) for r in parent_rows
                            if r.get(ref_col) is not None
                        ))
                        if value not in valid_values:
                            value = random.choice(valid_values) if valid_values else 1
                    else:
                        if value is None:
                            value = 1
                    fixed_row[col.name] = value
                continue

            # When primary key column is missing, use max_id_tracker to assign new ID (avoid default 0)
            if col.is_primary_key and col.is_integer_type() and value is None:
                self.max_id_tracker[table_info.name] = self.max_id_tracker.get(table_info.name, 0) + 1
                fixed_row[col.name] = self.max_id_tracker[table_info.name]
                continue

            # typecorrect
            if value is not None:
                # Oracle treats empty string as NULL, NOT NULL columns cannot have empty string
                # Here uniformly replace empty string in NOT NULL columns with meaningful default value
                if not col.is_nullable and isinstance(value, str) and value.strip() == '':
                    fixed_row[col.name] = self._default_value(col)
                else:
                    fixed_row[col.name] = self._coerce_type(value, col)
            elif not col.is_nullable:
                fixed_row[col.name] = self._default_value(col)
            else:
                fixed_row[col.name] = None

        # Quality check: discard rows where core identifier column is empty
        # Core identifier column = first non-PK, non-FK text column in table (usually Name/Title/ShortName etc.)
        # Auxiliary columns (Description, Notes, HomePage etc.) allow NULL
        if self._is_empty_row(fixed_row, table_info):
            return None

        return fixed_row

    def _is_empty_row(self, row: Dict, table_info: TableInfo) -> bool:
        """
        Check if it is a meaningless null row

        Judgment criteria: first non-PK, non-FK text column in table (i.e. core identifier column) is empty
        e.g.：Author.Name、Conference.ShortName、Paper.Title
        """
        fk_col_names = set()
        for fk in table_info.foreign_keys:
            fk_col_names.update(fk.columns)

        for col in table_info.columns:
            if col.is_primary_key or col.name in fk_col_names:
                continue
            if col.is_string_type():
                # This is the core identifier column
                val = row.get(col.name)
                if val is None or (isinstance(val, str) and not val.strip()):
                    return True  # Core identifier column is empty → invalid row
                return False  # Core identifier column has value → valid row

        return False  # No text column table (pure numeric table) → keep

    # =========================================================================
    # helperutility method
    # =========================================================================

    # Content dedup key field names (lowercase matching)
    _DEDUP_FIELD_NAMES = {
        'name', 'shortname', 'short_name', 'title', 'fullname', 'full_name',
    }

    def _is_content_duplicate(
        self, table_name: str, table_info: TableInfo, row: Dict
    ) -> bool:
        """
        Content-level dedup: check if Name/ShortName/Title etc. key field already exists

        For bridge tables (composite primary key with only FK columns), skip this check.
        For dimension tables (Author/Conference/Journal etc.), if key field value is duplicated, skip this row.

        Returns:
            True = duplicated, should skip
        """
        # Bridge tables do not do content dedup (like PaperAuthor)
        pk_cols = table_info.primary_key
        if len(pk_cols) >= 2:
            # If all PK columns are foreign keys, it is a bridge table
            fk_cols_set = set()
            for fk in table_info.foreign_keys:
                fk_cols_set.update(fk.columns)
            if all(c in fk_cols_set for c in pk_cols):
                return False

        # Find key dedup fields in this table
        dedup_fields = []
        for col in table_info.columns:
            if col.name.lower() in self._DEDUP_FIELD_NAMES and col.is_string_type():
                dedup_fields.append(col.name)

        if not dedup_fields:
            return False

        # initializeindex
        if table_name not in self.content_index:
            self.content_index[table_name] = {}

        is_dup = False
        for field in dedup_fields:
            val = row.get(field)
            if val is None or not isinstance(val, str):
                continue
            val_normalized = val.strip().lower()
            if not val_normalized:
                continue

            if field not in self.content_index[table_name]:
                self.content_index[table_name][field] = set()

            if val_normalized in self.content_index[table_name][field]:
                logger.debug(
                    f"    Skipping content duplicate: {table_name}.{field}='{val}'"
                )
                is_dup = True
                break

        # If not duplicate, record to index
        if not is_dup:
            for field in dedup_fields:
                val = row.get(field)
                if val is not None and isinstance(val, str) and val.strip():
                    if field not in self.content_index[table_name]:
                        self.content_index[table_name][field] = set()
                    self.content_index[table_name][field].add(val.strip().lower())

        return is_dup

    def _fix_all_foreign_keys(self, schema: SchemaInfo):
        """
        Global FK cleanup — executed after all data is generated

        For each foreign key column, check if the referenced value exists in the parent table's referenced column:
        - exist → keep
        - does not exist -> randomly remap to existing valid value in parent table referenced column

        Note: FK can reference PK column, and can also reference UNIQUE column (like cards.uuid).
        """
        logger.info("  Global FK cleanup...")
        total_fixed = 0

        # build (ref_table, ref_col) → (value_set, value_list) index
        # Build on demand, collect only when encountering a new (ref_table, ref_col) combination
        ref_cache: Dict[Tuple[str, str], Tuple[set, list]] = {}

        def _get_ref_values(ref_table: str, ref_col: str) -> Tuple[set, list]:
            """Get the set of valid values for the specified column in the parent table."""
            key = (ref_table, ref_col)
            if key not in ref_cache:
                rows = self.generated_data.get(ref_table, [])
                values = set()
                value_list = []
                for r in rows:
                    v = r.get(ref_col)
                    if v is not None:
                        values.add(v)
                        value_list.append(v)
                        # Also add int/str forms to ensure matching
                        if isinstance(v, int):
                            values.add(str(v))
                        elif isinstance(v, str) and v.isdigit():
                            values.add(int(v))
                ref_cache[key] = (values, value_list)
            return ref_cache[key]

        # Iterate all tables and all foreign keys
        for table in schema.tables:
            rows = self.generated_data.get(table.name, [])
            if not rows:
                continue

            for fk in table.foreign_keys:
                ref_table = fk.ref_table

                for i, fk_col in enumerate(fk.columns):
                    ref_col = fk.ref_columns[i] if i < len(fk.ref_columns) else fk.ref_columns[0]
                    valid_values, valid_list = _get_ref_values(ref_table, ref_col)

                    if not valid_list:
                        continue

                    fixed_count = 0
                    for row in rows:
                        val = row.get(fk_col)
                        if val is not None and val not in valid_values:
                            # Remap to a random valid value
                            row[fk_col] = random.choice(valid_list)
                            fixed_count += 1
                    if fixed_count > 0:
                        total_fixed += fixed_count
                        logger.info(
                            f"    {table.name}.{fk_col} → {ref_table}.{ref_col}: "
                            f"fix {fixed_count} invalid references"
                        )

        if total_fixed > 0:
            logger.info(f"  FK cleanup done: fixed {total_fixed} invalid references")
        else:
            logger.info(f"  FK cleanup done: no fix needed")

    def _fix_unique_columns(self, schema: SchemaInfo):
        """
        UNIQUE column dedup — executed after FK cleanup

        For each non-PK column with UNIQUE constraint:
          1. Detect duplicate values
          2. Generate new random unique value for duplicate values
          3. Cascade update all FKs referencing this column

        Typical scenario: cards.uuid (VARCHAR UNIQUE) gets duplicate values generated by LLM,
        and foreign_data.uuid / legalities.uuid / rulings.uuid reference it via FK.
        """
        logger.info("  UNIQUE column dedup...")
        total_fixed = 0

        # Build "which columns are FK-referenced by other tables" index
        # {(parent_table, parent_col): [(child_table, child_col), ...]}
        fk_references: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for table in schema.tables:
            for fk in table.foreign_keys:
                for i, ref_col in enumerate(fk.ref_columns):
                    key = (fk.ref_table, ref_col)
                    child = (table.name, fk.columns[i])
                    fk_references.setdefault(key, []).append(child)

        for table in schema.tables:
            rows = self.generated_data.get(table.name, [])
            if not rows:
                continue

            for col in table.columns:
                # Skip PK columns (PK dedup already has logic)
                if col.name in table.primary_key:
                    continue

                # Check if has UNIQUE constraint
                if not col.is_unique:
                    continue

                # Collect current column values, find duplicates
                seen: Dict[Any, int] = {}  # value -> first_row_index
                duplicates: List[Tuple[int, Any]] = []  # (row_index, old_value)

                for i, row in enumerate(rows):
                    val = row.get(col.name)
                    if val is None:
                        continue
                    if val in seen:
                        duplicates.append((i, val))
                    else:
                        seen[val] = i

                if not duplicates:
                    continue

                # Generate new unique value for each duplicate value
                existing_vals = set(seen.keys())
                fixed_count = 0

                for row_idx, old_val in duplicates:
                    # Generate new value: for UUID format columns use uuid4, otherwise append suffix to original value
                    new_val = self._generate_unique_value(
                        old_val, col, existing_vals
                    )
                    existing_vals.add(new_val)

                    # Update this row
                    rows[row_idx][col.name] = new_val

                    # Cascade update all FKs referencing this column
                    ref_key = (table.name, col.name)
                    if ref_key in fk_references:
                        for child_table, child_col in fk_references[ref_key]:
                            child_rows = self.generated_data.get(child_table, [])
                            for child_row in child_rows:
                                if child_row.get(child_col) == old_val:
                                    child_row[child_col] = new_val

                    fixed_count += 1

                if fixed_count > 0:
                    total_fixed += fixed_count
                    logger.info(
                        f"    {table.name}.{col.name}: "
                        f"fixed {fixed_count} duplicate values"
                    )

        if total_fixed > 0:
            logger.info(f"  UNIQUE dedup done: fixed {total_fixed} duplicate values")
        else:
            logger.info(f"  UNIQUE dedup done: no fix needed")

    def _generate_unique_value(
        self, old_val: Any, col: ColumnInfo, existing: set
    ) -> Any:
        """
        Generate a new value for UNIQUE column that does not conflict with existing values

        policy:
          - UUID format (VARCHAR(36) and value looks like UUID) → uuid4()
          - string → original value + random suffix
          - integer → original value + random offset
        """
        # Detect if it is a UUID format column
        is_uuid_col = (
            'uuid' in col.name.lower()
            or (col.data_type.upper().startswith('VARCHAR')
                and '36' in col.data_type)
        )

        if is_uuid_col:
            for _ in range(100):
                new_val = str(uuid.uuid4())
                if new_val not in existing:
                    return new_val

        if isinstance(old_val, str):
            for _ in range(100):
                suffix = f"_{random.randint(1000, 9999)}"
                new_val = old_val + suffix
                # If exceeds column length limit, truncate original value
                if col.max_length and len(new_val) > col.max_length:
                    new_val = old_val[:col.max_length - len(suffix)] + suffix
                if new_val not in existing:
                    return new_val

        if isinstance(old_val, (int, float)):
            for _ in range(100):
                new_val = old_val + random.randint(1, 100000)
                if new_val not in existing:
                    return new_val

        # fallback
        return str(uuid.uuid4())

    def _get_fk_ref(self, col_name: str, table_info: TableInfo) -> Optional[Tuple[str, str]]:
        """Get column foreign key reference info."""
        for fk in table_info.foreign_keys:
            if col_name in fk.columns:
                idx = fk.columns.index(col_name)
                return (fk.ref_table, fk.ref_columns[idx])
        return None

    # integertypevalidrangemapping
    _INTEGER_RANGES = {
        'TINYINT': (-128, 127),
        'SMALLINT': (-32768, 32767),
        'MEDIUMINT': (-8388608, 8388607),
        'INT': (-2147483648, 2147483647),
        'INTEGER': (-2147483648, 2147483647),
        'BIGINT': (-9223372036854775808, 9223372036854775807),
    }

    def _clamp_integer(self, value: int, col: ColumnInfo) -> int:
        """Clamp numeric value to within column type allowed range."""
        bt = col.base_type()
        # Oracle NUMBER(N) determine range based on precision
        if bt == 'NUMBER':
            m = re.search(r'NUMBER\((\d+)(?:,\s*(\d+))?\)', col.data_type.upper())
            if m:
                precision = int(m.group(1))
                max_val = 10 ** precision - 1
                return max(-max_val, min(value, max_val))
            return value
        range_tuple = self._INTEGER_RANGES.get(bt)
        if range_tuple:
            min_val, max_val = range_tuple
            return max(min_val, min(value, max_val))
        return value

    def _coerce_type(self, value: Any, col: ColumnInfo) -> Any:
        """Type coercion — ensure values satisfy schema constraints (cross-engine data consistency guarantee)."""
        try:
            # ENUM column: validate value is within allowed_values range
            if col.allowed_values:
                str_val = str(value).strip()
                if str_val in col.allowed_values:
                    return str_val
                # Try case-insensitive matching
                lower_map = {v.lower(): v for v in col.allowed_values}
                if str_val.lower() in lower_map:
                    return lower_map[str_val.lower()]
                # Try removing spaces/underscores fuzzy matching
                normalized_map = {v.lower().replace(' ', '_').replace('-', '_'): v for v in col.allowed_values}
                normalized_input = str_val.lower().replace(' ', '_').replace('-', '_')
                if normalized_input in normalized_map:
                    return normalized_map[normalized_input]
                # Value not in allowed range: randomly select a valid value
                return random.choice(col.allowed_values)

            if col.is_integer_type():
                int_val = None
                if isinstance(value, (int, float)):
                    int_val = int(value)
                elif isinstance(value, str) and value.replace('-', '').isdigit():
                    int_val = int(value)
                if int_val is not None:
                    return self._clamp_integer(int_val, col)
                return value

            if col.is_decimal_type():
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        return value
                return value

            if col.is_boolean_type():
                if isinstance(value, bool):
                    return value
                if isinstance(value, int):
                    return bool(value)
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes')
                return value

            if col.is_json_type():
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value)

            # Date/time type: fix pure time value written to DATETIME/TIMESTAMP column issue
            # LLM sometimes generates pure time value like "08:00:00" for DATETIME column,
            # MySQL DATETIME column can reluctantly accept but semantically wrong, Oracle TO_TIMESTAMP will directly error
            if col.is_date_type() and isinstance(value, str):
                stripped = value.strip()
                bt = col.base_type()
                # Pure time value (HH:MM:SS) written to DATETIME/TIMESTAMP/DATE column → prepend date prefix
                if re.match(r'^\d{2}:\d{2}:\d{2}(?:\.\d+)?$', stripped):
                    if bt in ('DATETIME', 'TIMESTAMP', 'TIMESTAMPTZ', 'DATE'):
                        return f"2024-01-01 {re.sub(r'\\.\\d+', '', stripped)}"
                # Pure date value (YYYY-MM-DD) written to DATETIME/TIMESTAMP column → append time suffix
                if re.match(r'^\d{4}-\d{2}-\d{2}$', stripped):
                    if bt in ('DATETIME', 'TIMESTAMP', 'TIMESTAMPTZ'):
                        return f"{stripped} 00:00:00"

            # stringtype: checking max_length truncate
            if col.is_string_type() and col.max_length and isinstance(value, str):
                if len(value) > col.max_length:
                    return value[:col.max_length]

            return value

        except (ValueError, TypeError):
            return value

    def _default_value(self, col: ColumnInfo) -> Any:
        """Generate default value for NOT NULL column."""
        if col.allowed_values:
            return col.allowed_values[0]
        if col.is_integer_type():
            return 0
        if col.is_decimal_type():
            return 0.0
        if col.is_boolean_type():
            return False
        if col.is_date_type():
            bt = col.base_type()
            if bt in ("DATETIME", "TIMESTAMP", "TIMESTAMPTZ"):
                return "2024-01-01 00:00:00"
            return "2024-01-01"
        if col.is_json_type():
            return '{}'
        return "N/A"

    # =========================================================================
    # Fallback generation (not dependent on LLM)
    # =========================================================================

    def _generate_fallback_rows(
        self,
        table_info: TableInfo,
        count: int,
        start_id: int,
    ) -> List[Dict]:
        """Pure rule-based fallback generation (used when LLM generated data is insufficient)."""
        rows = []
        for i in range(count):
            row = {}
            current_id = start_id + i

            for col in table_info.columns:
                # Auto-increment primary key
                if col.is_auto_increment or (col.is_primary_key and col.is_integer_type()):
                    row[col.name] = current_id
                    continue

                # foreign key
                fk_ref = self._get_fk_ref(col.name, table_info)
                if fk_ref:
                    ref_table, ref_col = fk_ref
                    if ref_table.lower() == table_info.name.lower():
                        row[col.name] = None if i < 2 else random.randint(start_id, start_id + i - 1)
                    else:
                        parent_rows = self.generated_data.get(ref_table, [])
                        if parent_rows:
                            valid = [r.get(ref_col) for r in parent_rows if r.get(ref_col) is not None]
                            row[col.name] = random.choice(valid) if valid else 1
                        else:
                            row[col.name] = 1
                    continue

                # Generate default value by type
                row[col.name] = self._fallback_value(col, i)

            rows.append(row)

        return rows

    def _fallback_value(self, col: ColumnInfo, row_idx: int) -> Any:
        """fallbackvaluegenerate"""
        if col.allowed_values:
            return col.allowed_values[row_idx % len(col.allowed_values)]
        if col.is_integer_type():
            return random.randint(1, 1000)
        if col.is_decimal_type():
            return round(random.uniform(0.01, 9999.99), 2)
        if col.is_boolean_type():
            return random.choice([True, False])
        if col.is_date_type():
            bt = col.base_type()
            year = random.randint(2020, 2025)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            if bt in ("DATETIME", "TIMESTAMP", "TIMESTAMPTZ"):
                h, m, s = random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)
                return f"{year:04d}-{month:02d}-{day:02d} {h:02d}:{m:02d}:{s:02d}"
            return f"{year:04d}-{month:02d}-{day:02d}"
        if col.is_json_type():
            return json.dumps({"key": f"value_{row_idx}"}, ensure_ascii=False)
        if col.is_string_type():
            return f"{col.name}_{row_idx + 1}"
        return f"val_{row_idx + 1}"
