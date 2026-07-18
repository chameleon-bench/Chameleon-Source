"""
Query synthesizer core logic

Workflow:
1. Load query_requirements + query_patterns per database from diff allocation file
2. Read DDL file for the corresponding dialect
3. Read sample data from database
4. Call LLM to generate queries covering allocated differences
5. Save results
"""

import json
import os
import re
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger
from utils.db_utils import DatabaseManager, DatabaseConfig
from llm.client import LLMClient
from query_synthesis.diff_allocation_loader import DiffAllocationLoader
from query_synthesis.dialect_parser import BuiltinFunctionLoader
from query_synthesis.prompts import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    REFLECTION_SYSTEM_PROMPT, REFLECTION_USER_PROMPT,
)
from query_synthesis.fewshot_examples import (
    DIFFICULTY_DEFINITIONS,
    DEFAULT_DIFFICULTY_WEIGHTS,
    get_examples_for_difficulty,
    format_examples_for_prompt,
)
from data_synthesis.schema_parser import SchemaParser

logger = get_logger(__name__)


class QuerySynthesizer:
    """
    Query synthesizer

    Iterates over all databases, generating SQL queries via LLM that cover syntax differences and built-in functions.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize synthesizer

        Args:
            config_path: Config file path, defaults to config/query_synthesis.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / 'src' / 'config' / 'query_synthesis.yaml'

        self.config = self._load_config(config_path)
        self.target_dialect = self.config['target_dialect']  # "mysql" or "postgresql"

        # Synthesis parameters
        syn_cfg = self.config['synthesis']
        self.queries_per_call = syn_cfg['queries_per_call']
        self.loops_per_difficulty = syn_cfg.get('loops_per_difficulty', syn_cfg.get('outer_loops', 3))
        self.sample_rows = syn_cfg['sample_rows']
        self.seed = syn_cfg.get('seed', 42)

        # Difficulty distribution weights (read from config, with default fallback)
        self.difficulty_weights = self.config.get('difficulty_weights', DEFAULT_DIFFICULTY_WEIGHTS)

        # Parallelism config
        parallel_cfg = self.config.get('parallel', {})
        self.max_llm_workers = parallel_cfg.get('max_llm_workers', 5)
        self.max_validation_workers = parallel_cfg.get('max_validation_workers', 10)

        # Random number generator
        self.rng = random.Random(self.seed)

        # Initialize LLM client
        llm_cfg = self.config['llm']
        self.llm_client = LLMClient(
            provider=llm_cfg['provider'],
            model=llm_cfg['model'],
        )
        self.llm_temperature = llm_cfg.get('temperature', 0.8)
        self.llm_max_tokens = llm_cfg.get('max_tokens', 16384)

        # Initialize database manager
        self.db_manager = self._init_db_manager()

        # Initialize diff allocation loader
        alloc_cfg = self.config.get('diff_allocation', {})
        allocation_file = PROJECT_ROOT / alloc_cfg.get('allocation_file', 'output/schema_expansion/allocation_dev.json')
        requirements_file = PROJECT_ROOT / alloc_cfg.get('requirements_file', 'data/database_sync_requirements.json')
        self.diff_allocation = DiffAllocationLoader(str(allocation_file), str(requirements_file))

        # Schema directory (contains split subdirectory)
        self.database_dir = PROJECT_ROOT / self.config['schema']['database_dir']
        self.split = alloc_cfg.get('split', 'dev')

        # Determine actual schema path (database_dir/split/ or database_dir/)
        self.schema_dir = self.database_dir / self.split
        if not self.schema_dir.exists():
            # Fall back to database_dir itself
            self.schema_dir = self.database_dir

        # Output directory (subdirectory by split: output/query_synthesis/{split}/)
        self.output_base_dir = PROJECT_ROOT / self.config['output']['output_dir']
        self.output_dir = self.output_base_dir / self.split
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reflection (SQL self-repair) config
        refl_cfg = self.config.get('reflection', {})
        self.reflection_enabled = refl_cfg.get('enabled', False)
        self.reflection_max_retries = refl_cfg.get('max_retries', 3)
        self.reflection_query_timeout = refl_cfg.get('query_timeout', 10)

        # Load KB knowledge base for target dialect (built-in functions & keyword reference)
        kb_file_map = {
            'mysql': 'data/mysql_8_kb.json',
            'postgresql': 'data/pg_14_kb.json',
            'oracle': 'data/oracle_11_kb.json',
        }
        kb_file = PROJECT_ROOT / kb_file_map.get(self.target_dialect, kb_file_map['mysql'])
        self.kb_loader = BuiltinFunctionLoader(str(kb_file))
        # KB reference injection config
        kb_cfg = self.config.get('kb_reference', {})
        self.kb_items_per_call = kb_cfg.get('items_per_call', 50)
        self.kb_stratified_sampling = kb_cfg.get('stratified_sampling', True)
        logger.info(f"KB knowledge base: {kb_file.name}, entries={len(self.kb_loader.functions)}, "
                     f"items_per_call={self.kb_items_per_call}, stratified_sampling={self.kb_stratified_sampling}")

        # Result statistics
        self.stats: Dict[str, Dict] = {}

        logger.info(
            f"QuerySynthesizer initialization complete: "
            f"dialect={self.target_dialect}, "
            f"split={self.split}, "
            f"queries_per_call={self.queries_per_call}, "
            f"loops_per_difficulty={self.loops_per_difficulty}, "
            f"difficulty_weights={self.difficulty_weights}, "
            f"parallel=LLM×{self.max_llm_workers}/DB×{self.max_validation_workers}, "
            f"allocated_databases={len(self.diff_allocation.allocation)}, "
            f"total_diffs={len(self.diff_allocation.diff_map)}, "
            f"reflection={'ON (max_retries=' + str(self.reflection_max_retries) + ')' if self.reflection_enabled else 'OFF'}"
        )

    def _load_config(self, config_path) -> Dict:
        """Load config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_db_manager(self) -> DatabaseManager:
        """Initialize database manager."""
        db_key = self.target_dialect
        db_cfg = self.config['database'][db_key]

        config = DatabaseConfig(
            host=db_cfg['host'],
            port=db_cfg['port'],
            user=db_cfg['user'],
            password=db_cfg['password'],
        )

        return DatabaseManager(config, db_type=self.target_dialect)

    def _load_dialect_assignment(self) -> Optional[Dict[str, List[str]]]:
        """
        Load dialect assignment file (dialect_assignment.json)

        Returns:
            {dialect: [db_names]} dict, returns None if file does not exist
        """
        # Prefer reading path from config, default to database/{split}/dialect_assignment.json
        assign_cfg = self.config.get('dialect_assignment', {})
        default_path = self.database_dir / self.split / 'dialect_assignment.json'
        assign_file = Path(assign_cfg.get('file', str(default_path)))

        if not assign_file.is_absolute():
            assign_file = PROJECT_ROOT / assign_file

        if assign_file.exists():
            with open(assign_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            result = {}
            for dialect, info in data.get('allocation', {}).items():
                result[dialect] = info.get('databases', [])
            logger.info(f"Loading dialect assignment file: {assign_file} "
                        f"({', '.join(f'{k}={len(v)}' for k, v in result.items())})")
            return result
        else:
            logger.info(f"Dialect assignment file not found: {assign_file}，will use all databases")
            return None

    def _discover_databases(self) -> List[str]:
        """
        Discover databases needing query synthesis

        Prefer getting current dialect's database list from dialect assignment file,
        if no assignment file, get full list from diff allocation file.
        Also verify corresponding schema file exists.

        Returns:
            Database name list
        """
        # Try to get current dialect's databases from dialect assignment file
        dialect_assignment = self._load_dialect_assignment()
        if dialect_assignment is not None:
            assigned_dbs = dialect_assignment.get(self.target_dialect, [])
            logger.info(f"Dialect assignment: {self.target_dialect} assigned {len(assigned_dbs)} databases")
        else:
            # Fallback: get full list from diff allocation file
            assigned_dbs = self.diff_allocation.get_all_database_names()

        # Verify schema file exists
        databases = []
        for db_name in sorted(assigned_dbs):
            if self.target_dialect == 'mysql':
                suffix = 'mysql'
            elif self.target_dialect == 'postgresql':
                suffix = 'pg'
            elif self.target_dialect == 'oracle':
                suffix = 'oracle'
            else:
                suffix = self.target_dialect

            schema_file = self.schema_dir / db_name / f"{db_name}_schema_{suffix}.sql"
            if schema_file.exists():
                databases.append(db_name)
            else:
                logger.warning(f"Database {db_name} assigned but missing schema file: {schema_file}")

        logger.info(f"Found {len(databases)} databases (dialect={self.target_dialect})")
        return databases

    def _read_ddl(self, db_name: str) -> str:
        """
        Read database DDL file

        Args:
            db_name: Database name

        Returns:
            DDL content
        """
        if self.target_dialect == 'mysql':
            suffix = 'mysql'
        elif self.target_dialect == 'postgresql':
            suffix = 'pg'
        elif self.target_dialect == 'oracle':
            suffix = 'oracle'
        else:
            suffix = self.target_dialect
            
        schema_file = self.schema_dir / db_name / f"{db_name}_schema_{suffix}.sql"

        if not schema_file.exists():
            logger.warning(f"Schema file not found: {schema_file}")
            return ""

        return schema_file.read_text(encoding='utf-8')

    def _get_table_names_from_ddl(self, ddl: str) -> List[str]:
        """
        Extract table names from DDL

        Args:
            ddl: DDL content

        Returns:
            Table name list
        """
        # Compatible with quoted and unquoted: CREATE TABLE `t`, CREATE TABLE "t", CREATE TABLE t, CREATE UNLOGGED TABLE "t"
        pattern = r'CREATE\s+(?:UNLOGGED\s+)?TABLE\s+(?:`([^`]+)`|"([^"]+)"|(\w+))'
        matches = re.findall(pattern, ddl, re.IGNORECASE)
        # re.findall returns tuple list, take first non-empty group
        tables = [m[0] or m[1] or m[2] for m in matches]
        return tables

    def _fetch_sample_data(self, db_name: str, table_names: List[str]) -> str:
        """
        Read sample data for each table from database

        Args:
            db_name: Database name
            table_names: Table name list

        Returns:
            Formatted sample data text
        """
        sample_texts = []

        for table_name in table_names:
            try:
                if self.target_dialect == 'mysql':
                    sql = f"SELECT * FROM `{table_name}` LIMIT {self.sample_rows}"
                elif self.target_dialect == 'oracle':
                    sql = f'SELECT * FROM "{table_name}" FETCH FIRST {self.sample_rows} ROWS ONLY'
                else:
                    sql = f'SELECT * FROM "{table_name}" LIMIT {self.sample_rows}'

                rows = self.db_manager.query(sql, database=db_name)

                if rows:
                    sample_texts.append(f"### Table: {table_name} (total {len(rows)} rows sample)")
                    # Format as concise table
                    columns = list(rows[0].keys())
                    sample_texts.append("| " + " | ".join(columns) + " |")
                    sample_texts.append("| " + " | ".join(["---"] * len(columns)) + " |")
                    for row in rows:
                        values = []
                        for col in columns:
                            val = row[col]
                            if val is None:
                                values.append("NULL")
                            else:
                                val_str = str(val)
                                # Truncate overly long values
                                if len(val_str) > 50:
                                    val_str = val_str[:50] + "..."
                                values.append(val_str)
                        sample_texts.append("| " + " | ".join(values) + " |")
                    sample_texts.append("")
                else:
                    sample_texts.append(f"### Table: {table_name} (no data)")
                    sample_texts.append("")

            except Exception as e:
                logger.warning(f"Failed to read table {db_name}.{table_name} sample data failed: {e}")
                sample_texts.append(f"### Table: {table_name} (read failed)")
                sample_texts.append("")

        return "\n".join(sample_texts)

    def _fetch_column_stats(self, db_name: str, ddl: str) -> str:
        """
        Query MIN/MAX statistics for numeric and date/time columns from database.

        Parse DDL via SchemaParser to identify column types,
        for numeric columns (INT/DECIMAL/FLOAT etc.) and date columns (DATE/DATETIME/TIMESTAMP etc.)
        query real min and max values.
        Injecting this info into prompts helps LLM generate WHERE conditions with reasonable ranges,
        significantly reducing empty result queries.

        Args:
            db_name: Database name
            ddl: Current dialect DDL text

        Returns:
            Formatted stats text (Markdown), empty string if no stats available
        """
        # Parse DDL via SchemaParser to get structured info
        parser = SchemaParser()
        schema = parser.parse_sql(ddl, db_name)

        stats_texts = []

        for table in schema.tables:
            # Collect FK referenced column names (MIN/MAX meaningless for these)
            fk_col_names = set()
            for fk in table.foreign_keys:
                for col_name in fk.columns:
                    fk_col_names.add(col_name.lower())

            # Collect columns needing stats: numeric + date columns
            # Exclude: auto-increment PK, FK columns, PK columns (bridge table PK is also FK)
            target_cols = []
            for col in table.columns:
                if col.is_auto_increment:
                    continue
                if col.name.lower() in fk_col_names:
                    continue
                if col.is_primary_key:
                    continue
                if col.is_numeric_type() or col.is_date_type():
                    target_cols.append(col)

            if not target_cols:
                continue

            # Build SELECT MIN(...), MAX(...) query
            select_parts = []
            col_names = []
            for col in target_cols:
                if self.target_dialect == 'mysql':
                    qname = f"`{col.name}`"
                    select_parts.append(f"MIN({qname}) AS `min_{col.name}`")
                    select_parts.append(f"MAX({qname}) AS `max_{col.name}`")
                else:
                    qname = f'"{col.name}"'
                    select_parts.append(f'MIN({qname}) AS "min_{col.name}"')
                    select_parts.append(f'MAX({qname}) AS "max_{col.name}"')
                col_names.append(col.name)

            if self.target_dialect == 'mysql':
                tname = f"`{table.name}`"
            else:
                tname = f'"{table.name}"'

            sql = f"SELECT {', '.join(select_parts)} FROM {tname}"

            try:
                result = self.db_manager.query_one(sql, database=db_name)
                if not result:
                    continue

                # Format
                table_stats = []
                for col in target_cols:
                    min_val = result.get(f"min_{col.name}")
                    max_val = result.get(f"max_{col.name}")
                    if min_val is None and max_val is None:
                        continue
                    type_label = col.data_type
                    table_stats.append(
                        f"  - `{col.name}` ({type_label}): "
                        f"MIN = {min_val}, MAX = {max_val}"
                    )

                if table_stats:
                    stats_texts.append(f"### Table: {table.name}")
                    stats_texts.extend(table_stats)
                    stats_texts.append("")

            except Exception as e:
                logger.warning(f"Query table {db_name}.{table.name} column stats failed: {e}")

        return "\n".join(stats_texts) if stats_texts else ""

    def _format_allocated_diffs(self, diffs: List[Dict[str, Any]]) -> str:
        """
        Format allocated differences (query_requirements + query_patterns) as text

        Args:
            diffs: Difference info list (from DiffAllocationLoader.get_query_requirements_and_patterns)

        Returns:
            Formatted text
        """
        texts = []
        for diff in diffs:
            texts.append(f"**{diff['id']} — {diff['feature']} ({diff['category']})**\n")
            texts.append(f"Description: {diff['description']}\n")

            qr = diff.get('query_requirements', [])
            if qr:
                texts.append("Query requirements:")
                for req in qr:
                    texts.append(f"  - {req}")

            qp = diff.get('query_patterns', [])
            if qp:
                texts.append("Reference patterns:")
                for pat in qp:
                    texts.append(f"  - {pat}")

            texts.append("\n---\n")
        return "\n".join(texts)

    def _format_builtin_functions_reference(self) -> str:
        """
        Randomly select built-in functions and keyword references from KB, format as prompt text.

        Each call selects different random subset, multiple calls cover more functions, avoiding excessive token cost.
        Supports stratified_sampling: sample by type to ensure each category is represented.

        Returns:
            Formatted reference text
        """
        import random

        all_items = self.kb_loader.functions
        if not all_items:
            return "(No KB data loaded)"

        max_items = self.kb_items_per_call

        if self.kb_stratified_sampling:
            # Group by type, sample proportionally per group
            type_groups = {}
            for item in all_items:
                t = item.get('type', 'unknown')
                type_groups.setdefault(t, []).append(item)

            # Calculate sample count per group (proportional, at least 1)
            selected = []
            total = len(all_items)
            for t, items in type_groups.items():
                n = max(1, round(len(items) / total * max_items))
                n = min(n, len(items))
                selected.extend(random.sample(items, n))

            # If total exceeds max_items, truncate
            if len(selected) > max_items:
                selected = random.sample(selected, max_items)
        else:
            # Fully random
            selected = random.sample(all_items, min(max_items, len(all_items)))

        # Sort output by type priority (function > keyword > operator > type)
        priority_order = {'function': 0, 'keyword': 1, 'operator': 2, 'type': 3}
        selected.sort(key=lambda x: priority_order.get(x.get('type', 'type'), 9))

        texts = []
        for item in selected:
            item_type = item.get('type', 'unknown')
            keyword = item.get('keyword', '')
            description = item.get('description', '') or ''
            examples = item.get('example', [])

            # Clean HTML tags
            clean_desc = re.sub(r'<[^>]+>', '', str(description))

            # Truncate overly long descriptions
            if len(clean_desc) > 200:
                clean_desc = clean_desc[:200] + '...'

            line = f"- **{keyword}** [{item_type}]: {clean_desc}"
            if examples and examples[0]:
                # Take first example, truncate
                ex = str(examples[0])[:120]
                line += f"\n  Example: {ex}"
            texts.append(line)

        total = len(all_items)
        shown = len(selected)
        header = f"(total {total} entries, randomly selected {shown} entries)\n"
        return header + "\n".join(texts)

    def _call_llm(
        self,
        db_name: str,
        ddl: str,
        sample_data: str,
        allocated_diffs: List[Dict[str, Any]],
        difficulty: str = "medium",
        column_stats: str = "",
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """
        Call LLM to generate queries

        Args:
            db_name: Database name
            ddl: DDL content
            sample_data: Sample data text
            allocated_diffs: Allocated difference info for this database (query_requirements + query_patterns)
            difficulty: Difficulty level (easy/medium/hard/extra)
            column_stats: MIN/MAX stats for numeric/date columns

        Returns:
            (query list, token count, cost)
        """
        dialect_name = {"mysql": "MySQL", "postgresql": "PostgreSQL", "oracle": "Oracle"}.get(self.target_dialect, self.target_dialect.capitalize())
        fewshot_dialect = {"mysql": "mysql", "postgresql": "pg", "oracle": "oracle"}.get(self.target_dialect, self.target_dialect)

        system_prompt = SYSTEM_PROMPT.format(dialect=dialect_name)

        # Get difficulty constraints
        diff_def = DIFFICULTY_DEFINITIONS.get(difficulty, DIFFICULTY_DEFINITIONS["medium"])
        difficulty_constraints = (
            f"**{diff_def['label']}** (translation difficulty: {diff_def['translation_difficulty']}, "
            f"dialect difference points: {diff_def['dialect_points']})\n\n"
            f"{diff_def['constraints']}"
        )

        # Get few-shot examples
        examples = get_examples_for_difficulty(difficulty, n=2, dialect=fewshot_dialect)
        fewshot_text = format_examples_for_prompt(examples) if examples else "(no examples)"

        # Get KB built-in functions & keyword references (different random subset each call)
        kb_reference = self._format_builtin_functions_reference()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            queries_per_call=self.queries_per_call,
            dialect=dialect_name,
            db_name=db_name,
            ddl=ddl,
            sample_rows=self.sample_rows,
            sample_data=sample_data,
            column_stats=column_stats if column_stats else "(no stats info)",
            allocated_diffs=self._format_allocated_diffs(allocated_diffs),
            builtin_functions_reference=kb_reference,
            difficulty_constraints=difficulty_constraints,
            fewshot_examples=fewshot_text,
            difficulty=difficulty,
        )

        try:
            response = self.llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                save_response=self.config['output'].get('save_raw_response', True),
            )

            # Parse JSON response
            content = response.content.strip()
            # Try to extract JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            # Try direct parse
            queries = json.loads(content)

            if isinstance(queries, list):
                # Clean sql field: remove comments and non-SQL content
                for q in queries:
                    if 'sql' in q and isinstance(q['sql'], str):
                        q['sql'] = self._clean_sql(q['sql'])
                logger.info(
                    f"LLM generated {len(queries)} queries "
                    f"(db={db_name}, tokens={response.total_tokens}, "
                    f"cost=${response.cost:.4f})"
                )
                return queries, response.total_tokens, response.cost
            else:
                logger.warning(f"LLM did not return a list format: {type(queries)}")
                return [], 0, 0.0

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response JSON: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            return [], getattr(response, 'total_tokens', 0), getattr(response, 'cost', 0.0)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return [], 0, 0.0

    @staticmethod
    def _clean_sql(sql: str) -> str:
        """
        Clean SQL field, remove comments and non-SQL content,
        ensure only directly executable pure SQL statements are kept.
        """
        lines = sql.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip pure comment lines (starting with -- or #)
            if stripped.startswith('--') or stripped.startswith('#'):
                continue
            # Remove trailing comments (simple: part after --, but avoid deleting -- inside strings)
            # Only handle obvious trailing comments here
            cleaned.append(line)

        result = '\n'.join(cleaned).strip()

        # Remove /* ... */ block comments
        result = re.sub(r'/\*.*?\*/', '', result, flags=re.DOTALL).strip()

        # Remove trailing extra semicolons
        result = result.rstrip(';').strip()

        return result

    # -- Reflection: validate + self-repair ─────────────────────────────────

    def _validate_query(self, sql: str, db_name: str) -> Dict[str, Any]:
        """
        Execute SQL on target database for validation

        Args:
            sql: SQL to validate
            db_name: Database name

        Returns:
            {'status': 'valid'|'empty'|'error', 'row_count': int, 'error': str}
        """
        timeout_ms = self.reflection_query_timeout * 1000
        try:
            with self.db_manager.get_connection(database=db_name) as conn:
                # Set query timeout to prevent slow query blocking
                if self.target_dialect == 'postgresql':
                    with conn.cursor() as cur:
                        cur.execute(f"SET statement_timeout = {timeout_ms}")
                    conn.commit()
                elif self.target_dialect == 'mysql':
                    with conn.cursor() as cur:
                        cur.execute(f"SET SESSION max_execution_time = {timeout_ms}")
                elif self.target_dialect == 'oracle':
                    conn.call_timeout = timeout_ms
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    row_count = len(rows)
                    if row_count > 0:
                        return {'status': 'valid', 'row_count': row_count}
                    else:
                        return {'status': 'empty', 'row_count': 0}
        except Exception as e:
            err_str = str(e)[:500]
            # Timeout errors uniformly marked as timeout
            if 'timeout' in err_str.lower() or 'cancelled' in err_str.lower() or 'ORA-00406' in err_str or 'maximum execution time' in err_str.lower():
                return {'status': 'error', 'row_count': 0, 'error': f'[TIMEOUT] {err_str}'}
            return {'status': 'error', 'row_count': 0, 'error': err_str}

    def _reflect_and_fix(self, query: Dict[str, Any], ddl: str, db_name: str) -> Dict[str, Any]:
        """
        LLM reflection repair for queries that failed execution

        Max self.reflection_max_retries repair rounds, each round:
        1. Send original SQL + error info + DDL to LLM
        2. LLM returns repaired SQL
        3. Validate again, if passed then done, otherwise continue next round

        Args:
            query: Query dict containing 'sql' and 'validation'
            ddl: Database DDL
            db_name: Database name

        Returns:
            (updated query dict, reflection token count, reflection cost)
        """
        dialect_name = {"mysql": "MySQL", "postgresql": "PostgreSQL", "oracle": "Oracle"}.get(self.target_dialect, self.target_dialect.capitalize())
        system_prompt = REFLECTION_SYSTEM_PROMPT.format(dialect=dialect_name)

        current_sql = query.get('sql', '')
        if not current_sql:
            logger.warning(f"  ⚠️ Query object missing 'sql' key, skipping repair")
            return query, 0, 0.0
        current_error = query.get('validation', {}).get('error', '')
        original_sql = current_sql  # Keep original SQL for logging
        refl_tokens = 0
        refl_cost = 0.0

        for attempt in range(1, self.reflection_max_retries + 1):
            logger.info(
                f"  🔄 Reflection repair (attempt {attempt}/{self.reflection_max_retries} round): "
                f"error={current_error[:80]}"
            )

            user_prompt = REFLECTION_USER_PROMPT.format(
                dialect=dialect_name,
                ddl=ddl,
                original_sql=current_sql,
                error_message=current_error,
            )

            try:
                response = self.llm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,  # Use medium temperature for repair
                    max_tokens=16384,
                    save_response=False,
                )

                refl_tokens += response.total_tokens
                refl_cost += response.cost

                # Parse LLM returned JSON
                content = response.content.strip()
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)

                fix_result = json.loads(content)
                fixed_sql = fix_result.get('fixed_sql', '').strip()
                analysis = fix_result.get('analysis', '')

                if not fixed_sql:
                    logger.warning(f"  ⚠️ Reflection returned empty SQL, skipping")
                    break

                # Clean repaired SQL
                fixed_sql = self._clean_sql(fixed_sql)

                logger.info(f"  📝 Analysis: {analysis}")

                # Validate repaired SQL
                validation = self._validate_query(fixed_sql, db_name)

                if validation['status'] == 'valid':
                    logger.info(
                        f"  ✅ Repair successful! (attempt {attempt} round, "
                        f"{validation['row_count']} rows result)"
                    )
                    query['sql'] = fixed_sql
                    query['validation'] = validation
                    query['reflection'] = {
                        'fixed': True,
                        'attempts': attempt,
                        'original_sql': original_sql,
                        'analysis': analysis,
                    }
                    return query, refl_tokens, refl_cost

                elif validation['status'] == 'empty':
                    logger.info(f"  ⚠️ Repaired and executable but result is empty (attempt {attempt} round)")
                    # Although empty result is not an error, repair is considered successful (syntax is fine)
                    query['sql'] = fixed_sql
                    query['validation'] = validation
                    query['reflection'] = {
                        'fixed': True,
                        'attempts': attempt,
                        'original_sql': original_sql,
                        'analysis': analysis,
                    }
                    return query, refl_tokens, refl_cost

                else:
                    # Still errors, continue next round
                    current_sql = fixed_sql
                    current_error = validation.get('error', '')
                    logger.info(f"  ❌ Still errors after repair: {current_error[:80]}")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"  ⚠️ Reflection response parse failed: {e}")
                break
            except Exception as e:
                logger.warning(f"  ⚠️ Reflection LLM call failed: {e}")
                break

        # All rounds failed
        logger.info(
            f"  ❌ Reflection repair failed (attempted {self.reflection_max_retries} round)"
        )
        query['reflection'] = {
            'fixed': False,
            'attempts': self.reflection_max_retries,
            'original_sql': original_sql,
        }
        return query, refl_tokens, refl_cost

    def _validate_and_fix_queries(
        self,
        queries: List[Dict[str, Any]],
        db_name: str,
        ddl: str,
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """
        Validate a batch of queries, execute reflection repair on errors

        Stage 1: Concurrently validate all queries (ThreadPoolExecutor)
        Stage 2: Concurrently execute reflection repair on error queries (ThreadPoolExecutor)

        Args:
            queries: Query list
            db_name: Database name
            ddl: DDL content

        Returns:
            (validated/repaired query list, reflection token count, reflection cost)
        """
        # -- Stage 1: Concurrent validation --
        def _validate_one(idx: int) -> Tuple[int, Dict[str, Any]]:
            sql = queries[idx].get('sql', '')
            if not sql:
                return idx, {'status': 'error', 'row_count': 0, 'error': 'empty sql'}
            return idx, self._validate_query(sql, db_name)

        with ThreadPoolExecutor(max_workers=self.max_validation_workers) as pool:
            futures = {pool.submit(_validate_one, i): i for i in range(len(queries))}
            for fut in as_completed(futures):
                idx, validation = fut.result()
                queries[idx]['validation'] = validation

        # Stats and collect queries needing repair
        error_indices = []
        valid_count = 0
        empty_count = 0

        for i, q in enumerate(queries):
            status = q.get('validation', {}).get('status', 'error')
            if status == 'valid':
                valid_count += 1
            elif status == 'empty':
                empty_count += 1
            else:
                error_indices.append(i)
                logger.info(
                    f"  [{i}] ❌ Execution error: {q['validation'].get('error', '')[:80]}"
                )

        logger.info(
            f"  Stage 1 validation complete: ✅={valid_count}, ⚠️={empty_count}, "
            f"❌={len(error_indices)} (to repair)"
        )

        # -- Stage 2: Concurrent reflection repair --
        fixed_count = 0
        refl_total_tokens = 0
        refl_total_cost = 0.0
        if self.reflection_enabled and error_indices:
            def _fix_one(idx: int) -> Tuple[int, Dict[str, Any], int, float]:
                q, tokens, cost = self._reflect_and_fix(queries[idx], ddl, db_name)
                return idx, q, tokens, cost

            # reflection repair also calls LLM, use LLM worker count
            reflection_workers = min(self.max_llm_workers, len(error_indices))
            logger.info(
                f"  Starting concurrent repair {len(error_indices)} error queries "
                f"(workers={reflection_workers})"
            )

            with ThreadPoolExecutor(max_workers=reflection_workers) as pool:
                futures = {pool.submit(_fix_one, i): i for i in error_indices}
                for fut in as_completed(futures):
                    idx, fixed_q, rtokens, rcost = fut.result()
                    queries[idx] = fixed_q
                    refl_total_tokens += rtokens
                    refl_total_cost += rcost
                    if fixed_q.get('reflection', {}).get('fixed', False):
                        fixed_count += 1
                        if fixed_q['validation']['status'] == 'valid':
                            valid_count += 1
                        elif fixed_q['validation']['status'] == 'empty':
                            empty_count += 1

        final_errors = len(error_indices) - fixed_count
        logger.info(
            f"  Validation results: ✅ valid={valid_count}, ⚠️ empty={empty_count}, "
            f"❌ error={final_errors}"
            f"{f', 🔄 repaired={fixed_count}' if fixed_count > 0 else ''}"
        )

        return queries, refl_total_tokens, refl_total_cost

    def synthesize_for_database(self, db_name: str) -> List[Dict[str, Any]]:
        """
        Synthesize queries for a single database (concurrent LLM calls + concurrent validation)

        Workflow:
        1. Read DDL / sample data / column stats + load allocated differences
        2. Flatten all rounds of all difficulties into independent tasks
        3. Use ThreadPoolExecutor to concurrently call LLM to generate queries
        4. Collect all results then unified concurrent validation + serial reflection repair

        Args:
            db_name: Database name

        Returns:
            List of all generated queries
        """
        logger.info(f"===== Start synthesizing for database {db_name} synthesize queries =====")

        # 1. Read DDL
        ddl = self._read_ddl(db_name)
        if not ddl:
            logger.warning(f"Skipping database {db_name}: DDL is empty")
            return []

        # 2. Extract table names
        table_names = self._get_table_names_from_ddl(ddl)
        if not table_names:
            logger.warning(f"Skipping database {db_name}: no tables found")
            return []

        logger.info(f"Database {db_name} contains {len(table_names)} tables: {table_names}")

        # 3. Read sample data
        sample_data = self._fetch_sample_data(db_name, table_names)

        # 4. Get MIN/MAX stats for numeric/date columns (helps LLM generate reasonable range query conditions)
        column_stats = self._fetch_column_stats(db_name, ddl)
        if column_stats:
            logger.info(f"Database {db_name} column stats info obtained")

        # 5. Load allocated differences for this database (query_requirements + query_patterns)
        allocated_diffs = self.diff_allocation.get_query_requirements_and_patterns(db_name)
        if not allocated_diffs:
            logger.warning(f"Skipping database {db_name}: no allocated differences")
            return []

        diff_ids = [d['id'] for d in allocated_diffs]
        logger.info(
            f"Database {db_name} assigned {len(allocated_diffs)} differences: "
            f"{diff_ids}"
        )

        # 6. Calculate difficulty round allocation
        difficulties = ["easy", "medium", "hard", "extra"]
        weights = self.difficulty_weights
        min_weight = min(weights.values())

        difficulty_loops = {}
        for d in difficulties:
            difficulty_loops[d] = max(1, round(self.loops_per_difficulty * weights.get(d, 0.25) / min_weight))

        total_loops = sum(difficulty_loops.values())
        total_estimated = total_loops * self.queries_per_call
        logger.info(
            f"Difficulty round allocation: {difficulty_loops} (total {total_loops} round, "
            f"estimated ~{total_estimated} queries)"
        )

        # 7. Build flattened task list: [(difficulty, loop_idx), ...]
        # Each task uses same allocated_diffs (same database's differences are fixed)
        tasks = []
        for difficulty in difficulties:
            n_loops = difficulty_loops[difficulty]
            for loop_idx in range(n_loops):
                tasks.append((difficulty, loop_idx, n_loops))

        logger.info(
            f"Starting concurrent generation: {len(tasks)} LLM tasks, "
            f"max_workers={self.max_llm_workers}"
        )

        # 8. Concurrent LLM calls
        gen_total_tokens = 0
        gen_total_cost = 0.0

        def _generate_one(task_info):
            difficulty, loop_idx, n_loops = task_info
            task_label = f"{db_name}/{difficulty}/{loop_idx+1}"

            logger.info(
                f"[{task_label}] start generating | "
                f"allocated_diffs=[{', '.join(d['id'] for d in allocated_diffs)}]"
            )

            queries, tokens, cost = self._call_llm(
                db_name=db_name,
                ddl=ddl,
                sample_data=sample_data,
                allocated_diffs=allocated_diffs,
                difficulty=difficulty,
                column_stats=column_stats,
            )

            # Add metadata
            for q in queries:
                q['database'] = db_name
                q['dialect'] = self.target_dialect
                q['difficulty'] = difficulty
                q['loop_index'] = loop_idx
                q['allocated_diff_ids'] = diff_ids

            logger.info(f"[{task_label}] complete, generated {len(queries)} queries")
            return queries, tokens, cost

        all_queries = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_llm_workers) as pool:
            futures = {pool.submit(_generate_one, task): task for task in tasks}
            for fut in as_completed(futures):
                try:
                    queries, tokens, cost = fut.result()
                    all_queries.extend(queries)
                    gen_total_tokens += tokens
                    gen_total_cost += cost
                except Exception as e:
                    task = futures[fut]
                    logger.error(f"LLM task failed ({task[0]}/{task[1]}): {e}")

        gen_time = time.time() - start_time
        logger.info(
            f"LLM generation complete: {len(all_queries)} queries, "
            f"{gen_total_tokens:,} tokens, ${gen_total_cost:.4f}, "
            f"elapsed {gen_time:.1f}s (concurrent {self.max_llm_workers} workers)"
        )

        # 9. Unified validation + reflection repair
        refl_total_tokens = 0
        refl_total_cost = 0.0
        if self.reflection_enabled:
            logger.info(f"Starting validation and repair {len(all_queries)} queries...")
            val_start = time.time()
            all_queries, refl_total_tokens, refl_total_cost = self._validate_and_fix_queries(
                all_queries, db_name, ddl
            )
            val_time = time.time() - val_start
            logger.info(
                f"Validation+repair complete, elapsed {val_time:.1f}s, "
                f"reflection: {refl_total_tokens:,} tokens, ${refl_total_cost:.4f}"
            )

        # 10. Statistics
        difficulty_counts = {}
        for q in all_queries:
            d = q.get('difficulty', 'unknown')
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        logger.info(f"Database {db_name} Synthesis complete: difficulty distribution={difficulty_counts}")

        status_counts = {}
        reflection_stats = {'fixed': 0, 'failed': 0}
        if self.reflection_enabled:
            for q in all_queries:
                s = q.get('validation', {}).get('status', 'unknown')
                status_counts[s] = status_counts.get(s, 0) + 1
                refl = q.get('reflection', {})
                if refl.get('fixed'):
                    reflection_stats['fixed'] += 1
                elif refl and not refl.get('fixed'):
                    reflection_stats['failed'] += 1

            logger.info(
                f"Database {db_name} validation stats: "
                f"✅ valid={status_counts.get('valid', 0)}, "
                f"⚠️ empty={status_counts.get('empty', 0)}, "
                f"❌ error={status_counts.get('error', 0)}, "
                f"🔄 reflectionrepaired={reflection_stats['fixed']}, "
                f"reflection failed={reflection_stats['failed']}"
            )

        elapsed = time.time() - start_time

        # Save statistics
        self.stats[db_name] = {
            "llm_provider": self.llm_client.provider,
            "llm_model": self.llm_client.model,
            "llm_thinking": self.llm_client.model_config.get('thinking', {}),
            "total_queries": len(all_queries),
            "difficulty_counts": difficulty_counts,
            "allocated_diffs": len(allocated_diffs),
            "gen_tokens": gen_total_tokens,
            "gen_cost": gen_total_cost,
            "refl_tokens": refl_total_tokens,
            "refl_cost": refl_total_cost,
            "total_tokens": gen_total_tokens + refl_total_tokens,
            "total_cost": gen_total_cost + refl_total_cost,
            "elapsed": elapsed,
            "validation": status_counts if self.reflection_enabled else {},
            "reflection": reflection_stats if self.reflection_enabled else {},
        }

        logger.info(
            f"Database {db_name} Total elapsed: {elapsed:.1f}s, "
            f"Total {gen_total_tokens + refl_total_tokens:,} tokens, "
            f"${gen_total_cost + refl_total_cost:.4f}"
        )

        return all_queries

    def synthesize_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synthesize queries for all databases

        Returns:
            {db_name: [queries]} dict
        """
        databases = self._discover_databases()
        all_results = {}

        for db_name in databases:
            queries = self.synthesize_for_database(db_name)
            all_results[db_name] = queries

            # Save single database results
            self._save_results(db_name, queries)

        # Save summary results
        self._save_summary(all_results)

        # Print final summary
        self._print_summary()

        return all_results

    def _print_summary(self):
        """Print final summary and save statistics file."""
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Query Synthesis complete: {len(self.stats)} databases (this run)")
        logger.info(f"{'#' * 60}")

        # Print this run's statistics
        for db_name, stat in sorted(self.stats.items()):
            logger.info(
                f"  {db_name}: {stat.get('total_queries', 0)} queries, "
                f"{stat.get('total_tokens', 0):,} tokens, "
                f"${stat.get('total_cost', 0.0):.4f}, "
                f"{stat.get('elapsed', 0.0):.1f}s"
            )

        # Save statistics file (merge mode: append new, overwrite existing)
        stats_file = self.output_dir / f"synthesis_stats_{self.target_dialect}.json"

        existing_dbs = {}
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                existing_dbs = existing.get("databases", {})
            except Exception:
                pass

        # Merge: new records overwrite old (update not accumulate)
        existing_dbs.update(self.stats)

        # Recalculate totals based on all merged databases
        total_queries = 0
        total_gen_tokens = 0
        total_gen_cost = 0.0
        total_refl_tokens = 0
        total_refl_cost = 0.0
        total_elapsed = 0.0

        for stat in existing_dbs.values():
            total_queries += stat.get("total_queries", 0)
            total_gen_tokens += stat.get("gen_tokens", 0)
            total_gen_cost += stat.get("gen_cost", 0.0)
            total_refl_tokens += stat.get("refl_tokens", 0)
            total_refl_cost += stat.get("refl_cost", 0.0)
            total_elapsed += stat.get("elapsed", 0.0)

        total_tokens = total_gen_tokens + total_refl_tokens
        total_cost = total_gen_cost + total_refl_cost

        logger.info(f"\n  {'=' * 50}")
        logger.info(f"  Total (including history): {len(existing_dbs)} databases, {total_queries:,} queries")
        logger.info(f"  Query generation:       {total_gen_tokens:,} tokens, ${total_gen_cost:.4f}")
        logger.info(f"  Reflection repair:  {total_refl_tokens:,} tokens, ${total_refl_cost:.4f}")
        logger.info(f"  Total:             {total_tokens:,} tokens, ${total_cost:.4f}")
        logger.info(f"  Cumulative elapsed: {total_elapsed:.1f}s")

        stats_output = {
            "split": self.split,
            "dialect": self.target_dialect,
            "llm_provider": self.llm_client.provider,
            "llm_model": self.llm_client.model,
            "llm_thinking": self.llm_client.model_config.get('thinking', {}),
            "total_databases": len(existing_dbs),
            "total_queries": total_queries,
            "total_gen_tokens": total_gen_tokens,
            "total_gen_cost": total_gen_cost,
            "total_refl_tokens": total_refl_tokens,
            "total_refl_cost": total_refl_cost,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_elapsed": total_elapsed,
            "databases": existing_dbs,
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)
        logger.info(f"Statistics file saved: {stats_file}")

    def _save_results(self, db_name: str, queries: List[Dict[str, Any]]):
        """Save single database synthesis results"""
        dialect_short = {'mysql': 'mysql', 'postgresql': 'pg', 'oracle': 'oracle'}.get(self.target_dialect, self.target_dialect)
        output_file = self.output_dir / f"{db_name}_{dialect_short}_queries.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved: {output_file}")

    def _save_summary(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """Save summary results."""
        summary = {
            'target_dialect': self.target_dialect,
            'total_databases': len(all_results),
            'total_queries': sum(len(qs) for qs in all_results.values()),
            'per_database': {
                db_name: len(queries) for db_name, queries in all_results.items()
            },
            'config': {
                'queries_per_call': self.queries_per_call,
                'loops_per_difficulty': self.loops_per_difficulty,
                'difficulty_weights': self.difficulty_weights,
                'sample_rows': self.sample_rows,
                'split': self.split,
                'llm_provider': self.llm_client.provider,
                'llm_model': self.llm_client.model,
                'llm_thinking': self.llm_client.model_config.get('thinking', {}),
            },
        }

        summary_file = self.output_dir / f"synthesis_summary_{self.target_dialect}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved: {summary_file}")
