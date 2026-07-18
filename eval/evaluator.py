"""
Translation evaluator: executes source and translated SQL on their respective
database engines and compares result sets.

Input format:
  - Source queries file: JSON array of {query_id, difficulty, sql, database, dialect}
  - Translated queries file: JSON array of {query_id, translated_sql}

The evaluator does NOT call any LLM for translation. It assumes the translated
SQL is already provided. This decouples evaluation from translation, allowing
any translation system (LLM, rule-based, etc.) to be benchmarked.

Usage:
    from eval.evaluator import TranslationEvaluator

    evaluator = TranslationEvaluator(
        db_config_path='config/database_sync.yaml',
        dataset_dir='dataset',          # contains {split}/{db_id}/schema/
        split='dev',
    )
    report = evaluator.evaluate(
        source_queries_file='dev_source_queries.json',
        translated_queries_file='dev_translated_queries.json',
        output_file='eval_results.json',
    )
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from eval.compare import compare_results
from eval.sql_utils import pin_random_functions, pin_time_functions
from utils.db_utils import DatabaseConfig, DatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)

DIALECT_DISPLAY = {
    'mysql': 'MySQL',
    'pg': 'PostgreSQL',
    'postgresql': 'PostgreSQL',
    'oracle': 'Oracle',
}

DIRECTION_CONFIG = {
    'mysql2pg':       {'source': 'mysql',  'target': 'pg'},
    'pg2mysql':       {'source': 'pg',     'target': 'mysql'},
    'mysql2oracle':   {'source': 'mysql',  'target': 'oracle'},
    'oracle2mysql':   {'source': 'oracle', 'target': 'mysql'},
    'pg2oracle':      {'source': 'pg',     'target': 'oracle'},
    'oracle2pg':      {'source': 'oracle', 'target': 'pg'},
}

SCHEMA_FILE = {
    'mysql':  '{db}_schema_mysql.sql',
    'pg':     '{db}_schema_pg.sql',
    'oracle': '{db}_schema_oracle.sql',
}


class TranslationEvaluator:
    """
    Evaluates SQL translation quality by executing source and translated SQL
    on their respective database engines and comparing result sets.
    """

    def __init__(
        self,
        db_config_path: str = 'config/database_sync.yaml',
        dataset_dir: str = 'dataset',
        split: str = 'dev',
        query_timeout: int = 30,
        max_workers: int = 8,
    ):
        self.split = split
        self.dataset_dir = Path(dataset_dir)
        self.query_timeout = query_timeout
        self.max_workers = max_workers

        with open(db_config_path, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.safe_load(f)

        db_cfg = raw_cfg.get('database', raw_cfg)

        self._db_managers: Dict[str, DatabaseManager] = {}
        self._oracle_password = db_cfg.get('oracle', {}).get('import_password', 'YOUR_ORACLE_USER_PASSWORD')

        mysql_cfg = db_cfg['mysql']
        self._db_managers['mysql'] = DatabaseManager(
            DatabaseConfig(
                host=mysql_cfg['host'],
                port=mysql_cfg['port'],
                user=mysql_cfg['user'],
                password=str(mysql_cfg['password']),
            ),
            db_type='mysql',
        )

        pg_cfg = db_cfg['postgresql']
        self._db_managers['pg'] = DatabaseManager(
            DatabaseConfig(
                host=pg_cfg['host'],
                port=pg_cfg['port'],
                user=pg_cfg['user'],
                password=str(pg_cfg['password']),
            ),
            db_type='postgresql',
        )

        if 'oracle' in db_cfg:
            oracle_cfg = db_cfg['oracle']
            service_name = oracle_cfg.get('service_name', 'XE')
            self._oracle_cfg = {
                'host': oracle_cfg['host'],
                'port': oracle_cfg['port'],
                'service_name': service_name,
                'password': self._oracle_password,
            }
        else:
            self._oracle_cfg = None

        logger.info(
            f"TranslationEvaluator initialized: split={split}, "
            f"dataset_dir={dataset_dir}, workers={max_workers}"
        )

    def _get_oracle_manager(self, db_name: str) -> DatabaseManager:
        """Create an Oracle manager for a specific database (schema = user)."""
        oracle_user = db_name.upper()[:30]
        config = DatabaseConfig(
            host=self._oracle_cfg['host'],
            port=self._oracle_cfg['port'],
            user=oracle_user,
            password=self._oracle_cfg['password'],
            database=None,
        )
        config.service_name = self._oracle_cfg['service_name']
        return DatabaseManager(config, db_type='oracle')

    def _load_ddl(self, db_name: str, dialect: str) -> str:
        """Load DDL text for a database and dialect from the dataset directory."""
        schema_file = self.dataset_dir / self.split / db_name / 'schema' / \
            SCHEMA_FILE[dialect].format(db=db_name)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        return schema_file.read_text(encoding='utf-8')

    def _execute_sql(
        self,
        mgr: DatabaseManager,
        sql: str,
        dialect: str,
        db_name: str,
    ) -> Tuple[str, List[Dict], str]:
        """
        Execute SQL on the specified database engine.

        Returns:
            (status, rows, error) where status is 'valid', 'empty', or 'error'.
        """
        pinned_sql = pin_time_functions(sql, dialect)
        pinned_sql = pin_random_functions(pinned_sql, dialect)

        target_db = None if dialect == 'oracle' else db_name

        try:
            with mgr.get_connection(database=target_db) as conn:
                cursor = conn.cursor()
                if self.query_timeout and self.query_timeout > 0:
                    timeout_ms = self.query_timeout * 1000
                    if dialect == 'mysql':
                        cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {timeout_ms}")
                    elif dialect in ('pg', 'postgresql'):
                        cursor.execute(f"SET statement_timeout = '{timeout_ms}'")
                    elif dialect == 'oracle':
                        conn.call_timeout = timeout_ms

                cursor.execute(pinned_sql)
                if cursor.description:
                    if dialect == 'mysql':
                        rows = list(cursor.fetchall())
                    else:
                        columns = [desc[0] for desc in cursor.description]
                        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    rows = []
                cursor.close()
                return ('valid' if rows else 'empty'), rows, ''
        except Exception as e:
            return 'error', [], str(e)[:500]

    def evaluate_one(
        self,
        query: Dict[str, Any],
        translated_sql: str,
        direction: str,
        db_name: str,
        source_ddl: str,
        target_ddl: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query pair.

        Args:
            query: Source query dict with {query_id, difficulty, sql, ...}
            translated_sql: Pre-translated target SQL
            direction: Translation direction (e.g. 'mysql2pg')
            db_name: Database name
            source_ddl: Source dialect DDL text
            target_ddl: Target dialect DDL text
        """
        dir_cfg = DIRECTION_CONFIG[direction]
        source_dialect = dir_cfg['source']
        target_dialect = dir_cfg['target']
        source_name = DIALECT_DISPLAY.get(source_dialect, source_dialect)
        target_name = DIALECT_DISPLAY.get(target_dialect, target_dialect)

        eval_id = f"eval_{query.get('query_id', '?')}"
        source_sql = query['sql']
        difficulty = query.get('difficulty', '?')

        result = {
            'eval_id': eval_id,
            'query_id': query.get('query_id', '?'),
            'difficulty': difficulty,
            'direction': direction,
            'database': db_name,
            'source_sql': source_sql,
            'target_sql': translated_sql,
            'translate_ok': bool(translated_sql),
            'source_exec_ok': False,
            'target_exec_ok': False,
            'results_match': False,
            'comparison': {},
            'error': '',
        }

        if not translated_sql:
            result['error'] = 'Empty translated SQL'
            return result

        source_mgr = self._db_managers.get(source_dialect)
        if source_dialect == 'oracle' and self._oracle_cfg:
            source_mgr = self._get_oracle_manager(db_name)

        target_mgr = self._db_managers.get(target_dialect)
        if target_dialect == 'oracle' and self._oracle_cfg:
            target_mgr = self._get_oracle_manager(db_name)

        s_status, s_rows, s_err = self._execute_sql(source_mgr, source_sql, source_dialect, db_name)
        result['source_exec_ok'] = s_status != 'error'

        t_status, t_rows, t_err = self._execute_sql(target_mgr, translated_sql, target_dialect, db_name)
        result['target_exec_ok'] = t_status != 'error'

        if result['source_exec_ok'] and result['target_exec_ok']:
            comparison = compare_results(s_rows, t_rows, source_name, target_name)
            result['results_match'] = comparison['match']
            result['comparison'] = comparison
        elif t_status == 'error':
            result['comparison'] = {
                'match': False,
                'details': f"{target_name} execution failed: {t_err}",
            }
            result['error'] = f"{target_name} execution error: {t_err}"
        else:
            result['comparison'] = {
                'match': False,
                'details': f"{source_name} execution failed: {s_err}",
            }
            result['error'] = f"{source_name} execution error: {s_err}"

        status_icon = '[OK]' if result['results_match'] else (
            '[WARN]' if result['target_exec_ok'] else '[FAIL]'
        )
        logger.info(
            f"[{eval_id}] {status_icon} {difficulty} | "
            f"{source_name}={s_status}({len(s_rows)}) | "
            f"{target_name}={'ok' if result['target_exec_ok'] else 'err'} | "
            f"match={result['results_match']}"
        )

        return result

    def evaluate(
        self,
        source_queries_file: str,
        translated_queries_file: str,
        output_file: Optional[str] = None,
        max_queries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full evaluation from two input files.

        Args:
            source_queries_file: JSON file containing all source queries.
                Format: [{query_id, difficulty, sql, database, dialect}, ...]
            translated_queries_file: JSON file containing all translated queries.
                Format: [{query_id, translated_sql}, ...]
            output_file: Optional path to save results JSON.
            max_queries: Optional limit on number of queries to evaluate.

        Returns:
            Evaluation report dict.
        """
        with open(source_queries_file, 'r', encoding='utf-8') as f:
            source_queries = json.load(f)
        with open(translated_queries_file, 'r', encoding='utf-8') as f:
            translated_list = json.load(f)

        translated_map = {
            item['query_id']: item.get('translated_sql', '')
            for item in translated_list
        }

        # Group queries by database
        queries_by_db: Dict[str, List[Dict]] = {}
        for q in source_queries:
            db_name = q.get('database', '')
            if db_name not in queries_by_db:
                queries_by_db[db_name] = []
            queries_by_db[db_name].append(q)

        # Determine direction from first query's dialect
        first_query = source_queries[0]
        source_dialect = first_query.get('dialect', '')
        # Infer target dialect from translated queries file
        # (The user should specify it, but we try to infer)
        # For now, we need the direction to be specified externally

        logger.info(
            f"Loaded {len(source_queries)} source queries across {len(queries_by_db)} databases, "
            f"{len(translated_map)} translated queries"
        )

        all_results: List[Dict[str, Any]] = []
        start_time = time.time()

        for db_name, queries in sorted(queries_by_db.items()):
            if not db_name:
                continue

            # Determine direction from query dialect
            source_dialect = queries[0].get('dialect', '')
            direction = self._infer_direction(source_dialect, translated_queries_file)

            # Load DDL files
            try:
                source_ddl = self._load_ddl(db_name, source_dialect)
                target_dialect = DIRECTION_CONFIG[direction]['target']
                target_ddl = self._load_ddl(db_name, target_dialect)
            except FileNotFoundError as e:
                logger.error(f"DDL not found for {db_name}: {e}")
                continue

            # Filter queries that have translations
            eval_queries = []
            for q in queries:
                qid = q.get('query_id')
                if qid in translated_map:
                    eval_queries.append((q, translated_map[qid]))

            if max_queries:
                eval_queries = eval_queries[:max_queries]

            if not eval_queries:
                logger.warning(f"No translated queries found for {db_name}")
                continue

            logger.info(
                f"Evaluating {db_name}: {len(eval_queries)} queries "
                f"({DIALECT_DISPLAY.get(source_dialect, source_dialect)} -> "
                f"{DIALECT_DISPLAY.get(target_dialect, target_dialect)})"
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self.evaluate_one, q, tsql, direction, db_name, source_ddl, target_ddl
                    ): q
                    for q, tsql in eval_queries
                }
                for fut in as_completed(futures):
                    try:
                        all_results.append(fut.result())
                    except Exception as e:
                        q = futures[fut]
                        logger.error(f"Evaluation error for query {q.get('query_id')}: {e}")
                        all_results.append({
                            'eval_id': f"eval_{q.get('query_id', '?')}",
                            'query_id': q.get('query_id', '?'),
                            'difficulty': q.get('difficulty', '?'),
                            'direction': direction,
                            'database': db_name,
                            'source_sql': q.get('sql', ''),
                            'target_sql': '',
                            'translate_ok': False,
                            'source_exec_ok': False,
                            'target_exec_ok': False,
                            'results_match': False,
                            'comparison': {},
                            'error': str(e),
                        })

        elapsed = time.time() - start_time
        report = self._build_report(all_results, elapsed)

        if output_file:
            output_data = {
                'report': report,
                'results': all_results,
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")

        return report

    def _infer_direction(self, source_dialect: str, translated_file: str) -> str:
        """
        Infer translation direction from source dialect and translated file name.
        Override this or pass direction explicitly for custom logic.
        """
        fname = Path(translated_file).stem.lower()
        for d in DIRECTION_CONFIG:
            source, target = DIRECTION_CONFIG[d]['source'], DIRECTION_CONFIG[d]['target']
            if source in source_dialect and target in fname:
                return d
        # Default: mysql2pg
        if source_dialect == 'mysql':
            return 'mysql2pg'
        if source_dialect == 'oracle':
            return 'oracle2mysql'
        if source_dialect == 'pg':
            return 'pg2mysql'
        return 'mysql2pg'

    def _build_report(self, results: List[Dict], elapsed: float) -> Dict[str, Any]:
        """Build summary report from evaluation results."""
        total = len(results)
        translate_ok = sum(1 for r in results if r.get('translate_ok'))
        source_ok = sum(1 for r in results if r.get('source_exec_ok'))
        target_ok = sum(1 for r in results if r.get('target_exec_ok'))
        match = sum(1 for r in results if r.get('results_match'))

        by_difficulty: Dict[str, Dict[str, int]] = {}
        for r in results:
            d = r.get('difficulty', 'unknown')
            if d not in by_difficulty:
                by_difficulty[d] = {'total': 0, 'target_ok': 0, 'match': 0}
            by_difficulty[d]['total'] += 1
            if r.get('target_exec_ok'):
                by_difficulty[d]['target_ok'] += 1
            if r.get('results_match'):
                by_difficulty[d]['match'] += 1

        by_database: Dict[str, Dict[str, int]] = {}
        for r in results:
            db = r.get('database', 'unknown')
            if db not in by_database:
                by_database[db] = {'total': 0, 'match': 0}
            by_database[db]['total'] += 1
            if r.get('results_match'):
                by_database[db]['match'] += 1

        failures = [
            {
                'eval_id': r.get('eval_id', r.get('query_id', '?')),
                'query_id': r.get('query_id', '?'),
                'difficulty': r.get('difficulty'),
                'database': r.get('database'),
                'error': r.get('error', ''),
                'comparison_details': r.get('comparison', {}).get('details', ''),
            }
            for r in results
            if not r.get('results_match')
        ][:20]

        return {
            'total_queries': total,
            'translate_ok': translate_ok,
            'source_exec_ok': source_ok,
            'target_exec_ok': target_ok,
            'results_match': match,
            'execution_score': f"{target_ok / total * 100:.1f}%" if total > 0 else '0%',
            'execution_match': f"{match / total * 100:.1f}%" if total > 0 else '0%',
            'match_rate': match / total if total > 0 else 0,
            'by_difficulty': by_difficulty,
            'by_database': by_database,
            'elapsed_seconds': f"{elapsed:.1f}",
            'failures_sample': failures,
        }
