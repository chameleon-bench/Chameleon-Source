"""
Data synthesis main engine (Synthesizer).

LLM business operator mode:
  Phase 0 (Search): EntityHarvester searches and collects entity pools
  Phase 1 (LLM): Pass DDL + seed data + entity pool + data_requirements -> LLM generates K business operations in a single call
  Phase 2 (Format): Format -> tri-dialect INSERT SQL files (MySQL + PG + Oracle)

Data requirements source:
  allocation_{split}.json -> database_sync_requirements.json's data_requirements

Supports:
  - Specified split (--split dev/train/test)
  - Full synthesis (all databases in the specified split)
  - Single database synthesis (--db academic_research_and_evaluation)
  - Reproducible (--seed 42)
"""

import json
import sqlite3
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_synthesis.schema_parser import (
    SchemaInfo, TripleSchemaInfo,
    parse_all_triple_schemas,
)
from data_synthesis.llm_data_generator import LLMDataGenerator
from data_synthesis.entity_harvester import EntityHarvester
from data_synthesis.insert_formatter import InsertFormatter
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Config file path
_CONFIG_PATH = PROJECT_ROOT / "config" / "data_synthesis.yaml"

def _load_synthesis_config() -> Dict[str, Any]:
    """Load synthesis configuration."""
    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_allocation(allocation_path: str) -> Dict[str, Any]:
    """Load allocation file (diff_ids allocation)."""
    try:
        with open(allocation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("allocation", {})
    except Exception as e:
        logger.warning(f"Failed to load allocation file: {e}")
        return {}


def _load_sync_requirements(requirements_path: str) -> Dict[str, Dict]:
    """
    Load database_sync_requirements.json, return {diff_id: diff_info} mapping.
    """
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        diffs = data.get("differences", [])
        return {d["id"]: d for d in diffs if "id" in d}
    except Exception as e:
        logger.warning(f"Failed to load sync requirements file: {e}")
        return {}


def _read_sqlite_seed_data(sqlite_path: str) -> Dict[str, List[Dict]]:
    """
    Read SQLite seed data, return {table_name: [row_dict, ...]}.
    Each table has 2 rows of sample data from the SynSQL-2.5M dataset.
    """
    seed_data = {}
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        for (table_name,) in tables:
            cursor.execute(f'SELECT * FROM "{table_name}"')
            cols = [desc[0] for desc in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
            seed_data[table_name] = rows

        conn.close()
    except Exception as e:
        logger.warning(f"Failed to read SQLite seed data: {e}")

    return seed_data


def _extract_data_requirements(
    diff_ids: List[str],
    sync_requirements: Dict[str, Dict],
) -> List[Dict]:
    """
    Extract data_requirements from allocated diff_ids.

    Returns:
        [{"diff_id": "DIFF_0002", "feature": "...", "requirements": ["...", ...]}]
    """
    result = []
    for diff_id in diff_ids:
        diff_info = sync_requirements.get(diff_id, {})
        test_reqs = diff_info.get("test_requirements", {})
        data_reqs = test_reqs.get("data_requirements", [])

        if data_reqs:
            result.append({
                "diff_id": diff_id,
                "feature": diff_info.get("feature", ""),
                "category": diff_info.get("category", ""),
                "requirements": data_reqs,
            })

    return result


class DataSynthesizer:
    """
    Data synthesis main engine.

    Uses LLM business operator mode (LLMDataGenerator -> InsertFormatter) to generate data.
    Each database calls LLM only once, generating data for K=50 business operations.
    """

    def __init__(
        self,
        split: str = "dev",
        database_dir: str = None,
        output_dir: str = None,
        seed: int = 42,
        llm_provider: str = "aliyun",
        llm_model: str = "qwen3.5-flash",
        allocation_path: str = None,
        requirements_path: str = None,
        concurrent_workers: int = None,
    ):
        self.split = split
        self.database_dir = Path(database_dir or str(PROJECT_ROOT / "database" / split))
        self.output_dir = Path(output_dir or str(PROJECT_ROOT / "output" / "synthesized_data"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.concurrent_workers = concurrent_workers

        # Load config
        config = _load_synthesis_config()

        # If concurrency not explicitly specified, read from config file
        if self.concurrent_workers is None:
            self.concurrent_workers = config.get("synthesis", {}).get("concurrent_workers", 4)

        # Load allocation and sync requirements
        self.allocation = _load_allocation(
            allocation_path or str(PROJECT_ROOT / "output" / "schema_expansion" / f"allocation_{split}.json")
        )
        self.sync_requirements = _load_sync_requirements(
            requirements_path or str(PROJECT_ROOT / "data" / "database_sync_requirements.json")
        )
        logger.info(f"  Allocation: {len(self.allocation)} databases")
        logger.info(f"  Sync Requirements: {len(self.sync_requirements)} differences")

        # Save LLM config (create independent LLMDataGenerator instance per synthesize_database call)
        llm_data_cfg = config.get("llm_data_gen", {})
        self.llm_config = {
            "provider": llm_provider,
            "model": llm_model,
            "seed": seed,
            "scenarios_per_completion": llm_data_cfg.get("scenarios_per_completion", 50),
            "temperature": llm_data_cfg.get("temperature", 0.9),
            "max_tokens": llm_data_cfg.get("max_tokens", 20480),
        }

        # Entity harvester (supports independent LLM config)
        harvester_cfg = config.get("entity_harvester", {})
        harvester_llm_cfg = harvester_cfg.get("llm", {})
        harvester_provider = harvester_llm_cfg.get("provider", llm_provider)
        harvester_model = harvester_llm_cfg.get("model", llm_model)
        self.entity_harvester = EntityHarvester(
            provider=harvester_provider,
            model=harvester_model,
            queries_per_table=harvester_cfg.get("queries_per_table", 4),
            concurrent_tables=harvester_cfg.get("concurrent_tables", 4),
            concurrent_queries=harvester_cfg.get("concurrent_queries", 3),
        )

        self.insert_formatter = InsertFormatter(
            batch_size=config.get("synthesis", {}).get("batch_size", 50)
        )

        # Result statistics
        self.stats: Dict[str, Dict] = {}

    # =================================================================
    # Cleanup
    # =================================================================

    def clean(self, db_names: Optional[List[str]] = None):
        """
        Clear data synthesis result files.

        Deletes database/dev/{db}/*_data_mysql.sql, *_data_pg.sql, *_data_oracle.sql
        """
        removed = 0

        if db_names:
            search_dirs = [self.database_dir / db for db in db_names
                          if (self.database_dir / db).is_dir()]
        else:
            search_dirs = [d for d in self.database_dir.iterdir() if d.is_dir()]

        for d in search_dirs:
            for pattern in ("*_data_mysql.sql", "*_data_pg.sql", "*_data_oracle.sql"):
                for f in d.glob(pattern):
                    f.unlink()
                    removed += 1
                    logger.info(f"  Deleted: {f.name}")

        logger.info(f"\n  Cleanup complete: {removed} files deleted")
        return removed

    def synthesize_database(
        self,
        db_name: str,
        schema: TripleSchemaInfo,
    ) -> bool:
        """
        Synthesize data for a database (full pipeline).

        Phase 0: Entity harvesting
        Phase 1: LLM single-call data generation
        Phase 2: Tri-dialect INSERT SQL formatting
        """
        start_time = time.time()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Synthesizing data: {db_name}")
        logger.info(f"{'=' * 60}")

        primary_schema = schema.mysql_schema or schema.pg_schema or schema.oracle_schema
        if primary_schema is None:
            logger.error(f"  No available schema, skipping")
            return False

        # === Load data_requirements (from allocation -> sync_requirements) ===
        alloc_info = self.allocation.get(db_name, {})
        diff_ids = alloc_info.get("diff_ids", [])
        data_requirements = _extract_data_requirements(diff_ids, self.sync_requirements)
        logger.info(f"  Allocation diffs: {len(diff_ids)}, with data_requirements: {len(data_requirements)}")

        # === Read SQLite seed data ===
        sqlite_path = self.database_dir / db_name / f"{db_name}.sqlite"
        seed_data = {}
        if sqlite_path.exists():
            seed_data = _read_sqlite_seed_data(str(sqlite_path))
            total_seed_rows = sum(len(rows) for rows in seed_data.values())
            logger.info(f"  SQLite seed data: {len(seed_data)} tables, {total_seed_rows} rows")
        else:
            logger.warning(f"  SQLite seed file not found: {sqlite_path}")

        # === Phase 0: Entity harvesting ===
        logger.info(f"  Phase 0: Searching and collecting entity pool...")
        harvest_tokens_before = self.entity_harvester.stats["total_tokens"]
        harvest_cost_before = self.entity_harvester.stats["total_cost"]
        entity_pool = self.entity_harvester.harvest(primary_schema)
        harvest_tokens = self.entity_harvester.stats["total_tokens"] - harvest_tokens_before
        harvest_cost = self.entity_harvester.stats["total_cost"] - harvest_cost_before

        # === Phase 1: LLM single-call data generation ===
        logger.info(f"  Phase 1: LLM data generation (single call)...")

        # Build data_requirements text (replaces the old constraint_text)
        data_req_text = self._build_data_requirements_text(data_requirements)

        # Create independent LLMDataGenerator instance per database (concurrency-safe)
        llm_generator = LLMDataGenerator(**self.llm_config)

        data = llm_generator.generate_all(
            schema=primary_schema,
            selected_constraints=None,
            constraint_map=None,
            entity_pool=entity_pool,
            seed_data=seed_data,
            data_requirements_text=data_req_text,
            pg_schema=schema.pg_schema,
        )

        if not data:
            logger.error(f"  LLM data generation returned empty data")
            return False

        # === Phase 2: Tri-dialect INSERT formatting ===
        logger.info(f"  Phase 2: Formatting tri-dialect INSERT SQL...")
        generation_order = primary_schema.topological_order()

        db_output_dir = str(self.database_dir / db_name)

        files = self.insert_formatter.save_all(
            db_name=db_name,
            data=data,
            mysql_schema=primary_schema,
            pg_schema=schema.pg_schema,
            oracle_schema=schema.oracle_schema,
            generation_order=generation_order,
            output_dir=db_output_dir,
        )

        # === Statistics ===
        elapsed = time.time() - start_time
        total_rows = sum(len(rows) for rows in data.values())
        llm_stats = llm_generator.stats

        synth_tokens = llm_stats.get("total_tokens", 0)
        synth_cost = llm_stats.get("total_cost", 0.0)

        self.stats[db_name] = {
            "tables": len(data),
            "total_rows": total_rows,
            "diff_ids": len(diff_ids),
            "data_requirements": len(data_requirements),
            "elapsed": elapsed,
            "harvest_tokens": harvest_tokens,
            "harvest_cost": harvest_cost,
            "synth_tokens": synth_tokens,
            "synth_cost": synth_cost,
            "total_tokens": harvest_tokens + synth_tokens,
            "total_cost": harvest_cost + synth_cost,
            "files": files,
        }

        logger.info(
            f"  {db_name} complete: {len(data)} tables, {total_rows} rows, "
            f"harvest={harvest_tokens:,}+synth={synth_tokens:,}={harvest_tokens+synth_tokens:,} tokens, "
            f"${harvest_cost + synth_cost:.4f}, {elapsed:.1f}s"
        )
        logger.info(f"     MySQL:  {files.get('mysql')}")
        logger.info(f"     PG:     {files.get('pg')}")
        logger.info(f"     Oracle: {files.get('oracle')}")

        return True

    def _build_data_requirements_text(self, data_requirements: List[Dict]) -> str:
        """Build data_requirements into LLM prompt text."""
        if not data_requirements:
            return ""

        lines = []
        for req in data_requirements:
            diff_id = req["diff_id"]
            feature = req["feature"]
            category = req["category"]
            requirements = req["requirements"]

            lines.append(f"- [{diff_id}] {feature} ({category}):")
            for r in requirements:
                lines.append(f"    {r}")

        return "\n".join(lines)

    def synthesize_all(
        self,
        schemas: Dict[str, TripleSchemaInfo],
        order: Optional[List[str]] = None,
    ):
        """
        Synthesize data for all databases (concurrent execution).

        Uses ThreadPoolExecutor to process multiple databases simultaneously,
        each database creates an independent LLMDataGenerator instance, no interference.
        """
        if order is None:
            order = sorted(schemas.keys())

        # Filter out non-existent databases
        valid_order = [db for db in order if db in schemas]
        skipped = len(order) - len(valid_order)
        if skipped > 0:
            logger.warning(f"  Skipped {skipped} unknown databases")

        workers = self.concurrent_workers
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Starting data synthesis: {len(valid_order)} databases, concurrency={workers}")
        logger.info(f"# split={self.split}, seed={self.seed}")
        logger.info(f"{'#' * 60}")

        success = 0
        fail = 0

        if workers <= 1:
            # Serial mode
            for db_name in valid_order:
                try:
                    if self.synthesize_database(db_name, schemas[db_name]):
                        success += 1
                    else:
                        fail += 1
                except Exception as e:
                    logger.error(f"  {db_name} synthesis failed: {e}")
                    import traceback
                    traceback.print_exc()
                    fail += 1
        else:
            # Concurrent mode
            def _process_db(db_name):
                try:
                    return (db_name, self.synthesize_database(db_name, schemas[db_name]), None)
                except Exception as e:
                    return (db_name, False, e)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_db, db_name): db_name
                    for db_name in valid_order
                }

                for future in as_completed(futures):
                    db_name, ok, error = future.result()
                    if error is not None:
                        logger.error(f"  {db_name} synthesis failed: {error}")
                        fail += 1
                    elif ok:
                        success += 1
                    else:
                        fail += 1

                    logger.info(
                        f"  Progress: {success + fail}/{len(valid_order)} "
                        f"(success {success}, fail {fail})"
                    )

        self._print_summary(success, fail)

    def _print_summary(self, success: int, fail: int):
        """Print final summary and save statistics file."""
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Data synthesis complete: success {success}, fail {fail}")
        logger.info(f"{'#' * 60}")

        total_rows = 0
        total_tables = 0
        total_harvest_tokens = 0
        total_synth_tokens = 0
        total_harvest_cost = 0.0
        total_synth_cost = 0.0
        total_elapsed = 0.0

        for db_name, stat in sorted(self.stats.items()):
            total_rows += stat["total_rows"]
            total_tables += stat["tables"]
            total_harvest_tokens += stat.get("harvest_tokens", 0)
            total_synth_tokens += stat.get("synth_tokens", 0)
            total_harvest_cost += stat.get("harvest_cost", 0.0)
            total_synth_cost += stat.get("synth_cost", 0.0)
            total_elapsed += stat.get("elapsed", 0.0)
            logger.info(
                f"  {db_name}: {stat['tables']} tables, "
                f"{stat['total_rows']} rows, "
                f"{stat.get('total_tokens', 0):,} tokens, "
                f"${stat.get('total_cost', 0.0):.4f}, "
                f"{stat['elapsed']:.1f}s"
            )

        total_tokens = total_harvest_tokens + total_synth_tokens
        total_cost = total_harvest_cost + total_synth_cost

        logger.info(f"\n  {'=' * 50}")
        logger.info(f"  Total: {len(self.stats)} databases, {total_tables} tables, {total_rows:,} rows")
        logger.info(f"  Entity Harvester: {total_harvest_tokens:,} tokens, ${total_harvest_cost:.4f}")
        logger.info(f"  Data Synthesis:   {total_synth_tokens:,} tokens, ${total_synth_cost:.4f}")
        logger.info(f"  Combined:         {total_tokens:,} tokens, ${total_cost:.4f}")
        logger.info(f"  Total elapsed: {total_elapsed:.1f}s")

        # Save statistics file (append-merge mode: merge existing records + current records)
        stats_file = self.output_dir / f"synthesis_stats_{self.split}.json"

        # Read existing statistics
        existing_dbs = {}
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                existing_dbs = existing.get("databases", {})
                logger.info(f"  Existing stats file: {len(existing_dbs)} database records")
            except Exception:
                pass

        # Merge: current results overwrite existing same-name databases (update on re-run)
        merged_dbs = dict(existing_dbs)
        for db_name, stat in self.stats.items():
            merged_dbs[db_name] = {k: v for k, v in stat.items() if k != "files"}

        # Recalculate summary
        m_rows = sum(s.get("total_rows", 0) for s in merged_dbs.values())
        m_tables = sum(s.get("tables", 0) for s in merged_dbs.values())
        m_harvest_tokens = sum(s.get("harvest_tokens", 0) for s in merged_dbs.values())
        m_synth_tokens = sum(s.get("synth_tokens", 0) for s in merged_dbs.values())
        m_harvest_cost = sum(s.get("harvest_cost", 0.0) for s in merged_dbs.values())
        m_synth_cost = sum(s.get("synth_cost", 0.0) for s in merged_dbs.values())
        m_elapsed = sum(s.get("elapsed", 0.0) for s in merged_dbs.values())

        stats_output = {
            "split": self.split,
            "summary": {
                "databases": len(merged_dbs),
                "total_tables": m_tables,
                "total_rows": m_rows,
                "harvest_tokens": m_harvest_tokens,
                "harvest_cost": round(m_harvest_cost, 6),
                "synth_tokens": m_synth_tokens,
                "synth_cost": round(m_synth_cost, 6),
                "total_tokens": m_harvest_tokens + m_synth_tokens,
                "total_cost": round(m_harvest_cost + m_synth_cost, 6),
                "total_elapsed": round(m_elapsed, 1),
            },
            "databases": {k: merged_dbs[k] for k in sorted(merged_dbs.keys())},
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)
        logger.info(f"  Statistics file saved: {stats_file}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Data synthesis main engine (LLM business operator mode)")
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "test"],
                        help="Dataset split (train/dev/test, default dev)")
    parser.add_argument("--database-dir", default=None,
                        help="Database schema directory (default database/{split})")
    parser.add_argument("--output-dir", default=None,
                        help="Synthesized data output directory")
    parser.add_argument("--db", type=str, default=None,
                        help="Process only specified databases (comma-separated)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--clean", action="store_true",
                        help="Clear all data synthesis results (INSERT SQL)")
    parser.add_argument("--llm-provider", default=None,
                        help="LLM provider (default from config file)")
    parser.add_argument("--llm-model", default=None,
                        help="LLM model (default from config file)")
    parser.add_argument("--allocation", default=None,
                        help="Allocation file path")
    parser.add_argument("--requirements", default=None,
                        help="database_sync_requirements.json file path")
    parser.add_argument("--concurrent", type=int, default=None,
                        help="Number of concurrent databases (default from config file, set to 1 for serial)")

    args = parser.parse_args()

    split = args.split
    db_names = [db.strip() for db in args.db.split(',')] if args.db else None

    # database_dir defaults based on split
    database_dir = args.database_dir or str(PROJECT_ROOT / "database" / split)

    # Read LLM defaults from config file
    config = _load_synthesis_config()
    llm_cfg = config.get("llm", {})
    llm_provider = args.llm_provider or llm_cfg.get("provider", "aliyun")
    llm_model = args.llm_model or llm_cfg.get("model", "qwen3.5-flash")

    # Create synthesizer
    synthesizer = DataSynthesizer(
        split=split,
        database_dir=database_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        llm_provider=llm_provider,
        llm_model=llm_model,
        allocation_path=args.allocation,
        requirements_path=args.requirements,
        concurrent_workers=args.concurrent,
    )

    # --clean: cleanup mode
    if args.clean:
        synthesizer.clean(db_names=db_names)
        return

    # Parse all schemas (tri-dialect)
    logger.info(f"Parsing all database schemas (MySQL + PostgreSQL + Oracle) [split={split}]...")
    schemas = parse_all_triple_schemas(database_dir)
    logger.info(f"  Parsed {len(schemas)} databases")

    # Determine processing order
    if db_names:
        order = db_names
    else:
        order = sorted(schemas.keys())

    # Execute synthesis
    synthesizer.synthesize_all(schemas, order=order)


if __name__ == "__main__":
    main()
