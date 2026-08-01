"""
Query synthesis entry script.

Usage:
    # Synthesize MySQL queries
    python -m query_synthesis.run --dialect mysql

    # Synthesize Oracle queries
    python -m query_synthesis.run --dialect oracle

    # Specify config file
    python -m query_synthesis.run --config config/query_synthesis.yaml --dialect mysql

    # Synthesize for specific databases only
    python -m query_synthesis.run --dialect mysql --databases authors books

    # Specify split
    python -m query_synthesis.run --dialect mysql --split test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import setup_logger, get_logger
from query_synthesis.query_synthesizer import QuerySynthesizer


def main():
    parser = argparse.ArgumentParser(description="SQL query synthesis tool")
    parser.add_argument(
        '--dialect', type=str, choices=['mysql', 'postgresql', 'oracle'],
        help='Target SQL dialect (mysql, postgresql, or oracle), overrides config file setting'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Config file path (default: config/query_synthesis.yaml)'
    )
    parser.add_argument(
        '--databases', type=str, nargs='+', default=None,
        help='Synthesize only specified databases (space-separated)'
    )
    parser.add_argument(
        '--split', type=str, choices=['dev', 'test', 'train'], default=None,
        help='Dataset split (overrides config file setting)'
    )
    parser.add_argument(
        '--loops-per-difficulty', type=int, default=None,
        help='Base number of loops per difficulty level (overrides config file)'
    )
    parser.add_argument(
        '--queries-per-call', type=int, default=None,
        help='Number of queries generated per LLM call (overrides config file)'
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logger(level=20)  # INFO
    logger = get_logger(__name__)

    # Create synthesizer
    synthesizer = QuerySynthesizer(config_path=args.config)

    # Command-line arguments override config
    if args.dialect:
        synthesizer.target_dialect = args.dialect
        # Re-initialize database manager
        synthesizer.db_manager = synthesizer._init_db_manager()

    if args.split:
        # Switching split requires reloading allocation file and schema paths
        synthesizer.split = args.split
        alloc_cfg = synthesizer.config.get('diff_allocation', {})
        alloc_file = str(PROJECT_ROOT / f"output/schema_expansion/allocation_{args.split}.json")
        req_file = str(PROJECT_ROOT / alloc_cfg.get('requirements_file', 'data/database_sync_requirements.json'))
        from query_synthesis.diff_allocation_loader import DiffAllocationLoader
        synthesizer.diff_allocation = DiffAllocationLoader(alloc_file, req_file)
        # Update schema path
        synthesizer.schema_dir = synthesizer.database_dir / args.split
        if not synthesizer.schema_dir.exists():
            synthesizer.schema_dir = synthesizer.database_dir
        # Update output directory (subdirectory per split)
        synthesizer.output_dir = synthesizer.output_base_dir / args.split
        synthesizer.output_dir.mkdir(parents=True, exist_ok=True)

    if args.loops_per_difficulty is not None:
        synthesizer.loops_per_difficulty = args.loops_per_difficulty
    if args.queries_per_call is not None:
        synthesizer.queries_per_call = args.queries_per_call

    logger.info(
        f"Start synthesizing {synthesizer.target_dialect} queries, "
        f"loops_per_difficulty={synthesizer.loops_per_difficulty}, "
        f"difficulty_weights={synthesizer.difficulty_weights}, "
        f"queries_per_call={synthesizer.queries_per_call}"
    )

    # Execute synthesis
    if args.databases:
        # Synthesize only specified databases
        all_results = {}
        for db_name in args.databases:
            queries = synthesizer.synthesize_for_database(db_name)
            all_results[db_name] = queries
            synthesizer._save_results(db_name, queries)
        synthesizer._save_summary(all_results)
        synthesizer._print_summary()
    else:
        # Synthesize all databases
        synthesizer.synthesize_all()

    logger.info("Synthesis complete!")


if __name__ == '__main__':
    main()
