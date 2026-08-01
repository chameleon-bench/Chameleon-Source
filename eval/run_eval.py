"""
CLI entry point for SQL translation evaluation.

Usage:
    python -m eval.run_eval \
        --source-queries dev_source_queries.json \
        --translated-queries dev_translated_queries.json \
        --output eval_results.json \
        --split dev \
        --dataset-dir dataset \
        --db-config config/database_sync.yaml \
        --workers 8 \
        --query-timeout 30

Input file formats:
    Source queries file (JSON array):
        [
            {
                "query_id": 1,
                "difficulty": "easy",
                "sql": "SELECT * FROM ...",
                "database": "my_db",
                "dialect": "mysql"
            },
            ...
        ]

    Translated queries file (JSON array):
        [
            {
                "query_id": 1,
                "translated_sql": "SELECT * FROM ..."
            },
            ...
        ]

Output:
    A JSON file with:
    {
        "report": {
            "total_queries": 100,
            "execution_score": "95.0%",   # ES: target SQL executes successfully
            "execution_match": "82.0%",   # EM: result sets match
            "by_difficulty": {...},
            "by_database": {...},
            ...
        },
        "results": [...]
    }
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluator import TranslationEvaluator
from utils.logging_config import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SQL translation quality by executing source and translated SQL '
                    'on their respective database engines and comparing result sets.'
    )
    parser.add_argument(
        '--source-queries', required=True,
        help='JSON file containing all source queries',
    )
    parser.add_argument(
        '--translated-queries', required=True,
        help='JSON file containing all translated queries',
    )
    parser.add_argument(
        '--output', default='eval_results.json',
        help='Output results file path (default: eval_results.json)',
    )
    parser.add_argument(
        '--split', default='dev', choices=['train', 'dev', 'test'],
        help='Dataset split (default: dev)',
    )
    parser.add_argument(
        '--dataset-dir', default='dataset',
        help='Dataset directory containing {split}/{db_id}/schema/ (default: dataset)',
    )
    parser.add_argument(
        '--db-config', default='config/database_sync.yaml',
        help='Database connection config YAML (default: config/database_sync.yaml)',
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel workers (default: 8)',
    )
    parser.add_argument(
        '--query-timeout', type=int, default=30,
        help='SQL execution timeout in seconds (default: 30)',
    )
    parser.add_argument(
        '--max-queries', type=int, default=None,
        help='Maximum number of queries to evaluate (default: all)',
    )

    args = parser.parse_args()

    logger.info(f"Starting evaluation")
    logger.info(f"  Source queries:     {args.source_queries}")
    logger.info(f"  Translated queries: {args.translated_queries}")
    logger.info(f"  Output:             {args.output}")
    logger.info(f"  Split:              {args.split}")
    logger.info(f"  Dataset dir:        {args.dataset_dir}")
    logger.info(f"  Workers:            {args.workers}")

    evaluator = TranslationEvaluator(
        db_config_path=args.db_config,
        dataset_dir=args.dataset_dir,
        split=args.split,
        query_timeout=args.query_timeout,
        max_workers=args.workers,
    )

    report = evaluator.evaluate(
        source_queries_file=args.source_queries,
        translated_queries_file=args.translated_queries,
        output_file=args.output,
        max_queries=args.max_queries,
    )

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete")
    print(f"{'='*60}")
    print(f"  Total queries:    {report['total_queries']}")
    print(f"  Translate OK:     {report['translate_ok']}")
    print(f"  Source exec OK:   {report['source_exec_ok']}")
    print(f"  Target exec OK:   {report['target_exec_ok']}")
    print(f"  Results match:    {report['results_match']}")
    print(f"  Execution Score:  {report['execution_score']}")
    print(f"  Execution Match:  {report['execution_match']}")
    print(f"  Elapsed:          {report['elapsed_seconds']}s")
    print(f"{'='*60}")

    if report['by_difficulty']:
        print(f"\n  By difficulty:")
        for diff, stats in sorted(report['by_difficulty'].items()):
            es = f"{stats['target_ok']}/{stats['total']}"
            em = f"{stats['match']}/{stats['total']}"
            print(f"    {diff:10s}  ES={es:10s}  EM={em:10s}")

    print()


if __name__ == '__main__':
    main()
