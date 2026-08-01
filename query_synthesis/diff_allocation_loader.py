"""
Difference allocation loader.

Responsible for:
1. Loading allocated difference IDs per database from allocation_{split}.json
2. Looking up corresponding test_requirements from database_sync_requirements.json
3. Providing per-database difference information for query synthesis (query_requirements + query_patterns)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DiffAllocationLoader:
    """
    Loads database-level difference information from allocation and requirements files.

    Replaces the old DialectDifferenceParser random sampling approach,
    using differences precisely allocated to each database during the schema expansion stage.
    """

    def __init__(
        self,
        allocation_file: str,
        requirements_file: str,
    ):
        """
        Args:
            allocation_file: Path to the difference allocation file (e.g., output/schema_expansion/allocation_dev.json)
            requirements_file: Path to the difference requirements file (e.g., data/database_sync_requirements.json)
        """
        self.allocation_file = Path(allocation_file)
        self.requirements_file = Path(requirements_file)

        # allocation: {db_name: {diff_ids: [...], diffs: [...]}}
        self.allocation: Dict[str, Dict[str, Any]] = {}
        # diff_map: {diff_id: diff_full_info}
        self.diff_map: Dict[str, Dict[str, Any]] = {}

        self._load()

    def _load(self):
        """Load allocation and requirements files."""
        # Load requirements file
        if not self.requirements_file.exists():
            logger.error(f"Difference requirements file not found: {self.requirements_file}")
            return

        try:
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                req_data = json.load(f)

            for diff in req_data.get('differences', []):
                self.diff_map[diff['id']] = diff

            logger.info(f"Loaded {len(self.diff_map)} difference requirements from {self.requirements_file.name}")
        except Exception as e:
            logger.error(f"Failed to load difference requirements file: {e}")
            return

        # Load allocation file
        if not self.allocation_file.exists():
            logger.error(f"Difference allocation file not found: {self.allocation_file}")
            return

        try:
            with open(self.allocation_file, 'r', encoding='utf-8') as f:
                alloc_data = json.load(f)

            self.allocation = alloc_data.get('allocation', {})
            self.stats = alloc_data.get('stats', {})

            logger.info(
                f"Loaded difference allocations for {len(self.allocation)} databases from {self.allocation_file.name}"
            )
        except Exception as e:
            logger.error(f"Failed to load difference allocation file: {e}")

    def get_allocated_diffs(self, db_name: str) -> List[Dict[str, Any]]:
        """
        Get the list of differences allocated to a database.

        Args:
            db_name: Database name

        Returns:
            List of difference info, each containing id, category, feature, description, test_requirements, etc.
        """
        db_alloc = self.allocation.get(db_name)
        if not db_alloc:
            logger.warning(f"Database {db_name} not found in allocation file")
            return []

        result = []
        for diff_summary in db_alloc.get('diffs', []):
            diff_id = diff_summary['id']
            full_diff = self.diff_map.get(diff_id)
            if full_diff:
                result.append(full_diff)
            else:
                logger.warning(f"Difference {diff_id} not found in requirements file")

        return result

    def get_diff_ids(self, db_name: str) -> List[str]:
        """Get the list of difference IDs allocated to a database."""
        db_alloc = self.allocation.get(db_name)
        if not db_alloc:
            return []
        return db_alloc.get('diff_ids', [])

    def get_query_requirements_and_patterns(self, db_name: str) -> List[Dict[str, Any]]:
        """
        Get query_requirements + query_patterns for a database.

        Return format:
        [
            {
                "id": "DIFF_0001",
                "feature": "INDEX_OFFSET",
                "category": "dialect property",
                "description": "Array base index offset",
                "query_requirements": [...],
                "query_patterns": [...],
            },
            ...
        ]
        """
        diffs = self.get_allocated_diffs(db_name)
        result = []
        for diff in diffs:
            tr = diff.get('test_requirements', {})
            result.append({
                'id': diff['id'],
                'feature': diff['feature'],
                'category': diff.get('category', ''),
                'description': diff.get('description', ''),
                'query_requirements': tr.get('query_requirements', []),
                'query_patterns': tr.get('query_patterns', []),
            })
        return result

    def get_all_database_names(self) -> List[str]:
        """Get all database names that have allocated differences."""
        return list(self.allocation.keys())
