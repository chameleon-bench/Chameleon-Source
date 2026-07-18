"""
Dialect difference loader & built-in functions loader.

Responsible for:
1. Loading all dialect difference items from dialect_differences.json
2. Loading built-in functions from mysql_8_kb.json / pg_14_kb.json
3. Providing random sampling interface
"""

import json
import re
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DialectDifferenceParser:
    """
    Loads dialect difference items from dialect_differences.json.

    JSON format:
    [
      {
        "id": "D1.1",
        "category": "D1",
        "title": "Identifier quoting characters",
        "content": "... difference description ...",
        "mysql_examples": ["SELECT `order` ..."],
        "pg_examples": ["SELECT \"order\" ..."]
      },
      ...
    ]
    """

    def __init__(self, json_file_path: str):
        """
        Args:
            json_file_path: Path to dialect_differences.json
        """
        self.json_file_path = Path(json_file_path)
        self.items: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load difference items from JSON file."""
        if not self.json_file_path.exists():
            logger.error(f"Dialect differences file not found: {self.json_file_path}")
            return

        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                self.items = data
            else:
                logger.error(f"Invalid dialect differences file format, expected JSON array: {self.json_file_path}")
                return

            logger.info(f"Loaded {len(self.items)} difference items from {self.json_file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load dialect differences file: {e}")

    def get_all_items(self) -> List[Dict[str, str]]:
        """Return all difference items."""
        return self.items

    def sample_items(self, n: int, rng: random.Random = None) -> List[Dict[str, str]]:
        """
        Randomly sample n difference items.

        Args:
            n: Number of items to sample
            rng: Random number generator (optional, for reproducibility)

        Returns:
            List of sampled difference items
        """
        if not self.items:
            return []
        n = min(n, len(self.items))
        if rng:
            return rng.sample(self.items, n)
        return random.sample(self.items, n)


class BuiltinFunctionLoader:
    """
    Loads built-in functions from mysql_8_kb.json / pg_14_kb.json.
    """

    def __init__(self, kb_file_path: str):
        """
        Args:
            kb_file_path: Path to the JSON knowledge base file
        """
        self.kb_file_path = Path(kb_file_path)
        self.functions: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load JSON knowledge base."""
        if not self.kb_file_path.exists():
            logger.error(f"Knowledge base file not found: {self.kb_file_path}")
            return

        try:
            with open(self.kb_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                self.functions = data
            else:
                logger.warning(f"Knowledge base file format is not as expected: {self.kb_file_path}")
                return

            logger.info(f"Loaded {len(self.functions)} entries from knowledge base {self.kb_file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base file: {e}")

    def get_all_functions(self) -> List[Dict[str, Any]]:
        """Return all function entries."""
        return self.functions

    def sample_functions(self, n: int, rng: random.Random = None) -> List[Dict[str, Any]]:
        """
        Randomly sample n built-in functions.

        Args:
            n: Number of functions to sample
            rng: Random number generator

        Returns:
            List of sampled functions
        """
        if not self.functions:
            return []
        n = min(n, len(self.functions))
        if rng:
            return rng.sample(self.functions, n)
        return random.sample(self.functions, n)

    def format_function_info(self, func_item: Dict[str, Any]) -> str:
        """
        Format a single function entry into readable text.

        Args:
            func_item: A single function entry

        Returns:
            Formatted text
        """
        keyword = func_item.get('keyword', '')
        description = func_item.get('description', '') or ''
        examples = func_item.get('example', [])

        # Ensure description is a string
        if not isinstance(description, str):
            description = str(description)

        # Clean HTML tags
        description = re.sub(r'<[^>]+>', '', description)

        text = f"**{keyword}**\n"
        text += f"Description: {description[:300]}\n"  # Truncate overly long descriptions
        if examples:
            text += f"Example: {examples[0]}\n"

        return text
