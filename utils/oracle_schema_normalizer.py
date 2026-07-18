#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle schema post-normalization tool.

Called automatically in schema_expander._save_result() to fix common issues
in LLM-generated Oracle DDL:

1. **Identifier case**:
   - Oracle double-quoted identifiers are case-sensitive
   - LLM sometimes outputs all uppercase (e.g., "EXCHANGES"), but MySQL/PG use lowercase
   - Force all double-quoted identifiers to lowercase

2. **Identifier truncation fix**:
   - Oracle 11g identifier length limit is 30 characters
   - LLM sometimes truncates table names causing inconsistency with MySQL/PG
   - Fix by comparing with MySQL schema table names

3. **Table/column name tri-engine consistency verification**:
   - Extract table name lists from MySQL and Oracle DDL, check for full consistency
   - If Oracle has misspelled table names, attempt auto-fix via fuzzy matching

Note: NUMBER(p,s) is no longer blindly replaced with BINARY_DOUBLE at this stage.
LLM can choose appropriate types based on business semantics; if NUMBER(p,s)
overflows during data import, data_import.py will automatically ALTER TABLE to
upgrade the column to BINARY_DOUBLE and retry.

Public API:
    normalize_oracle_schema(oracle_ddl, mysql_ddl=None, pg_ddl=None) -> str
"""

import re
from typing import List, Optional, Set, Tuple

from utils.logging_config import get_logger

logger = get_logger(__name__)


# ===========================================================================
# 1. Identifier lowercasing
# ===========================================================================

def _lowercase_quoted_identifiers(sql: str) -> str:
    """
    Convert all double-quoted identifiers in Oracle DDL to lowercase.

    Skips:
    - /* ... */ block comments
    - -- line comments
    - '...' single-quoted strings (data values)

    Examples:
      "EXCHANGES" -> "exchanges"
      "API_VERSIONS" -> "api_versions"
    """
    result = []
    i = 0
    n = len(sql)
    in_single_quote = False

    while i < n:
        ch = sql[i]

        # /* ... */ block comment: preserve as-is
        if not in_single_quote and ch == '/' and i + 1 < n and sql[i + 1] == '*':
            end = sql.find('*/', i + 2)
            if end != -1:
                result.append(sql[i:end + 2])
                i = end + 2
            else:
                result.append(sql[i:])
                break
            continue

        # -- line comment: preserve as-is
        if not in_single_quote and ch == '-' and i + 1 < n and sql[i + 1] == '-':
            end = sql.find('\n', i)
            if end != -1:
                result.append(sql[i:end])
                i = end
            else:
                result.append(sql[i:])
                break
            continue

        # Single-quoted string
        if ch == "'" and not in_single_quote:
            in_single_quote = True
            result.append(ch)
            i += 1
            continue
        if in_single_quote:
            if ch == "'" and i + 1 < n and sql[i + 1] == "'":
                result.append(ch)
                result.append(sql[i + 1])
                i += 2
                continue
            elif ch == "'":
                in_single_quote = False
            result.append(ch)
            i += 1
            continue

        # Double-quoted identifier -> lowercase
        if ch == '"':
            j = i + 1
            while j < n and sql[j] != '"':
                j += 1
            if j < n:
                identifier = sql[i + 1:j].lower()
                result.append('"')
                result.append(identifier)
                result.append('"')
                i = j + 1
            else:
                result.append(ch)
                i += 1
        else:
            result.append(ch)
            i += 1

    return ''.join(result)


def _quote_bare_identifiers_in_create(sql: str, mysql_ddl: Optional[str] = None) -> str:
    """
    Add double quotes to bare identifiers (unquoted column/table names) inside
    CREATE TABLE statements in Oracle DDL.

    LLM sometimes omits double quotes for certain columns (especially Oracle
    reserved words/sensitive words like action, comments, comment, level, type,
    status, etc.), causing Oracle to store identifiers in uppercase. When INSERT
    statements reference them with lowercase double quotes, the column is not
    found (ORA-00904).

    Strategy: scan column definition lines within CREATE TABLE body line by line;
    if a column name is not wrapped in double quotes, add them and lowercase it.
    Also handles CREATE TABLE table names.
    """
    lines = sql.split('\n')
    result = []
    in_create = False
    paren_depth = 0

    for line in lines:
        stripped = line.strip()
        upper_stripped = stripped.upper()

        # Detect CREATE TABLE line - fix table name
        create_m = re.match(
            r'(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)(\w+)(\s*\(.*)',
            stripped, re.IGNORECASE
        )
        if create_m and not re.match(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"',
            stripped, re.IGNORECASE
        ):
            # Bare table name -> add double quotes, lowercase
            prefix = create_m.group(1)
            table_name = create_m.group(2)
            suffix = create_m.group(3)
            # Preserve original indentation
            indent = line[:len(line) - len(line.lstrip())]
            line = f'{indent}{prefix}"{table_name.lower()}"{suffix}'
            in_create = True
            paren_depth = line.count('(') - line.count(')')
            result.append(line)
            continue

        # Detect CREATE TABLE "xxx" ( - standard format
        if re.match(r'CREATE\s+TABLE\s', stripped, re.IGNORECASE):
            in_create = True
            paren_depth = line.count('(') - line.count(')')
            result.append(line)
            continue

        if in_create:
            paren_depth += line.count('(') - line.count(')')

            # Skip constraint lines, comment lines
            if any(upper_stripped.startswith(kw) for kw in (
                'PRIMARY', 'CONSTRAINT', 'FOREIGN', 'KEY', 'CHECK', 'UNIQUE',
                'INDEX', '--', '/*', 'ENGINE', 'ALTER', ')'
            )):
                if paren_depth <= 0:
                    in_create = False
                result.append(line)
                continue

            # Empty line
            if not stripped:
                result.append(line)
                continue

            # Column definition line: if starts with a bare identifier (no double quotes) -> add quotes
            col_m = re.match(r'^(\s*)(\w+)(\s+\w)', line)
            if col_m:
                indent = col_m.group(1)
                col_name = col_m.group(2)
                rest_start = col_m.group(3)
                rest = line[col_m.end():]
                # Check if already has double quotes (shouldn't reach here, but just in case)
                # Also exclude keywords (like PRIMARY KEY sub-lines)
                if col_name.upper() not in (
                    'PRIMARY', 'CONSTRAINT', 'FOREIGN', 'KEY', 'CHECK',
                    'UNIQUE', 'INDEX', 'ALTER', 'CREATE', 'DROP'
                ):
                    line = f'{indent}"{col_name.lower()}"{rest_start}{rest}'

            if paren_depth <= 0:
                in_create = False

        result.append(line)

    # Also handle bare identifiers in ALTER TABLE
    # ALTER TABLE xxx ADD CONSTRAINT ... -> ALTER TABLE "xxx" ADD CONSTRAINT ...
    joined = '\n'.join(result)
    joined = re.sub(
        r'(ALTER\s+TABLE\s+)(?!")(\w+)',
        lambda m: f'{m.group(1)}"{m.group(2).lower()}"',
        joined, flags=re.IGNORECASE
    )

    return joined


# ===========================================================================
# 2. Tri-engine table name consistency fix
# ===========================================================================

def _extract_table_names(ddl: str) -> List[str]:
    """Extract all CREATE TABLE table names (bare names, lowercased) from DDL."""
    # Match CREATE TABLE `xxx` or CREATE TABLE "xxx" or CREATE TABLE xxx
    tables = []
    for m in re.finditer(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?',
        ddl, re.IGNORECASE
    ):
        tables.append(m.group(1).lower())
    return tables


def _fix_truncated_table_names(oracle_ddl: str, mysql_ddl: str) -> str:
    """
    Fix truncated table names in Oracle DDL.

    Oracle 11g identifier limit is 30 characters; LLM sometimes truncates
    table names, causing Oracle table names to be inconsistent with MySQL/PG.

    Strategy: compare with MySQL table name list; if Oracle has a table name
    that doesn't match, check if it's a prefix of a MySQL table name (truncation),
    and auto-fix.
    """
    if not mysql_ddl:
        return oracle_ddl

    mysql_tables = set(_extract_table_names(mysql_ddl))
    oracle_tables = set(_extract_table_names(oracle_ddl))

    # Find table names in Oracle but not in MySQL
    oracle_only = oracle_tables - mysql_tables
    mysql_only = mysql_tables - oracle_tables

    if not oracle_only or not mysql_only:
        return oracle_ddl

    # Attempt matching: Oracle wrong name -> MySQL correct name
    replacements = {}
    for o_name in oracle_only:
        best_match = None
        best_score = 0
        for m_name in mysql_only:
            # Case 1: Oracle name is a prefix of MySQL name (truncation)
            if m_name.startswith(o_name):
                score = len(o_name)
                if score > best_score:
                    best_score = score
                    best_match = m_name
            # Case 2: Very small edit distance (1-2 character difference, typo)
            elif _edit_distance(o_name, m_name) <= 2:
                score = len(o_name) * 10  # High priority
                if score > best_score:
                    best_score = score
                    best_match = m_name

        if best_match:
            replacements[o_name] = best_match

    if not replacements:
        return oracle_ddl

    # Execute replacement
    fixed = oracle_ddl
    for old_name, new_name in replacements.items():
        # Replace both quoted and unquoted occurrences
        fixed = fixed.replace(f'"{old_name}"', f'"{new_name}"')
        fixed = fixed.replace(f'"{old_name.upper()}"', f'"{new_name}"')
        logger.info(f"  [Fix] Oracle normalization: table name fix '{old_name}' -> '{new_name}'")

    return fixed


def _edit_distance(s1: str, s2: str) -> int:
    """Compute the edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ===========================================================================
# 3. Column name consistency verification (log warnings)
# ===========================================================================

def _extract_columns_per_table(ddl: str) -> dict:
    """Extract column name lists per table."""
    tables = {}

    # Simplified version: chunk by CREATE TABLE
    for m in re.finditer(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?\s*\(',
        ddl, re.IGNORECASE
    ):
        table_name = m.group(1).lower()
        # Find content between matching parentheses
        start = m.end()
        depth = 1
        i = start
        while i < len(ddl) and depth > 0:
            if ddl[i] == '(':
                depth += 1
            elif ddl[i] == ')':
                depth -= 1
            i += 1

        body = ddl[start:i - 1]

        # Extract column names (skip constraint lines)
        cols = []
        for line in body.split('\n'):
            line = line.strip().rstrip(',')
            if not line:
                continue
            upper = line.upper()
            if any(upper.startswith(kw) for kw in (
                'PRIMARY', 'CONSTRAINT', 'FOREIGN', 'KEY', 'CHECK', 'UNIQUE',
                'INDEX', '--', '/*', 'ENGINE', 'ALTER'
            )):
                continue
            # Extract column name (backtick, double quote, or bare)
            col_m = re.match(r'[`"]?(\w+)[`"]?\s+\w', line)
            if col_m:
                cols.append(col_m.group(1).lower())

        tables[table_name] = cols

    return tables


def _warn_column_mismatches(oracle_ddl: str, mysql_ddl: str):
    """Compare Oracle and MySQL column names, log warnings for inconsistencies."""
    if not mysql_ddl:
        return

    mysql_tables = _extract_columns_per_table(mysql_ddl)
    oracle_tables = _extract_columns_per_table(oracle_ddl)

    for table in mysql_tables:
        if table not in oracle_tables:
            continue

        mysql_cols = set(mysql_tables[table])
        oracle_cols = set(oracle_tables[table])

        mysql_only = mysql_cols - oracle_cols
        oracle_only = oracle_cols - mysql_cols

        if mysql_only or oracle_only:
            logger.warning(
                f"  [Warning] Table '{table}' column name mismatch: "
                f"MySQL only={mysql_only or 'none'}, Oracle only={oracle_only or 'none'}"
            )


# ===========================================================================
# Main entry
# ===========================================================================

def normalize_oracle_schema(
    oracle_ddl: str,
    mysql_ddl: Optional[str] = None,
    pg_ddl: Optional[str] = None,
) -> str:
    """
    Oracle schema post-normalization.

    Called automatically before saving in schema_expander._save_result().

    Processes:
    1. Force double-quoted identifiers to lowercase
    1.5. Add double quotes to bare identifiers (unquoted column/table names)
    2. Fix truncated table names (compare with MySQL schema)
    3. Column name consistency warnings

    Note: NUMBER(p,s) is preserved as-is, not automatically replaced with BINARY_DOUBLE.
    If overflow occurs during data import, data_import.py will automatically upgrade
    the column type and retry.

    Args:
        oracle_ddl: Oracle DDL text
        mysql_ddl: MySQL DDL (optional, for tri-engine consistency verification)
        pg_ddl: PG DDL (optional, reserved for extension)

    Returns:
        Normalized Oracle DDL
    """
    if not oracle_ddl:
        return oracle_ddl

    original = oracle_ddl

    # Step 1: Lowercase identifiers (already double-quoted)
    oracle_ddl = _lowercase_quoted_identifiers(oracle_ddl)

    # Step 1.5: Add double quotes to bare identifiers
    # LLM sometimes omits double quotes for certain columns (especially Oracle
    # reserved words like action/comments), causing Oracle to store them in
    # uppercase while INSERT references them with lowercase double quotes -> ORA-00904
    oracle_ddl = _quote_bare_identifiers_in_create(oracle_ddl, mysql_ddl)

    # Step 2: Fix truncated table names
    if mysql_ddl:
        oracle_ddl = _fix_truncated_table_names(oracle_ddl, mysql_ddl)

    # Step 3: Column name consistency warnings
    if mysql_ddl:
        _warn_column_mismatches(oracle_ddl, mysql_ddl)

    if oracle_ddl != original:
        logger.info("  [Fix] Oracle schema normalized")

    return oracle_ddl
