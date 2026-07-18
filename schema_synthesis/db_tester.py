#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real database DDL table creation tester.

Executes synthesized DDL in MySQL / PostgreSQL / Oracle to verify tables can be created correctly.
- MySQL: creates a temporary database, DROPs after testing
- PostgreSQL: creates a temporary schema, DROP CASCADE after testing
- Oracle: creates tables directly as system user (with prefix to avoid conflicts), DROP TABLE one by one after testing
"""

import re
import yaml
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from utils.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class DBTestResult:
    """Single dialect test result."""
    dialect: str          # "mysql" / "postgresql" / "oracle"
    success: bool
    tables_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_feedback(self) -> str:
        """Generate feedback text for LLM reflection."""
        if self.success:
            return f"{self.dialect}: [OK] Table creation succeeded ({len(self.tables_created)} tables)"

        lines = [f"{self.dialect}: [FAIL] Table creation failed"]
        if self.tables_created:
            lines.append(f"  Tables created: {len(self.tables_created)}")
        for err in self.errors:
            lines.append(f"  - {err}")
        return "\n".join(lines)


@dataclass
class AllDBTestResult:
    """Combined test results for all three dialects."""
    mysql: Optional[DBTestResult] = None
    pg: Optional[DBTestResult] = None
    oracle: Optional[DBTestResult] = None

    @property
    def all_passed(self) -> bool:
        results = [r for r in [self.mysql, self.pg, self.oracle] if r is not None]
        return all(r.success for r in results)

    def to_feedback(self) -> str:
        """Generate combined feedback for LLM reflection."""
        parts = []
        for r in [self.mysql, self.pg, self.oracle]:
            if r:
                parts.append(r.to_feedback())
        return "\n\n".join(parts)

    @property
    def failed_dialects(self) -> List[str]:
        """Return list of failed dialects."""
        result = []
        for r in [self.mysql, self.pg, self.oracle]:
            if r and not r.success:
                result.append(r.dialect)
        return result


# =============================================================================
# SQL utility functions
# =============================================================================

def split_statements(sql_text: str) -> List[str]:
    """Split SQL text into statements by semicolons, skipping empty statements and comments."""
    statements = []
    for raw_block in sql_text.split(";"):
        cleaned_lines = []
        for line in raw_block.split("\n"):
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            cleaned_lines.append(line)
        stmt = "\n".join(cleaned_lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


def _extract_table_name(stmt: str) -> Optional[str]:
    """Extract table name from a CREATE TABLE statement."""
    m = re.search(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\']?(\w+)[`"\']?',
        stmt, re.IGNORECASE
    )
    return m.group(1).lower() if m else None


def _extract_fk_references(stmt: str) -> set:
    """Extract foreign key referenced table names from a CREATE TABLE statement."""
    refs = set()
    for m in re.finditer(r'REFERENCES\s+[`"\']?(\w+)[`"\']?', stmt, re.IGNORECASE):
        refs.add(m.group(1).lower())
    return refs


def _extract_comment_table(stmt: str) -> Optional[str]:
    """Extract table name from a COMMENT ON statement."""
    m = re.search(
        r'COMMENT\s+ON\s+(?:COLUMN|TABLE)\s+[`"\']?(\w+)[`"\']?',
        stmt, re.IGNORECASE
    )
    return m.group(1).lower() if m else None


def _strip_fk_constraints(stmt: str, table_name: str) -> Tuple[str, List[str]]:
    """
    Strip FOREIGN KEY constraints from a CREATE TABLE statement,
    return (cleaned CREATE TABLE statement, list of ALTER TABLE statements).

    Supports two FK formats:
    1. CONSTRAINT fk_name FOREIGN KEY (col) REFERENCES other_table(col)
    2. FOREIGN KEY (col) REFERENCES other_table(col)
    """
    alter_stmts = []

    # Extract table name quoting style (preserve original quotes)
    m = re.search(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([`"\']?\w+[`"\']?)',
        stmt, re.IGNORECASE
    )
    quoted_table = m.group(1) if m else table_name

    # Match FK constraint lines (with optional CONSTRAINT name)
    # Pattern: [CONSTRAINT name] FOREIGN KEY (cols) REFERENCES table(cols) [ON DELETE/UPDATE ...]
    fk_pattern = re.compile(
        r',?\s*'
        r'(?:CONSTRAINT\s+[`"\']?\w+[`"\']?\s+)?'
        r'FOREIGN\s+KEY\s*\([^)]+\)\s*'
        r'REFERENCES\s+[`"\']?\w+[`"\']?\s*\([^)]+\)'
        r'(?:\s+ON\s+(?:DELETE|UPDATE)\s+(?:CASCADE|SET\s+NULL|SET\s+DEFAULT|NO\s+ACTION|RESTRICT))*',
        re.IGNORECASE
    )

    fk_matches = list(fk_pattern.finditer(stmt))
    if not fk_matches:
        return stmt, []

    # Generate ALTER TABLE statements
    fk_counter = 0
    for fk_match in fk_matches:
        fk_text = fk_match.group(0).strip().lstrip(',').strip()

        # If no CONSTRAINT name, generate one
        if not re.match(r'CONSTRAINT\s+', fk_text, re.IGNORECASE):
            fk_counter += 1
            fk_name = f"fk_{table_name}_{fk_counter}"
            # Oracle 11g limit 30 chars
            if len(fk_name) > 30:
                fk_name = fk_name[:30]
            fk_text = f"CONSTRAINT {fk_name} {fk_text}"

        alter_stmts.append(f"ALTER TABLE {quoted_table} ADD {fk_text}")

    # Remove FK constraints from original statement
    cleaned = stmt
    for fk_match in reversed(fk_matches):  # Delete from back to front to avoid offset shifts
        start, end = fk_match.span()
        cleaned = cleaned[:start] + cleaned[end:]

    # Clean up possible trailing comma (before ) )
    # Find content before the last ) and remove trailing comma
    cleaned = re.sub(r',\s*\)', ')', cleaned)

    return cleaned, alter_stmts


def topo_sort_statements(statements: List[str]) -> List[str]:
    """
    Topologically sort CREATE TABLE statements by foreign key dependencies.
    - CREATE TYPE / CREATE SEQUENCE and other pre-statements go first
    - COMMENT ON statements follow their corresponding CREATE TABLE
    - If circular dependencies exist, strip FK constraints from involved tables,
      convert to ALTER TABLE ADD CONSTRAINT statements executed at the end
    """
    create_table_stmts = []   # (index, table_name, stmt, fk_refs)
    pre_stmts = []            # CREATE TYPE/SEQUENCE and other pre-statements
    comment_stmts = {}        # table_name -> [stmt, ...]
    post_stmts = []           # other post-statements

    for i, stmt in enumerate(statements):
        tbl = _extract_table_name(stmt)
        if tbl and re.match(r'\s*CREATE\s+TABLE', stmt, re.IGNORECASE):
            refs = _extract_fk_references(stmt)
            create_table_stmts.append((i, tbl, stmt, refs))
        elif re.match(r'\s*COMMENT\s+ON', stmt, re.IGNORECASE):
            ctbl = _extract_comment_table(stmt)
            if ctbl:
                comment_stmts.setdefault(ctbl, []).append(stmt)
            else:
                post_stmts.append(stmt)
        elif re.match(r'\s*CREATE\s+(TYPE|SEQUENCE|EXTENSION)', stmt, re.IGNORECASE):
            pre_stmts.append(stmt)
        elif re.match(r'\s*CREATE\s+INDEX', stmt, re.IGNORECASE):
            post_stmts.append(stmt)
        else:
            post_stmts.append(stmt)

    # Kahn topological sort
    table_to_idx = {item[1]: idx for idx, item in enumerate(create_table_stmts)}
    n = len(create_table_stmts)
    in_degree = [0] * n
    adj = [[] for _ in range(n)]

    for idx, (_, tbl, _, refs) in enumerate(create_table_stmts):
        for ref in refs:
            if ref in table_to_idx and ref != tbl:
                dep_idx = table_to_idx[ref]
                adj[dep_idx].append(idx)
                in_degree[idx] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    sorted_indices = []

    while queue:
        node = queue.popleft()
        sorted_indices.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Handle circular dependencies: strip FK constraints, convert to ALTER TABLE
    deferred_alter_stmts = []  # ALTER TABLE ADD CONSTRAINT statements
    if len(sorted_indices) < n:
        cycle_indices = set(range(n)) - set(sorted_indices)
        logger.info(f"    Detected circular FK dependency, involving tables: "
                     f"{[create_table_stmts[i][1] for i in cycle_indices]}")

        for ci in cycle_indices:
            orig_i, tbl, stmt, refs = create_table_stmts[ci]
            cleaned_stmt, alter_stmts = _strip_fk_constraints(stmt, tbl)
            # Replace original statement with FK-stripped version
            create_table_stmts[ci] = (orig_i, tbl, cleaned_stmt, set())
            deferred_alter_stmts.extend(alter_stmts)

        # Re-run topological sort (cycle broken)
        in_degree2 = [0] * n
        adj2 = [[] for _ in range(n)]
        for idx, (_, tbl, _, refs) in enumerate(create_table_stmts):
            for ref in refs:
                if ref in table_to_idx and ref != tbl:
                    dep_idx = table_to_idx[ref]
                    adj2[dep_idx].append(idx)
                    in_degree2[idx] += 1

        queue2 = deque([i for i in range(n) if in_degree2[i] == 0])
        sorted_indices = []
        while queue2:
            node = queue2.popleft()
            sorted_indices.append(node)
            for neighbor in adj2[node]:
                in_degree2[neighbor] -= 1
                if in_degree2[neighbor] == 0:
                    queue2.append(neighbor)

        # If there are still remaining (theoretically shouldn't happen), append
        if len(sorted_indices) < n:
            remaining = set(range(n)) - set(sorted_indices)
            sorted_indices.extend(sorted(remaining))

    # Assemble: pre-statements -> topologically sorted CREATE TABLE (+COMMENT ON) -> ALTER TABLE -> post-statements
    result = list(pre_stmts)
    for idx in sorted_indices:
        _, tbl, stmt, _ = create_table_stmts[idx]
        result.append(stmt)
        if tbl in comment_stmts:
            result.extend(comment_stmts[tbl])
    result.extend(deferred_alter_stmts)
    result.extend(post_stmts)

    return result


# =============================================================================
# Database connection config loading
# =============================================================================

def _load_db_config() -> Dict:
    """Load database connection config from database_sync.yaml."""
    config_path = PROJECT_ROOT / "config" / "database_sync.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["database"]


# =============================================================================
# MySQL test
# =============================================================================

def test_mysql(db_name: str, ddl_sql: str, db_config: Dict) -> DBTestResult:
    """Test DDL table creation in MySQL."""
    import pymysql

    result = DBTestResult(dialect="mysql", success=False)
    # Use full database name (not truncated) to expose overly long names early
    # MySQL database name limit is 64 characters
    schema_name = f"test_{db_name}"

    cfg = db_config["mysql"]
    conn = None
    try:
        conn = pymysql.connect(
            host=cfg["host"], port=cfg["port"],
            user=cfg["user"], password=cfg["password"],
            charset="utf8mb4"
        )
        cur = conn.cursor()

        # Pre-clean + create
        cur.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")
        cur.execute(f"CREATE DATABASE `{schema_name}` CHARACTER SET utf8mb4")
        cur.execute(f"USE `{schema_name}`")

        # Execute after topological sort
        statements = topo_sort_statements(split_statements(ddl_sql))
        for i, stmt in enumerate(statements):
            try:
                cur.execute(stmt)
            except Exception as e:
                result.errors.append(
                    f"Statement {i+1}: {str(e)[:200]}\n  SQL: {stmt[:200]}"
                )
        conn.commit()

        # Verify
        cur.execute("SHOW TABLES")
        result.tables_created = [row[0] for row in cur.fetchall()]
        result.success = len(result.tables_created) > 0 and len(result.errors) == 0

        # Cleanup
        cur.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")
        conn.commit()

    except Exception as e:
        result.errors.append(f"Connection/execution error: {str(e)[:300]}")
        # Attempt cleanup
        try:
            if conn:
                cur = conn.cursor()
                cur.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")
                conn.commit()
        except:
            pass
    finally:
        if conn:
            conn.close()

    return result


# =============================================================================
# PostgreSQL test
# =============================================================================

def test_postgresql(db_name: str, ddl_sql: str, db_config: Dict) -> DBTestResult:
    """Test DDL table creation in PostgreSQL."""
    import psycopg2

    result = DBTestResult(dialect="postgresql", success=False)
    # Use full database name (not truncated) to expose overly long names early
    # PostgreSQL identifier limit is 63 characters
    schema_name = f"test_{db_name}"

    cfg = db_config["postgresql"]
    conn = None
    try:
        conn = psycopg2.connect(
            host=cfg["host"], port=cfg["port"],
            user=cfg["user"], password=cfg["password"],
            dbname="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Pre-clean + create schema
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        cur.execute(f'CREATE SCHEMA "{schema_name}"')
        cur.execute(f'SET search_path TO "{schema_name}"')

        # Execute after topological sort
        statements = topo_sort_statements(split_statements(ddl_sql))
        for i, stmt in enumerate(statements):
            try:
                cur.execute(stmt)
            except Exception as e:
                result.errors.append(
                    f"Statement {i+1}: {str(e)[:200]}\n  SQL: {stmt[:200]}"
                )

        # Verify
        cur.execute(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{schema_name}'
            ORDER BY table_name
        """)
        result.tables_created = [row[0] for row in cur.fetchall()]
        result.success = len(result.tables_created) > 0 and len(result.errors) == 0

        # Cleanup
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')

    except Exception as e:
        result.errors.append(f"Connection/execution error: {str(e)[:300]}")
        try:
            if conn:
                cur = conn.cursor()
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        except:
            pass
    finally:
        if conn:
            conn.close()

    return result


# =============================================================================
# Oracle test (system user direct table creation, with prefix to avoid conflicts)
# =============================================================================

def test_oracle(db_name: str, ddl_sql: str, db_config: Dict) -> DBTestResult:
    """
    Test DDL table creation in Oracle.

    Uses system user to create tables directly, with prefix T_{db_name_hash}_ to avoid conflicts.
    After testing, DROP TABLE / DROP TYPE one by one for cleanup.
    """
    import oracledb

    result = DBTestResult(dialect="oracle", success=False)

    # Generate short prefix from db_name hash (Oracle identifier limit 30 chars, keep prefix short)
    import hashlib
    prefix = "T" + hashlib.md5(db_name.encode()).hexdigest()[:3].upper() + "_"  # 5-char prefix

    cfg = db_config["oracle"]
    dsn = oracledb.makedsn(cfg["host"], cfg["port"], service_name=cfg["service_name"])
    conn = None
    created_tables = []
    created_types = []

    try:
        conn = oracledb.connect(user=cfg["user"], password=cfg["password"], dsn=dsn)
        cur = conn.cursor()

        # Pre-clean: delete residual tables and types with same prefix
        try:
            cur.execute(f"SELECT table_name FROM user_tables WHERE table_name LIKE '{prefix.upper()}%'")
            old_tables = [r[0] for r in cur.fetchall()]
            for t in old_tables:
                try:
                    cur.execute(f'DROP TABLE "{t}" CASCADE CONSTRAINTS')
                    conn.commit()
                except:
                    pass
            # Also clean residual TYPEs
            cur.execute(f"SELECT type_name FROM user_types WHERE type_name LIKE '{prefix.upper()}%'")
            old_types = [r[0] for r in cur.fetchall()]
            for t in old_types:
                try:
                    cur.execute(f'DROP TYPE "{t}" FORCE')
                    conn.commit()
                except:
                    pass
        except:
            pass

        # Topological sort
        statements = topo_sort_statements(split_statements(ddl_sql))

        # Add prefix to all table names
        prefixed_stmts = _add_oracle_prefix(statements, prefix)

        for i, stmt in enumerate(prefixed_stmts):
            try:
                cur.execute(stmt)
                conn.commit()
                # Record created objects for cleanup
                tbl = _extract_table_name(stmt)
                if tbl and re.match(r'\s*CREATE\s+TABLE', stmt, re.IGNORECASE):
                    created_tables.append(tbl.upper())
                elif re.match(r'\s*CREATE\s+TYPE', stmt, re.IGNORECASE):
                    m = re.search(r'CREATE\s+TYPE\s+"?(\w+)"?', stmt, re.IGNORECASE)
                    if m:
                        created_types.append(m.group(1).upper())
            except Exception as e:
                result.errors.append(
                    f"Statement {i+1}: {str(e)[:200]}\n  SQL: {stmt[:200]}"
                )

        # Verify
        cur.execute(f"""
            SELECT table_name FROM user_tables
            WHERE table_name LIKE '{prefix.upper()}%'
            ORDER BY table_name
        """)
        result.tables_created = [row[0] for row in cur.fetchall()]
        result.success = len(result.tables_created) > 0 and len(result.errors) == 0

        # Cleanup: drop tables first (reverse order for FK dependencies), then types
        for tbl in reversed(created_tables):
            try:
                cur.execute(f'DROP TABLE "{tbl}" CASCADE CONSTRAINTS')
                conn.commit()
            except:
                pass
        for tp in reversed(created_types):
            try:
                cur.execute(f'DROP TYPE "{tp}"')
                conn.commit()
            except:
                pass

    except Exception as e:
        result.errors.append(f"Connection/execution error: {str(e)[:300]}")
        # Attempt cleanup
        try:
            if conn:
                cur = conn.cursor()
                for tbl in reversed(created_tables):
                    try:
                        cur.execute(f'DROP TABLE "{tbl}" CASCADE CONSTRAINTS')
                        conn.commit()
                    except:
                        pass
                for tp in reversed(created_types):
                    try:
                        cur.execute(f'DROP TYPE "{tp}"')
                        conn.commit()
                    except:
                        pass
        except:
            pass
    finally:
        if conn:
            conn.close()

    return result


def _truncate_long_identifiers(ddl_sql: str, max_len: int = 30) -> str:
    """
    Truncate all double-quoted identifiers in Oracle DDL exceeding max_len characters.
    Only processes identifiers that would be too long without prefix, uses hash suffix for uniqueness.
    """
    import hashlib as _hl

    # Collect all double-quoted identifiers
    long_ids = {}  # original -> truncated
    used = set()

    for m in re.finditer(r'"(\w+)"', ddl_sql):
        name = m.group(1)
        if len(name) > max_len and name not in long_ids:
            h = _hl.md5(name.encode()).hexdigest()[:4]
            truncated = name[:max_len - 4] + h
            base = truncated
            counter = 0
            while truncated.upper() in used:
                counter += 1
                truncated = base[:max_len - 2] + f"{counter:02d}"
            used.add(truncated.upper())
            long_ids[name] = truncated

    if not long_ids:
        return ddl_sql

    # Replace in descending order of name length to avoid short names matching substrings of long names
    for name in sorted(long_ids.keys(), key=len, reverse=True):
        ddl_sql = ddl_sql.replace(f'"{name}"', f'"{long_ids[name]}"')

    return ddl_sql


def _add_oracle_prefix(statements: List[str], prefix: str) -> List[str]:
    """
    Add prefix to all table names, type names, and constraint names in Oracle DDL.
    This allows concurrent testing of multiple databases under the same system schema without conflicts.
    If the prefixed name exceeds 30 characters (Oracle 11g limit), it is automatically truncated.
    Also truncates other identifiers (e.g., column names) exceeding 30 characters.
    """
    # Collect original table names and type names
    table_names = set()
    type_names = set()
    constraint_names = set()

    for stmt in statements:
        if re.match(r'\s*CREATE\s+TABLE', stmt, re.IGNORECASE):
            tbl = _extract_table_name(stmt)
            if tbl:
                table_names.add(tbl)
            # Extract CONSTRAINT names
            for m in re.finditer(r'CONSTRAINT\s+[`"\']?(\w+)[`"\']?', stmt, re.IGNORECASE):
                constraint_names.add(m.group(1))
        elif re.match(r'\s*CREATE\s+TYPE', stmt, re.IGNORECASE):
            m = re.search(r'CREATE\s+TYPE\s+"?(\w+)"?', stmt, re.IGNORECASE)
            if m:
                type_names.add(m.group(1))
        elif re.match(r'\s*ALTER\s+TABLE', stmt, re.IGNORECASE):
            # ALTER TABLE also has CONSTRAINT names and table name references
            for m in re.finditer(r'CONSTRAINT\s+[`"\']?(\w+)[`"\']?', stmt, re.IGNORECASE):
                constraint_names.add(m.group(1))
            # Extract ALTER TABLE target table name
            m = re.search(r'ALTER\s+TABLE\s+[`"\']?(\w+)[`"\']?', stmt, re.IGNORECASE)
            if m:
                table_names.add(m.group(1))
            # Extract REFERENCES table names
            for m in re.finditer(r'REFERENCES\s+[`"\']?(\w+)[`"\']?', stmt, re.IGNORECASE):
                table_names.add(m.group(1))

    # Build mapping table (30-char truncation, hash suffix to avoid name collisions)
    import hashlib as _hl
    name_mapping = {}
    used_names = set()
    for name in (table_names | type_names | constraint_names):
        prefixed = prefix + name
        if len(prefixed) > 30:
            # Keep prefix + as much of original name as possible + 4-char hash suffix for uniqueness
            h = _hl.md5(name.encode()).hexdigest()[:4]
            prefixed = prefix + name[:30 - len(prefix) - 4] + h
        # Handle extreme cases where names still collide
        base = prefixed
        counter = 0
        while prefixed.upper() in used_names:
            counter += 1
            prefixed = base[:28] + f"{counter:02d}"
        used_names.add(prefixed.upper())
        name_mapping[name] = prefixed

    # Replace in descending order of name length to avoid short names matching substrings of long names
    sorted_names = sorted(name_mapping.keys(), key=len, reverse=True)

    # Oracle SQL keyword list (lowercase), used to avoid accidentally replacing SQL syntax keywords
    _ORACLE_KEYWORDS = {
        'table', 'create', 'alter', 'drop', 'index', 'constraint', 'primary',
        'foreign', 'key', 'references', 'not', 'null', 'unique', 'check',
        'default', 'number', 'varchar2', 'char', 'clob', 'blob', 'date',
        'timestamp', 'integer', 'int', 'cascade', 'constraints', 'on',
        'delete', 'update', 'set', 'restrict', 'action', 'add', 'if',
        'exists', 'in', 'is', 'and', 'or', 'force', 'type', 'as', 'select',
        'from', 'where', 'insert', 'into', 'values', 'binary_double',
    }

    result = []
    for stmt in statements:
        new_stmt = stmt
        for name in sorted_names:
            target = name_mapping[name]
            # Replace quoted (priority, unambiguous)
            new_stmt = re.sub(
                rf'"({re.escape(name)})"',
                f'"{target}"',
                new_stmt,
                flags=re.IGNORECASE
            )
            # Replace unquoted standalone words (only when name is not a SQL keyword,
            # or name is indeed a table/constraint name we collected)
            # Use negative lookahead/lookbehind to exclude already-quoted cases
            # Important: wrap replaced name in double quotes, because CREATE TABLE "T98E_xxx" creates a lowercase table name,
            # if REFERENCES T98E_xxx is unquoted, Oracle parses it as uppercase T98E_XXX, causing case mismatch
            if name.lower() not in _ORACLE_KEYWORDS:
                # Determine if it's a constraint name (constraint names are not quoted, keep original replacement behavior)
                is_constraint = name in constraint_names and name not in (table_names | type_names)
                replacement = target if is_constraint else f'"{target}"'
                new_stmt = re.sub(
                    rf'\b({re.escape(name)})\b',
                    replacement,
                    new_stmt,
                    flags=re.IGNORECASE
                )
        # Truncate overly long column names and other double-quoted identifiers
        new_stmt = _truncate_long_identifiers(new_stmt, max_len=30)
        result.append(new_stmt)

    return result


# =============================================================================
# Combined test entry
# =============================================================================

def run_db_tests(
    db_name: str,
    mysql_ddl: str,
    pg_ddl: str,
    oracle_ddl: str,
) -> AllDBTestResult:
    """
    Execute real database table creation tests for tri-dialect DDL.

    Returns:
        AllDBTestResult containing test results for all three dialects
    """
    db_config = _load_db_config()
    result = AllDBTestResult()

    if mysql_ddl:
        logger.info(f"    [MySQL] Table creation test...")
        result.mysql = test_mysql(db_name, mysql_ddl, db_config)
        status = "[OK]" if result.mysql.success else "[FAIL]"
        logger.info(f"    {status} MySQL: {len(result.mysql.tables_created)} tables, "
                     f"{len(result.mysql.errors)} errors")

    if pg_ddl:
        logger.info(f"    [PostgreSQL] Table creation test...")
        result.pg = test_postgresql(db_name, pg_ddl, db_config)
        status = "[OK]" if result.pg.success else "[FAIL]"
        logger.info(f"    {status} PostgreSQL: {len(result.pg.tables_created)} tables, "
                     f"{len(result.pg.errors)} errors")

    if oracle_ddl:
        logger.info(f"    [Oracle] Table creation test...")
        result.oracle = test_oracle(db_name, oracle_ddl, db_config)
        status = "[OK]" if result.oracle.success else "[FAIL]"
        logger.info(f"    {status} Oracle: {len(result.oracle.tables_created)} tables, "
                     f"{len(result.oracle.errors)} errors")

    return result
