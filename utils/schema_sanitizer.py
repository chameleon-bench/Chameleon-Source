#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema post-processing sanitizer.

Purpose: Fix the issue where LLM-generated tri-dialect DDL may have
"self-referencing foreign key columns marked NOT NULL".
Self-referencing foreign keys (FOREIGN KEY (col) REFERENCES self_table(pk)) are typically
used to represent tree/graph structures, where the root node's parent reference
must be NULL, so the column must be NULLABLE.

Behavior:
- Only removes NOT NULL from self-referencing foreign key columns;
- NOT NULL constraints on all other columns remain unchanged;
- Supports all three dialect DDLs (MySQL / PostgreSQL / Oracle), preserving column name
  case and original quoting style;
- Pure string-level processing, does not depend on sqlparse, idempotent (multiple calls
  produce the same result);
- If no self-referencing foreign keys exist in the DDL, returns the original string
  (does not change a single byte).

Public API:
    sanitize_self_ref_fk_nullable(ddl: str, dialect: str) -> str
    sanitize_schema_files(db_dir: Path, db_id: str, dialects=("mysql", "pg", "oracle")) -> dict

Note:
    This is a forward-compatible sanitization logic - it only makes changes when it
    detects the contradiction of "self-referencing FK + column NOT NULL".
    Existing correct schema files remain byte-identical after processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------

_DIALECT_QUOTES: Dict[str, Tuple[str, str]] = {
    # dialect -> (left_quote, right_quote)
    "mysql": ("`", "`"),
    "pg": ('"', '"'),
    "postgresql": ('"', '"'),
    "oracle": ('"', '"'),
}

# CREATE TABLE statement start (header + left paren); body and right paren need
# parenthesis count matching, cannot use simple non-greedy regex (otherwise the
# right paren of PRIMARY KEY(...) would be mistaken as the end of table definition).
_CREATE_TABLE_HEAD_RE = re.compile(
    r"CREATE\s+TABLE\s+(?P<header>[^(]+?)\(",
    re.IGNORECASE,
)


def _strip_quotes(ident: str) -> str:
    """Remove backticks or double quotes from both sides of an identifier, return the bare name."""
    ident = ident.strip()
    if len(ident) >= 2 and ident[0] in ("`", '"') and ident[-1] == ident[0]:
        ident = ident[1:-1]
    return ident


def _extract_table_name(header: str) -> Optional[str]:
    """Extract the bare table name from the header part of `CREATE TABLE xxx `."""
    # Could be: `my_db`.`t1`  "schema"."t1"   t1   IF NOT EXISTS `t1`
    header = re.sub(r"IF\s+NOT\s+EXISTS", "", header, flags=re.IGNORECASE).strip()
    # Take the last segment (considering schema.table)
    tokens = re.split(r"\s+", header)
    if not tokens:
        return None
    last = tokens[-1]
    # schema.table case
    if "." in last:
        last = last.split(".")[-1]
    return _strip_quotes(last)


def _split_body_items(body: str) -> List[str]:
    """
    Split the parenthesis body of CREATE TABLE by top-level commas.

    Handles:
      - Parenthesis nesting (e.g., DECIMAL(10,2), ENUM('a','b'))
      - Block comments /* ... */ (commas/quotes inside comments are not parsed)
      - Line comments -- ...
      - Strings '...' / identifiers `...` / "..."
    """
    items: List[str] = []
    buf: List[str] = []
    depth = 0
    in_str: Optional[str] = None
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]

        if in_str is not None:
            buf.append(ch)
            if ch == in_str:
                if i + 1 < n and body[i + 1] == in_str:
                    buf.append(body[i + 1])
                    i += 2
                    continue
                in_str = None
            i += 1
            continue

        # Block comment: collect entire block into buf as-is
        if ch == "/" and i + 1 < n and body[i + 1] == "*":
            end = body.find("*/", i + 2)
            if end == -1:
                buf.append(body[i:])
                i = n
            else:
                buf.append(body[i : end + 2])
                i = end + 2
            continue

        # Line comment
        if ch == "-" and i + 1 < n and body[i + 1] == "-":
            nl = body.find("\n", i + 2)
            if nl == -1:
                buf.append(body[i:])
                i = n
            else:
                buf.append(body[i : nl + 1])
                i = nl + 1
            continue

        if ch in ("'", "`", '"'):
            in_str = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


# Top-level FK constraint (table-level):
#   [CONSTRAINT xxx] FOREIGN KEY (col1[, col2]) REFERENCES ref_table (ref_col1[, ref_col2])
_TABLE_FK_RE = re.compile(
    r"""
    ^\s*
    (?:CONSTRAINT\s+[`"]?[\w$]+[`"]?\s+)?
    FOREIGN\s+KEY\s*
    \(\s*(?P<cols>[^)]+)\)\s*
    REFERENCES\s+
    (?P<ref_table>(?:[`"][\w$.]+[`"])|[\w$.]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Inline FK in column definition (less common, both MySQL/PG support it):
#   col_name TYPE ... REFERENCES ref_table (ref_col) ...
_INLINE_REFERENCES_RE = re.compile(
    r"REFERENCES\s+(?P<ref_table>(?:[`\"][\w$.]+[`\"])|[\w$.]+)\s*\(\s*(?P<ref_col>[^)]+?)\s*\)",
    re.IGNORECASE,
)

# Identify column definition lines that are not constraint starts (excludes PRIMARY KEY / UNIQUE / FOREIGN KEY / CONSTRAINT / CHECK / KEY / INDEX etc.)
_CONSTRAINT_PREFIXES = (
    "PRIMARY ", "UNIQUE ", "FOREIGN ", "CONSTRAINT ",
    "CHECK", "KEY ", "INDEX ", "FULLTEXT ", "SPATIAL ",
)


def _is_constraint_line(item: str) -> bool:
    upper = item.lstrip().upper()
    return any(upper.startswith(p) for p in _CONSTRAINT_PREFIXES)


def _parse_col_name(item: str) -> Optional[str]:
    """
    Extract the bare column name from the beginning of a column definition line.
    Examples:
      `parent_id` INT NOT NULL      -> parent_id
      "parent_id" INTEGER            -> parent_id
      parent_id NUMBER(10) NOT NULL  -> parent_id
    """
    stripped = item.lstrip()
    if stripped.startswith("`"):
        m = re.match(r"`([^`]+)`", stripped)
    elif stripped.startswith('"'):
        m = re.match(r'"([^"]+)"', stripped)
    else:
        m = re.match(r"([A-Za-z_][\w$]*)", stripped)
    return m.group(1) if m else None


def _extract_self_ref_cols_from_body(
    table_name: str, items: List[str]
) -> Set[str]:
    """
    From split column/constraint items, identify all columns involved in
    "self-referencing foreign keys" (bare column names, lowercased for comparison).
    """
    table_lower = table_name.lower()
    self_ref_cols: Set[str] = set()

    for item in items:
        # (a) Table-level FOREIGN KEY
        m = _TABLE_FK_RE.match(item)
        if m:
            ref_table = _strip_quotes(m.group("ref_table").split(".")[-1])
            if ref_table.lower() == table_lower:
                cols = [
                    _strip_quotes(c).lower()
                    for c in m.group("cols").split(",")
                    if c.strip()
                ]
                self_ref_cols.update(cols)
            continue

        # (b) Inline REFERENCES in column definition (not a constraint line)
        if not _is_constraint_line(item):
            ref_m = _INLINE_REFERENCES_RE.search(item)
            if ref_m:
                ref_table = _strip_quotes(ref_m.group("ref_table").split(".")[-1])
                if ref_table.lower() == table_lower:
                    col = _parse_col_name(item)
                    if col:
                        self_ref_cols.add(col.lower())

    return self_ref_cols


# Match NOT NULL near the end of a column definition (cannot match "NOT NULL" inside comments).
# Conservative approach: replace /* ... */ comments before searching.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_NOT_NULL_RE = re.compile(r"\bNOT\s+NULL\b", re.IGNORECASE)


def _remove_not_null_in_col_item(item: str) -> Tuple[str, bool]:
    """
    Remove NOT NULL from a column definition, but preserve any "NOT NULL"
    literal inside comment blocks (do not touch comments).
    Returns (new_text, whether modification occurred).
    """
    # Record comment positions, replace with placeholders, then restore after processing
    placeholders: List[str] = []

    def _stash(match):
        placeholders.append(match.group(0))
        return f"\x00CMT{len(placeholders) - 1}\x00"

    stashed = _BLOCK_COMMENT_RE.sub(_stash, item)

    if not _NOT_NULL_RE.search(stashed):
        return item, False

    new_stashed = _NOT_NULL_RE.sub("", stashed)
    # Merge extra whitespace (do not cross newlines, stay conservative)
    new_stashed = re.sub(r"[ \t]{2,}", " ", new_stashed)
    new_stashed = re.sub(r"\s+,", ",", new_stashed)

    # Restore comments
    def _unstash(match):
        idx = int(match.group(1))
        return placeholders[idx]

    restored = re.sub(r"\x00CMT(\d+)\x00", _unstash, new_stashed)
    return restored, restored != item


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _find_matching_paren(text: str, open_idx: int) -> int:
    """
    Given the position of a left parenthesis open_idx, return the index of
    the matching right parenthesis.

    Scanning correctly skips:
      - Block comments /* ... */ (quotes/parens inside are literal characters)
      - Line comments -- ... to end of line
      - String literals '...' (including '' escape)
      - Identifier quotes `...` and "..."

    Returns -1 if not found.
    """
    assert text[open_idx] == "("
    depth = 0
    in_str: Optional[str] = None  # Current string/identifier start character (' ` ")
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]

        if in_str is not None:
            if ch == in_str:
                # '' / `` / "" means escape
                if i + 1 < n and text[i + 1] == in_str:
                    i += 2
                    continue
                in_str = None
            i += 1
            continue

        # Block comment /* ... */
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            if end == -1:
                return -1
            i = end + 2
            continue

        # Line comment -- ... \n
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            nl = text.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue

        # String/identifier start
        if ch in ("'", "`", '"'):
            in_str = ch
            i += 1
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def sanitize_self_ref_fk_nullable(ddl: str, dialect: str = "mysql") -> str:
    """
    Scan each table in the DDL; if a self-referencing foreign key exists,
    remove NOT NULL from the column definitions involved in that FK.

    - Only removes NOT NULL from column definitions; does not touch
      PRIMARY KEY / CHECK / FK constraint clauses.
    - NOT NULL on non-self-referencing columns remains unchanged.
    - Idempotent.
    """
    if not ddl:
        return ddl
    upper = ddl.upper()
    if "FOREIGN KEY" not in upper and "REFERENCES" not in upper:
        return ddl

    _ = dialect  # Parameter retained for future extension

    result_parts: List[str] = []
    cursor = 0
    while cursor < len(ddl):
        m = _CREATE_TABLE_HEAD_RE.search(ddl, cursor)
        if not m:
            result_parts.append(ddl[cursor:])
            break

        # Append original text before CREATE TABLE
        result_parts.append(ddl[cursor : m.start()])

        header = m.group("header")
        open_paren_idx = m.end() - 1  # Position of `(`
        close_paren_idx = _find_matching_paren(ddl, open_paren_idx)
        if close_paren_idx == -1:
            # Mismatched parens, output as-is
            result_parts.append(ddl[m.start():])
            break

        body = ddl[open_paren_idx + 1 : close_paren_idx]
        # tail: the part between right paren and the next semicolon (inclusive)
        tail_end = ddl.find(";", close_paren_idx + 1)
        if tail_end == -1:
            tail = ddl[close_paren_idx + 1 :]
            next_cursor = len(ddl)
        else:
            tail = ddl[close_paren_idx + 1 : tail_end]
            next_cursor = tail_end + 1  # Skip semicolon

        table_name = _extract_table_name(header)
        items = _split_body_items(body) if table_name else []
        self_ref_cols = (
            _extract_self_ref_cols_from_body(table_name, items)
            if (table_name and items)
            else set()
        )

        if not self_ref_cols:
            # No changes, output entire segment (including semicolon) as-is
            result_parts.append(ddl[m.start() : next_cursor])
            cursor = next_cursor
            continue

        changed_any = False
        new_items: List[str] = []
        for item in items:
            if _is_constraint_line(item):
                new_items.append(item)
                continue
            col = _parse_col_name(item)
            if col and col.lower() in self_ref_cols:
                new_item, changed = _remove_not_null_in_col_item(item)
                if changed:
                    changed_any = True
                new_items.append(new_item)
            else:
                new_items.append(item)

        if not changed_any:
            result_parts.append(ddl[m.start() : next_cursor])
            cursor = next_cursor
            continue

        # Reassemble: comma + newline + two-space indent
        new_body = ",\n  ".join(s.strip() for s in new_items)
        new_create = (
            f"CREATE TABLE {header.strip()} (\n  {new_body}\n){tail};"
        )
        result_parts.append(new_create)
        cursor = next_cursor

    return "".join(result_parts)


def sanitize_schema_files(
    db_dir: Path,
    db_id: str,
    dialects: Iterable[str] = ("mysql", "pg", "oracle"),
) -> Dict[str, bool]:
    """
    Sanitize the three schema files under a database directory in-place.

    Returns a {dialect: changed?} dictionary. Unmodified files are not rewritten.
    """
    results: Dict[str, bool] = {}
    for dialect in dialects:
        path = db_dir / f"{db_id}_schema_{dialect}.sql"
        if not path.exists():
            results[dialect] = False
            continue
        original = path.read_text(encoding="utf-8")
        updated = sanitize_self_ref_fk_nullable(original, dialect=dialect)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            results[dialect] = True
        else:
            results[dialect] = False
    return results
