"""
SQL preprocessing utilities for deterministic cross-engine evaluation.

Replaces time-sensitive functions (NOW(), CURDATE(), SYSDATE, etc.) and
random functions (RAND(), RANDOM(), DBMS_RANDOM.VALUE) with fixed constants
so that source and target engines produce deterministic, comparable results.
"""

import re

# Fixed time anchor: a mid-dataset timestamp ensuring queries hit data
FIXED_TIMESTAMP = '2024-06-15 00:00:00'
FIXED_DATE = '2024-06-15'

# Fixed random value (subquery form to avoid ORDER BY column-index ambiguity)
FIXED_RANDOM = '(SELECT 0.5)'

# ---------------------------------------------------------------------------
# Time-sensitive function replacements per dialect
# ---------------------------------------------------------------------------

_MYSQL_TIME_REPLACEMENTS = [
    (re.compile(r'\bNOW\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'"),
    (re.compile(r'\bCURDATE\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_DATE}'"),
    (re.compile(r'\bCURRENT_DATE\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_DATE}'"),
    (re.compile(r'\bCURRENT_DATE\b(?!\s*\()', re.IGNORECASE), f"'{FIXED_DATE}'"),
    (re.compile(r'\bCURRENT_TIMESTAMP\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'"),
    (re.compile(r'\bCURRENT_TIMESTAMP\b(?!\s*\()', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'"),
    (re.compile(r'\bCURTIME\s*\(\s*\)', re.IGNORECASE), "'00:00:00'"),
    (re.compile(r'\bUTC_DATE\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_DATE}'"),
    (re.compile(r'\bUTC_TIMESTAMP\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'"),
    (re.compile(r'\bSYSDATE\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'"),
]

_PG_TIME_REPLACEMENTS = [
    (re.compile(r'\bNOW\s*\(\s*\)', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'::timestamp"),
    (re.compile(r'\bCURRENT_DATE\b', re.IGNORECASE), f"'{FIXED_DATE}'::date"),
    (re.compile(r'\bCURRENT_TIMESTAMP\b', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'::timestamp"),
    (re.compile(r'\bLOCALTIMESTAMP\b', re.IGNORECASE), f"'{FIXED_TIMESTAMP}'::timestamp"),
    (re.compile(r'\bLOCALTIME\b', re.IGNORECASE), "'00:00:00'::time"),
    (re.compile(r'\bCURRENT_TIME\b', re.IGNORECASE), "'00:00:00'::time"),
]

_ORACLE_TIME_REPLACEMENTS = [
    (re.compile(r'\bSYSDATE\b', re.IGNORECASE), f"TO_TIMESTAMP('{FIXED_TIMESTAMP}', 'YYYY-MM-DD HH24:MI:SS')"),
    (re.compile(r'\bCURRENT_DATE\b', re.IGNORECASE), f"TO_DATE('{FIXED_DATE}', 'YYYY-MM-DD')"),
    (re.compile(r'\bCURRENT_TIMESTAMP\b', re.IGNORECASE), f"TO_TIMESTAMP('{FIXED_TIMESTAMP}', 'YYYY-MM-DD HH24:MI:SS')"),
    (re.compile(r'\bLOCALTIMESTAMP\b', re.IGNORECASE), f"TO_TIMESTAMP('{FIXED_TIMESTAMP}', 'YYYY-MM-DD HH24:MI:SS')"),
    (re.compile(r'\bSYSTIMESTAMP\b', re.IGNORECASE), f"TO_TIMESTAMP_TZ('{FIXED_TIMESTAMP} +00:00', 'YYYY-MM-DD HH24:MI:SS TZH:TZM')"),
]

# ---------------------------------------------------------------------------
# Random function replacements per dialect
# ---------------------------------------------------------------------------

_MYSQL_RAND_REPLACEMENTS = [
    (re.compile(r'\bRAND\s*\(\s*\d+\s*\)', re.IGNORECASE), FIXED_RANDOM),
    (re.compile(r'\bRAND\s*\(\s*\)', re.IGNORECASE), FIXED_RANDOM),
]

_PG_RAND_REPLACEMENTS = [
    (re.compile(r'\bRANDOM\s*\(\s*\)', re.IGNORECASE), FIXED_RANDOM),
]

_ORACLE_RAND_REPLACEMENTS = [
    (re.compile(r'\bDBMS_RANDOM\.VALUE\s*\(\s*[\d.]+\s*,\s*[\d.]+\s*\)', re.IGNORECASE), FIXED_RANDOM),
    (re.compile(r'\bDBMS_RANDOM\.VALUE\b(?!\s*\()', re.IGNORECASE), FIXED_RANDOM),
    (re.compile(r'\bDBMS_RANDOM\.VALUE\s*\(\s*\)', re.IGNORECASE), FIXED_RANDOM),
    (re.compile(r'\bDBMS_RANDOM\.RANDOM\b', re.IGNORECASE), '(SELECT 42 FROM DUAL)'),
]


def pin_time_functions(sql: str, dialect: str) -> str:
    """Replace time-sensitive functions with fixed constants for deterministic execution."""
    _alias_placeholders = {}
    _alias_counter = [0]

    def _protect_alias(m):
        key = f"__ALIAS_PLACEHOLDER_{_alias_counter[0]}__"
        _alias_counter[0] += 1
        _alias_placeholders[key] = m.group(0)
        return key

    sql = re.sub(
        r'\bAS\s+(current_time|current_date|current_timestamp)\b',
        _protect_alias, sql, flags=re.IGNORECASE,
    )

    if dialect == 'mysql':
        replacements = _MYSQL_TIME_REPLACEMENTS
    elif dialect == 'oracle':
        replacements = _ORACLE_TIME_REPLACEMENTS
    else:
        replacements = _PG_TIME_REPLACEMENTS
    for pattern, replacement in replacements:
        sql = pattern.sub(replacement, sql)

    for key, original in _alias_placeholders.items():
        sql = sql.replace(key, original)

    return sql


def pin_random_functions(sql: str, dialect: str) -> str:
    """Replace random functions with fixed values for deterministic execution."""
    if dialect == 'mysql':
        replacements = _MYSQL_RAND_REPLACEMENTS
    elif dialect == 'oracle':
        replacements = _ORACLE_RAND_REPLACEMENTS
    else:
        replacements = _PG_RAND_REPLACEMENTS
    for pattern, replacement in replacements:
        sql = pattern.sub(replacement, sql)
    return sql
