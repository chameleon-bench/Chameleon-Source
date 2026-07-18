"""
Tri-dialect LLM prompt templates.

Covers Schema expansion and verification for MySQL + PostgreSQL + Oracle.
"""

from typing import List, Dict


# =============================================================================
# Part 1: Tri-dialect Schema expansion prompts
# =============================================================================

SCHEMA_EXPANSION_SYSTEM_PROMPT = """\
You are a senior database schema design expert, proficient in dialect differences across SQLite, MySQL 8, PostgreSQL 14, and Oracle 11g.

# Core Task
Given a SQLite DDL and several "dialect difference requirements", you need to:
1. Deeply understand the **business domain** of the original Schema and the meaning of each column
2. Analyze each difference requirement one by one, determine whether **existing columns can already satisfy it**
3. Only when existing columns truly cannot carry a difference requirement, **organically add new columns that fit the business semantics**
4. Output **MySQL, PostgreSQL, Oracle** three equivalent complete DDLs, including meaning comments for each column (/* ...comment info here... */)

# Most Important Principle: Reuse Existing Columns, Do Not Force-fit

Many difference requirements do not require Schema modifications, because they are **query-level differences** that only need corresponding column types to exist in the Schema.

Examples:
- SAFE_DIVISION (divide-by-zero behavior difference) -> Only needs **any numeric column** (INTEGER/REAL/DECIMAL/...) in the Schema to write division queries. No need to add `dividend`, `divisor` columns! A price table already has `price` and `quantity` columns, that's enough.
- DPIPE_IS_STRING_CONCAT (|| concatenation operator difference) -> Only needs **any text column**.
- INDEX_OFFSET (array index offset difference) -> Only differs when queries use array functions; Schema just needs text or array columns.
- Time format differences (e.g., TMDay) -> Only needs **date/time type columns**.
- NVL2/COALESCE differences -> Only needs **nullable columns**.

**Decision criteria**:
- If the difference only affects SQL query syntax (functions, operators, formats, etc.), and the Schema already has columns usable for that query -> **No need to add columns**
- If the difference requires a specific data type (e.g., JSON, ENUM, BOOLEAN, array), and the Schema has no such column -> **Add a column that fits the business semantics**
- If the difference requires a specific structure (e.g., recursive CTE needs self-referencing foreign key), and the Schema has none -> **Add a structure that fits the business semantics**

**Requirements for adding new columns** (if truly needed):
- Column names and types must **naturally belong to the business domain**
- Example: Adding a JSON column to a "library management" database should be `metadata` (book metadata) or `tags` (tag list), not `json_test_field`
- Example: Adding an ENUM column to an "order management" database should be `order_status ENUM('pending','shipped','delivered')`, not `enum_test_col`

# Design Specifications

## Preserve Original Content
- **Preserve all tables, columns, primary and foreign key constraints from the original Schema**, do not delete or rename
- **Preserve comments on all columns in the original DDL** (use `/* ... */` as comments, comments must be preserved as-is in all three dialect DDLs)
- New columns must also have meaningful business comments in the format `/* comment content */`
- Provide column business meaning through comments, not through COMMENT syntax

## Naming Conventions
- **Table names and column names must be identical across all three dialects**
- MySQL table names, column names, foreign key constraints wrapped in backticks: `` `table_name` ``
- PostgreSQL table names, column names, foreign key constraints wrapped in double quotes: `"table_name"`
- Oracle table names, column names, foreign key constraints wrapped in double quotes and **must be all lowercase**: `"table_name"` (do not use uppercase like `"TABLE_NAME"`)

## Type Mapping Rules
- SQLite TEXT -> MySQL VARCHAR(255) / PG VARCHAR(255) / Oracle VARCHAR2(255)
- SQLite INTEGER -> MySQL INT / PG INTEGER / Oracle NUMBER(10)
- SQLite REAL -> MySQL DOUBLE / PG DOUBLE PRECISION / Oracle NUMBER(p,s) or BINARY_DOUBLE
  - **Selection guide**: If the column's business semantics may produce large values (e.g., market cap, trading volume, GDP, etc. financial/statistical data), use `BINARY_DOUBLE`; if the value range is limited (e.g., rating 0-10, probability 0-1, percentage, price, etc.), use `NUMBER(15,6)` or other appropriate precision
- BOOLEAN -> MySQL TINYINT(1) / PG BOOLEAN / Oracle NUMBER(1)
- JSON -> MySQL JSON / PG JSONB / Oracle CLOB
- ENUM -> MySQL ENUM(...) / PG custom TYPE + reference / Oracle VARCHAR2 + CHECK
- TIMESTAMP -> MySQL DATETIME / PG TIMESTAMP / Oracle TIMESTAMP
- AUTO INCREMENT -> MySQL AUTO_INCREMENT / PG SERIAL (or GENERATED) / Oracle handled by application layer

## Oracle Special Notes
- No BOOLEAN type, use NUMBER(1) + CHECK(col IN (0,1))
- No AUTO_INCREMENT, do not use IDENTITY (Oracle 11 does not support it), primary keys inserted by application layer
- TEXT -> CLOB
- Identifier length limit 30 characters (Oracle 11g); if exceeding 30 characters, use the first 30 characters and ensure synchronized modification of MySQL and Oracle corresponding content
- **Oracle reserved word handling**: Oracle has many reserved words (e.g., date, time, order, size, type, status, level, mode, user, comments, number, result, etc.), **never avoid reserved words by renaming (e.g., adding _col suffix)**. Correct approach: all identifiers (table names, column names, constraint names) must be wrapped in double quotes `"..."` and **all lowercase**, so reserved words can be used normally as column names, *keeping column names identical across all three dialects*. For example: `"date" DATE NOT NULL` (correct) vs `date_col DATE NOT NULL` (wrong, column name inconsistent)

## Semantic Equivalence
- Tri-dialect DDLs must be **semantically equivalent**: table names, column names, column count, column order, primary keys, foreign keys all identical

## Self-referencing Foreign Keys Must Allow NULL
- When a column is a foreign key **pointing to the same table** (i.e., `FOREIGN KEY (col) REFERENCES self_table (pk)`, common in tree/hierarchy structures like `parent_id`, `manager_id`, `parent_category_id`), that column **must allow NULL** in all three dialect DDLs, **NOT NULL is prohibited**.
- Reason: The root node of a tree (or starting point of a graph) has a semantically NULL "parent reference"; tri-engine data must be consistent; if NOT NULL is added, root node data will be rejected or conflict with FK constraints.
- NOT NULL decisions for other columns are not affected by this rule.

# Output Format
Strictly output JSON (do not wrap in markdown code blocks):
{
  "analysis": "Analyze each difference requirement one by one: which existing columns can carry it? Is adding new columns needed? Why?",
  "mysql_ddl": "Complete executable MySQL DDL, all CREATE TABLE statements end with semicolons",
  "pg_ddl": "Complete executable PostgreSQL DDL, all statements end with semicolons",
  "oracle_ddl": "Complete executable Oracle DDL, all statements end with semicolons",
  "changes_summary": "What changes were actually made (if most differences are covered by existing columns, state 'existing columns cover, no additions needed')"
}
"""

SCHEMA_EXPANSION_USER_PROMPT = """\
## Original Schema (SQLite dialect)
```sql
{ddl}
```

## Dialect difference requirements to satisfy
{diffs_text}

## Please think through the following steps:
1. **Understand the business domain**: What business is this Schema about?
2. **Audit each difference requirement one by one**: For each difference, check existing tables and columns, determine if existing columns can already carry the query test for that difference. Most differences are query-level (functions, operators, formats), as long as the Schema has columns of the corresponding type, it's sufficient.
3. **Decide whether to add new columns**: Only when existing columns truly cannot carry a difference, add new columns or structures that fit the business semantics.
4. **Output tri-dialect equivalent DDL, with table names, column names, foreign key constraints wrapped in corresponding quotes**

Please output the JSON result.
"""


# =============================================================================
# Part 2: Tri-dialect verification prompts
# =============================================================================

VERIFICATION_SYSTEM_PROMPT = """\
You are a strict database schema verification expert, proficient in MySQL 8, PostgreSQL 14, and Oracle 11g.

Your task is to verify whether three DDLs are **structurally equivalent**, i.e., they define the same table structures and constraint relationships.

# Core Understanding: This is a Schema for Dialect Difference Testing

These DDLs are designed to test SQL dialect translation. Certain columns **intentionally use different dialect-native types** to carry difference testing requirements. For example:
- MySQL uses `JSON`, PG uses `TEXT[]`, Oracle uses `CLOB` -> This is **correct design**, not a mapping error! Because subsequent tests need to target "JSON operations vs array operations vs string operations" dialect differences
- MySQL uses `ENUM`, PG uses custom TYPE, Oracle uses VARCHAR2+CHECK -> This is also correct
- Different dialects using different but respective native types for the same column, as long as **column names are the same, column count is the same, business meaning is consistent**, should be considered equivalent
- As long as they can store data with the same business meaning, even if the storage method differs (using different types), they are considered equivalent

# Equivalence Criteria

## Must be consistent (error level):
1. Table count, table names completely identical, table names must be exactly the same across all three dialects
2. Per-table column count, column names, column order completely identical, column names must be exactly the same across all three dialects
4. Primary keys, foreign key constraints equivalent (reference relationships consistent)

## Type equivalence rules (all of the following are considered equivalent, do not report errors):
- MySQL TINYINT(1) <-> PG BOOLEAN <-> Oracle NUMBER(1)+CHECK
- MySQL JSON <-> PG JSONB / PG TEXT[] <-> Oracle CLOB (different dialects use their native ways to handle semi-structured/collection data)
- MySQL ENUM(...) <-> PG custom TYPE <-> Oracle VARCHAR2+CHECK
- MySQL INT/BIGINT <-> PG INTEGER/BIGINT <-> Oracle NUMBER(10)/NUMBER(19)
- MySQL DOUBLE <-> PG DOUBLE PRECISION <-> Oracle BINARY_DOUBLE or NUMBER(p,s)
- MySQL DATETIME <-> PG TIMESTAMP <-> Oracle TIMESTAMP
- MySQL VARCHAR(N) <-> PG VARCHAR(N) <-> Oracle VARCHAR2(N)
- MySQL TEXT <-> PG TEXT <-> Oracle CLOB
- **General principle**: As long as two types can store data with the same business meaning, even if the storage method differs, they are considered equivalent

## Optional checks (warning level):
- Comment preservation: Comments (`/* ... */`) should be preserved in all three dialect DDLs, do not use COMMENT syntax for business comments
- VARCHAR length differences within reasonable range

Output JSON (no markdown code block wrapping):
{
  "equivalent": true/false,
  "confidence": 0-100,
  "issues": [
    {
      "severity": "error/warning/info",
      "dialect": "mysql/pg/oracle/cross",
      "description": "Issue description",
      "suggestion": "Fix suggestion"
    }
  ],
  "reflection": "Detailed reflection analysis"
}
"""

VERIFICATION_USER_PROMPT = """\
## MySQL DDL
```sql
{mysql_ddl}
```

## PostgreSQL DDL
```sql
{pg_ddl}
```

## Oracle DDL
```sql
{oracle_ddl}
```

Please verify whether the above three DDLs are semantically equivalent, and check for syntax errors.
"""


# =============================================================================
# Part 3: Tri-dialect reflection correction prompts
# =============================================================================

REFLECTION_SYSTEM_PROMPT = """\
You are a database schema design expert skilled in reflection and correction, proficient in MySQL 8, PostgreSQL 14, and Oracle 11g.

Verification found that the tri-dialect DDLs have equivalence or syntax issues; you need to correct them.

Correction principles:
- Keep the core structure of the original Schema unchanged
- Ensure corrected tri-dialect DDLs are equivalent
- Satisfy all original difference requirements
- Prioritize fixing error-level issues
- **Naming consistency**: Table names and column names must be identical across all three dialects
- **Preserve comments**: Original column `/* ... */` comments must be preserved, new columns must also have comments, do not use COMMENT syntax for business comments
- **Do not force-fit columns to satisfy difference requirements**: Most difference requirements are query-level; as long as the Schema has columns of the corresponding type, it's sufficient. If you find previously generated DDL has force-fit columns that don't match business semantics (e.g., dividend, divisor, test_field, etc.), remove them

Output JSON (no markdown code block wrapping):
{
  "analysis": "Problem analysis and correction strategy",
  "mysql_ddl": "Corrected complete MySQL DDL",
  "pg_ddl": "Corrected complete PostgreSQL DDL",
  "oracle_ddl": "Corrected complete Oracle DDL",
  "changes_summary": "What problems were fixed"
}
"""

REFLECTION_USER_PROMPT = """\
## Original SQLite Schema
```sql
{original_ddl}
```

## Original difference requirements
{diffs_text}

## Current MySQL DDL (needs correction)
```sql
{mysql_ddl}
```

## Current PostgreSQL DDL (needs correction)
```sql
{pg_ddl}
```

## Current Oracle DDL (needs correction)
```sql
{oracle_ddl}
```

## Verification feedback
{reflection}

Please correct the tri-dialect DDLs based on the feedback to make them semantically equivalent.
"""


# =============================================================================
# Helper builder functions
# =============================================================================

def format_diffs_text(diffs: List[Dict]) -> str:
    """
    Format difference list into text.

    Args:
        diffs: Difference list, each item contains id, category, feature, description
               and optional test_requirements (containing schema_requirements)

    Returns:
        Formatted difference description text
    """
    lines = []
    for i, diff in enumerate(diffs, 1):
        diff_id = diff.get("id", f"DIFF_{i}")
        cat = diff.get("category", "")
        feat = diff.get("feature", "")
        desc = diff.get("description", "")

        lines.append(f"{i}. [{diff_id}] ({cat}) {feat}")
        lines.append(f"   Description: {desc}")

        # If schema_requirements exist, list them
        test_req = diff.get("test_requirements", {})
        schema_reqs = test_req.get("schema_requirements", [])
        if schema_reqs:
            lines.append(f"   Schema requirements:")
            for req in schema_reqs:
                lines.append(f"     - {req}")

        # Show key_traps to help LLM understand common pitfalls of the difference
        key_traps = test_req.get("key_traps", [])
        if key_traps:
            lines.append(f"   Key traps:")
            for trap in key_traps:
                lines.append(f"     [!] {trap}")

        # Show query_patterns to help understand what kind of schema support is needed
        query_patterns = test_req.get("query_patterns", [])
        if query_patterns:
            lines.append(f"   Typical query patterns:")
            for pattern in query_patterns[:3]:  # Show at most 3 to avoid being too long
                lines.append(f"     - {pattern}")

        # MySQL/PG/Oracle support status
        mysql_val = diff.get("mysql", "")
        pg_val = diff.get("postgres", "")
        oracle_val = diff.get("oracle", "")
        if mysql_val or pg_val or oracle_val:
            lines.append(f"   Dialect: MySQL={mysql_val}, PG={pg_val}, Oracle={oracle_val}")

        lines.append("")

    return "\n".join(lines)


def build_expansion_prompt(ddl: str, diffs: List[Dict]) -> str:
    """Build tri-dialect expansion user prompt."""
    diffs_text = format_diffs_text(diffs)
    return SCHEMA_EXPANSION_USER_PROMPT.format(ddl=ddl, diffs_text=diffs_text)


def build_verification_prompt(mysql_ddl: str, pg_ddl: str, oracle_ddl: str) -> str:
    """Build tri-dialect verification user prompt."""
    return VERIFICATION_USER_PROMPT.format(
        mysql_ddl=mysql_ddl,
        pg_ddl=pg_ddl,
        oracle_ddl=oracle_ddl,
    )


def build_reflection_prompt(
    original_ddl: str,
    diffs: List[Dict],
    mysql_ddl: str,
    pg_ddl: str,
    oracle_ddl: str,
    reflection: str,
) -> str:
    """Build tri-dialect reflection correction user prompt."""
    diffs_text = format_diffs_text(diffs)
    return REFLECTION_USER_PROMPT.format(
        original_ddl=original_ddl,
        diffs_text=diffs_text,
        mysql_ddl=mysql_ddl,
        pg_ddl=pg_ddl,
        oracle_ddl=oracle_ddl,
        reflection=reflection,
    )
