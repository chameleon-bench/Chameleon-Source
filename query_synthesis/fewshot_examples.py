"""
Few-shot Examples for SQL Query Synthesis
==========================================

This file provides few-shot examples for LLM SQL query generation, supporting **MySQL**, **PostgreSQL**, and **Oracle** dialects.

## Core Design Philosophy

**Difficulty levels are based on translation difficulty, not just SQL structural complexity.**

This benchmark tests LLM's MySQL<->PostgreSQL<->Oracle dialect translation ability, so difficulty should reflect
how hard it is to translate this query to another dialect. A structurally simple query with semantic traps
(e.g., LOG(x) same name different meaning), its translation difficulty is far higher than a query using CTE + window functions
where all functions have consistent syntax across dialects.

## Difficulty Grading Dimensions

| Dimension | Description |
|------|------|
| Dialect difference count | Number of tri-dialect differences in a single query |
| Translation mode | Simple rename -> syntax rewrite -> structural rewrite -> semantic trap |
| Nesting depth | Difference at top level vs inside window function/subquery/PARTITION BY |
| Direction specificity | Bidirectional simple substitution vs unidirectional full rewrite (e.g., PG->MySQL DISTINCT ON, Oracle->MySQL ROWNUM) |
| Boundary traps | Whether involving NULL semantic differences, integer division, divide-by-zero behavior, etc. |

## Difficulty Definition (v3.0 - based on tri-dialect translation difficulty)

### Easy (Simple Translation)
- **Dialect difference points**: 0-1
- **Translation mode**: Simple rename or no difference
- **Structural complexity**: 1-2 tables, no subquery/window function/CTE
- **Typical translation**: IFNULL->COALESCE/NVL, CURDATE()->CURRENT_DATE/SYSDATE,
  backtick->double quote, LIMIT->ROWNUM/ROW_NUMBER subquery
- **Why easy**: One-to-one function name replacement, no need to understand semantic differences

### Medium (Regular Translation)
- **Dialect difference points**: 1-2
- **Translation mode**: Syntax rewrite (need to understand syntax structure differences across 2/3 dialects)
- **Structural complexity**: 2-3 tables, may have GROUP BY / HAVING / simple subquery
- **Typical translation**: DATE_FORMAT->TO_CHAR (with format specifier mapping), GROUP_CONCAT->STRING_AGG/LISTAGG,
  IF()->CASE WHEN/DECODE, DATEDIFF->date subtraction/MONTHS_BETWEEN,
  INTERVAL quoting differences, NVL->COALESCE/IFNULL
- **Why medium**: Not just renaming, but also changing syntax structure and format specifiers;
  but each difference point is independent, no interference

### Hard (Difficult Translation)
- **Dialect difference points**: 2-3, with at least 1 being 3-star difficulty
- **Translation mode**: Structural rewrite + multiple difference point stacking
- **Structural complexity**: 3+ tables, may have window functions or subqueries
- **Typical translation**:
  - PG->MySQL/Oracle: DISTINCT ON->ROW_NUMBER subquery, FILTER->CASE WHEN,
    generate_series->recursive CTE/CONNECT BY, BOOL_AND->MIN simulation, AGE()->TIMESTAMPDIFF/MONTHS_BETWEEN
  - MySQL->PG/Oracle: integer division semantics, REGEXP_REPLACE default behavior differences
  - Oracle→MySQL/PG: ROWNUM→LIMIT/OFFSET, DECODE→CASE WHEN, NVL→IFNULL/COALESCE,
    (+)outer join->LEFT JOIN, CONNECT BY->recursive CTE
  - Multiple difference point stacking: e.g., date formatting + INTERVAL quoting + type conversion in same query
- **Why hard**: Need to understand multi-dialect structural differences, do non-one-to-one rewrites;
  easy to miss when multiple difference points appear simultaneously

### Extra (Extremely Difficult Translation)
- **Dialect difference points**: 4+, including semantic traps or high-difficulty structural rewrites
- **Translation mode**: Semantic traps + multiple difference point stacking + nested context + structural rewrite
- **Structural complexity**: CTE / nested subquery / window function / set operation (at least 2)
- **Typical translation challenges**:
  - A) Semantic traps: LOG(x) same name different meaning (MySQL=ln, PG=log10, Oracle=ln),
       LENGTH() bytes vs chars, GREATEST(NULL) behavior differences,
       integer division 5/2 (MySQL=2.5, PG=2, Oracle=2.5), divide by zero 1/0 NULL vs ERROR
  - B) Full format specifier mapping: single TO_CHAR/DATE_FORMAT with 5+ format codes
       (Oracle and PG TO_CHAR format codes also differ)
  - C) Nested context: dialect-specific functions appear inside window function PARTITION BY,
       inside CTE, inside subquery, need to process layer by layer when translating
  - D) Multiple difference point stacking: single query triggers 4-6 difference points simultaneously
  - E) JSON path syntax differences + array expansion + regex differences stacking
- **Why extremely hard**: Even if the model knows each difference's translation rule, when they
  stack in complex structures, the model can easily miss a difference point or mistranslate semantic traps

## Example Schema (shared by all examples)

```sql
-- Example schema for few-shot, not a real database
CREATE TABLE department (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(100),
    budget DECIMAL(12,2),
    created_at DATE
);

CREATE TABLE employee (
    emp_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    hire_date DATE,
    salary DECIMAL(10,2),
    dept_id INT REFERENCES department(dept_id),
    manager_id INT REFERENCES employee(emp_id),
    is_active BOOLEAN,
    metadata JSON
);

CREATE TABLE project (
    proj_id INT PRIMARY KEY,
    proj_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    status VARCHAR(20),  -- 'active', 'completed', 'cancelled'
    budget DECIMAL(12,2)
);

CREATE TABLE assignment (
    emp_id INT REFERENCES employee(emp_id),
    proj_id INT REFERENCES project(proj_id),
    role VARCHAR(50),
    hours_allocated DECIMAL(6,1),
    PRIMARY KEY (emp_id, proj_id)
);

CREATE TABLE timesheet (
    ts_id INT PRIMARY KEY,
    emp_id INT REFERENCES employee(emp_id),
    proj_id INT REFERENCES project(proj_id),
    work_date DATE,
    hours_worked DECIMAL(4,1),
    description TEXT
);
```
"""

# =============================================================================
# Difficulty level definitions (injected into prompt)
# Core change: difficulty based on translation difficulty not just SQL structural complexity
# =============================================================================

DIFFICULTY_DEFINITIONS = {
    "easy": {
        "label": "Easy (Simple Translation)",
        "translation_difficulty": "⭐",
        "dialect_points": "0-1",
        "constraints": (
            "The **translation difficulty** of queries generated this time is Easy:\n"
            "[Structural Constraints]\n"
            "- Only 1-2 tables (max 1 JOIN)\n"
            "- No subqueries, window functions, CTE\n"
            "- Can use: SELECT, WHERE, ORDER BY, LIMIT/ROWNUM/ROW_NUMBER pagination, "
            "simple aggregate functions (COUNT, SUM, AVG, MAX, MIN)\n"
            "[Translation Difficulty Constraints]\n"
            "- Contains 0-1 tri-dialect difference points\n"
            "- Difference types limited to simple function name replacement: e.g., IFNULL->COALESCE/NVL, "
            "CURDATE()->CURRENT_DATE/SYSDATE, CONCAT->||, LIMIT->ROWNUM/ROW_NUMBER pagination\n"
            "- No semantic traps (e.g., LOG/LENGTH same name different meaning)\n"
            "- Encourage using provided real data values to construct WHERE filter conditions"
        ),
    },
    "medium": {
        "label": "Medium (Regular Translation)",
        "translation_difficulty": "⭐⭐",
        "dialect_points": "1-2",
        "constraints": (
            "The **translation difficulty** of queries generated this time is Medium:\n"
            "[Structural Constraints]\n"
            "- Involves 2-3 tables (using JOIN)\n"
            "- Can use GROUP BY, HAVING, ORDER BY\n"
            "- Can use simple subqueries (IN / EXISTS)\n"
            "- No window functions, CTE\n"
            "[Translation Difficulty Constraints]\n"
            "- Include 1-2 tri-dialect difference points\n"
            "- Difference types include syntax rewrite (not just renaming):\n"
            "  * Date formatting: DATE_FORMAT('%Y-%m') ↔ TO_CHAR('YYYY-MM')(PG) ↔ TO_CHAR('YYYY-MM')(Oracle)\n"
            "  * Aggregation concatenation: GROUP_CONCAT(col ORDER BY col SEPARATOR ',') ↔ STRING_AGG(col, ',') ↔ LISTAGG(col, ',')\n"
            "  * Conditional function: IF(cond, a, b) ↔ CASE WHEN cond THEN a ELSE b END ↔ DECODE(expr, val1, res1, default)\n"
            "  * NULL handling: IFNULL(a, b) ↔ COALESCE(a, b) ↔ NVL(a, b)\n"
            "  * Date difference: DATEDIFF(d1, d2) ↔ d1::date - d2::date ↔ (d1 - d2)\n"
            "  * INTERVAL quoting: INTERVAL 30 DAY ↔ INTERVAL '30 days' ↔ INTERVAL '30' DAY\n"
            "  * Date extraction: YEAR(d)/MONTH(d) ↔ EXTRACT(YEAR FROM d) ↔ EXTRACT(YEAR FROM d)\n"
            "- Each difference point appears independently, without interfering with each other\n"
            "- Encourage using real data values in WHERE/HAVING clauses"
        ),
    },
    "hard": {
        "label": "Hard (Difficult Translation)",
        "translation_difficulty": "⭐⭐⭐",
        "dialect_points": "2-3",
        "constraints": (
            "The **translation difficulty** of queries generated this time is Hard:\n"
            "[Structural Constraints]\n"
            "- Involves 3+ tables\n"
            "- Must include at least 1 of: subquery (can be nested), window function\n"
            "- Can use HAVING, multiple aggregations\n"
            "[Translation Difficulty Constraints]\n"
            "- Include 2-3 dialect difference points, and at least 1 belongs to ⭐⭐⭐ high-difficulty differences\n"
            "- Difference types include **structural rewrite** or **multiple difference point stacking**:\n"
            "  * Structural rewrite (PG→MySQL/Oracle): DISTINCT ON→ROW_NUMBER subquery, "
            "FILTER clause→CASE WHEN, BOOL_AND→MIN simulation, AGE()→TIMESTAMPDIFF/MONTHS_BETWEEN\n"
            "  * Structural rewrite (Oracle→MySQL/PG): ROWNUM→LIMIT/OFFSET, "
            "DECODE→CASE WHEN, NVL→IFNULL/COALESCE, (+) outer join→LEFT JOIN\n"
            "  * Structural rewrite (MySQL→PG/Oracle): integer division 5/2 semantics, "
            "REGEXP_REPLACE default behavior difference (MySQL global / PG first-only)\n"
            "  * Multiple difference stacking: date formatting + INTERVAL quoting + type cast appearing simultaneously\n"
            "  * Semi-structured text extraction: doc->>'$.key' ↔ doc->>'key' ↔ Oracle 11g available REGEXP_SUBSTR/INSTR/SUBSTR etc. compatible syntax\n"
            "- Difference points can appear inside window functions or subqueries"
        ),
    },
    "extra": {
        "label": "Extra (Extremely Difficult Translation)",
        "translation_difficulty": "⭐⭐⭐+",
        "dialect_points": "4+",
        "constraints": (
            "The **translation difficulty** of queries generated this time is Extra:\n"
            "[Structural Constraints]\n"
            "- Must include at least 2 of: CTE, nested subquery, window function, "
            "set operation (UNION/EXCEPT/INTERSECT/MINUS)\n"
            "- Can use recursive query, nested window function, complex aggregation and other advanced features; if target is Oracle 11g, avoid LATERAL JOIN, FETCH FIRST, JSON_VALUE etc. high-version syntax\n"
            "[Translation Difficulty Constraints — this is key!]\n"
            "- Include 4+ dialect difference points, and at least 2 of the following:\n"
            "  (A) Semantic trap — same-name function with different semantics, model must know semantic differences to translate correctly:\n"
            "      * LOG(x): MySQL=natural log ln(x), PG=base-10 log₁₀(x), Oracle=natural log ln(x)\n"
            "      * LENGTH(str): MySQL=byte count, PG=char count, Oracle=char count\n"
            "      * GREATEST(5, NULL, 3): MySQL→NULL, PG→5, Oracle→5\n"
            "      * Integer division 5/2: MySQL→2.5, PG→2, Oracle→2.5\n"
            "      * Division by zero 1/0: MySQL→NULL, PG→ERROR, Oracle→ERROR\n"
            "      * REGEXP_REPLACE: MySQL default global replace, PG default first-only, Oracle default first-only\n"
            "  (B) Format specifier full mapping — single date formatting expression with 4+ format codes:\n"
            "      * DATE_FORMAT(d, '%W, %M %d, %Y %h:%i %p')\n"
            "        ↔ TO_CHAR(d, 'Day, Month DD, YYYY HH12:MI AM')(PG)\n"
            "        ↔ TO_CHAR(d, 'Day, Month DD, YYYY HH12:MI AM')(Oracle)\n"
            "  (C) Nested context — dialect-specific function appears inside window function/PARTITION BY/CTE:\n"
            "      * RANK() OVER (PARTITION BY DATE_FORMAT(d, '%Y-%m') ORDER BY ...)\n"
            "      * CTE inner uses GROUP_CONCAT/LISTAGG + INTERVAL + IF/DECODE etc. multi-dialect functions\n"
            "  (D) Structural rewrite stacking — single query needs multiple non-trivial structural rewrites:\n"
            "      * PG→MySQL: generate_series→recursive CTE + FILTER→CASE WHEN + ::→CAST\n"
            "      * MySQL→PG: LAST_DAY→composite expression + DATE_ADD→INTERVAL + IFNULL→COALESCE\n"
            "      * Oracle→MySQL/PG: CONNECT BY→recursive CTE + ROWNUM→LIMIT + NVL→IFNULL/COALESCE\n"
            "  (E) JSON + timestamp + array and other advanced feature difference stacking\n"
            "- Query should exhibit multi-step analysis logic\n"
            "- Goal: make the model **miss or mistranslate at least 1-2 difference points** when translating"
        ),
    },
}

# Default difficulty distribution weights
DEFAULT_DIFFICULTY_WEIGHTS = {
    "easy": 0.25,
    "medium": 0.25,
    "hard": 0.35,
    "extra": 0.15,
}


# =============================================================================
# MySQL Few-shot Examples
# =============================================================================
# Examples arranged in increasing translation difficulty.
# Easy: 0-1 difference points, simple rename (IFNULL→COALESCE/NVL, CONCAT→||)
# Medium: 1-2 difference points, syntax rewrite (DATE_FORMAT→TO_CHAR, IF→CASE/DECODE)
# Hard: 2-3 difference points, structural rewrite + multiple difference stacking (ROLLUP, JSON, REGEXP)
# Extra: 4+ difference points, semantic trap + nested context + format specifier full mapping
# =============================================================================

FEWSHOT_EXAMPLES_MYSQL = [

    # ==================================================================
    # EASY: 0-1 dialect difference points, simple function name replacement
    # ==================================================================

    {
        "difficulty": "easy",
        "query_id": "mysql_easy_1",
        "comment": "Count employees hired in 2024, treat NULL salary as 0",
        "sql": (
            "SELECT COUNT(*) AS cnt, AVG(IFNULL(salary, 0)) AS avg_salary "
            "FROM employee "
            "WHERE hire_date >= '2024-01-01' AND hire_date < '2025-01-01'"
        ),
        "dialect_features_used": ["D7.2"],
        "builtin_functions_used": ["IFNULL", "COUNT", "AVG"],
    },

    {
        "difficulty": "easy",
        "query_id": "mysql_easy_2",
        "comment": "Concatenate names, find employees whose email domain is gmail.com",
        "sql": (
            "SELECT CONCAT(first_name, ' ', last_name) AS full_name, email "
            "FROM employee "
            "WHERE email LIKE '%@gmail.com'"
        ),
        "dialect_features_used": ["D3.1"],
        "builtin_functions_used": ["CONCAT"],
    },

    {
        "difficulty": "easy",
        "query_id": "mysql_easy_3",
        "comment": "Get employees ranked 11-20 by salary descending (pagination)",
        "sql": (
            "SELECT emp_id, first_name, last_name, salary "
            "FROM employee "
            "ORDER BY salary DESC "
            "LIMIT 10, 10"
        ),
        "dialect_features_used": ["D2.1"],
        "builtin_functions_used": [],
    },

    # ==================================================================
    # MEDIUM: 1-2 dialect difference points, syntax rewrite (not just renaming)
    # ==================================================================

    {
        "difficulty": "medium",
        "query_id": "mysql_medium_1",
        "comment": "Count active employee hours per month by project (DATE_FORMAT format specifier mapping)",
        "sql": (
            "SELECT DATE_FORMAT(t.work_date, '%Y-%m') AS month, "
            "       e.first_name, SUM(t.hours_worked) AS total_hours "
            "FROM timesheet t "
            "INNER JOIN employee e ON t.emp_id = e.emp_id "
            "WHERE t.proj_id IN (SELECT proj_id FROM project WHERE status = 'active') "
            "GROUP BY month, e.emp_id, e.first_name "
            "ORDER BY month, total_hours DESC"
        ),
        "dialect_features_used": ["D4.2"],
        "builtin_functions_used": ["DATE_FORMAT", "SUM"],
    },

    {
        "difficulty": "medium",
        "query_id": "mysql_medium_2",
        "comment": "Count high/low salary employees per department + concatenate names (IF→CASE, GROUP_CONCAT→STRING_AGG)",
        "sql": (
            "SELECT d.dept_name, "
            "       SUM(IF(e.salary >= 80000, 1, 0)) AS high_salary_count, "
            "       SUM(IF(e.salary < 80000, 1, 0)) AS low_salary_count, "
            "       GROUP_CONCAT(e.last_name ORDER BY e.last_name SEPARATOR ', ') AS all_names "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_id, d.dept_name"
        ),
        "dialect_features_used": ["D7.3", "D3.8"],
        "builtin_functions_used": ["IF", "SUM", "GROUP_CONCAT"],
    },

    {
        "difficulty": "medium",
        "query_id": "mysql_medium_3",
        "comment": "Find projects overdue by 30 days (DATEDIFF→subtraction, INTERVAL unquoted→quoted)",
        "sql": (
            "SELECT p.proj_name, p.end_date, "
            "       DATEDIFF(CURDATE(), p.end_date) AS overdue_days "
            "FROM project p "
            "WHERE p.status = 'active' "
            "  AND p.end_date < DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
            "ORDER BY overdue_days DESC"
        ),
        "dialect_features_used": ["D4.4", "D4.5", "D16.8"],
        "builtin_functions_used": ["DATEDIFF", "DATE_SUB", "CURDATE"],
    },

    {
        "difficulty": "medium",
        "query_id": "mysql_medium_4",
        "comment": "Count new hires by year/month (YEAR/MONTH→EXTRACT)",
        "sql": (
            "SELECT d.dept_name, "
            "       YEAR(e.hire_date) AS hire_year, "
            "       MONTH(e.hire_date) AS hire_month, "
            "       COUNT(*) AS new_hires "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_name, hire_year, hire_month "
            "HAVING COUNT(*) >= 2 "
            "ORDER BY hire_year DESC, hire_month DESC"
        ),
        "dialect_features_used": ["D4.6"],
        "builtin_functions_used": ["YEAR", "MONTH", "COUNT"],
    },

    # ==================================================================
    # HARD: 2-3 dialect difference points, including ⭐⭐⭐ differences, structural rewrite or multiple difference stacking
    # ==================================================================

    {
        "difficulty": "hard",
        "query_id": "mysql_hard_1",
        "comment": "Window function inner difference: backtick+DATEDIFF+INTERVAL triple difference stacking",
        "sql": (
            "SELECT emp_id, work_date, hours_worked, "
            "       DATEDIFF(work_date, LAG(work_date) OVER ("
            "         PARTITION BY emp_id ORDER BY work_date"
            "       )) AS `gap_days` "
            "FROM timesheet "
            "WHERE work_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
        ),
        "dialect_features_used": ["D1.1", "D4.5", "D4.4", "D16.8"],
        "builtin_functions_used": ["LAG", "DATEDIFF", "DATE_SUB", "CURDATE"],
    },

    {
        "difficulty": "hard",
        "query_id": "mysql_hard_2",
        "comment": "Multiple format code DATE_FORMAT + IF + CAST SIGNED — three different rewrites",
        "sql": (
            "SELECT d.dept_name, "
            "       DATE_FORMAT(e.hire_date, '%W, %M %d, %Y') AS hire_date_fmt, "
            "       CAST(AVG(DATEDIFF(CURDATE(), e.hire_date)) AS SIGNED) AS avg_tenure, "
            "       IF(AVG(DATEDIFF(CURDATE(), e.hire_date)) > 1000, 'senior', 'junior') AS team_type "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_name, e.hire_date "
            "ORDER BY avg_tenure DESC"
        ),
        "dialect_features_used": ["D4.2", "D7.3", "D6.2", "D4.5"],
        "builtin_functions_used": ["DATE_FORMAT", "CAST", "AVG", "DATEDIFF", "IF", "CURDATE"],
    },

    {
        "difficulty": "hard",
        "query_id": "mysql_hard_3",
        "comment": "JSONpathextract + REGEXP + IF — three syntax rewrites",
        "sql": (
            "SELECT d.dept_name, "
            "       JSON_UNQUOTE(JSON_EXTRACT(e.metadata, '$.primary_skill')) AS skill, "
            "       COUNT(*) AS cnt, "
            "       IF(COUNT(*) >= 3, 'core', 'rare') AS skill_type "
            "FROM employee e "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "WHERE JSON_EXTRACT(e.metadata, '$.primary_skill') IS NOT NULL "
            "  AND JSON_UNQUOTE(JSON_EXTRACT(e.metadata, '$.primary_skill')) REGEXP '^(Python|Java|Go)' "
            "GROUP BY d.dept_name, skill "
            "HAVING cnt >= 1 "
            "ORDER BY d.dept_name, cnt DESC"
        ),
        "dialect_features_used": ["D12.2", "D13.1", "D7.3"],
        "builtin_functions_used": ["JSON_EXTRACT", "JSON_UNQUOTE", "REGEXP", "IF", "COUNT"],
    },

    {
        "difficulty": "hard",
        "query_id": "mysql_hard_4",
        "comment": "ROLLUP syntax difference + IFNULL + 4-table JOIN",
        "sql": (
            "SELECT IFNULL(d.dept_name, '[Total]') AS dept_name, "
            "       IFNULL(p.status, '[Subtotal]') AS status, "
            "       COUNT(*) AS assignment_count, "
            "       SUM(a.hours_allocated) AS total_hours "
            "FROM assignment a "
            "INNER JOIN employee e ON a.emp_id = e.emp_id "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "INNER JOIN project p ON a.proj_id = p.proj_id "
            "GROUP BY d.dept_name, p.status WITH ROLLUP"
        ),
        "dialect_features_used": ["D8.2", "D7.2", "D3.1"],
        "builtin_functions_used": ["IFNULL", "COUNT", "SUM"],
    },

    # ==================================================================
    # EXTRA: 4+ dialect difference points
    # semantic trap + nested context + format specifier full mapping + multiple difference depth stacking
    # ==================================================================

    {
        "difficulty": "extra",
        "query_id": "mysql_extra_1",
        "comment": "⚠️semantic trap: LOG same-name different-semantics + integer division + GREATEST(NULL) behavior + TRUNCATE naming",
        "sql": (
            "WITH salary_analysis AS ("
            "  SELECT e.emp_id, d.dept_name, e.salary, "
            "         LOG(e.salary) AS log_salary, "
            "         TRUNCATE(e.salary / IFNULL(a.hours_allocated, 1), 2) AS hourly_rate, "
            "         GREATEST(e.salary, ("
            "           SELECT AVG(e2.salary) FROM employee e2 WHERE e2.dept_id = e.dept_id"
            "         ), NULL) AS effective_salary "
            "  FROM employee e "
            "  INNER JOIN department d ON e.dept_id = d.dept_id "
            "  LEFT JOIN assignment a ON e.emp_id = a.emp_id"
            ") "
            "SELECT dept_name, COUNT(*) AS emp_count, "
            "       TRUNCATE(AVG(log_salary), 4) AS avg_log_salary, "
            "       TRUNCATE(AVG(hourly_rate), 2) AS avg_hourly_rate, "
            "       SUM(IF(effective_salary IS NULL, 1, 0)) AS null_salary_count "
            "FROM salary_analysis "
            "GROUP BY dept_name ORDER BY avg_log_salary DESC"
        ),
        "dialect_features_used": ["D5.4", "D5.2", "D7.4", "D7.2", "D5.1", "D7.3"],
        "builtin_functions_used": ["LOG", "TRUNCATE", "IFNULL", "GREATEST", "AVG", "COUNT", "SUM", "IF"],
        "translation_traps": [
            "LOG(salary): MySQL=ln(salary), PG LOG=log10(x), translate to PG must use LN(salary)",
            "salary/hours_allocated: PG integer division truncates, MySQL does not",
            "GREATEST(..., NULL): MySQLreturnNULL, PGignoreNULL",
            "TRUNCATE(x,n) → PG: TRUNC(x,n)",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "mysql_extra_2",
        "comment": "Nested context: 6 format code DATE_FORMAT + CTE inner INTERVAL + GROUP_CONCAT + IF multiple stacking",
        "sql": (
            "WITH monthly_work AS ("
            "  SELECT t.emp_id, "
            "         DATE_FORMAT(t.work_date, '%Y-%m') AS work_month, "
            "         DATE_FORMAT(t.work_date, '%W, %M %d, %Y') AS formatted_date, "
            "         SUM(t.hours_worked) AS monthly_hours "
            "  FROM timesheet t "
            "  WHERE t.work_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) "
            "  GROUP BY t.emp_id, work_month, formatted_date"
            "), "
            "ranked AS ("
            "  SELECT mw.*, e.first_name, d.dept_name, "
            "         RANK() OVER (PARTITION BY mw.work_month ORDER BY mw.monthly_hours DESC) AS month_rank, "
            "         IF(mw.monthly_hours > 160, 'overtime', 'normal') AS work_type "
            "  FROM monthly_work mw "
            "  INNER JOIN employee e ON mw.emp_id = e.emp_id "
            "  INNER JOIN department d ON e.dept_id = d.dept_id"
            ") "
            "SELECT dept_name, work_month, "
            "       GROUP_CONCAT(first_name ORDER BY month_rank SEPARATOR ', ') AS top_workers, "
            "       SUM(monthly_hours) AS dept_total_hours, "
            "       SUM(IF(work_type = 'overtime', 1, 0)) AS overtime_count "
            "FROM ranked WHERE month_rank <= 3 "
            "GROUP BY dept_name, work_month "
            "ORDER BY work_month DESC, dept_total_hours DESC"
        ),
        "dialect_features_used": ["D4.2", "D4.4", "D16.8", "D3.8", "D7.3", "D4.5"],
        "builtin_functions_used": ["DATE_FORMAT", "DATE_SUB", "CURDATE", "SUM", "RANK", "IF", "GROUP_CONCAT"],
        "translation_traps": [
            "DATE_FORMAT('%W, %M %d, %Y') needs mapping 6 format codes",
            "CTE inner DATE_FORMAT+INTERVAL easily missed",
            "INTERVAL 12 MONTH → PG needs quoting: INTERVAL '12 months'",
            "GROUP_CONCAT(...SEPARATOR...) → STRING_AGGrewrite",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "mysql_extra_3",
        "comment": "⚠️Division by zero trap: recursive CTE + LAST_DAY + <=> NULL-safe + CAST difference + window GROUP_CONCAT",
        "sql": (
            "WITH RECURSIVE org_tree AS ("
            "  SELECT emp_id, first_name, last_name, manager_id, salary, dept_id, 0 AS depth "
            "  FROM employee WHERE manager_id IS NULL "
            "  UNION ALL "
            "  SELECT e.emp_id, e.first_name, e.last_name, e.manager_id, e.salary, e.dept_id, ot.depth + 1 "
            "  FROM employee e INNER JOIN org_tree ot ON e.manager_id = ot.emp_id"
            "), "
            "budget_analysis AS ("
            "  SELECT d.dept_name, d.budget, "
            "         LAST_DAY(d.created_at) AS budget_month_end, "
            "         COUNT(ot.emp_id) AS headcount, "
            "         d.budget / COUNT(ot.emp_id) AS budget_per_head "
            "  FROM department d "
            "  LEFT JOIN org_tree ot ON ot.dept_id = d.dept_id "
            "  GROUP BY d.dept_id, d.dept_name, d.budget, d.created_at"
            ") "
            "SELECT dept_name, headcount, "
            "       CAST(budget AS CHAR) AS budget_str, "
            "       CAST(IFNULL(budget_per_head, 0) AS SIGNED) AS per_head_int, "
            "       budget_month_end "
            "FROM budget_analysis "
            "WHERE headcount <=> headcount "
            "ORDER BY budget DESC"
        ),
        "dialect_features_used": ["D16.6", "D4.10", "D7.5", "D6.2", "D7.2"],
        "builtin_functions_used": ["LAST_DAY", "COUNT", "CAST", "IFNULL", "<=>(NULL_SAFE_EQUAL)"],
        "translation_traps": [
            "budget/COUNT(emp_id): when COUNT=0 MySQL→NULL, PG→division by zero ERROR",
            "LAST_DAY(): PG has no this function, need (DATE_TRUNC('month',d)+INTERVAL '1 month'-INTERVAL '1 day')::DATE",
            "CAST AS CHAR→PG: ::TEXT; CAST AS SIGNED→PG: ::INTEGER",
            "<=> → IS NOT DISTINCT FROM",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "mysql_extra_4",
        "comment": "⚠️LENGTHsemantic: bytevschar + REGEXP_REPLACEdefaultbehavior + FIND_IN_SET + HEX",
        "sql": (
            "WITH cleaned AS ("
            "  SELECT e.emp_id, e.first_name, e.last_name, e.dept_id, e.hire_date, "
            "         LENGTH(e.first_name) AS name_byte_len, "
            "         CHAR_LENGTH(e.first_name) AS name_char_len, "
            "         REGEXP_REPLACE(e.email, '[^a-zA-Z0-9@.]', '') AS clean_email, "
            "         HEX(LEFT(e.last_name, 4)) AS name_hex "
            "  FROM employee e"
            "), "
            "with_tags AS ("
            "  SELECT c.*, d.dept_name, "
            "         CONCAT(c.first_name, ' ', c.last_name) AS full_name, "
            "         DATE_FORMAT(c.hire_date, '%Y-%m-01') AS hire_month "
            "  FROM cleaned c "
            "  INNER JOIN department d ON c.dept_id = d.dept_id "
            "  WHERE FIND_IN_SET(d.dept_name, 'Engineering,Sales,Marketing') > 0"
            ") "
            "SELECT dept_name, full_name, name_byte_len, name_char_len, clean_email, "
            "       ROW_NUMBER() OVER (PARTITION BY hire_month ORDER BY name_byte_len DESC) AS byte_rank "
            "FROM with_tags ORDER BY hire_month, byte_rank"
        ),
        "dialect_features_used": ["D3.7", "D13.2", "D14.3", "D3.1", "D4.2", "D3.10"],
        "builtin_functions_used": [
            "LENGTH", "CHAR_LENGTH", "REGEXP_REPLACE", "HEX", "LEFT",
            "CONCAT", "DATE_FORMAT", "FIND_IN_SET", "ROW_NUMBER",
        ],
        "translation_traps": [
            "LENGTH(str): MySQL=byte count, PG=char count; need OCTET_LENGTH to preserve byte semantics",
            "REGEXP_REPLACE: MySQL default global, PG default first-only (need 'g' flag)",
            "FIND_IN_SET → PG: dept_name = ANY(STRING_TO_ARRAY(...))",
            "HEX(str) → PG: UPPER(ENCODE(str::bytea, 'hex'))",
        ],
    },
]


# =============================================================================
# PostgreSQL Few-shot Examples
# =============================================================================
# Corresponds to MySQL examples, uses PG native syntax. Sorted by increasing translation difficulty。
# Easy: 0-1 difference points, simple rename
# Medium: 1-2 difference points, syntax rewrite (FILTER, TO_CHAR, STRING_AGG)
# Hard: 2-3 difference points, PG→MySQL/Oracle structural rewrite (DISTINCT ON, AGE, BOOL_AND)
# Extra: 4+ difference points, semantic trap + generate_series + ARRAY_AGG + multiple feature stacking
# =============================================================================

FEWSHOT_EXAMPLES_PG = [

    # ==================================================================
    # EASY: 0-1 dialect difference points, simple function name replacement
    # ==================================================================

    {
        "difficulty": "easy",
        "query_id": "pg_easy_1",
        "comment": "Count employees hired in 2024, treat NULL salary as 0 (COALESCE)",
        "sql": (
            "SELECT COUNT(*) AS cnt, AVG(COALESCE(salary, 0)) AS avg_salary "
            "FROM employee "
            "WHERE hire_date >= '2024-01-01' AND hire_date < '2025-01-01'"
        ),
        "dialect_features_used": ["D7.2"],
        "builtin_functions_used": ["COALESCE", "COUNT", "AVG"],
    },

    {
        "difficulty": "easy",
        "query_id": "pg_easy_2",
        "comment": "|| string concatenation + ILIKE case-insensitive",
        "sql": (
            "SELECT first_name || ' ' || last_name AS full_name, email "
            "FROM employee "
            "WHERE email ILIKE '%@gmail.com'"
        ),
        "dialect_features_used": ["D3.1", "D16.3"],
        "builtin_functions_used": [],
    },

    {
        "difficulty": "easy",
        "query_id": "pg_easy_3",
        "comment": "LIMIT/OFFSET standard pagination form",
        "sql": (
            "SELECT emp_id, first_name, last_name, salary "
            "FROM employee "
            "ORDER BY salary DESC "
            "LIMIT 10 OFFSET 10"
        ),
        "dialect_features_used": ["D2.1"],
        "builtin_functions_used": [],
    },

    # ==================================================================
    # MEDIUM: 1-2 dialect difference points, syntax rewrite
    # ==================================================================

    {
        "difficulty": "medium",
        "query_id": "pg_medium_1",
        "comment": "TO_CHAR format specifiermapping (→MySQL: DATE_FORMAT + %Y-%m)",
        "sql": (
            "SELECT TO_CHAR(t.work_date, 'YYYY-MM') AS month, "
            "       e.first_name, SUM(t.hours_worked) AS total_hours "
            "FROM timesheet t "
            "INNER JOIN employee e ON t.emp_id = e.emp_id "
            "WHERE t.proj_id IN (SELECT proj_id FROM project WHERE status = 'active') "
            "GROUP BY month, e.emp_id, e.first_name "
            "ORDER BY month, total_hours DESC"
        ),
        "dialect_features_used": ["D4.2"],
        "builtin_functions_used": ["TO_CHAR", "SUM"],
    },

    {
        "difficulty": "medium",
        "query_id": "pg_medium_2",
        "comment": "CASE WHEN + STRING_AGG (→MySQL: IF + GROUP_CONCAT)",
        "sql": (
            "SELECT d.dept_name, "
            "       SUM(CASE WHEN e.salary >= 80000 THEN 1 ELSE 0 END) AS high_salary_count, "
            "       SUM(CASE WHEN e.salary < 80000 THEN 1 ELSE 0 END) AS low_salary_count, "
            "       STRING_AGG(e.last_name, ', ' ORDER BY e.last_name) AS all_names "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_id, d.dept_name"
        ),
        "dialect_features_used": ["D7.3", "D3.8"],
        "builtin_functions_used": ["CASE", "SUM", "STRING_AGG"],
    },

    {
        "difficulty": "medium",
        "query_id": "pg_medium_3",
        "comment": "INTERVAL with quote + date subtraction (→MySQL: DATEDIFF + INTERVAL unquoted)",
        "sql": (
            "SELECT p.proj_name, p.end_date, "
            "       (CURRENT_DATE - p.end_date) AS overdue_days "
            "FROM project p "
            "WHERE p.status = 'active' "
            "  AND p.end_date < CURRENT_DATE - INTERVAL '30 days' "
            "ORDER BY overdue_days DESC"
        ),
        "dialect_features_used": ["D4.4", "D4.5", "D16.8"],
        "builtin_functions_used": ["CURRENT_DATE"],
    },

    {
        "difficulty": "medium",
        "query_id": "pg_medium_4",
        "comment": "FILTER clause (PG-only, →MySQL: SUM(CASE WHEN...))",
        "sql": (
            "SELECT d.dept_name, "
            "       SUM(t.hours_worked) FILTER (WHERE p.status = 'active') AS active_hours, "
            "       SUM(t.hours_worked) FILTER (WHERE p.status = 'completed') AS completed_hours "
            "FROM timesheet t "
            "INNER JOIN employee e ON t.emp_id = e.emp_id "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "INNER JOIN project p ON t.proj_id = p.proj_id "
            "GROUP BY d.dept_id, d.dept_name "
            "ORDER BY d.dept_name"
        ),
        "dialect_features_used": ["D8.3"],
        "builtin_functions_used": ["SUM", "FILTER"],
    },

    # ==================================================================
    # HARD: 2-3 dialect difference points, PG->MySQL directional structural rewrite
    # ==================================================================

    {
        "difficulty": "hard",
        "query_id": "pg_hard_1",
        "comment": "DISTINCT ON (PG-only→MySQL needs ROW_NUMBER rewrite) + ::cast + ||concatenation",
        "sql": (
            "SELECT DISTINCT ON (d.dept_name) "
            "       d.dept_name, "
            "       e.first_name || ' ' || e.last_name AS full_name, "
            "       e.salary::NUMERIC(10,2) AS salary "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "INNER JOIN assignment a ON e.emp_id = a.emp_id "
            "ORDER BY d.dept_name, e.salary DESC"
        ),
        "dialect_features_used": ["D1.4", "D6.4", "D3.1"],
        "builtin_functions_used": [],
    },

    {
        "difficulty": "hard",
        "query_id": "pg_hard_2",
        "comment": "FILTER clause×2 + INTERVAL + ::cast — multiple difference stacking",
        "sql": (
            "SELECT d.dept_name, "
            "       COUNT(*) FILTER (WHERE t.work_date >= CURRENT_DATE - INTERVAL '90 days') "
            "         AS recent_entries, "
            "       SUM(t.hours_worked) FILTER (WHERE t.work_date >= CURRENT_DATE - INTERVAL '90 days') "
            "         AS recent_hours, "
            "       SUM(t.hours_worked) AS total_hours, "
            "       (CURRENT_DATE - MIN(e.hire_date))::INTEGER AS max_tenure_days "
            "FROM timesheet t "
            "INNER JOIN employee e ON t.emp_id = e.emp_id "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_id, d.dept_name "
            "ORDER BY recent_hours DESC NULLS LAST"
        ),
        "dialect_features_used": ["D8.3", "D4.4", "D4.5", "D6.4", "D16.8"],
        "builtin_functions_used": ["COUNT", "SUM", "FILTER", "MIN", "CURRENT_DATE"],
    },

    {
        "difficulty": "hard",
        "query_id": "pg_hard_3",
        "comment": "AGE()structurerewrite + DATE_TRUNC + JSONBextract — three different rewrite modes",
        "sql": (
            "SELECT d.dept_name, "
            "       DATE_TRUNC('quarter', e.hire_date)::DATE AS hire_quarter, "
            "       e.first_name, "
            "       EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.hire_date)) AS tenure_years, "
            "       e.metadata->>'primary_skill' AS skill "
            "FROM employee e "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "INNER JOIN assignment a ON e.emp_id = a.emp_id "
            "WHERE e.metadata->>'primary_skill' IS NOT NULL "
            "ORDER BY tenure_years DESC, d.dept_name"
        ),
        "dialect_features_used": ["D4.9", "D4.7", "D12.2", "D6.4"],
        "builtin_functions_used": ["AGE", "DATE_TRUNC", "EXTRACT", "JSONB_EXTRACT(->>, ->)"],
    },

    {
        "difficulty": "hard",
        "query_id": "pg_hard_4",
        "comment": "BOOL_ANDstructurerewrite + GROUP BY ROLLUP() + COALESCE",
        "sql": (
            "SELECT COALESCE(d.dept_name, '[Total]') AS dept_name, "
            "       COALESCE(p.status, '[Subtotal]') AS status, "
            "       COUNT(*) AS assignment_count, "
            "       SUM(a.hours_allocated) AS total_hours, "
            "       BOOL_AND(e.is_active) AS all_active "
            "FROM assignment a "
            "INNER JOIN employee e ON a.emp_id = e.emp_id "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "INNER JOIN project p ON a.proj_id = p.proj_id "
            "GROUP BY ROLLUP(d.dept_name, p.status)"
        ),
        "dialect_features_used": ["D8.4", "D8.2", "D7.2"],
        "builtin_functions_used": ["COALESCE", "COUNT", "SUM", "BOOL_AND"],
    },

    # ==================================================================
    # EXTRA: 4+ dialect difference points
    # semantic trap + PG-only structural rewrite stacking + format specifier full mapping
    # ==================================================================

    {
        "difficulty": "extra",
        "query_id": "pg_extra_1",
        "comment": "⚠️semantic trap: LOG same-name different-semantics (PG=log10) + integer division + GREATEST(NULL) ignore + TRUNC naming",
        "sql": (
            "WITH salary_analysis AS ("
            "  SELECT e.emp_id, d.dept_name, e.salary, "
            "         LOG(e.salary) AS log_salary, "
            "         TRUNC(e.salary / COALESCE(a.hours_allocated, 1), 2) AS hourly_rate, "
            "         GREATEST(e.salary, ("
            "           SELECT AVG(e2.salary) FROM employee e2 WHERE e2.dept_id = e.dept_id"
            "         ), NULL) AS effective_salary "
            "  FROM employee e "
            "  INNER JOIN department d ON e.dept_id = d.dept_id "
            "  LEFT JOIN assignment a ON e.emp_id = a.emp_id"
            ") "
            "SELECT dept_name, COUNT(*) AS emp_count, "
            "       TRUNC(AVG(log_salary), 4) AS avg_log_salary, "
            "       TRUNC(AVG(hourly_rate), 2) AS avg_hourly_rate, "
            "       SUM(CASE WHEN effective_salary IS NULL THEN 1 ELSE 0 END) AS null_salary_count "
            "FROM salary_analysis "
            "GROUP BY dept_name ORDER BY avg_log_salary DESC"
        ),
        "dialect_features_used": ["D5.4", "D5.2", "D7.4", "D7.2", "D5.1"],
        "builtin_functions_used": ["LOG", "TRUNC", "COALESCE", "GREATEST", "AVG", "COUNT", "SUM", "CASE"],
        "translation_traps": [
            "LOG(salary): PG=log₁₀(salary), MySQL LOG=ln; translate to MySQL keep LOG (i.e. ln semantics) or use LOG10",
            "salary/hours_allocated: PG integer division truncates, MySQL does not truncate",
            "GREATEST(..., NULL): PGignoreNULL, MySQLreturnNULL",
            "TRUNC(x,n) → MySQL: TRUNCATE(x,n)",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "pg_extra_2",
        "comment": "generate_series (→MySQL needs recursive CTE) + FILTER×2 + multiple format code TO_CHAR + ::cast",
        "sql": (
            "WITH months AS ("
            "  SELECT gs::DATE AS month_start, "
            "         TO_CHAR(gs, 'Month YYYY') AS month_label "
            "  FROM generate_series("
            "    DATE_TRUNC('year', CURRENT_DATE)::DATE, "
            "    (DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '11 months')::DATE, "
            "    INTERVAL '1 month'"
            "  ) AS gs"
            "), "
            "monthly_stats AS ("
            "  SELECT DATE_TRUNC('month', t.work_date)::DATE AS work_month, "
            "         d.dept_name, "
            "         SUM(t.hours_worked) AS total_hours, "
            "         COUNT(DISTINCT t.emp_id) FILTER (WHERE t.hours_worked > 4) AS productive_emps, "
            "         SUM(t.hours_worked) FILTER (WHERE p.status = 'active') AS active_hours "
            "  FROM timesheet t "
            "  INNER JOIN employee e ON t.emp_id = e.emp_id "
            "  INNER JOIN department d ON e.dept_id = d.dept_id "
            "  INNER JOIN project p ON t.proj_id = p.proj_id "
            "  GROUP BY work_month, d.dept_name"
            ") "
            "SELECT m.month_label, ms.dept_name, "
            "       COALESCE(ms.total_hours, 0) AS total_hours, "
            "       COALESCE(ms.productive_emps, 0)::INTEGER AS productive_emps, "
            "       COALESCE(ms.active_hours, 0) AS active_hours, "
            "       TO_CHAR(m.month_start, 'Day, DD Month YYYY') AS full_date "
            "FROM months m "
            "LEFT JOIN monthly_stats ms ON ms.work_month = m.month_start "
            "ORDER BY m.month_start, ms.dept_name"
        ),
        "dialect_features_used": ["D16.9", "D8.3", "D4.7", "D4.2", "D6.4", "D7.2"],
        "builtin_functions_used": [
            "generate_series", "DATE_TRUNC", "TO_CHAR", "CURRENT_DATE",
            "SUM", "COUNT", "FILTER", "COALESCE",
        ],
        "translation_traps": [
            "generate_series(): MySQL has no this function, need recursive CTE to generate date sequence",
            "FILTER(WHERE...): MySQL has no this syntax, need SUM(CASE WHEN...THEN...END)",
            "TO_CHAR(gs, 'Month YYYY') → DATE_FORMAT('%M %Y')",
            "TO_CHAR(d, 'Day, DD Month YYYY') → DATE_FORMAT('%W, %d %M %Y')",
            "::DATE/::INTEGER → CAST(... AS DATE)/CAST(... AS SIGNED)",
            "DATE_TRUNC('month', d)::DATE → DATE_FORMAT(d, '%Y-%m-01')",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "pg_extra_3",
        "comment": "DISTINCT ON + AGE() + ARRAY_AGG + ~regex + FILTER — PG-only feature stacking",
        "sql": (
            "WITH active_projects AS ("
            "  SELECT DISTINCT ON (a.emp_id) "
            "         a.emp_id, p.proj_name, p.budget, a.role "
            "  FROM assignment a "
            "  INNER JOIN project p ON a.proj_id = p.proj_id "
            "  WHERE p.status = 'active' "
            "  ORDER BY a.emp_id, p.budget DESC"
            "), "
            "dept_summary AS ("
            "  SELECT d.dept_name, "
            "         ARRAY_AGG(DISTINCT e.first_name || ' ' || e.last_name "
            "                   ORDER BY e.first_name || ' ' || e.last_name) AS member_names, "
            "         EXTRACT(YEAR FROM AGE(CURRENT_DATE, MIN(e.hire_date)))::INTEGER AS max_tenure, "
            "         COUNT(*) FILTER (WHERE ap.proj_name IS NOT NULL) AS active_count "
            "  FROM employee e "
            "  INNER JOIN department d ON e.dept_id = d.dept_id "
            "  LEFT JOIN active_projects ap ON e.emp_id = ap.emp_id "
            "  WHERE e.email ~ '^[a-z]+\\.[a-z]+@' "
            "  GROUP BY d.dept_id, d.dept_name"
            ") "
            "SELECT dept_name, member_names, max_tenure, active_count "
            "FROM dept_summary WHERE max_tenure >= 2 "
            "ORDER BY active_count DESC, dept_name"
        ),
        "dialect_features_used": ["D1.4", "D4.9", "D14.1", "D13.1", "D6.4", "D3.1", "D8.3"],
        "builtin_functions_used": [
            "DISTINCT_ON", "ARRAY_AGG", "AGE", "EXTRACT",
            "FILTER", "COUNT", "MIN", "CURRENT_DATE",
        ],
        "translation_traps": [
            "DISTINCT ON(emp_id): MySQL has no this syntax, need ROW_NUMBER() OVER(PARTITION BY...) subquery",
            "AGE(d1,d2): MySQL has no this function, need TIMESTAMPDIFF(YEAR, d2, d1)",
            "ARRAY_AGG(DISTINCT ...): MySQL has no ARRAY, need GROUP_CONCAT(DISTINCT ... SEPARATOR ',')",
            "FILTER(WHERE...): MySQL needs SUM(CASE WHEN...THEN 1 ELSE 0 END)",
            "~ regex: MySQL needs REGEXP",
            "::INTEGER: MySQL needs CAST(... AS SIGNED)",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "pg_extra_4",
        "comment": "⚠️LENGTHsemantic trap: charvsbyte + REGEXP_REPLACEdefaultbehavior + ANY(ARRAY) + ENCODE",
        "sql": (
            "WITH cleaned AS ("
            "  SELECT e.emp_id, e.first_name, e.last_name, e.dept_id, e.hire_date, "
            "         LENGTH(e.first_name) AS name_char_len, "
            "         OCTET_LENGTH(e.first_name) AS name_byte_len, "
            "         REGEXP_REPLACE(e.email, '[^a-zA-Z0-9@.]', '', 'g') AS clean_email, "
            "         UPPER(ENCODE(LEFT(e.last_name, 4)::BYTEA, 'hex')) AS name_hex "
            "  FROM employee e"
            "), "
            "with_tags AS ("
            "  SELECT c.*, d.dept_name, "
            "         c.first_name || ' ' || c.last_name AS full_name, "
            "         TO_CHAR(c.hire_date, 'YYYY-MM') || '-01' AS hire_month "
            "  FROM cleaned c "
            "  INNER JOIN department d ON c.dept_id = d.dept_id "
            "  WHERE d.dept_name = ANY(ARRAY['Engineering', 'Sales', 'Marketing'])"
            ") "
            "SELECT dept_name, full_name, name_byte_len, name_char_len, clean_email, "
            "       ROW_NUMBER() OVER (PARTITION BY hire_month ORDER BY name_byte_len DESC) AS byte_rank "
            "FROM with_tags ORDER BY hire_month, byte_rank"
        ),
        "dialect_features_used": ["D3.7", "D13.2", "D14.2", "D3.1", "D4.2", "D3.10", "D6.4"],
        "builtin_functions_used": [
            "LENGTH", "OCTET_LENGTH", "REGEXP_REPLACE", "ENCODE", "LEFT",
            "TO_CHAR", "ROW_NUMBER", "UPPER",
        ],
        "translation_traps": [
            "LENGTH(str): PG=char count, MySQL=byte count; to preserve char semantics use MySQL CHAR_LENGTH",
            "REGEXP_REPLACE(e, pat, '', 'g'): MySQL default global no need 'g'; PG without 'g' only replaces first",
            "ANY(ARRAY[...]): MySQL needs FIND_IN_SET or IN(...)",
            "ENCODE(str::BYTEA,'hex'): MySQL needs HEX(str)",
            "TO_CHAR inside CTE easily missed when translating",
        ],
    },
]


# =============================================================================
# Oracle Few-shot Examples
# =============================================================================
# Corresponds to MySQL/PG examples, uses Oracle native syntax. Sorted by increasing translation difficulty。
# Easy: 0-1 difference points, simple rename (NVL→COALESCE/IFNULL, ||→CONCAT)
# Medium: 1-2 difference points, syntax rewrite (TO_CHAR, LISTAGG, DECODE)
# Hard: 2-3 difference points, Oracle→MySQL/PG structural rewrite (ROWNUM, DECODE, (+) outer join, CONNECT BY)
# Extra: 4+ difference points, semantic trap + CONNECT BY + LISTAGG + multiple feature stacking
# =============================================================================

FEWSHOT_EXAMPLES_ORACLE = [

    # ==================================================================
    # EASY: 0-1 dialect difference points, simple function name replacement
    # ==================================================================

    {
        "difficulty": "easy",
        "query_id": "oracle_easy_1",
        "comment": "Count employees hired in 2024, treat NULL salary as 0 (NVL)",
        "sql": (
            "SELECT COUNT(*) AS cnt, AVG(NVL(salary, 0)) AS avg_salary "
            "FROM employee "
            "WHERE hire_date >= DATE '2024-01-01' AND hire_date < DATE '2025-01-01'"
        ),
        "dialect_features_used": ["D7.2"],
        "builtin_functions_used": ["NVL", "COUNT", "AVG"],
    },

    {
        "difficulty": "easy",
        "query_id": "oracle_easy_2",
        "comment": "|| string concatenation + get top 10 by salary descending (ROWNUM pagination)",
        "sql": (
            "SELECT full_name, email "
            "FROM ("
            "  SELECT first_name || ' ' || last_name AS full_name, email "
            "  FROM employee "
            "  WHERE email LIKE '%@gmail.com' "
            "  ORDER BY salary DESC"
            ") "
            "WHERE ROWNUM <= 10"
        ),
        "dialect_features_used": ["D3.1", "D2.1"],
        "builtin_functions_used": ["ROWNUM"],
    },

    {
        "difficulty": "easy",
        "query_id": "oracle_easy_3",
        "comment": "Pagination query: get employees ranked 11-20 (ROW_NUMBER pagination)",
        "sql": (
            "SELECT emp_id, first_name, last_name, salary "
            "FROM ("
            "  SELECT emp_id, first_name, last_name, salary, "
            "         ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "  FROM employee"
            ") "
            "WHERE rn BETWEEN 11 AND 20 "
            "ORDER BY rn"
        ),
        "dialect_features_used": ["D2.1"],
        "builtin_functions_used": ["ROW_NUMBER"],
    },

    # ==================================================================
    # MEDIUM: 1-2 dialect difference points, syntax rewrite
    # ==================================================================

    {
        "difficulty": "medium",
        "query_id": "oracle_medium_1",
        "comment": "TO_CHAR format specifiermapping (→MySQL: DATE_FORMAT + %Y-%m)",
        "sql": (
            "SELECT TO_CHAR(t.work_date, 'YYYY-MM') AS month, "
            "       e.first_name, SUM(t.hours_worked) AS total_hours "
            "FROM timesheet t "
            "INNER JOIN employee e ON t.emp_id = e.emp_id "
            "WHERE t.proj_id IN (SELECT proj_id FROM project WHERE status = 'active') "
            "GROUP BY TO_CHAR(t.work_date, 'YYYY-MM'), e.emp_id, e.first_name "
            "ORDER BY month, total_hours DESC"
        ),
        "dialect_features_used": ["D4.2"],
        "builtin_functions_used": ["TO_CHAR", "SUM"],
    },

    {
        "difficulty": "medium",
        "query_id": "oracle_medium_2",
        "comment": "CASE WHEN + LISTAGG (→MySQL: IF + GROUP_CONCAT, →PG: CASE + STRING_AGG)",
        "sql": (
            "SELECT d.dept_name, "
            "       SUM(CASE WHEN e.salary >= 80000 THEN 1 ELSE 0 END) AS high_salary_count, "
            "       SUM(CASE WHEN e.salary < 80000 THEN 1 ELSE 0 END) AS low_salary_count, "
            "       LISTAGG(e.last_name, ', ') WITHIN GROUP (ORDER BY e.last_name) AS all_names "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_id, d.dept_name"
        ),
        "dialect_features_used": ["D7.3", "D3.8"],
        "builtin_functions_used": ["CASE", "SUM", "LISTAGG"],
    },

    {
        "difficulty": "medium",
        "query_id": "oracle_medium_3",
        "comment": "INTERVAL syntax + date subtraction (→MySQL: DATEDIFF + INTERVAL unquoted)",
        "sql": (
            "SELECT p.proj_name, p.end_date, "
            "       TRUNC(SYSDATE - p.end_date) AS overdue_days "
            "FROM project p "
            "WHERE p.status = 'active' "
            "  AND p.end_date < SYSDATE - INTERVAL '30' DAY "
            "ORDER BY overdue_days DESC"
        ),
        "dialect_features_used": ["D4.4", "D4.5", "D16.8"],
        "builtin_functions_used": ["SYSDATE", "TRUNC"],
    },

    {
        "difficulty": "medium",
        "query_id": "oracle_medium_4",
        "comment": "EXTRACT extractdatepartial + NVL (→MySQL: YEAR/MONTH + IFNULL, →PG: EXTRACT + COALESCE)",
        "sql": (
            "SELECT d.dept_name, "
            "       EXTRACT(YEAR FROM e.hire_date) AS hire_year, "
            "       EXTRACT(MONTH FROM e.hire_date) AS hire_month, "
            "       COUNT(*) AS new_hires "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_name, EXTRACT(YEAR FROM e.hire_date), EXTRACT(MONTH FROM e.hire_date) "
            "HAVING COUNT(*) >= 2 "
            "ORDER BY hire_year DESC, hire_month DESC"
        ),
        "dialect_features_used": ["D4.6"],
        "builtin_functions_used": ["EXTRACT", "COUNT"],
    },

    # ==================================================================
    # HARD: 2-3 dialect difference points, Oracle->MySQL/PG structural rewrite
    # ==================================================================

    {
        "difficulty": "hard",
        "query_id": "oracle_hard_1",
        "comment": "Window function inner difference: double quote+date subtraction+INTERVAL triple difference stacking",
        "sql": (
            "SELECT emp_id, work_date, hours_worked, "
            "       TRUNC(work_date - LAG(work_date) OVER ("
            "         PARTITION BY emp_id ORDER BY work_date"
            "       )) AS \"gap_days\" "
            "FROM timesheet "
            "WHERE work_date >= SYSDATE - INTERVAL '90' DAY"
        ),
        "dialect_features_used": ["D1.1", "D4.5", "D4.4", "D16.8"],
        "builtin_functions_used": ["LAG", "TRUNC", "SYSDATE"],
    },

    {
        "difficulty": "hard",
        "query_id": "oracle_hard_2",
        "comment": "Multiple format code TO_CHAR + DECODE + CAST — three different rewrite modes",
        "sql": (
            "SELECT d.dept_name, "
            "       TO_CHAR(e.hire_date, 'Day, Month DD, YYYY') AS hire_date_fmt, "
            "       TRUNC(AVG(SYSDATE - e.hire_date)) AS avg_tenure, "
            "       DECODE(SIGN(AVG(SYSDATE - e.hire_date) - 1000), 1, 'senior', 'junior') AS team_type "
            "FROM department d "
            "INNER JOIN employee e ON d.dept_id = e.dept_id "
            "GROUP BY d.dept_name, e.hire_date "
            "ORDER BY avg_tenure DESC"
        ),
        "dialect_features_used": ["D4.2", "D7.3", "D6.2", "D4.5"],
        "builtin_functions_used": ["TO_CHAR", "TRUNC", "AVG", "DECODE", "SYSDATE"],
    },

    {
        "difficulty": "hard",
        "query_id": "oracle_hard_3",
        "comment": " JSON text extraction + REGEXP_LIKE + DECODE — Oracle 11g compatible syntax",
        "sql": (
            "SELECT d.dept_name, "
            "       REGEXP_SUBSTR(e.metadata, '\"primary_skill\"[[:space:]]*:[[:space:]]*\"([^\"]+)\"', 1, 1, NULL, 1) AS skill, "
            "       COUNT(*) AS cnt, "
            "       DECODE(SIGN(COUNT(*) - 3), -1, 'rare', 'core') AS skill_type "
            "FROM employee e "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "WHERE REGEXP_SUBSTR(e.metadata, '\"primary_skill\"[[:space:]]*:[[:space:]]*\"([^\"]+)\"', 1, 1, NULL, 1) IS NOT NULL "
            "  AND REGEXP_LIKE(REGEXP_SUBSTR(e.metadata, '\"primary_skill\"[[:space:]]*:[[:space:]]*\"([^\"]+)\"', 1, 1, NULL, 1), '^(Python|Java|Go)') "
            "GROUP BY d.dept_name, REGEXP_SUBSTR(e.metadata, '\"primary_skill\"[[:space:]]*:[[:space:]]*\"([^\"]+)\"', 1, 1, NULL, 1) "
            "HAVING COUNT(*) >= 1 "
            "ORDER BY d.dept_name, cnt DESC"
        ),
        "dialect_features_used": ["D13.1", "D7.3"],
        "builtin_functions_used": ["REGEXP_SUBSTR", "REGEXP_LIKE", "DECODE", "COUNT"],
    },

    {
        "difficulty": "hard",
        "query_id": "oracle_hard_4",
        "comment": "GROUP BY ROLLUP() + NVL + 4-table JOIN",
        "sql": (
            "SELECT NVL(d.dept_name, '[Total]') AS dept_name, "
            "       NVL(p.status, '[Subtotal]') AS status, "
            "       COUNT(*) AS assignment_count, "
            "       SUM(a.hours_allocated) AS total_hours "
            "FROM assignment a "
            "INNER JOIN employee e ON a.emp_id = e.emp_id "
            "INNER JOIN department d ON e.dept_id = d.dept_id "
            "INNER JOIN project p ON a.proj_id = p.proj_id "
            "GROUP BY ROLLUP(d.dept_name, p.status)"
        ),
        "dialect_features_used": ["D8.4", "D7.2", "D3.1"],
        "builtin_functions_used": ["NVL", "COUNT", "SUM"],
    },

    # ==================================================================
    # EXTRA: 4+ dialect difference points
    # semantic trap + Oracle-only structural rewrite stacking + format specifier full mapping
    # ==================================================================

    {
        "difficulty": "extra",
        "query_id": "oracle_extra_1",
        "comment": "⚠️semantic trap: LOG same-name different-semantics (Oracle=ln) + integer division + GREATEST(NULL) ignore + TRUNC naming",
        "sql": (
            "WITH salary_analysis AS ("
            "  SELECT e.emp_id, d.dept_name, e.salary, "
            "         LOG(e.salary) AS log_salary, "
            "         TRUNC(e.salary / NVL(a.hours_allocated, 1), 2) AS hourly_rate, "
            "         GREATEST(e.salary, ("
            "           SELECT AVG(e2.salary) FROM employee e2 WHERE e2.dept_id = e.dept_id"
            "         ), NULL) AS effective_salary "
            "  FROM employee e "
            "  INNER JOIN department d ON e.dept_id = d.dept_id "
            "  LEFT JOIN assignment a ON e.emp_id = a.emp_id"
            ") "
            "SELECT dept_name, COUNT(*) AS emp_count, "
            "       TRUNC(AVG(log_salary), 4) AS avg_log_salary, "
            "       TRUNC(AVG(hourly_rate), 2) AS avg_hourly_rate, "
            "       SUM(CASE WHEN effective_salary IS NULL THEN 1 ELSE 0 END) AS null_salary_count "
            "FROM salary_analysis "
            "GROUP BY dept_name ORDER BY avg_log_salary DESC"
        ),
        "dialect_features_used": ["D5.4", "D5.2", "D7.4", "D7.2", "D5.1"],
        "builtin_functions_used": ["LOG", "TRUNC", "NVL", "GREATEST", "AVG", "COUNT", "SUM", "CASE"],
        "translation_traps": [
            "LOG(salary): Oracle=ln(salary), PG LOG=log10(x), translate to PG must use LN(salary)",
            "salary/hours_allocated: PG integer division truncates, Oracle and MySQL do not",
            "GREATEST(..., NULL): Oracle ignores NULL (same as PG), MySQL returns NULL",
            "TRUNC(x,n) → MySQL: TRUNCATE(x,n); PG: TRUNC(x,n)",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "oracle_extra_2",
        "comment": "CONNECT BY recursion + multiple format code TO_CHAR + LISTAGG + INTERVAL multiple difference stacking",
        "sql": (
            "WITH monthly_work AS ("
            "  SELECT t.emp_id, "
            "         TO_CHAR(t.work_date, 'YYYY-MM') AS work_month, "
            "         TO_CHAR(t.work_date, 'Day, Month DD, YYYY') AS formatted_date, "
            "         SUM(t.hours_worked) AS monthly_hours "
            "  FROM timesheet t "
            "  WHERE t.work_date >= ADD_MONTHS(TRUNC(SYSDATE, 'MM'), -12) "
            "  GROUP BY t.emp_id, TO_CHAR(t.work_date, 'YYYY-MM'), TO_CHAR(t.work_date, 'Day, Month DD, YYYY')"
            "), "
            "ranked AS ("
            "  SELECT mw.*, e.first_name, d.dept_name, "
            "         RANK() OVER (PARTITION BY mw.work_month ORDER BY mw.monthly_hours DESC) AS month_rank, "
            "         CASE WHEN mw.monthly_hours > 160 THEN 'overtime' ELSE 'normal' END AS work_type "
            "  FROM monthly_work mw "
            "  INNER JOIN employee e ON mw.emp_id = e.emp_id "
            "  INNER JOIN department d ON e.dept_id = d.dept_id"
            ") "
            "SELECT dept_name, work_month, "
            "       LISTAGG(first_name, ', ') WITHIN GROUP (ORDER BY month_rank) AS top_workers, "
            "       SUM(monthly_hours) AS dept_total_hours, "
            "       SUM(CASE WHEN work_type = 'overtime' THEN 1 ELSE 0 END) AS overtime_count "
            "FROM ranked WHERE month_rank <= 3 "
            "GROUP BY dept_name, work_month "
            "ORDER BY work_month DESC, dept_total_hours DESC"
        ),
        "dialect_features_used": ["D4.2", "D4.4", "D16.8", "D3.8", "D7.3", "D4.5"],
        "builtin_functions_used": ["TO_CHAR", "ADD_MONTHS", "TRUNC", "SYSDATE", "SUM", "RANK", "LISTAGG"],
        "translation_traps": [
            "TO_CHAR('Day, Month DD, YYYY') needs mapping multiple format codes, Oracle and PG format codes also differ",
            "ADD_MONTHS(): MySQLnothisfunction→DATE_ADD, PGnothisfunction→INTERVAL",
            "LISTAGG(...WITHIN GROUP...) → MySQL: GROUP_CONCAT(...ORDER BY...SEPARATOR...), PG: STRING_AGG(...ORDER BY...)",
            "TRUNC(SYSDATE, 'MM') → MySQL: DATE_FORMAT(CURDATE(), '%Y-%m-01'), PG: DATE_TRUNC('month', CURRENT_DATE)::DATE",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "oracle_extra_3",
        "comment": "⚠️Division by zero trap: CONNECT BY recursion + LAST_DAY + NVL + CAST difference + ROWNUM",
        "sql": (
            "SELECT dept_name, headcount, "
            "       TO_CHAR(budget, 'FM999,999,999.00') AS budget_str, "
            "       CAST(NVL(budget_per_head, 0) AS INTEGER) AS per_head_int, "
            "       budget_month_end "
            "FROM ("
            "  SELECT d.dept_name, d.budget, "
            "         LAST_DAY(d.created_at) AS budget_month_end, "
            "         COUNT(ot.emp_id) AS headcount, "
            "         d.budget / COUNT(ot.emp_id) AS budget_per_head "
            "  FROM department d "
            "  LEFT JOIN ("
            "    SELECT emp_id, first_name, last_name, manager_id, salary, dept_id, LEVEL AS depth "
            "    FROM employee "
            "    START WITH manager_id IS NULL "
            "    CONNECT BY PRIOR emp_id = manager_id"
            "  ) ot ON ot.dept_id = d.dept_id "
            "  GROUP BY d.dept_id, d.dept_name, d.budget, d.created_at"
            ") "
            "WHERE ROWNUM <= 100 "
            "ORDER BY budget DESC"
        ),
        "dialect_features_used": ["D16.6", "D4.10", "D7.5", "D6.2", "D7.2", "D2.1"],
        "builtin_functions_used": ["LAST_DAY", "COUNT", "CAST", "NVL", "TO_CHAR", "ROWNUM", "CONNECT_BY"],
        "translation_traps": [
            "budget/COUNT(emp_id): when COUNT=0 Oracle→ERROR (same as PG), MySQL→NULL",
            "LAST_DAY(): PG has no this function, need composite expression; MySQL has this function",
            "CAST AS INTEGER→MySQL: CAST AS SIGNED; PG: ::INTEGER",
            "CONNECT BY → MySQL/PG: recursive CTE",
            "ROWNUM <= N → MySQL: LIMIT N; PG: FETCH FIRST N ROWS ONLY",
            "TO_CHAR(num, 'FM999,999,999.00') → MySQL: FORMAT(num, 2); PG: TO_CHAR(num, 'FM999,999,999.00')(format codes same)",
        ],
    },

    {
        "difficulty": "extra",
        "query_id": "oracle_extra_4",
        "comment": "⚠️LENGTHsemantic: charvsbyte + REGEXP_REPLACEdefaultbehavior + RAWTOHEX + ROW_NUMBER",
        "sql": (
            "WITH cleaned AS ("
            "  SELECT e.emp_id, e.first_name, e.last_name, e.dept_id, e.hire_date, "
            "         LENGTH(e.first_name) AS name_char_len, "
            "         LENGTHB(e.first_name) AS name_byte_len, "
            "         REGEXP_REPLACE(e.email, '[^a-zA-Z0-9@.]', '') AS clean_email, "
            "         RAWTOHEX(UTL_RAW.CAST_TO_RAW(SUBSTR(e.last_name, 1, 4))) AS name_hex "
            "  FROM employee e"
            "), "
            "with_tags AS ("
            "  SELECT c.*, d.dept_name, "
            "         c.first_name || ' ' || c.last_name AS full_name, "
            "         TO_CHAR(c.hire_date, 'YYYY-MM') || '-01' AS hire_month "
            "  FROM cleaned c "
            "  INNER JOIN department d ON c.dept_id = d.dept_id "
            "  WHERE d.dept_name IN ('Engineering', 'Sales', 'Marketing')"
            ") "
            "SELECT dept_name, full_name, name_byte_len, name_char_len, clean_email, "
            "       ROW_NUMBER() OVER (PARTITION BY hire_month ORDER BY name_byte_len DESC) AS byte_rank "
            "FROM with_tags ORDER BY hire_month, byte_rank"
        ),
        "dialect_features_used": ["D3.7", "D13.2", "D14.3", "D3.1", "D4.2", "D3.10"],
        "builtin_functions_used": [
            "LENGTH", "LENGTHB", "REGEXP_REPLACE", "RAWTOHEX", "UTL_RAW",
            "TO_CHAR", "ROW_NUMBER",
        ],
        "translation_traps": [
            "LENGTH(str): Oracle=char count (same as PG), MySQL=byte count; byte semantics need LENGTHB/MySQL LENGTH",
            "REGEXP_REPLACE: Oracle default first-only (same as PG), MySQL default global",
            "RAWTOHEX(str) → MySQL: HEX(str); PG: UPPER(ENCODE(str::bytea, 'hex'))",
            "LENGTHB() → MySQL: LENGTH(); PG: OCTET_LENGTH()",
        ],
    },
]


# =============================================================================
# Index grouped by dialect and difficulty for easy level-based retrieval
# =============================================================================

_ALL_EXAMPLES = {
    "mysql": FEWSHOT_EXAMPLES_MYSQL,
    "pg": FEWSHOT_EXAMPLES_PG,
    "oracle": FEWSHOT_EXAMPLES_ORACLE,
}

EXAMPLES_BY_DIFFICULTY = {}          # Legacy interface kept — default uses MySQL
for ex in FEWSHOT_EXAMPLES_MYSQL:
    d = ex["difficulty"]
    EXAMPLES_BY_DIFFICULTY.setdefault(d, []).append(ex)

EXAMPLES_BY_DIALECT_DIFFICULTY: dict[str, dict[str, list]] = {}
for dialect, examples in _ALL_EXAMPLES.items():
    EXAMPLES_BY_DIALECT_DIFFICULTY[dialect] = {}
    for ex in examples:
        d = ex["difficulty"]
        EXAMPLES_BY_DIALECT_DIFFICULTY[dialect].setdefault(d, []).append(ex)


def get_examples_for_difficulty(
    difficulty: str,
    n: int = 2,
    dialect: str = "mysql",
) -> list:
    """
    getspecifydifficultyanddialect few-shot example

    Args:
        difficulty: easy / medium / hard / extra
        n: returnexamplequantity
        dialect: "mysql", "pg" or "oracle"

    Returns:
        Example list
    """
    store = EXAMPLES_BY_DIALECT_DIFFICULTY.get(dialect, EXAMPLES_BY_DIFFICULTY)
    examples = store.get(difficulty, [])
    return examples[:n]


def format_examples_for_prompt(examples: list) -> str:
    """
    Will format examples to embed in prompt text

    Args:
        examples: Example list

    Returns:
        formattext
    """
    import json as _json
    parts = []
    for ex in examples:
        parts.append(
            _json.dumps(ex, ensure_ascii=False, indent=2)
        )
    return "[\n" + ",\n".join(parts) + "\n]"


# =============================================================================
# Self-check: statistics coverage situation
# =============================================================================

def _report_for_dialect(dialect_name: str, examples: list) -> None:
    """Print single dialect coverage statistics."""
    all_features: set[str] = set()
    all_functions: set[str] = set()
    difficulty_counts: dict[str, int] = {}

    for ex in examples:
        d = ex["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        for f in ex.get("dialect_features_used", []):
            all_features.add(f)
        for f in ex.get("builtin_functions_used", []):
            all_functions.add(f)

    has_filter = sum(
        1 for ex in examples
        if any(kw in ex["sql"].upper() for kw in ["WHERE", "HAVING", "LIKE", "ILIKE", "REGEXP", "~"])
    )

    print(f"\n{'=' * 55}")
    print(f"  [{dialect_name.upper()}] Few-shot Examples Coverage Report")
    print(f"{'=' * 55}")
    print(f"\n  Total examples: {len(examples)}")
    print(f"\n  difficultydistribution:")
    for d in ["easy", "medium", "hard", "extra"]:
        print(f"    {d}: {difficulty_counts.get(d, 0)} entries")
    print(f"\n  Covered difference points: {len(all_features)} ")
    print(f"    {sorted(all_features)}")
    print(f"\n  overwritebuilt-in function: {len(all_functions)} ")
    print(f"    {sorted(all_functions)}")
    print(f"\n  Contains data filtering (WHERE/HAVING/LIKE/ILIKE/REGEXP/~): "
          f"{has_filter}/{len(examples)}")


def print_coverage_report():
    """printalldialect few-shot exampleoverwritestatistics"""
    _report_for_dialect("MySQL", FEWSHOT_EXAMPLES_MYSQL)
    _report_for_dialect("PostgreSQL", FEWSHOT_EXAMPLES_PG)
    _report_for_dialect("Oracle", FEWSHOT_EXAMPLES_ORACLE)

    # Merging difference point coverage
    all_features: set[str] = set()
    for examples in _ALL_EXAMPLES.values():
        for ex in examples:
            for f in ex.get("dialect_features_used", []):
                all_features.add(f)
    print(f"\n{'=' * 55}")
    print(f"  tri-dialect total coverage difference points: {len(all_features)} ")
    print(f"    {sorted(all_features)}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    print_coverage_report()
