# Chameleon: A Large-scale Benchmark for SQL Dialect Translation via Divergence-Driven LLM Synthesis

## Overview

**Chameleon** is a large-scale benchmark for evaluating SQL dialect translation across MySQL, PostgreSQL, and Oracle. It is constructed through a divergence-driven synthesis framework that systematically mines over 500 cross-dialect divergences and derives their stage-specific requirements on schema construction, data population, and query synthesis.

The benchmark comprises:
- **1,200** cross-domain databases with cross-engine equivalent schemas and data
- **100K+** execution-validated queries across four difficulty levels (easy, medium, hard, extra)
- **6** translation directions among three database engines
- Standardized **train/dev/test** split (1,000 / 100 / 100 databases) for reproducible evaluation

## Repository Structure

```
source/
├── llm/                  # Unified LLM client supporting multiple providers
├── schema_synthesis/     # Divergence-driven schema construction pipeline
├── data_synthesis/       # Constraint-aware data population pipeline
├── query_synthesis/      # Divergence-guided query synthesis pipeline
├── eval/                 # Evaluation toolkit (ES / EM metrics)
├── utils/                # Database utilities and logging
├── config/               # Configuration files (database, LLM, synthesis)
├── data/                 # Dialect divergence inventory and knowledge bases
└── requirements.txt      # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database Connections

Edit `config/database_sync.yaml` with your MySQL, PostgreSQL, and Oracle connection details:

```yaml
database:
  mysql:
    host: YOUR_MYSQL_HOST
    port: 3306
    user: root
    password: YOUR_MYSQL_PASSWORD
  postgresql:
    host: YOUR_PG_HOST
    port: 5432
    user: postgres
    password: YOUR_PG_PASSWORD
  oracle:
    host: YOUR_ORACLE_HOST
    port: 1521
    user: system
    password: YOUR_ORACLE_PASSWORD
    service_name: XE
    import_password: "YOUR_ORACLE_USER_PASSWORD"
```

### 3. Import the Dataset

Download the dataset from the [project page](https://chameleon-bench.github.io/Chameleon/) and import the schema and data files into your databases using standard tools:

```bash
# MySQL
mysql -u root -p < database/<db_name>/<db_name>_schema_mysql.sql
mysql -u root -p <db_name> < database/<db_name>/<db_name>_data_mysql.sql

# PostgreSQL
psql -U postgres -d <db_name> -f database/<db_name>/<db_name>_schema_pg.sql
psql -U postgres -d <db_name> -f database/<db_name>/<db_name>_data_pg.sql

# Oracle
sqlplus system/<password>@<host>:1521/XE @ database/<db_name>/<db_name>_schema_oracle.sql
sqlplus <db_name_upper>/<password>@<host>:1521/XE @ database/<db_name>/<db_name>_data_oracle.sql
```

### 4. Configure LLM (Optional)

Edit `config/llm_config.yaml` to set up your LLM provider and API key for running the synthesis pipelines.

## Evaluation

The evaluation toolkit measures translation quality using two metrics:

- **Execution Success (ES):** The percentage of translated SQL queries that execute successfully on the target database engine.
- **Execution Match (EM):** The percentage of translated SQL queries whose result sets match the source query's results.

### Input Format

The evaluator takes two JSON files as input:

**Source queries file:**
```json
[
  {
    "query_id": 1,
    "difficulty": "easy",
    "sql": "SELECT * FROM ...",
    "database": "my_db",
    "dialect": "mysql"
  }
]
```

**Translated queries file:**
```json
[
  {
    "query_id": 1,
    "translated_sql": "SELECT * FROM ..."
  }
]
```

### Running Evaluation

```bash
python -m eval.run_eval \
    --source-queries source_queries.json \
    --translated-queries translated_queries.json \
    --output eval_results.json \
    --split dev \
    --dataset-dir dataset \
    --db-config config/database_sync.yaml \
    --workers 8
```

### Output

The evaluation produces a JSON file containing:

- Overall ES and EM scores
- Per-difficulty breakdown
- Per-database breakdown
- Sample failure cases with error details

## Dataset

The Chameleon dataset is available for download from the [project page](https://chameleon-bench.github.io/Chameleon/). It includes:

- **Schema files:** Tri-dialect DDL (MySQL, PostgreSQL, Oracle) for each database
- **Data files:** Tri-dialect INSERT scripts for each database
- **Query files:** Execution-validated SQL queries with difficulty labels
- **Dialect assignment:** Per-database dialect allocation for query synthesis

The train and dev splits are publicly available. The test split is withheld to prevent contamination.

## Citation

If you use Chameleon in your research, please cite:

```bibtex
@inproceedings{chameleon2026,
  title     = {Chameleon: A Large-scale Benchmark for SQL Dialect Translation via Divergence-Driven LLM Synthesis},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference on Very Large Data Bases (VLDB)},
  year      = {2026},
  note      = {Anonymous submission}
}
```

## License

This project is released under the MIT License.
