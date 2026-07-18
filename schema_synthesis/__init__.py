"""
Schema synthesis module.

Dialect-aware schema generation pipeline:
1. SchemaExpander uses LLM to generate tri-dialect DDL (MySQL/PG/Oracle)
2. DBTester validates DDL against real databases
"""
