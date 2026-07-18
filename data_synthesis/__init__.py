"""
Data synthesis module.

LLM-driven constraint-aware data synthesis pipeline:
1. Parse database schema
2. LLM selects the most suitable K constraints from the constraint library for each database
3. Synthesize data based on the selected constraints
"""
