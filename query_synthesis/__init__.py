"""
Query synthesis module.

Generates MySQL / PostgreSQL / Oracle queries via LLM based on database DDL, sample data, dialect differences, and built-in functions.
"""

from .query_synthesizer import QuerySynthesizer
from .dialect_parser import DialectDifferenceParser, BuiltinFunctionLoader

__all__ = [
    'QuerySynthesizer',
    'DialectDifferenceParser',
    'BuiltinFunctionLoader',
]
