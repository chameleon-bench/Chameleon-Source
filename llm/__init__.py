"""
LLM interaction module.

Provides a unified LLM calling interface supporting multiple providers (OpenAI, Azure, Anthropic, etc.).
"""

from .client import LLMClient, LLMResponse

__all__ = [
    'LLMClient',
    'LLMResponse',
]