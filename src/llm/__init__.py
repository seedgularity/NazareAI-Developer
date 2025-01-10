# src/llm/__init__.py
"""LLM module initialization."""
# This will be imported after httpx is properly installed
from .router import OpenRouterClient

__all__ = ['OpenRouterClient']