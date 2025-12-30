"""
Local LLM Module

Provides embedded LLM capability without requiring external services.
Uses llama-cpp-python to run quantized models directly.

Copyright (c) 2024-2025 Ihsan Mokhlis
Licensed under CC-BY-NC-SA-4.0
"""

from .local_llm import LocalLLM, get_llm

__all__ = ["LocalLLM", "get_llm"]

