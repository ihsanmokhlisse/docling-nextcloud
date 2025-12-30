"""
Backend Abstraction Layer

Copyright (c) 2024-2025 Ihsan Mokhlis
Licensed under Non-Commercial Open Source License

Supports two modes:
1. LOCAL (default) - Everything runs inside the ExApp container
2. CLOUD (future SaaS) - Processing offloaded to cloud backend

This abstraction allows seamless switching between modes.
"""

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum


class BackendMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class BaseBackend(ABC):
    """Abstract base class for all backends."""
    
    @abstractmethod
    async def process_document(self, content: bytes, filename: str) -> dict:
        """Process a document and return structured data."""
        pass
    
    @abstractmethod
    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for text chunks."""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search across documents."""
        pass
    
    @abstractmethod
    async def chat(self, question: str, context: str) -> str:
        """Generate response using LLM."""
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check backend health status."""
        pass

