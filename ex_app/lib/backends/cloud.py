"""
Cloud Backend - SaaS Backend Service (Phase 2)

Copyright (c) 2024-2025 Ihsan Mokhlis
Licensed under Non-Commercial Open Source License

This backend offloads processing to a managed cloud service.
Useful for:
- Users without powerful hardware
- Enterprise deployments wanting managed infrastructure
- High-volume processing needs

The cloud backend communicates with a REST API service that you host.
"""

import logging
import os
from typing import Any

import httpx

from . import BaseBackend

logger = logging.getLogger("docling_kb.backend.cloud")


class CloudBackend(BaseBackend):
    """
    Cloud/SaaS backend that offloads processing to external service.
    
    Configuration via environment variables:
    - DOCLING_CLOUD_URL: Base URL of the cloud service
    - DOCLING_CLOUD_API_KEY: API key for authentication
    - DOCLING_CLOUD_WORKSPACE: Workspace/tenant ID
    """
    
    def __init__(self):
        self.base_url = os.environ.get("DOCLING_CLOUD_URL", "https://api.docling-cloud.example.com")
        self.api_key = os.environ.get("DOCLING_CLOUD_API_KEY", "")
        self.workspace = os.environ.get("DOCLING_CLOUD_WORKSPACE", "default")
        self._client: httpx.AsyncClient | None = None
    
    async def initialize(self):
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Workspace": self.workspace,
            },
            timeout=300.0,  # 5 minutes for large documents
        )
        logger.info(f"☁️ Cloud Backend initialized: {self.base_url}")
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
    
    async def process_document(self, content: bytes, filename: str) -> dict:
        """Upload and process document via cloud API."""
        if not self._client:
            raise RuntimeError("Cloud backend not initialized")
        
        response = await self._client.post(
            "/api/v1/process",
            files={"file": (filename, content)},
        )
        response.raise_for_status()
        return response.json()
    
    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings via cloud API."""
        if not self._client:
            raise RuntimeError("Cloud backend not initialized")
        
        response = await self._client.post(
            "/api/v1/embeddings",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    
    async def search(self, query: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[dict]:
        """Semantic search via cloud API."""
        if not self._client:
            raise RuntimeError("Cloud backend not initialized")
        
        response = await self._client.post(
            "/api/v1/search",
            json={
                "query": query,
                "top_k": top_k,
                "doc_ids": doc_ids,
            },
        )
        response.raise_for_status()
        return response.json()["results"]
    
    async def chat(self, question: str, context: str) -> str:
        """Generate response via cloud LLM."""
        if not self._client:
            raise RuntimeError("Cloud backend not initialized")
        
        response = await self._client.post(
            "/api/v1/chat",
            json={
                "question": question,
                "context": context,
            },
        )
        response.raise_for_status()
        return response.json()["answer"]
    
    async def health_check(self) -> dict:
        """Check cloud service health."""
        if not self._client:
            return {"mode": "cloud", "status": "not_initialized"}
        
        try:
            response = await self._client.get("/api/v1/health")
            response.raise_for_status()
            data = response.json()
            return {
                "mode": "cloud",
                "status": "healthy",
                "service": data,
            }
        except Exception as e:
            return {
                "mode": "cloud",
                "status": "unhealthy",
                "error": str(e),
            }


# =============================================================================
# Cloud Service API Specification (for Phase 2 SaaS implementation)
# =============================================================================
"""
The cloud backend expects the following REST API endpoints:

POST /api/v1/process
    - Input: multipart/form-data with file
    - Output: { text, markdown, tables, metadata }
    
POST /api/v1/embeddings
    - Input: { texts: string[] }
    - Output: { embeddings: float[][] }

POST /api/v1/search
    - Input: { query: string, top_k: int, doc_ids?: string[] }
    - Output: { results: [{ content, metadata, relevance }] }

POST /api/v1/chat
    - Input: { question: string, context: string }
    - Output: { answer: string }

GET /api/v1/health
    - Output: { status: string, ... }

Authentication: Bearer token in Authorization header
Multi-tenancy: X-Workspace header for tenant isolation
"""

