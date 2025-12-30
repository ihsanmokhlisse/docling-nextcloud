"""
Local Backend - Everything runs inside the ExApp container

Copyright (c) 2024-2025 Ihsan Mokhlis
Licensed under Non-Commercial Open Source License

This is the default, fully self-contained mode where:
- Docling processes documents locally
- ChromaDB stores vectors locally
- Sentence Transformers creates embeddings locally
- Ollama (or compatible) handles LLM inference locally

No external services required!
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from . import BaseBackend

logger = logging.getLogger("docling_kb.backend.local")


class LocalBackend(BaseBackend):
    """
    Fully self-contained local backend.
    
    All processing happens within the container:
    - Document conversion: Docling
    - Vector storage: ChromaDB
    - Embeddings: Sentence Transformers
    - LLM: Ollama (local) or OpenAI-compatible API
    """
    
    def __init__(self):
        self.converter = None
        self.collection = None
        self.embedding_model = None
        self.llm_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all local services."""
        if self._initialized:
            return
        
        logger.info("🚀 Initializing Local Backend...")
        
        # Initialize Docling
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("✅ Docling initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Docling not available: {e}")
        
        # Initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.environ.get("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
                anonymized_telemetry=False,
            ))
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ ChromaDB initialized")
        except ImportError as e:
            logger.warning(f"⚠️ ChromaDB not available: {e}")
        
        # Initialize Embedding Model
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"✅ Embedding model loaded: {model_name}")
        except ImportError as e:
            logger.warning(f"⚠️ Sentence Transformers not available: {e}")
        
        # Initialize LLM Client
        try:
            from openai import AsyncOpenAI
            
            self.llm_client = AsyncOpenAI(
                base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("LLM_API_KEY", "ollama"),
            )
            logger.info("✅ LLM client initialized")
        except ImportError as e:
            logger.warning(f"⚠️ OpenAI client not available: {e}")
        
        self._initialized = True
        logger.info("🎉 Local Backend ready!")
    
    async def process_document(self, content: bytes, filename: str) -> dict:
        """Process document with Docling locally."""
        if not self.converter:
            raise RuntimeError("Docling not initialized")
        
        # Write to temp file
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = self.converter.convert(tmp_path)
            doc = result.document
            
            return {
                "text": doc.export_to_text(),
                "markdown": doc.export_to_markdown(),
                "tables": self._extract_tables(doc),
                "metadata": {
                    "filename": filename,
                    "pages": getattr(result, 'num_pages', None),
                }
            }
        finally:
            os.unlink(tmp_path)
    
    def _extract_tables(self, doc) -> list[dict]:
        """Extract tables from document."""
        tables = []
        try:
            for item in doc.iterate_items():
                if hasattr(item, 'table') and item.table:
                    tables.append({
                        "type": "table",
                        "content": str(item.table),
                        "page": getattr(item, 'page', None),
                    })
        except:
            pass
        return tables
    
    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings using local model."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        return self.embedding_model.encode(texts).tolist()
    
    async def store_embeddings(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict
    ):
        """Store embeddings in local ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {**metadata, "chunk_id": i, "doc_id": doc_id}
            for i in range(len(chunks))
        ]
        
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
        )
    
    async def search(self, query: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[dict]:
        """Semantic search using local ChromaDB."""
        if not self.collection or not self.embedding_model:
            raise RuntimeError("Search services not initialized")
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            search_results.append({
                "content": doc,
                "metadata": metadata,
                "relevance": 1 - results["distances"][0][i],
            })
        
        return search_results
    
    async def chat(self, question: str, context: str) -> str:
        """Generate response using local LLM."""
        if not self.llm_client:
            # Fallback: return context without LLM
            return f"Based on your documents:\n\n{context}"
        
        model = os.environ.get("LLM_MODEL", "llama3.2")
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Only answer based on the provided context
- If the context doesn't contain relevant information, say so
- Cite sources when possible (mention document names)
- Be concise but thorough"""

        response = await self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    async def health_check(self) -> dict:
        """Check health of all local services."""
        return {
            "mode": "local",
            "services": {
                "docling": self.converter is not None,
                "vector_db": self.collection is not None,
                "embeddings": self.embedding_model is not None,
                "llm": self.llm_client is not None,
            },
            "status": "healthy" if self._initialized else "initializing",
        }
    
    async def delete_document(self, doc_id: str):
        """Remove document from local storage."""
        if self.collection:
            try:
                self.collection.delete(where={"doc_id": doc_id})
            except:
                pass


# Singleton instance
_backend: LocalBackend | None = None


async def get_local_backend() -> LocalBackend:
    """Get or create the local backend singleton."""
    global _backend
    if _backend is None:
        _backend = LocalBackend()
        await _backend.initialize()
    return _backend

