"""
Docling Knowledge Base ExApp for Nextcloud

A comprehensive document processing and knowledge base system that:
1. Automatically processes documents using Docling
2. Builds a searchable knowledge base with vector embeddings
3. Enables chat-based Q&A across all your documents
4. Extracts and queries structured data (tables, metadata)

Copyright (c) 2024-2025 Ihsan Mokhlis
Licensed under Non-Commercial Open Source License
See LICENSE file for details.
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Request, Response, UploadFile
from nc_py_api import NextcloudApp
from nc_py_api.ex_app import AppAPIAuthMiddleware, run_app, set_handlers

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "").lower() == "true" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("docling_kb")

# =============================================================================
# Library Imports with Fallbacks
# =============================================================================

# Docling
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import DoclingDocument
    DOCLING_AVAILABLE = True
    logger.info("✅ Docling library loaded")
except ImportError as e:
    DOCLING_AVAILABLE = False
    logger.warning(f"⚠️ Docling not available: {e}")

# Vector Database (ChromaDB)
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB loaded")
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("⚠️ ChromaDB not available")

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    logger.info("✅ Sentence Transformers loaded")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("⚠️ Sentence Transformers not available")

# LLM Integration (Ollama/OpenAI compatible)
try:
    from openai import AsyncOpenAI
    LLM_AVAILABLE = True
    logger.info("✅ OpenAI client loaded")
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("⚠️ OpenAI client not available")

# =============================================================================
# Global State
# =============================================================================

CONVERTER: DocumentConverter | None = None
CHROMA_CLIENT: Any = None
COLLECTION: Any = None
EMBEDDING_MODEL: Any = None
LLM_CLIENT: Any = None

# In-memory stores (for demo - use persistent DB in production)
PROCESSING_JOBS: dict[str, dict[str, Any]] = {}
PROCESSED_DOCUMENTS: dict[str, dict[str, Any]] = {}
STRUCTURED_DATA: dict[str, list[dict]] = {}  # doc_id -> list of tables/entities


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    global CONVERTER, CHROMA_CLIENT, COLLECTION, EMBEDDING_MODEL, LLM_CLIENT
    
    logger.info("🚀 Starting Docling Knowledge Base ExApp...")
    
    # Initialize Docling
    if DOCLING_AVAILABLE:
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            
            CONVERTER = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("✅ Docling DocumentConverter initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Docling: {e}")
    
    # Initialize Vector Database
    if CHROMADB_AVAILABLE:
        try:
            CHROMA_CLIENT = chromadb.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="/app/data/chroma",
                anonymized_telemetry=False,
            ))
            COLLECTION = CHROMA_CLIENT.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ ChromaDB initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
    
    # Initialize Embedding Model
    if EMBEDDINGS_AVAILABLE:
        try:
            model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            EMBEDDING_MODEL = SentenceTransformer(model_name)
            logger.info(f"✅ Embedding model loaded: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
    
    # Initialize LLM Client (Ollama or OpenAI-compatible)
    if LLM_AVAILABLE:
        try:
            llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
            LLM_CLIENT = AsyncOpenAI(
                base_url=llm_base_url,
                api_key=os.environ.get("LLM_API_KEY", "ollama"),  # Ollama doesn't need real key
            )
            logger.info(f"✅ LLM client initialized: {llm_base_url}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {e}")
    
    yield
    
    # Cleanup
    logger.info("👋 Shutting down Docling Knowledge Base ExApp...")
    if CHROMADB_AVAILABLE and CHROMA_CLIENT:
        try:
            CHROMA_CLIENT.persist()
        except:
            pass


# Create FastAPI app
APP = FastAPI(
    title="Docling Knowledge Base",
    description="AI-powered document processing and knowledge base for Nextcloud",
    version="1.0.0",
    lifespan=lifespan,
)
APP.add_middleware(AppAPIAuthMiddleware)


# =============================================================================
# ExApp Lifecycle Endpoints
# =============================================================================

@APP.get("/heartbeat")
async def heartbeat():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "docling": DOCLING_AVAILABLE and CONVERTER is not None,
            "vector_db": CHROMADB_AVAILABLE and COLLECTION is not None,
            "embeddings": EMBEDDINGS_AVAILABLE and EMBEDDING_MODEL is not None,
            "llm": LLM_AVAILABLE and LLM_CLIENT is not None,
        },
        "documents_processed": len(PROCESSED_DOCUMENTS),
    }


@APP.post("/init")
async def init_app():
    """Initialize ExApp - register UI elements with Nextcloud."""
    logger.info("Initializing Docling Knowledge Base...")
    
    try:
        nc = NextcloudApp()
        
        # Register top menu entry
        await nc.ui.top_menu.register(
            name="docling_kb",
            display_name="Docling KB",
            icon="app-icon",
            admin_required=False,
        )
        
        # Register file actions for auto-processing
        supported_mimes = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/html",
            "text/plain",
            "text/markdown",
            "image/png",
            "image/jpeg",
        ]
        
        # Add to Knowledge Base action
        await nc.ui.files_dropdown_menu.register(
            name="add_to_knowledge_base",
            display_name="📚 Add to Knowledge Base",
            mime=supported_mimes,
            permissions="read",
            action_handler_url="/api/process",
            icon="kb-icon",
        )
        
        # Chat with document action
        await nc.ui.files_dropdown_menu.register(
            name="chat_with_document",
            display_name="💬 Chat with Document",
            mime=supported_mimes,
            permissions="read",
            action_handler_url="/api/chat/document",
            icon="chat-icon",
        )
        
        logger.info("✅ ExApp initialized successfully")
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return {"status": "error", "error": str(e)}


@APP.put("/enabled")
async def enabled_handler(enabled: bool, request: Request):
    """Handle enable/disable events."""
    logger.info(f"ExApp {'enabled' if enabled else 'disabled'}")
    return {"status": "ok"}


# =============================================================================
# Document Processing API
# =============================================================================

@APP.post("/api/process")
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = None,
):
    """
    Process a document and add it to the knowledge base.
    
    This:
    1. Converts the document using Docling
    2. Extracts text, tables, and structure
    3. Creates embeddings for semantic search
    4. Stores everything in the knowledge base
    """
    if not DOCLING_AVAILABLE or CONVERTER is None:
        return Response(
            content='{"error": "Docling not available"}',
            status_code=503,
            media_type="application/json",
        )
    
    # Handle file from upload or Nextcloud file action
    try:
        data = await request.json() if file is None else None
    except:
        data = None
    
    if file is None and data:
        # File action from Nextcloud
        file_path = data.get("filePath")
        if file_path:
            nc = NextcloudApp()
            content = await nc.files.download(file_path)
            filename = Path(file_path).name
        else:
            return Response(
                content='{"error": "No file provided"}',
                status_code=400,
                media_type="application/json",
            )
    elif file:
        content = await file.read()
        filename = file.filename
        file_path = None
    else:
        return Response(
            content='{"error": "No file provided"}',
            status_code=400,
            media_type="application/json",
        )
    
    # Generate document ID
    doc_id = hashlib.md5(f"{filename}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    # Initialize job tracking
    PROCESSING_JOBS[doc_id] = {
        "status": "pending",
        "filename": filename,
        "file_path": file_path,
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "steps": [],
    }
    
    # Process in background
    background_tasks.add_task(
        process_document_pipeline,
        doc_id=doc_id,
        filename=filename,
        content=content,
        file_path=file_path,
    )
    
    return {
        "doc_id": doc_id,
        "status": "processing",
        "message": f"Processing '{filename}' - will be added to your knowledge base",
    }


@APP.post("/api/process/folder")
async def process_folder(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Process all documents in a folder."""
    data = await request.json()
    folder_path = data.get("folderPath", "/")
    
    nc = NextcloudApp()
    
    # List files in folder
    files = await nc.files.listdir(folder_path)
    
    # Supported extensions
    supported_ext = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".txt", ".md", ".png", ".jpg", ".jpeg"}
    
    doc_ids = []
    for f in files:
        if not f.is_dir and Path(f.name).suffix.lower() in supported_ext:
            # Queue each file for processing
            content = await nc.files.download(f.path)
            doc_id = hashlib.md5(f"{f.name}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
            
            PROCESSING_JOBS[doc_id] = {
                "status": "pending",
                "filename": f.name,
                "file_path": f.path,
                "progress": 0,
                "created_at": datetime.now().isoformat(),
                "steps": [],
            }
            
            background_tasks.add_task(
                process_document_pipeline,
                doc_id=doc_id,
                filename=f.name,
                content=content,
                file_path=f.path,
            )
            
            doc_ids.append(doc_id)
    
    return {
        "message": f"Processing {len(doc_ids)} documents",
        "doc_ids": doc_ids,
    }


@APP.get("/api/process/{doc_id}")
async def get_processing_status(doc_id: str):
    """Get the status of a document processing job."""
    if doc_id not in PROCESSING_JOBS:
        return Response(
            content='{"error": "Job not found"}',
            status_code=404,
            media_type="application/json",
        )
    return PROCESSING_JOBS[doc_id]


# =============================================================================
# Chat / Q&A API
# =============================================================================

@APP.post("/api/chat")
async def chat_with_knowledge_base(request: Request):
    """
    Chat with your entire knowledge base.
    
    Uses RAG (Retrieval-Augmented Generation):
    1. Find relevant document chunks via semantic search
    2. Provide context to the LLM
    3. Generate informed response
    """
    data = await request.json()
    question = data.get("question", "")
    doc_ids = data.get("doc_ids")  # Optional: limit to specific documents
    top_k = data.get("top_k", 5)
    
    if not question:
        return Response(
            content='{"error": "No question provided"}',
            status_code=400,
            media_type="application/json",
        )
    
    # Check services
    if not (COLLECTION and EMBEDDING_MODEL):
        return Response(
            content='{"error": "Knowledge base not available"}',
            status_code=503,
            media_type="application/json",
        )
    
    try:
        # 1. Create embedding for the question
        question_embedding = EMBEDDING_MODEL.encode(question).tolist()
        
        # 2. Search for relevant chunks
        where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        
        results = COLLECTION.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        # 3. Build context from results
        context_chunks = []
        sources = []
        
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            context_chunks.append(f"[Source: {metadata.get('filename', 'Unknown')}]\n{doc}")
            sources.append({
                "filename": metadata.get("filename"),
                "file_path": metadata.get("file_path"),
                "chunk_id": metadata.get("chunk_id"),
                "relevance": 1 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        context = "\n\n---\n\n".join(context_chunks)
        
        # 4. Generate response with LLM
        if LLM_CLIENT:
            response = await generate_llm_response(question, context)
        else:
            # Fallback: return context without LLM
            response = f"Based on your documents, here's relevant information:\n\n{context}"
        
        return {
            "answer": response,
            "sources": sources,
            "question": question,
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return Response(
            content=f'{{"error": "{str(e)}"}}',
            status_code=500,
            media_type="application/json",
        )


@APP.post("/api/chat/document")
async def chat_with_single_document(request: Request):
    """Chat with a specific document."""
    data = await request.json()
    file_path = data.get("filePath")
    question = data.get("question", "")
    
    if not file_path:
        return Response(
            content='{"error": "No file path provided"}',
            status_code=400,
            media_type="application/json",
        )
    
    # Find doc_id for this file
    doc_id = None
    for did, doc in PROCESSED_DOCUMENTS.items():
        if doc.get("file_path") == file_path:
            doc_id = did
            break
    
    if not doc_id:
        return Response(
            content='{"error": "Document not in knowledge base. Please process it first."}',
            status_code=404,
            media_type="application/json",
        )
    
    # Use main chat endpoint with doc_id filter
    return await chat_with_knowledge_base(Request(
        scope=request.scope,
        receive=lambda: {"type": "http.request", "body": json.dumps({
            "question": question,
            "doc_ids": [doc_id],
        }).encode()},
    ))


# =============================================================================
# Knowledge Base Query API
# =============================================================================

@APP.get("/api/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    return {
        "documents": list(PROCESSED_DOCUMENTS.values()),
        "total": len(PROCESSED_DOCUMENTS),
    }


@APP.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get detailed information about a processed document."""
    if doc_id not in PROCESSED_DOCUMENTS:
        return Response(
            content='{"error": "Document not found"}',
            status_code=404,
            media_type="application/json",
        )
    
    doc = PROCESSED_DOCUMENTS[doc_id]
    
    # Include structured data if available
    doc["structured_data"] = STRUCTURED_DATA.get(doc_id, [])
    
    return doc


@APP.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    if doc_id not in PROCESSED_DOCUMENTS:
        return Response(
            content='{"error": "Document not found"}',
            status_code=404,
            media_type="application/json",
        )
    
    # Remove from vector DB
    if COLLECTION:
        try:
            COLLECTION.delete(where={"doc_id": doc_id})
        except:
            pass
    
    # Remove from stores
    del PROCESSED_DOCUMENTS[doc_id]
    if doc_id in STRUCTURED_DATA:
        del STRUCTURED_DATA[doc_id]
    
    return {"status": "deleted", "doc_id": doc_id}


@APP.post("/api/search")
async def semantic_search(request: Request):
    """
    Semantic search across all documents.
    Find content by meaning, not just keywords.
    """
    data = await request.json()
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    
    if not query:
        return Response(
            content='{"error": "No query provided"}',
            status_code=400,
            media_type="application/json",
        )
    
    if not (COLLECTION and EMBEDDING_MODEL):
        return Response(
            content='{"error": "Search not available"}',
            status_code=503,
            media_type="application/json",
        )
    
    try:
        query_embedding = EMBEDDING_MODEL.encode(query).tolist()
        
        results = COLLECTION.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            search_results.append({
                "content": doc,
                "filename": metadata.get("filename"),
                "file_path": metadata.get("file_path"),
                "doc_id": metadata.get("doc_id"),
                "relevance": round(1 - results["distances"][0][i], 3),
            })
        
        return {
            "query": query,
            "results": search_results,
            "total": len(search_results),
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return Response(
            content=f'{{"error": "{str(e)}"}}',
            status_code=500,
            media_type="application/json",
        )


@APP.get("/api/tables")
async def list_extracted_tables():
    """List all tables extracted from documents."""
    all_tables = []
    for doc_id, tables in STRUCTURED_DATA.items():
        doc = PROCESSED_DOCUMENTS.get(doc_id, {})
        for table in tables:
            if table.get("type") == "table":
                all_tables.append({
                    **table,
                    "doc_id": doc_id,
                    "filename": doc.get("filename"),
                })
    return {"tables": all_tables, "total": len(all_tables)}


@APP.get("/api/stats")
async def get_knowledge_base_stats():
    """Get statistics about the knowledge base."""
    total_chunks = 0
    if COLLECTION:
        try:
            total_chunks = COLLECTION.count()
        except:
            pass
    
    total_tables = sum(
        len([t for t in tables if t.get("type") == "table"])
        for tables in STRUCTURED_DATA.values()
    )
    
    return {
        "total_documents": len(PROCESSED_DOCUMENTS),
        "total_chunks": total_chunks,
        "total_tables": total_tables,
        "processing_jobs": len([j for j in PROCESSING_JOBS.values() if j["status"] == "processing"]),
        "services": {
            "docling": DOCLING_AVAILABLE and CONVERTER is not None,
            "vector_db": CHROMADB_AVAILABLE and COLLECTION is not None,
            "embeddings": EMBEDDINGS_AVAILABLE and EMBEDDING_MODEL is not None,
            "llm": LLM_AVAILABLE and LLM_CLIENT is not None,
        },
    }


# =============================================================================
# Background Processing Pipeline
# =============================================================================

async def process_document_pipeline(
    doc_id: str,
    filename: str,
    content: bytes,
    file_path: str | None = None,
):
    """
    Full document processing pipeline:
    1. Convert with Docling
    2. Extract structure (text, tables, metadata)
    3. Chunk content
    4. Create embeddings
    5. Store in vector DB
    """
    job = PROCESSING_JOBS[doc_id]
    
    try:
        job["status"] = "processing"
        job["steps"].append({"step": "started", "time": datetime.now().isoformat()})
        
        # Step 1: Convert document with Docling
        job["progress"] = 10
        job["current_step"] = "Converting document with Docling..."
        
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = CONVERTER.convert(tmp_path)
            doc: DoclingDocument = result.document
            job["steps"].append({"step": "docling_conversion", "time": datetime.now().isoformat()})
        finally:
            os.unlink(tmp_path)
        
        job["progress"] = 30
        
        # Step 2: Extract content and structure
        job["current_step"] = "Extracting structure..."
        
        # Get markdown content
        markdown_content = doc.export_to_markdown()
        
        # Extract text for chunking
        full_text = doc.export_to_text()
        
        # Extract tables
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
        
        # Store structured data
        STRUCTURED_DATA[doc_id] = tables
        
        job["progress"] = 50
        job["steps"].append({"step": "structure_extraction", "time": datetime.now().isoformat()})
        
        # Step 3: Chunk content
        job["current_step"] = "Chunking content..."
        chunks = chunk_text(full_text, chunk_size=500, overlap=50)
        
        job["progress"] = 60
        
        # Step 4: Create embeddings and store
        if EMBEDDING_MODEL and COLLECTION:
            job["current_step"] = "Creating embeddings..."
            
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_metadatas.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_path": file_path or "",
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                })
            
            # Create embeddings
            embeddings = EMBEDDING_MODEL.encode(chunk_texts).tolist()
            
            job["progress"] = 80
            
            # Store in vector DB
            job["current_step"] = "Storing in knowledge base..."
            
            COLLECTION.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
            )
            
            job["steps"].append({"step": "vector_storage", "time": datetime.now().isoformat()})
        
        job["progress"] = 90
        
        # Step 5: Store document metadata
        PROCESSED_DOCUMENTS[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "chunks": len(chunks),
            "tables": len(tables),
            "markdown_preview": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
        }
        
        # Complete
        job["status"] = "completed"
        job["progress"] = 100
        job["current_step"] = "Done!"
        job["completed_at"] = datetime.now().isoformat()
        job["steps"].append({"step": "completed", "time": datetime.now().isoformat()})
        
        logger.info(f"✅ Processed document: {filename} ({len(chunks)} chunks, {len(tables)} tables)")
        
    except Exception as e:
        logger.error(f"❌ Failed to process {filename}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["steps"].append({"step": "failed", "error": str(e), "time": datetime.now().isoformat()})


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end
            for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


async def generate_llm_response(question: str, context: str) -> str:
    """Generate a response using the LLM with RAG context."""
    if not LLM_CLIENT:
        return f"Based on your documents:\n\n{context}"
    
    try:
        model = os.environ.get("LLM_MODEL", "llama3.2")
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided document context.
        
Rules:
- Only answer based on the provided context
- If the context doesn't contain relevant information, say so
- Cite sources when possible (mention document names)
- Be concise but thorough
- If asked for specific data (numbers, dates), quote them exactly from the context"""

        response = await LLM_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}"},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"I found relevant information in your documents, but couldn't generate a full response.\n\nRelevant context:\n{context[:1000]}..."


# =============================================================================
# UI
# =============================================================================

@APP.get("/")
async def index():
    """Serve the main UI."""
    return Response(content=get_index_html(), media_type="text/html")


def get_index_html() -> str:
    """Generate the Knowledge Base UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Docling Knowledge Base</title>
    <style>
        :root {
            --bg: #0f0f10;
            --surface: #1a1a1c;
            --surface-2: #242427;
            --border: #2d2d30;
            --text: #e4e4e7;
            --text-dim: #71717a;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --success: #22c55e;
            --error: #ef4444;
            --radius: 12px;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .app {
            display: grid;
            grid-template-columns: 280px 1fr;
            min-height: 100vh;
        }
        
        .sidebar {
            background: var(--surface);
            border-right: 1px solid var(--border);
            padding: 24px;
            display: flex;
            flex-direction: column;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 32px;
        }
        
        .logo-icon {
            font-size: 1.5rem;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: var(--radius);
            color: var(--text-dim);
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 4px;
        }
        
        .nav-item:hover, .nav-item.active {
            background: var(--surface-2);
            color: var(--text);
        }
        
        .nav-item.active {
            background: var(--accent);
            color: white;
        }
        
        .main {
            padding: 32px;
            overflow-y: auto;
        }
        
        .header {
            margin-bottom: 32px;
        }
        
        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-dim);
        }
        
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 24px;
        }
        
        .chat-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .message {
            max-width: 80%;
            padding: 16px;
            border-radius: var(--radius);
        }
        
        .message.user {
            background: var(--accent);
            align-self: flex-end;
        }
        
        .message.assistant {
            background: var(--surface-2);
            align-self: flex-start;
        }
        
        .chat-input-container {
            padding: 16px;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
        }
        
        .chat-input {
            flex: 1;
            padding: 14px 18px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text);
            font-size: 1rem;
            outline: none;
        }
        
        .chat-input:focus {
            border-color: var(--accent);
        }
        
        button {
            padding: 14px 24px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        button:hover {
            background: var(--accent-hover);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 600;
            color: var(--accent);
        }
        
        .stat-label {
            color: var(--text-dim);
            font-size: 0.9rem;
        }
        
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: var(--radius);
            padding: 48px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .upload-zone:hover {
            border-color: var(--accent);
            background: rgba(99, 102, 241, 0.05);
        }
        
        .upload-zone input {
            display: none;
        }
        
        .sources {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }
        
        .source-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px;
            background: var(--surface);
            border-radius: 8px;
            margin-top: 8px;
            font-size: 0.85rem;
        }
        
        .source-relevance {
            background: var(--accent);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        
        .loading {
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--text-dim);
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <div class="logo">
                <span class="logo-icon">📚</span>
                <span>Docling KB</span>
            </div>
            
            <nav>
                <div class="nav-item active" onclick="showSection('chat')">
                    <span>💬</span> Chat
                </div>
                <div class="nav-item" onclick="showSection('documents')">
                    <span>📄</span> Documents
                </div>
                <div class="nav-item" onclick="showSection('upload')">
                    <span>📤</span> Upload
                </div>
                <div class="nav-item" onclick="showSection('search')">
                    <span>🔍</span> Search
                </div>
            </nav>
            
            <div style="margin-top: auto; padding-top: 24px; border-top: 1px solid var(--border);">
                <div id="sidebarStats" class="loading">
                    <div class="spinner"></div>
                    Loading stats...
                </div>
            </div>
        </aside>
        
        <main class="main">
            <!-- Chat Section -->
            <section id="chatSection">
                <div class="header">
                    <h1>💬 Chat with your Knowledge Base</h1>
                    <p class="subtitle">Ask questions about all your documents</p>
                </div>
                
                <div class="card chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message assistant">
                            Hello! I'm your knowledge base assistant. Ask me anything about your documents, and I'll find the relevant information for you.
                        </div>
                    </div>
                    
                    <div class="chat-input-container">
                        <input type="text" class="chat-input" id="chatInput" 
                               placeholder="Ask a question about your documents..." 
                               onkeypress="if(event.key==='Enter')sendMessage()">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </section>
            
            <!-- Documents Section -->
            <section id="documentsSection" style="display: none;">
                <div class="header">
                    <h1>📄 Knowledge Base Documents</h1>
                    <p class="subtitle">All documents in your knowledge base</p>
                </div>
                
                <div id="documentsList" class="card">
                    <div class="loading"><div class="spinner"></div> Loading documents...</div>
                </div>
            </section>
            
            <!-- Upload Section -->
            <section id="uploadSection" style="display: none;">
                <div class="header">
                    <h1>📤 Add Documents</h1>
                    <p class="subtitle">Upload documents to your knowledge base</p>
                </div>
                
                <div class="card">
                    <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                        <input type="file" id="fileInput" multiple 
                               accept=".pdf,.docx,.pptx,.xlsx,.html,.txt,.md,.png,.jpg,.jpeg"
                               onchange="uploadFiles(this.files)">
                        <div style="font-size: 3rem; margin-bottom: 16px;">📁</div>
                        <div style="font-size: 1.1rem; font-weight: 500;">Drop files here or click to browse</div>
                        <div style="color: var(--text-dim); margin-top: 8px;">
                            Supports: PDF, DOCX, PPTX, XLSX, HTML, TXT, MD, Images
                        </div>
                    </div>
                </div>
                
                <div id="uploadProgress" class="card" style="display: none;">
                    <h3 style="margin-bottom: 16px;">Processing...</h3>
                    <div id="uploadProgressList"></div>
                </div>
            </section>
            
            <!-- Search Section -->
            <section id="searchSection" style="display: none;">
                <div class="header">
                    <h1>🔍 Semantic Search</h1>
                    <p class="subtitle">Search by meaning across all documents</p>
                </div>
                
                <div class="card">
                    <div style="display: flex; gap: 12px; margin-bottom: 24px;">
                        <input type="text" class="chat-input" id="searchInput" 
                               placeholder="Search for concepts, topics, or specific information..."
                               onkeypress="if(event.key==='Enter')doSearch()">
                        <button onclick="doSearch()">Search</button>
                    </div>
                    
                    <div id="searchResults"></div>
                </div>
            </section>
        </main>
    </div>
    
    <script>
        // State
        let stats = {};
        
        // Init
        document.addEventListener('DOMContentLoaded', () => {
            loadStats();
            loadDocuments();
        });
        
        // Navigation
        function showSection(name) {
            document.querySelectorAll('.main > section').forEach(s => s.style.display = 'none');
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById(name + 'Section').style.display = 'block';
            event.target.closest('.nav-item').classList.add('active');
            
            if (name === 'documents') loadDocuments();
        }
        
        // Stats
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                stats = await res.json();
                
                document.getElementById('sidebarStats').innerHTML = `
                    <div style="font-size: 0.85rem; color: var(--text-dim);">
                        <div style="margin-bottom: 8px;">📄 ${stats.total_documents} documents</div>
                        <div style="margin-bottom: 8px;">📊 ${stats.total_chunks} chunks</div>
                        <div>📋 ${stats.total_tables} tables</div>
                    </div>
                `;
            } catch (e) {
                document.getElementById('sidebarStats').innerHTML = '<span style="color: var(--error);">Error loading stats</span>';
            }
        }
        
        // Chat
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const question = input.value.trim();
            if (!question) return;
            
            // Add user message
            addMessage(question, 'user');
            input.value = '';
            
            // Show loading
            const loadingId = Date.now();
            addMessage('<div class="loading"><div class="spinner"></div> Thinking...</div>', 'assistant', loadingId);
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });
                
                const data = await res.json();
                
                // Remove loading
                document.getElementById('msg-' + loadingId)?.remove();
                
                // Add response with sources
                let html = data.answer || data.error;
                
                if (data.sources && data.sources.length > 0) {
                    html += '<div class="sources"><strong>Sources:</strong>';
                    data.sources.forEach(s => {
                        html += `<div class="source-item">
                            <span class="source-relevance">${Math.round(s.relevance * 100)}%</span>
                            ${s.filename}
                        </div>`;
                    });
                    html += '</div>';
                }
                
                addMessage(html, 'assistant');
                
            } catch (e) {
                document.getElementById('msg-' + loadingId)?.remove();
                addMessage('Error: ' + e.message, 'assistant');
            }
        }
        
        function addMessage(content, role, id) {
            const container = document.getElementById('chatMessages');
            const msg = document.createElement('div');
            msg.className = 'message ' + role;
            if (id) msg.id = 'msg-' + id;
            msg.innerHTML = content;
            container.appendChild(msg);
            container.scrollTop = container.scrollHeight;
        }
        
        // Documents
        async function loadDocuments() {
            try {
                const res = await fetch('/api/documents');
                const data = await res.json();
                
                if (data.documents.length === 0) {
                    document.getElementById('documentsList').innerHTML = `
                        <div style="text-align: center; color: var(--text-dim); padding: 48px;">
                            <div style="font-size: 3rem; margin-bottom: 16px;">📭</div>
                            <div>No documents yet. Upload some to get started!</div>
                        </div>
                    `;
                    return;
                }
                
                let html = '<table style="width: 100%; border-collapse: collapse;">';
                html += '<tr style="border-bottom: 1px solid var(--border);"><th style="text-align: left; padding: 12px;">File</th><th>Chunks</th><th>Tables</th><th>Processed</th></tr>';
                
                data.documents.forEach(doc => {
                    html += `<tr style="border-bottom: 1px solid var(--border);">
                        <td style="padding: 12px;">📄 ${doc.filename}</td>
                        <td style="text-align: center;">${doc.chunks}</td>
                        <td style="text-align: center;">${doc.tables}</td>
                        <td style="text-align: center; color: var(--text-dim);">${new Date(doc.processed_at).toLocaleDateString()}</td>
                    </tr>`;
                });
                
                html += '</table>';
                document.getElementById('documentsList').innerHTML = html;
                
            } catch (e) {
                document.getElementById('documentsList').innerHTML = `<span style="color: var(--error);">Error: ${e.message}</span>`;
            }
        }
        
        // Upload
        async function uploadFiles(files) {
            const progressDiv = document.getElementById('uploadProgress');
            const listDiv = document.getElementById('uploadProgressList');
            progressDiv.style.display = 'block';
            listDiv.innerHTML = '';
            
            for (const file of files) {
                const itemId = 'upload-' + Date.now();
                listDiv.innerHTML += `<div id="${itemId}" style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div class="spinner"></div>
                    <span>${file.name}</span>
                </div>`;
                
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const res = await fetch('/api/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await res.json();
                    
                    document.getElementById(itemId).innerHTML = `
                        <span style="color: var(--success);">✓</span>
                        <span>${file.name}</span>
                        <span style="color: var(--text-dim);">Processing...</span>
                    `;
                    
                    // Poll for completion
                    pollJobStatus(data.doc_id, itemId, file.name);
                    
                } catch (e) {
                    document.getElementById(itemId).innerHTML = `
                        <span style="color: var(--error);">✗</span>
                        <span>${file.name}</span>
                        <span style="color: var(--error);">${e.message}</span>
                    `;
                }
            }
        }
        
        async function pollJobStatus(docId, itemId, filename) {
            const res = await fetch('/api/process/' + docId);
            const job = await res.json();
            
            if (job.status === 'completed') {
                document.getElementById(itemId).innerHTML = `
                    <span style="color: var(--success);">✓</span>
                    <span>${filename}</span>
                    <span style="color: var(--success);">Added to knowledge base!</span>
                `;
                loadStats();
            } else if (job.status === 'failed') {
                document.getElementById(itemId).innerHTML = `
                    <span style="color: var(--error);">✗</span>
                    <span>${filename}</span>
                    <span style="color: var(--error);">${job.error}</span>
                `;
            } else {
                document.getElementById(itemId).innerHTML = `
                    <div class="spinner"></div>
                    <span>${filename}</span>
                    <span style="color: var(--text-dim);">${job.current_step || 'Processing...'} (${job.progress}%)</span>
                `;
                setTimeout(() => pollJobStatus(docId, itemId, filename), 1000);
            }
        }
        
        // Search
        async function doSearch() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;
            
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div> Searching...</div>';
            
            try {
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, top_k: 10})
                });
                
                const data = await res.json();
                
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<div style="color: var(--text-dim);">No results found.</div>';
                    return;
                }
                
                let html = '';
                data.results.forEach(r => {
                    html += `<div style="padding: 16px; background: var(--surface-2); border-radius: var(--radius); margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <strong>📄 ${r.filename}</strong>
                            <span class="source-relevance">${Math.round(r.relevance * 100)}% match</span>
                        </div>
                        <div style="color: var(--text-dim); font-size: 0.9rem;">${r.content.substring(0, 300)}...</div>
                    </div>`;
                });
                
                resultsDiv.innerHTML = html;
                
            } catch (e) {
                resultsDiv.innerHTML = `<span style="color: var(--error);">Error: ${e.message}</span>`;
            }
        }
    </script>
</body>
</html>"""


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    set_handlers(APP, enabled_handler, models_to_fetch={})
    run_app("main:APP", log_level="info")
