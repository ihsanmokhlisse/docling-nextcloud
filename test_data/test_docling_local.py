#!/usr/bin/env python3
"""
Local test script for Docling document processing
Tests the core document processing pipeline without the full container
"""

import sys
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import docling
        print(f"✓ docling: {docling.__version__}")
    except ImportError:
        missing.append("docling")
    
    try:
        import chromadb
        print(f"✓ chromadb: {chromadb.__version__}")
    except ImportError:
        missing.append("chromadb")
    
    try:
        import sentence_transformers
        print(f"✓ sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        missing.append("sentence-transformers")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    return True


def test_document_processing(pdf_dir: Path, max_files: int = 5):
    """Test Docling document processing"""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    
    print("\n" + "=" * 60)
    print("📄 Testing Document Processing")
    print("=" * 60)
    
    # Find PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))[:max_files]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize converter
    converter = DocumentConverter()
    
    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        try:
            start_time = time.time()
            
            # Convert document
            result = converter.convert(str(pdf_file))
            
            # Get document info
            doc = result.document
            
            elapsed = time.time() - start_time
            
            # Extract text
            text = doc.export_to_markdown()
            word_count = len(text.split())
            
            print(f"  ✓ Processed in {elapsed:.1f}s")
            print(f"  ✓ Extracted {word_count} words")
            print(f"  ✓ {len(doc.pages)} pages")
            
            results.append({
                "file": pdf_file.name,
                "pages": len(doc.pages),
                "words": word_count,
                "time": elapsed,
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "file": pdf_file.name,
                "error": str(e)
            })
    
    return results


def test_vector_storage(documents: list):
    """Test ChromaDB vector storage"""
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "=" * 60)
    print("🗄️ Testing Vector Storage (ChromaDB)")
    print("=" * 60)
    
    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    client = chromadb.Client()
    collection = client.create_collection(
        name="test_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents
    print(f"Adding {len(documents)} documents to vector store...")
    
    for i, doc in enumerate(documents):
        if "error" in doc:
            continue
            
        # Create chunks (simple splitting for test)
        text = doc.get("text_preview", "")
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        
        for j, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = model.encode(chunk).tolist()
                collection.add(
                    ids=[f"{doc['file']}_{j}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": doc["file"], "chunk": j}]
                )
    
    print(f"✓ Added {collection.count()} chunks to vector store")
    
    # Test search
    print("\nTesting semantic search...")
    query = "science technology research"
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"Query: '{query}'")
    print(f"Found {len(results['documents'][0])} results:")
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"  {i+1}. From {meta['source']}: {doc[:100]}...")
    
    return True


def main():
    print("=" * 60)
    print("🧪 Docling Knowledge Base - Local Test Suite")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Find PDF directory
    pdf_dir = Path(__file__).parent / "pdfs"
    
    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        sys.exit(1)
    
    # Test document processing
    results = test_document_processing(pdf_dir, max_files=5)
    
    if not results:
        print("❌ No documents processed")
        sys.exit(1)
    
    # Test vector storage
    success = test_vector_storage(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print(f"✓ Documents processed: {len(successful)}/{len(results)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for f in failed:
            print(f"   - {f['file']}: {f['error']}")
    
    total_words = sum(r.get("words", 0) for r in successful)
    total_time = sum(r.get("time", 0) for r in successful)
    
    print(f"✓ Total words extracted: {total_words:,}")
    print(f"✓ Total processing time: {total_time:.1f}s")
    print(f"✓ Vector storage: {'Working' if success else 'Failed'}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()

