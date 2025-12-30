#!/usr/bin/env python3
"""
Download public domain PDFs from Archive.org for testing Docling KB
"""

import os
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Archive.org search API
SEARCH_URL = "https://archive.org/advancedsearch.php"

# Categories of books to download (mix of topics for diverse testing)
SEARCH_QUERIES = [
    "subject:science AND mediatype:texts AND format:PDF",
    "subject:history AND mediatype:texts AND format:PDF", 
    "subject:philosophy AND mediatype:texts AND format:PDF",
    "subject:mathematics AND mediatype:texts AND format:PDF",
    "subject:literature AND mediatype:texts AND format:PDF",
    "subject:technology AND mediatype:texts AND format:PDF",
    "subject:economics AND mediatype:texts AND format:PDF",
    "subject:physics AND mediatype:texts AND format:PDF",
    "subject:chemistry AND mediatype:texts AND format:PDF",
    "subject:biology AND mediatype:texts AND format:PDF",
]

OUTPUT_DIR = Path(__file__).parent / "pdfs"
TARGET_COUNT = 100


def search_archive(query: str, rows: int = 20) -> list:
    """Search Archive.org for items matching query"""
    params = {
        "q": query,
        "fl[]": ["identifier", "title", "creator"],
        "sort[]": "downloads desc",
        "rows": rows,
        "page": 1,
        "output": "json",
    }
    
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params, doseq=True)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("response", {}).get("docs", [])
    except Exception as e:
        print(f"Search error for '{query}': {e}")
        return []


def get_pdf_url(identifier: str) -> str | None:
    """Get the PDF download URL for an Archive.org item"""
    metadata_url = f"https://archive.org/metadata/{identifier}"
    
    try:
        with urllib.request.urlopen(metadata_url, timeout=30) as response:
            data = json.loads(response.read().decode())
            files = data.get("files", [])
            
            # Find a PDF file (prefer smaller ones for testing)
            pdf_files = [f for f in files if f.get("name", "").lower().endswith(".pdf")]
            
            if pdf_files:
                # Sort by size, pick smallest under 10MB for faster testing
                pdf_files.sort(key=lambda x: int(x.get("size", 999999999)))
                for pdf in pdf_files:
                    size = int(pdf.get("size", 0))
                    if size < 10_000_000:  # Under 10MB
                        return f"https://archive.org/download/{identifier}/{pdf['name']}"
                # If all are large, take the smallest anyway
                if pdf_files:
                    return f"https://archive.org/download/{identifier}/{pdf_files[0]['name']}"
    except Exception as e:
        print(f"Metadata error for '{identifier}': {e}")
    
    return None


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF file"""
    try:
        print(f"  Downloading: {output_path.name}")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as filename"""
    # Remove/replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Limit length
    return name[:100]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📚 Archive.org PDF Downloader for Docling KB Testing")
    print("=" * 60)
    print(f"Target: {TARGET_COUNT} PDFs")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Collect items from different categories
    all_items = []
    items_per_query = TARGET_COUNT // len(SEARCH_QUERIES) + 5
    
    print("🔍 Searching Archive.org...")
    for query in SEARCH_QUERIES:
        print(f"  Searching: {query[:50]}...")
        items = search_archive(query, rows=items_per_query)
        all_items.extend(items)
        print(f"    Found {len(items)} items")
    
    # Remove duplicates
    seen = set()
    unique_items = []
    for item in all_items:
        if item["identifier"] not in seen:
            seen.add(item["identifier"])
            unique_items.append(item)
    
    print(f"\n📋 Total unique items: {len(unique_items)}")
    print()
    
    # Download PDFs
    print("📥 Downloading PDFs...")
    downloaded = 0
    
    for i, item in enumerate(unique_items):
        if downloaded >= TARGET_COUNT:
            break
            
        identifier = item["identifier"]
        title = item.get("title", identifier)
        
        # Get PDF URL
        pdf_url = get_pdf_url(identifier)
        if not pdf_url:
            continue
        
        # Create filename
        filename = f"{i+1:03d}_{sanitize_filename(title)}.pdf"
        output_path = OUTPUT_DIR / filename
        
        # Skip if already exists
        if output_path.exists():
            print(f"  Skipping (exists): {filename}")
            downloaded += 1
            continue
        
        # Download
        if download_pdf(pdf_url, output_path):
            downloaded += 1
            print(f"  ✓ [{downloaded}/{TARGET_COUNT}] {filename}")
    
    print()
    print("=" * 60)
    print(f"✅ Downloaded {downloaded} PDFs to {OUTPUT_DIR}")
    print("=" * 60)
    
    # List downloaded files
    pdf_files = list(OUTPUT_DIR.glob("*.pdf"))
    total_size = sum(f.stat().st_size for f in pdf_files)
    print(f"\nTotal files: {len(pdf_files)}")
    print(f"Total size: {total_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()

