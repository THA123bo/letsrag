"""Document ingestion pipeline into ChromaDB with enriched metadata."""

import os
import re
import shutil
from datetime import datetime

from chonkie import Pipeline
from chonkie.handshakes import ChromaHandshake

from rag_studio.config import Config

# Spanish month map to digits since input documents are in Spanish
_MONTHS = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}

def _extract_title(content: str) -> str:
    """Extracts the title (first H1 heading) of the document.

    Args:
        content (str): The document content.

    Returns:
        str: The extracted title, or an empty string if not found.
    """
    match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    return match.group(1).strip() if match else ""

def _extract_doc_date(content: str) -> str:
    """Extracts the document date looking for patterns like '**Fecha:** DD de MES de YYYY'.

    Args:
        content (str): The document content.

    Returns:
        str: Date in ISO format (YYYY-MM-DD) or an empty string if not found.
    """
    pattern = r"\*\*Fecha(?:\s+de\s+registro)?:\*\*\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})"
    match = re.search(pattern, content)
    if not match:
        return ""
    
    day, month_name, year = match.group(1).zfill(2), match.group(2).lower(), match.group(3)
    month = _MONTHS.get(month_name, "01")
    return f"{year}-{month}-{day}"

def _find_section(chunk_text: str, full_content: str) -> str:
    """Identifies the section (H2/H3 heading) to which a chunk belongs.

    Args:
        chunk_text (str): The text of the chunk.
        full_content (str): The full content of the document.

    Returns:
        str: The closest previous heading, or an empty string if none.
    """
    pos = full_content.find(chunk_text[:80])
    if pos == -1:
        return ""

    headings = list(re.finditer(r"^#{1,3}\s+(.+)", full_content[:pos], re.MULTILINE))
    return headings[-1].group(1).strip() if headings else ""

def run_ingestion(rebuild: bool = True) -> None:
    """Executes the ingestion pipeline with enriched metadata for each chunk.

    Args:
        rebuild (bool): If True, deletes existing collection before ingestion.
    """
    if rebuild and Config.CHROMA_DIR.exists():
        shutil.rmtree(Config.CHROMA_DIR)

    documents = []
    for fname in sorted(os.listdir(Config.INPUT_DIR)):
        if not fname.endswith(".md"):
            continue
            
        with open(Config.INPUT_DIR / fname, "r", encoding="utf-8") as f:
            content = f.read()
            
        documents.append({
            "filename": fname,
            "content": content,
            "title": _extract_title(content),
            "doc_date": _extract_doc_date(content),
        })

    if not documents:
        print("⚠️  No .md documents found in input directory")
        return

    pipe = (
        Pipeline()
        .chunk_with("recursive", recipe="markdown", chunk_size=Config.CHUNK_SIZE * 2)
        .chunk_with(
            "sentence",
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            delim=[".", "!", "?", "¿", "¡", "…", "\n", "\n\n"],
            include_delim="prev"
        )
    )

    handshake = ChromaHandshake(
        path=str(Config.CHROMA_DIR),
        collection_name=Config.COLLECTION_NAME,
        embedding_model=Config.EMBEDDING_MODEL,
    )
    
    collection = handshake.collection
    chunked_at = datetime.now().isoformat()
    total_chunks = 0

    for doc in documents:
        doc_result = pipe.run(texts=doc["content"])
        chunks = doc_result.chunks

        for idx, chunk in enumerate(chunks):
            metadata = {
                "source_file": doc["filename"],
                "doc_title": doc["title"],
                "doc_date": doc["doc_date"],
                "section": _find_section(chunk.text, doc["content"]),
                "chunk_index": idx,
                "chunked_at": chunked_at,
                "token_count": chunk.token_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }

            chunk_id = f"{doc['filename']}::chunk_{idx}"
            collection.upsert(ids=[chunk_id], documents=[chunk.text], metadatas=[metadata])
            total_chunks += 1

    print(f"✅ Ingestion complete: {total_chunks} chunks from {len(documents)} documents.")

if __name__ == "__main__":
    print("Starting document ingestion...")
    run_ingestion()
