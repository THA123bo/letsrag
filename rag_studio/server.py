"""FastAPI server — RAG chatbot and document manager."""

import re
import os
import logging
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_studio.config import Config
from rag_studio.llm import llm_chat
from rag_studio.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_studio.retrieval import search_hybrid, invalidate_bm25_cache
from rag_studio.ingestion import run_ingestion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger("rag_studio")

app = FastAPI(title="RAG Studio", version="1.0.0")

# In-memory document index for source attribution
_DOC_INDEX: list[dict] = []

def _build_doc_index() -> None:
    """Rebuilds the in-memory index using documents from the input directory."""
    global _DOC_INDEX
    _DOC_INDEX = []
    
    for fname in os.listdir(Config.INPUT_DIR):
        if not fname.endswith(".md"):
            continue
            
        with open(Config.INPUT_DIR / fname, "r", encoding="utf-8") as f:
            content = f.read()
            
        title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else fname.replace(".md", "")
        
        _DOC_INDEX.append({
            "filename": fname,
            "title": title,
            "content": content,
        })

_build_doc_index()

def identify_source(chunk_text: str) -> dict:
    """Identifies the document and section for a given chunk using the index.

    Args:
        chunk_text (str): The chunk text to search for.

    Returns:
        dict: Source metadata including 'doc_title', 'doc_file', and 'section'.
    """
    best_doc, best_overlap = None, 0
    
    for doc in _DOC_INDEX:
        if chunk_text[:80] in doc["content"]:
            if len(chunk_text) > best_overlap:
                best_overlap = len(chunk_text)
                best_doc = doc
                
    doc_title = best_doc["title"] if best_doc else "Unknown Document"
    doc_file = best_doc["filename"] if best_doc else ""
    
    sections = re.findall(r"^#{1,3}\s+(.+)", chunk_text, re.MULTILINE)
    section = sections[0].strip() if sections else ""
    
    return {"doc_title": doc_title, "doc_file": doc_file, "section": section}

# Pydantic Models
class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    text: str
    rerank_score: float
    rank: int
    doc_title: str
    doc_file: str
    section: str

class ChatResponse(BaseModel):
    response: str
    sources: list[Source]

class DocInfo(BaseModel):
    filename: str
    title: str
    size_kb: float

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Processes a user question, searches via RAG, and generates a response with sources."""
    try:
        logger.info(f"Question received: {req.message}")
        
        logger.info("Starting hybrid search...")
        chunks = search_hybrid(req.message, limit=Config.RETRIEVAL_LIMIT)
        retrieval_context = [c["text"] for c in chunks]
        logger.info(f"Search complete: {len(chunks)} chunks retrieved")

        logger.info("Generating response with LLM...")
        user_prompt = build_user_prompt(retrieval_context, req.message)
        answer = llm_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        logger.info("Response successfully generated")

        all_sources = []
        for rank, chunk in enumerate(chunks, start=1):
            chunk_meta = chunk.get("metadata", {})
            
            if chunk_meta.get("doc_title"):
                doc_title = chunk_meta["doc_title"]
                doc_file = chunk_meta.get("source_file", "")
                section = chunk_meta.get("section", "")
            else:
                meta = identify_source(chunk["text"])
                doc_title, doc_file, section = meta["doc_title"], meta["doc_file"], meta["section"]

            all_sources.append(Source(
                text=chunk["text"][:300],
                rerank_score=round(chunk.get("rerank_score", 0), 4),
                rank=rank,
                doc_title=doc_title,
                doc_file=doc_file,
                section=section,
            ))

        # Filter by threshold, guaranteeing at least the best source if available
        above = [s for s in all_sources if s.rerank_score >= Config.SOURCES_MIN_SCORE]
        sources = above if above else all_sources[:1]

        return ChatResponse(response=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


# Document Management Endpoints
@app.get("/api/docs", response_model=list[DocInfo])
def list_documents():
    """Lists all .md documents in the input directory."""
    docs = []
    for fname in sorted(os.listdir(Config.INPUT_DIR)):
        if not fname.endswith(".md"):
            continue
            
        path = Config.INPUT_DIR / fname
        size = os.path.getsize(path) / 1024
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else fname
        docs.append(DocInfo(filename=fname, title=title, size_kb=round(size, 1)))
        
    return docs

@app.get("/api/docs/{filename}")
def get_document(filename: str):
    """Retrieves the content of a specific .md document."""
    if not filename.endswith(".md"):
        raise HTTPException(400, "Only .md files are allowed")
        
    path = Config.INPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Document not found")
        
    with open(path, "r", encoding="utf-8") as f:
        return {"filename": filename, "content": f.read()}

@app.post("/api/docs/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a .md file to the input directory."""
    if not file.filename.endswith(".md"):
        raise HTTPException(400, "Only .md files are allowed")
        
    dest = Config.INPUT_DIR / file.filename
    content = await file.read()
    
    with open(dest, "wb") as f:
        f.write(content)
        
    _build_doc_index()
    return {"message": f"Document '{file.filename}' successfully uploaded", "filename": file.filename}

@app.delete("/api/docs/{filename}")
def delete_document(filename: str):
    """Deletes a .md document from the input directory."""
    path = Config.INPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Document not found")
        
    os.remove(path)
    _build_doc_index()
    return {"message": f"Document '{filename}' deleted"}

@app.post("/api/docs/ingest")
def ingest_documents():
    """Re-ingests all documents into ChromaDB."""
    run_ingestion(rebuild=True)
    invalidate_bm25_cache()
    _build_doc_index()
    return {"message": "Database rebuilt with all documents"}


# Frontend serving
@app.get("/")
def serve_frontend():
    """Serves the main frontend application."""
    return FileResponse(
        str(Config.STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

app.mount("/static", StaticFiles(directory=str(Config.STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
