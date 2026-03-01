"""Entry point for the RAG Studio server."""

import uvicorn

from rag_studio.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
