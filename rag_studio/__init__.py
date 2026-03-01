"""RAG Studio — Document Assistant with RAG over internal documents."""

from rag_studio.config import Config
from rag_studio.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_studio.retrieval import search_hybrid
from rag_studio.ingestion import run_ingestion

__all__ = [
    "Config",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "search_hybrid",
    "run_ingestion",
]
