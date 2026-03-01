"""Centralized configuration for the RAG Studio project."""

import os
from pathlib import Path
import yaml

_BASE_DIR = Path(__file__).resolve().parent.parent
_config_path = _BASE_DIR / "config.yaml"
_cfg = {}
if _config_path.exists():
    with open(_config_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}

def _get_val(section: str, key: str, default):
    """Helper to get a nested value from the yaml config."""
    return _cfg.get(section, {}).get(key, default)


class Config:
    """Project configuration constants.
    
    Attributes:
        BASE_DIR (Path): Base directory of the project.
        INPUT_DIR (Path): Directory for input documents.
        STATIC_DIR (Path): Directory for static frontend assets.
        CHROMA_DIR (Path): Directory for ChromaDB storage.
        EVAL_DIR (Path): Directory for evaluation data.
        LLM_PROVIDER (str): The configured LLM provider.
        LLM_MODEL (str): The configured LLM model.
        EMBEDDING_MODEL (str): The configured embedding model.
        RERANK_MODEL (str): The configured reranking model.
        COLLECTION_NAME (str): The ChromaDB collection name.
        CHUNK_SIZE (int): Document chunk size.
        CHUNK_OVERLAP (int): Document chunk overlap.
        SEMANTIC_THRESHOLD (float): Minimum semantic similarity threshold.
        RETRIEVAL_LIMIT (int): Final number of retrieved documents.
        RETRIEVAL_TOP_K (int): Top K documents to retrieve per method.
        RERANK_MIN_SCORE (float): Minimum score for the reranker.
        SOURCES_DISPLAY (int): Number of sources to display.
        SOURCES_MIN_SCORE (float): Minimum threshold to display sources.
        RRF_K (int): Reciprocal Rank Fusion constant.
        RRF_WEIGHT_SEMANTIC (float): Weight for semantic search in RRF.
        RRF_WEIGHT_BM25 (float): Weight for BM25 search in RRF.
    """

    BASE_DIR = _BASE_DIR
    INPUT_DIR = BASE_DIR / "input"
    STATIC_DIR = BASE_DIR / "static"
    CHROMA_DIR = BASE_DIR / "chroma_db"
    EVAL_DIR = BASE_DIR / "eval"

    LLM_PROVIDER = _get_val("llm", "provider", "ollama")
    LLM_MODEL = _get_val("llm", "model", "llama3.1:8b")
    EMBEDDING_MODEL = _get_val("llm", "embedding_model", "litellm://ollama/nomic-embed-text")
    RERANK_MODEL = _get_val("llm", "rerank_model", "BAAI/bge-reranker-base")

    COLLECTION_NAME = _get_val("chromadb", "collection_name", "rag_collection")

    CHUNK_SIZE = int(_get_val("pipeline", "chunk_size", 1024))
    CHUNK_OVERLAP = int(_get_val("pipeline", "chunk_overlap", 128))
    SEMANTIC_THRESHOLD = float(_get_val("pipeline", "semantic_threshold", 0.5))

    RETRIEVAL_LIMIT = int(_get_val("retrieval", "limit", 5))
    RETRIEVAL_TOP_K = int(_get_val("retrieval", "top_k", 10))
    RERANK_MIN_SCORE = float(_get_val("retrieval", "rerank_min_score", 0.0))
    SOURCES_DISPLAY = int(_get_val("retrieval", "sources_display", 3))
    SOURCES_MIN_SCORE = float(_get_val("retrieval", "sources_min_score", 0.3))
    RRF_K = int(_get_val("retrieval", "rrf_k", 60))
    RRF_WEIGHT_SEMANTIC = float(_get_val("retrieval", "rrf_weight_semantic", 0.7))
    RRF_WEIGHT_BM25 = float(_get_val("retrieval", "rrf_weight_bm25", 0.3))
