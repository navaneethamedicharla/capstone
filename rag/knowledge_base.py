"""
Knowledge base – manages the FAISS vector store lifecycle:
build, save, load, and query operations.
The Research Agent calls query_knowledge_base() to search uploaded docs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import rag_config
from rag.chunker import chunk_documents
from rag.document_loader import load_document, load_documents_from_directory
from rag.embeddings import get_embeddings
from rag.retriever import similarity_search

logger = logging.getLogger(__name__)

_KB_INDEX_DIR = rag_config.vectorstore_path


def build_knowledge_base(
    file_paths: Optional[List[Path]] = None,
    directory: Optional[Path] = None,
    persist: bool = True,
) -> Optional[Any]:
    """
    Build a FAISS vector store from documents.

    Args:
        file_paths: Optional list of individual file paths.
        directory: Optional directory to scan for documents.
        persist: Whether to save the index to disk.

    Returns:
        FAISS vectorstore instance, or None on failure.
    """
    from langchain_community.vectorstores import FAISS

    all_docs = []
    if file_paths:
        for p in file_paths:
            all_docs.extend(load_document(Path(p)))
    if directory:
        all_docs.extend(load_documents_from_directory(Path(directory)))

    if not all_docs:
        logger.warning("No documents provided for knowledge base build.")
        return None

    chunks = chunk_documents(all_docs)
    if not chunks:
        logger.warning("Chunking produced no output.")
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if persist:
        _KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(_KB_INDEX_DIR))
        logger.info("FAISS index saved to %s (%d chunks)", _KB_INDEX_DIR, len(chunks))

    return vectorstore


def load_knowledge_base() -> Optional[Any]:
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS vectorstore or None if not found.
    """
    index_file = _KB_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        logger.info("No saved FAISS index found at %s", _KB_INDEX_DIR)
        return None

    try:
        from langchain_community.vectorstores import FAISS

        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(_KB_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", _KB_INDEX_DIR)
        return vectorstore
    except Exception as exc:
        logger.error("Failed to load FAISS index: %s", exc)
        return None


def query_knowledge_base(
    query: str,
    top_k: int = 5,
    vectorstore=None,
) -> List[Dict[str, Any]]:
    """
    Query the knowledge base for relevant chunks.

    Args:
        query: Search query string.
        top_k: Number of chunks to return.
        vectorstore: Optional pre-loaded vectorstore (loads from disk if None).

    Returns:
        List of dicts with keys: text, score, metadata.
    """
    if not rag_config.enabled:
        return []

    vs = vectorstore or load_knowledge_base()
    if vs is None:
        return []

    results = similarity_search(vs, query, top_k=top_k)
    logger.debug("RAG query '%s' returned %d chunks", query[:60], len(results))
    return results


def is_knowledge_base_available() -> bool:
    """Check whether a saved FAISS index exists on disk."""
    return (_KB_INDEX_DIR / "index.faiss").exists()
