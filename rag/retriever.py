"""
Retriever – wraps the FAISS / Chroma vector store with a
similarity-search interface used by the Research Agent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import rag_config

logger = logging.getLogger(__name__)


def get_retriever(vectorstore, top_k: int = None):
    """
    Return a LangChain retriever from a vector store.

    Args:
        vectorstore: A FAISS or Chroma vector store instance.
        top_k: Number of documents to retrieve.

    Returns:
        LangChain VectorStoreRetriever.
    """
    k = top_k or rag_config.top_k
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def similarity_search(
    vectorstore,
    query: str,
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Run a similarity search and return plain dicts.

    Args:
        vectorstore: Initialized vector store.
        query: Search query string.
        top_k: Maximum number of results.

    Returns:
        List of dicts with keys: text, score, metadata.
    """
    k = top_k or rag_config.top_k
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        output = []
        for doc, score in results:
            output.append({
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            })
        return output
    except Exception as exc:
        logger.warning("Similarity search failed for query '%s': %s", query, exc)
        return []
