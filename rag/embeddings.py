"""
Embeddings – provides a LangChain embedding function.
Uses HuggingFace sentence-transformers (local, no API cost) by default,
with an OpenAI embedding fallback if configured.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import OPENAI_API_KEY, rag_config

logger = logging.getLogger(__name__)


def get_embeddings():
    """
    Return a LangChain-compatible embedding function.

    Tries HuggingFace embeddings first (local, free).
    Falls back to OpenAI embeddings if OPENAI_API_KEY is set and
    sentence-transformers is not installed.

    Returns:
        A LangChain Embeddings instance.
    """
    # Primary: HuggingFace sentence-transformers (local, no API cost)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=rag_config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Using HuggingFace embeddings: %s", rag_config.embedding_model)
        return embeddings
    except ImportError:
        logger.warning("langchain_huggingface not available, trying OpenAI embeddings")
    except Exception as exc:
        logger.warning("HuggingFace embeddings failed: %s. Trying OpenAI.", exc)

    # Fallback: OpenAI embeddings
    if OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=OPENAI_API_KEY,
            )
            logger.info("Using OpenAI embeddings: text-embedding-3-small")
            return embeddings
        except Exception as exc:
            logger.warning("OpenAI embeddings failed: %s", exc)

    raise RuntimeError(
        "No embeddings provider available. "
        "Install sentence-transformers: pip install sentence-transformers langchain-huggingface"
    )
