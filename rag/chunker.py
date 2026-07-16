"""
Chunker – splits LangChain Documents into smaller overlapping chunks
for vector store ingestion using RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import rag_config

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Characters per chunk (defaults to rag_config.chunk_size).
        chunk_overlap: Overlap between chunks (defaults to rag_config.chunk_overlap).

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    size = chunk_size or rag_config.chunk_size
    overlap = chunk_overlap or rag_config.chunk_overlap

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d document(s) into %d chunks (size=%d, overlap=%d)",
        len(documents), len(chunks), size, overlap,
    )
    return chunks
