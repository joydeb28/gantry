"""LlamaIndex-powered knowledge base retriever + retriever protocol.

Architecture:
    BaseRetriever       : Protocol defining the search(query) → Evidence interface.
    KnowledgeBaseRetriever : FileSystem implementation — reads Markdown files,
                             embeds with BAAI/bge-small-en-v1.5 (offline),
                             caches to disk.
    RemoteRetriever     : In-memory implementation — accepts raw document texts
                          at construction time. Useful for piping in content from
                          Confluence, Notion, SharePoint, or any external system.

Usage::

    # Default — filesystem KB
    retriever = KnowledgeBaseRetriever("examples/support/kb", use_case="support")
    docs = retriever.search("customer wants a refund")
    # Returns tuple[Evidence, ...] sorted by semantic relevance

    # Remote / in-memory KB
    retriever = RemoteRetriever(documents=["Policy A text...", "Policy B text..."])
    docs = retriever.search("refund policy")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import Evidence

logger = logging.getLogger(__name__)

# Suppress noisy LlamaIndex / tokeniser warnings
logging.getLogger("llama_index").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# BaseRetriever protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseRetriever(Protocol):
    """Protocol that every retriever implementation must satisfy.

    Any object with a ``search(query) → tuple[Evidence, ...]`` method
    is a valid retriever — no inheritance required.

    This enables callers to swap in:
        - ``KnowledgeBaseRetriever`` (file-system, offline, cached)
        - ``RemoteRetriever`` (in-memory, pre-loaded document texts)
        - Any custom retriever (e.g. wrapping Elasticsearch, Pinecone, Weaviate)
    """

    def search(self, query: str) -> tuple[Evidence, ...]:
        """Return ranked Evidence objects for the query."""
        ...


# ---------------------------------------------------------------------------
# File-System Retriever (LlamaIndex + HuggingFace)
# ---------------------------------------------------------------------------

class KnowledgeBaseRetriever:
    """Semantic vector-search retriever backed by LlamaIndex.

    On first run for a use case: loads markdown files, embeds them with
    BAAI/bge-small-en-v1.5 (local, no API key), and persists the index.

    On subsequent runs: loads the cached index from disk instantly.

    Args:
        kb_path:  Path to the directory of Markdown KB files.
        use_case: Name of the use case (used for cache directory naming).
        top_k:    Number of documents to retrieve per query.
    """

    _EMBED_MODEL = "BAAI/bge-small-en-v1.5"
    _CACHE_BASE = Path(".llama_index_cache")

    def __init__(self, kb_path: str | Path, use_case: str, top_k: int = 3) -> None:
        from llama_index.core import (
            Settings,
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # MUST set embed_model before any index operation;
        # default is OpenAI text-embedding-ada-002 which requires an API key.
        Settings.embed_model = HuggingFaceEmbedding(model_name=self._EMBED_MODEL)
        Settings.llm = None  # pure retrieval — no LLM needed in index operations

        cache_dir = self._CACHE_BASE / use_case
        kb_path = Path(kb_path)

        if cache_dir.exists():
            logger.info("Loading cached index for '%s' from %s", use_case, cache_dir)
            sc = StorageContext.from_defaults(persist_dir=str(cache_dir))
            index = load_index_from_storage(sc)
        else:
            logger.info("Building index for '%s' from %s", use_case, kb_path)
            docs = SimpleDirectoryReader(
                input_dir=str(kb_path),
                required_exts=[".md"],
                recursive=False,
            ).load_data()
            index = VectorStoreIndex.from_documents(docs)
            index.storage_context.persist(persist_dir=str(cache_dir))
            logger.info("Index persisted to %s", cache_dir)

        self._retriever = index.as_retriever(similarity_top_k=top_k)
        self._use_case = use_case

    def search(self, query: str) -> tuple[Evidence, ...]:
        """Return top-k semantically relevant Evidence objects for the query."""
        nodes = self._retriever.retrieve(query)
        evidence_list = []
        for n in nodes:
            title = str(n.metadata.get("file_name", ""))
            lines = n.text.strip().split("\n")
            if lines and lines[0].startswith("#"):
                title = lines[0].lstrip("#").strip()

            evidence_list.append(
                Evidence(
                    source=str(n.metadata.get("file_path", "")),
                    title=title,
                    text=n.text,
                    score=float(n.score or 0.0),
                )
            )
        return tuple(evidence_list)

    @classmethod
    def from_use_case(cls, use_case: str, kb_root: str | Path = "examples") -> "KnowledgeBaseRetriever":
        """Convenience constructor: build retriever from use-case name + root dir."""
        kb_path = Path(kb_root) / use_case / "kb"
        return cls(kb_path=kb_path, use_case=use_case)


# ---------------------------------------------------------------------------
# Remote / In-Memory Retriever
# ---------------------------------------------------------------------------

class RemoteRetriever:
    """In-memory retriever for externally provided document texts.

    Useful when KB content comes from an external system (Confluence, Notion,
    SharePoint, a REST API, etc.) rather than local Markdown files.

    Builds a LlamaIndex in-memory vector index from the supplied document
    strings using the same offline embedding model as ``KnowledgeBaseRetriever``.

    Args:
        documents:   List of document text strings to index.
        titles:      Optional list of document titles (same length as documents).
                     If omitted, titles default to "doc_0", "doc_1", etc.
        top_k:       Number of documents to retrieve per query. Default: 3.
        model_name:  HuggingFace embedding model. Default: bge-small-en-v1.5.

    Example::

        pages = fetch_confluence_pages(space="HR")
        retriever = RemoteRetriever(
            documents=[p["body"] for p in pages],
            titles=[p["title"] for p in pages],
        )
        # Drop this retriever straight into any weaver that accepts a retriever.
    """

    _EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        documents: list[str],
        titles: list[str] | None = None,
        top_k: int = 3,
        model_name: str = _EMBED_MODEL,
    ) -> None:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
        Settings.llm = None

        _titles = titles if titles and len(titles) == len(documents) else [
            f"doc_{i}" for i in range(len(documents))
        ]

        nodes = [
            TextNode(text=text, metadata={"title": title, "source": "remote"})
            for text, title in zip(documents, _titles)
        ]

        index = VectorStoreIndex(nodes)
        self._retriever = index.as_retriever(similarity_top_k=top_k)
        logger.info("RemoteRetriever: indexed %d documents in memory.", len(documents))

    def search(self, query: str) -> tuple[Evidence, ...]:
        """Return top-k semantically relevant Evidence objects for the query."""
        nodes = self._retriever.retrieve(query)
        return tuple(
            Evidence(
                source="remote",
                title=str(n.metadata.get("title", "unknown")),
                text=n.text,
                score=float(n.score or 0.0),
            )
            for n in nodes
        )
