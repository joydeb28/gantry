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

Thread-safety note:
    LlamaIndex ≤0.14 uses ``llama_index.core.Settings`` as a process-level
    singleton.  Mutating ``Settings.embed_model`` from two threads simultaneously
    causes a race condition where each retriever may end up using the wrong
    embedding model.

    **This file never touches ``Settings`` at all.**  Instead, the
    ``HuggingFaceEmbedding`` instance is constructed locally and passed
    directly to ``VectorStoreIndex(embed_model=...)``,
    ``VectorStoreIndex.from_documents(..., embed_model=...)``, and
    ``load_index_from_storage(..., embed_model=...)``.  Each retriever owns
    its own embedding object — no shared mutable global state.
"""

from __future__ import annotations

import hashlib
import json
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

    Cache invalidation:
        A SHA-256 fingerprint of the KB directory (file names + mtimes + sizes)
        is stored alongside the index.  If the fingerprint changes (files added,
        removed, or modified), the cache is automatically busted and rebuilt.

    Thread-safety:
        The ``HuggingFaceEmbedding`` instance is constructed locally and passed
        directly to LlamaIndex constructors — ``Settings`` is never mutated.
        Multiple ``KnowledgeBaseRetriever`` instances can be initialised
        concurrently without racing each other.

    Args:
        kb_path:  Path to the directory of Markdown KB files.
        use_case: Name of the use case (used for cache directory naming).
        top_k:    Number of documents to retrieve per query.
    """

    _EMBED_MODEL = "BAAI/bge-small-en-v1.5"
    _CACHE_BASE = Path(".llama_index_cache")
    _FINGERPRINT_FILE = "kb_fingerprint.json"

    def __init__(self, kb_path: str | Path, use_case: str, top_k: int = 3) -> None:
        from llama_index.core import (
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Build the embedding model locally — never mutate the global Settings singleton.
        # This makes concurrent retriever construction safe.
        embed_model = HuggingFaceEmbedding(model_name=self._EMBED_MODEL)

        cache_dir = self._CACHE_BASE / use_case
        kb_path = Path(kb_path)

        current_fp = self._compute_fingerprint(kb_path)
        cached_fp = self._load_cached_fingerprint(cache_dir)

        if cache_dir.exists() and current_fp == cached_fp:
            logger.info("Loading cached index for '%s' from %s", use_case, cache_dir)
            sc = StorageContext.from_defaults(persist_dir=str(cache_dir))
            # Pass embed_model directly — does NOT touch Settings
            index = load_index_from_storage(sc, embed_model=embed_model)
        else:
            if cache_dir.exists() and current_fp != cached_fp:
                logger.info(
                    "KB fingerprint changed for '%s' — rebuilding index "
                    "(cached=%s, current=%s)",
                    use_case, cached_fp[:8] if cached_fp else "none", current_fp[:8],
                )
            else:
                logger.info("Building index for '%s' from %s", use_case, kb_path)

            docs = SimpleDirectoryReader(
                input_dir=str(kb_path),
                required_exts=[".md"],
                recursive=False,
            ).load_data()
            # Pass embed_model directly — does NOT touch Settings
            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            index.storage_context.persist(persist_dir=str(cache_dir))
            self._save_fingerprint(cache_dir, current_fp)
            logger.info("Index persisted to %s (fingerprint=%s)", cache_dir, current_fp[:8])

        self._retriever = index.as_retriever(similarity_top_k=top_k)
        self._use_case = use_case

    # ------------------------------------------------------------------
    # Cache fingerprinting (O-7)
    # ------------------------------------------------------------------

    @classmethod
    def _compute_fingerprint(cls, kb_path: Path) -> str:
        """SHA-256 fingerprint of the KB directory contents.

        Hashes the sorted list of (filename, mtime_ns, size_bytes) tuples
        for all ``.md`` files in ``kb_path``.  Any add, remove, or modify
        operation changes the fingerprint and triggers a cache rebuild.
        """
        entries = sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in kb_path.glob("*.md")
            if p.is_file()
        )
        payload = json.dumps(entries, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _load_cached_fingerprint(cls, cache_dir: Path) -> str | None:
        """Read the stored fingerprint from the cache directory, or None."""
        fp_file = cache_dir / cls._FINGERPRINT_FILE
        if fp_file.exists():
            try:
                return json.loads(fp_file.read_text())["fingerprint"]
            except (KeyError, json.JSONDecodeError, OSError):
                return None
        return None

    @classmethod
    def _save_fingerprint(cls, cache_dir: Path, fingerprint: str) -> None:
        """Persist the fingerprint alongside the index."""
        fp_file = cache_dir / cls._FINGERPRINT_FILE
        fp_file.write_text(json.dumps({"fingerprint": fingerprint}))

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

    Thread-safety:
        The ``HuggingFaceEmbedding`` instance is constructed locally and passed
        directly to LlamaIndex constructors — ``Settings`` is never mutated.

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
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Build the embedding model locally — never mutate the global Settings singleton.
        embed_model = HuggingFaceEmbedding(model_name=model_name)

        _titles = titles if titles and len(titles) == len(documents) else [
            f"doc_{i}" for i in range(len(documents))
        ]

        nodes = [
            TextNode(text=text, metadata={"title": title, "source": "remote"})
            for text, title in zip(documents, _titles)
        ]

        # Pass embed_model directly — does NOT touch Settings
        index = VectorStoreIndex(nodes, embed_model=embed_model)
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
