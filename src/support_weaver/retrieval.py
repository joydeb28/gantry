from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Evidence

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


class TinyRetriever:
    """A dependency-free BM25-ish retriever for examples and tests."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.doc_tokens = [tokenize(doc.text + " " + doc.title) for doc in docs]
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(docs), 1)
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

    @classmethod
    def from_markdown_dir(cls, path: str | Path) -> "TinyRetriever":
        docs: list[Document] = []
        for file_path in sorted(Path(path).glob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            title = next((line.removeprefix("#").strip() for line in text.splitlines() if line.startswith("#")), file_path.stem)
            docs.append(Document(source=str(file_path), title=title, text=text))
        return cls(docs)

    def search(self, query: str, limit: int = 3) -> tuple[Evidence, ...]:
        query_tokens = tokenize(query)
        scored: list[Evidence] = []
        for doc, tokens in zip(self.docs, self.doc_tokens):
            score = self._score(query_tokens, tokens)
            if score > 0:
                scored.append(Evidence(doc.source, doc.title, doc.text, round(score, 4)))
        return tuple(sorted(scored, key=lambda item: item.score, reverse=True)[:limit])

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        counts: dict[str, int] = {}
        for token in doc_tokens:
            counts[token] = counts.get(token, 0) + 1
        k1 = 1.2
        b = 0.75
        score = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            idf = math.log(1 + (len(self.docs) - self.df.get(token, 0) + 0.5) / (self.df.get(token, 0) + 0.5))
            tf = counts[token]
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / max(self.avg_len, 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score
