from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import Evidence

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Document:
    source: str
    title: str
    text: str


class TinyRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self._tokens: list[list[str]] = [tokenize(d.text) for d in docs]
        self._avg_len: float = (
            sum(len(t) for t in self._tokens) / len(self._tokens) if self._tokens else 1.0
        )
        self._df: dict[str, int] = {}
        for token_list in self._tokens:
            for tok in set(token_list):
                self._df[tok] = self._df.get(tok, 0) + 1

    @classmethod
    def from_markdown_dir(cls, path: str | Path) -> "TinyRetriever":
        path = Path(path)
        docs: list[Document] = []
        if path.exists():
            for md_file in sorted(path.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                lines = text.splitlines()
                title = next(
                    (line.lstrip("# ").strip() for line in lines if line.startswith("#")),
                    md_file.stem,
                )
                docs.append(Document(source=str(md_file), title=title, text=text))
        return cls(docs)

    def search(self, query: str, limit: int = 3) -> tuple[Evidence, ...]:
        if not self._docs:
            return ()
        q_tokens = tokenize(query)
        scored = sorted(
            enumerate(self._docs),
            key=lambda pair: self._score(q_tokens, self._tokens[pair[0]]),
            reverse=True,
        )
        results = []
        for idx, doc in scored[:limit]:
            score = self._score(q_tokens, self._tokens[idx])
            if score > 0:
                results.append(Evidence(source=doc.source, title=doc.title, text=doc.text, score=score))
        return tuple(results)

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        n = len(self._docs)
        dl = len(doc_tokens)
        k1, b = 1.5, 0.75
        score = 0.0
        tf_map: dict[str, int] = {}
        for tok in doc_tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            df = self._df.get(tok, 1)
            import math
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / self._avg_len))
            score += idf * norm_tf
        return score
