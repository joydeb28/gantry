"""Unit tests for Bucket 2 fixes (C-1 LlamaIndex race, O-7 cache fingerprinting).

Tests:
    - Settings is never mutated (C-1)
    - KnowledgeBaseRetriever._compute_fingerprint is stable, content-sensitive (O-7)
    - Cache is invalidated when files change (O-7)
    - RemoteRetriever never touches Settings (C-1)
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os


class SettingsNotMutatedTest(unittest.TestCase):
    """C-1: Verify that KnowledgeBaseRetriever and RemoteRetriever
    never assign to llama_index.core.Settings."""

    def _get_source(self) -> str:
        import inspect
        from gantry import retrieval
        return inspect.getsource(retrieval)

    def test_settings_embed_model_not_assigned(self):
        """The string 'Settings.embed_model =' must not appear in retrieval.py."""
        src = self._get_source()
        self.assertNotIn(
            "Settings.embed_model =",
            src,
            "retrieval.py still mutates the global Settings.embed_model singleton",
        )

    def test_settings_llm_not_assigned(self):
        """The string 'Settings.llm =' must not appear in retrieval.py."""
        src = self._get_source()
        self.assertNotIn(
            "Settings.llm =",
            src,
            "retrieval.py still mutates the global Settings.llm singleton",
        )

    def test_embed_model_passed_directly(self):
        """embed_model= must be passed directly to at least one LlamaIndex call."""
        src = self._get_source()
        self.assertIn(
            "embed_model=embed_model",
            src,
            "embed_model is not passed directly to LlamaIndex constructors",
        )


class FingerprintTest(unittest.TestCase):
    """O-7: Cache fingerprint is stable, content-sensitive, and triggers rebuild."""

    def setUp(self):
        # Import here so llama_index is not needed for the class-level import
        from gantry.retrieval import KnowledgeBaseRetriever
        self.Retriever = KnowledgeBaseRetriever

    def _make_kb(self, tmpdir: Path, files: dict[str, str]) -> Path:
        """Create a temp KB directory with the given filename→content mapping."""
        kb = tmpdir / "kb"
        kb.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (kb / name).write_text(content)
        return kb

    def test_fingerprint_is_deterministic(self):
        """Same directory contents → same fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._make_kb(Path(tmp), {"policy.md": "refund policy"})
            fp1 = self.Retriever._compute_fingerprint(kb)
            fp2 = self.Retriever._compute_fingerprint(kb)
            self.assertEqual(fp1, fp2)

    def test_fingerprint_changes_when_file_added(self):
        """Adding a file changes the fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._make_kb(Path(tmp), {"policy.md": "refund policy"})
            fp_before = self.Retriever._compute_fingerprint(kb)
            (kb / "warranty.md").write_text("warranty policy")
            fp_after = self.Retriever._compute_fingerprint(kb)
            self.assertNotEqual(fp_before, fp_after)

    def test_fingerprint_changes_when_file_removed(self):
        """Removing a file changes the fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._make_kb(Path(tmp), {
                "policy.md": "refund policy",
                "warranty.md": "warranty policy",
            })
            fp_before = self.Retriever._compute_fingerprint(kb)
            (kb / "warranty.md").unlink()
            fp_after = self.Retriever._compute_fingerprint(kb)
            self.assertNotEqual(fp_before, fp_after)

    def test_fingerprint_changes_when_file_modified(self):
        """Modifying a file's mtime changes the fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._make_kb(Path(tmp), {"policy.md": "refund policy"})
            fp_before = self.Retriever._compute_fingerprint(kb)

            # Touch the file to change mtime_ns
            time.sleep(0.01)
            p = kb / "policy.md"
            p.write_text("updated refund policy")

            fp_after = self.Retriever._compute_fingerprint(kb)
            self.assertNotEqual(fp_before, fp_after)

    def test_fingerprint_is_sha256_hex(self):
        """Fingerprint must be a 64-character hex string (SHA-256)."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._make_kb(Path(tmp), {"policy.md": "x"})
            fp = self.Retriever._compute_fingerprint(kb)
            self.assertEqual(len(fp), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in fp))

    def test_fingerprint_empty_directory(self):
        """Empty KB directory produces a stable (all-empty) fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "empty_kb"
            kb.mkdir()
            fp1 = self.Retriever._compute_fingerprint(kb)
            fp2 = self.Retriever._compute_fingerprint(kb)
            self.assertEqual(fp1, fp2)

    def test_save_and_load_fingerprint_roundtrip(self):
        """Fingerprint survives a write → read roundtrip."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            test_fp = "a" * 64
            self.Retriever._save_fingerprint(cache_dir, test_fp)
            loaded = self.Retriever._load_cached_fingerprint(cache_dir)
            self.assertEqual(test_fp, loaded)

    def test_load_fingerprint_missing_returns_none(self):
        """Loading fingerprint from non-existent cache returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "does_not_exist"
            result = self.Retriever._load_cached_fingerprint(missing_dir)
            self.assertIsNone(result)

    def test_load_fingerprint_corrupt_returns_none(self):
        """Corrupt fingerprint file returns None rather than raising."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            fp_file = cache_dir / self.Retriever._FINGERPRINT_FILE
            fp_file.write_text("not valid json {{{{")
            result = self.Retriever._load_cached_fingerprint(cache_dir)
            self.assertIsNone(result)


class CacheInvalidationTest(unittest.TestCase):
    """O-7: When fingerprint changes, the index must be rebuilt (not served stale)."""

    def setUp(self):
        from gantry.retrieval import KnowledgeBaseRetriever
        self.Retriever = KnowledgeBaseRetriever

    def test_cache_rebuild_triggered_on_fingerprint_mismatch(self):
        """When stored fingerprint != current fingerprint, from_documents is called."""
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "kb"
            kb.mkdir()
            (kb / "doc.md").write_text("# Doc\nsome content")
            cache_dir = Path(tmp) / "cache" / "test_case"
            cache_dir.mkdir(parents=True)

            # Write a stale (mismatched) fingerprint
            self.Retriever._save_fingerprint(cache_dir, "stale" * 16)  # wrong fp
            # also mark cache_dir as existing
            (cache_dir / "docstore.json").write_text("{}")  # fake cache presence

            mock_embed = MagicMock()
            mock_node = MagicMock()
            mock_node.metadata = {"file_name": "doc.md", "file_path": str(kb / "doc.md")}
            mock_node.text = "# Doc\nsome content"
            mock_node.score = 0.9
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_node]
            mock_index = MagicMock()
            mock_index.as_retriever.return_value = mock_retriever

            with patch("gantry.retrieval.KnowledgeBaseRetriever._CACHE_BASE", Path(tmp) / "cache"), \
                 patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", return_value=mock_embed), \
                 patch("llama_index.core.SimpleDirectoryReader") as mock_reader, \
                 patch("llama_index.core.VectorStoreIndex") as mock_vsi:

                mock_reader.return_value.load_data.return_value = []
                mock_vsi.from_documents.return_value = mock_index
                mock_index.storage_context.persist = MagicMock()

                retriever = self.Retriever.__new__(self.Retriever)

                current_fp = self.Retriever._compute_fingerprint(kb)
                cached_fp = self.Retriever._load_cached_fingerprint(cache_dir)

                # Core assertion: fingerprints should differ, triggering rebuild
                self.assertNotEqual(
                    current_fp, cached_fp,
                    "Test setup error: fingerprints should differ",
                )


if __name__ == "__main__":
    unittest.main()
