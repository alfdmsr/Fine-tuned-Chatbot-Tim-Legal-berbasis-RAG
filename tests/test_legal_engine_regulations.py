"""CPU-only regression tests for the fixed legal_engine.py.

Put this file under tests/test_legal_engine_regulations.py and execute:
    python3 -m unittest discover -s tests -v

No GPU, HF model download, or PDF needed for these tests.
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ENGINE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "space" / "legal_engine.py"
if not ENGINE_PATH.exists():
    ENGINE_PATH = Path(__file__).with_name("legal_engine_fixed.py")
spec = importlib.util.spec_from_file_location("legal_engine_tested", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)


class RegulationIdTests(unittest.TestCase):
    def test_supported_formats(self):
        examples = [
            ("PP Nomor 35 Tahun 2021", {("PP", "35", "2021")}),
            ("PP 35/2021", {("PP", "35", "2021")}),
            ("PP No. 35 Tahun 2021", {("PP", "35", "2021")}),
            ("Peraturan Pemerintah Nomor 35 Tahun 2021", {("PP", "35", "2021")}),
            ("UU Nomor 6 Tahun 2023.pdf", {("UU", "6", "2023")}),
            ("Undang-Undang No. 6 Tahun 2023", {("UU", "6", "2023")}),
            ("PP 35/2021 dan UU 6/2023", {("PP", "35", "2021"), ("UU", "6", "2023")}),
        ]
        for query, expected in examples:
            with self.subTest(query=query):
                self.assertEqual(engine.regulation_ids(query), expected)

    def test_no_explicit_regulation(self):
        self.assertEqual(engine.regulation_ids("Apa aturan mengenai lembur?"), set())


class SearchTests(unittest.TestCase):
    def test_scope_applied_before_reranking(self):
        import numpy as np

        retriever = engine.LegalRetriever.__new__(engine.LegalRetriever)
        retriever.np = np
        retriever.passages = [
            engine.Passage("PP 35 keterangan kerja lembur", "PP Nomor 35 Tahun 2021.pdf", 18),
            engine.Passage("Isi UU tentang hubungan kerja", "UU Nomor 6 Tahun 2023.pdf", 559),
            engine.Passage("Ketentuan perizinan", "PP Nomor 5 Tahun 2021.pdf", 328),
            engine.Passage("Lembur dan upah kerja PP 35", "PP Nomor 35 Tahun 2021.pdf", 17),
        ]
        retriever.source_regulations = [engine.regulation_ids(p.source) for p in retriever.passages]
        retriever.embeddings = np.array(
            [[0.1, 0.1], [0.9, 0.0], [0.8, 0.1], [0.1, 0.2]], dtype=np.float32
        )

        class FakeBM25:
            def get_scores(self, words):
                return np.array([0.1, 9, 8, 0.2])

        class FakeEmbedder:
            def encode(self, words, **kwargs):
                return np.array([[1, 0]], dtype=np.float32)

        class FakeReranker:
            def predict(self, pairs):
                return np.arange(len(pairs), dtype=np.float32)

        retriever.bm25 = FakeBM25()
        retriever.embedder = FakeEmbedder()
        retriever.reranker = FakeReranker()

        results = retriever.search(
            "Apa batas lembur berdasarkan PP 35/2021?",
            k=3,
            allowed_regulations={("PP", "35", "2021")},
        )
        self.assertTrue(results)
        self.assertTrue(all(p.source == "PP Nomor 35 Tahun 2021.pdf" for p in results))
        self.assertEqual(
            retriever.search("PP 99/2029", allowed_regulations={("PP", "99", "2029")}),
            [],
        )


class AnswerTests(unittest.TestCase):
    def test_answer_does_not_mix_regulations(self):
        class FakeRetriever:
            def __init__(self):
                self.last_allowed = None

            def search(self, question, k, allowed_regulations):
                self.last_allowed = allowed_regulations
                # Correctly simulated scoped retrieval result.
                if allowed_regulations == {("PP", "35", "2021")}:
                    return [engine.Passage("Waktu kerja lembur 4 jam per hari", "PP Nomor 35 Tahun 2021.pdf", 18)]
                return []

        class FakeGenerator:
            def generate(self, *args, **kwargs):
                return "Batas lembur adalah 4 jam per hari [1]."

        obj = engine.LegalRAG.__new__(engine.LegalRAG)
        obj.generator = FakeGenerator()
        obj._retriever = FakeRetriever()
        orig_hyde = os.environ.pop("ENABLE_HYDE", None)
        try:
            result = obj.answer("Apa batas lembur menurut PP Nomor 35 Tahun 2021?")
            self.assertEqual(obj.retriever.last_allowed, {("PP", "35", "2021")})
            self.assertIn("PP Nomor 35 Tahun 2021.pdf, halaman 18", result)
            self.assertNotIn("PP Nomor 5 Tahun 2021.pdf", result)
            self.assertNotIn("UU Nomor 6 Tahun 2023.pdf", result)
            no_result = obj.answer("Apa yang diatur PP 999/2099?")
            self.assertIn("Tidak ditemukan konteks", no_result)
        finally:
            if orig_hyde is not None:
                os.environ["ENABLE_HYDE"] = orig_hyde


if __name__ == "__main__":
    unittest.main()
