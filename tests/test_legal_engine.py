import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "space"))
from legal_engine import Passage, format_references, keywords, reciprocal_rank_fusion, split_text


class BasicTests(unittest.TestCase):
    def test_chunks_and_overlap(self):
        self.assertEqual(split_text("", 10, 3), [])
        self.assertEqual(split_text("abcdefghijklm", 10, 3), ["abcdefghij", "hijklm"])

    def test_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            split_text("hello", 10, 10)

    def test_tokenizing_indonesian(self):
        self.assertEqual(keywords("Perizinan BERUSAHA?"), ["perizinan", "berusaha"])

    def test_rrf(self):
        results = reciprocal_rank_fusion([[0, 1], [1, 2]], [0.5, 0.5])
        self.assertEqual(results, [1, 0, 2])

    def test_references(self):
        passages = [Passage("one", "PP 35.pdf", 1), Passage("two", "PP 35.pdf", 1), Passage("three", "UU 6.pdf", 7)]
        self.assertEqual(format_references(passages), "- PP 35.pdf, halaman 1\n- UU 6.pdf, halaman 7")


if __name__ == "__main__":
    unittest.main()
