"""Additional CPU-only regression tests for retrieval context selection.

After copying legal_engine_context_fixed.py to deploy/space/legal_engine.py,
copy this file to tests/test_legal_context_selection.py, then run
  python3 -m unittest discover -s tests -v
"""
import importlib.util
from pathlib import Path
import sys
import unittest

ENGINE_PATH = Path(__file__).resolve().parents[1] / 'deploy' / 'space' / 'legal_engine.py'
if not ENGINE_PATH.exists():
    ENGINE_PATH = Path(__file__).with_name('legal_engine_context_fixed.py')
spec = importlib.util.spec_from_file_location('engine_context', ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)

class ContextSelectionTests(unittest.TestCase):
    def test_pas26_promoted_from_rank_four(self):
        source = 'PP Nomor 35 Tahun 2021.pdf'
        ranked = [
            engine.Passage('Perintah dan persetujuan pekerja yang bersedia bekerja lembur', source, 18),
            engine.Passage('Pekerja tidak dapat melakukan pekerjaan selama 6 bulan', source, 23),
            engine.Passage('Huruf b. Waktu kerja fleksibel untuk usaha tertentu', source, 49),
            engine.Passage('Bagian Ketiga Waktu Kerja Lembur Pasal 26: hanya dapat dilakukan paling lama 4 jam dalam satu hari dan 18 jam dalam satu minggu', source, 17),
            engine.Passage('Masa kerja dan hak pekerja', source, 27),
        ]
        selected = engine.select_context_passages('Apa ketentuan waktu kerja lembur berdasarkan PP Nomor 35 Tahun 2021?', ranked)
        self.assertEqual([p.page for p in selected], [17])

    def test_generic_query_preserves_reranker_order(self):
        p1 = engine.Passage('Ketentuan umum', 'PP Nomor 35 Tahun 2021.pdf', 18)
        p2 = engine.Passage('Lain-lain', 'PP Nomor 35 Tahun 2021.pdf', 17)
        self.assertEqual(engine.select_context_passages('Apa yang diatur PP Nomor 35 Tahun 2021?', [p1,p2]), [p1,p2])

    def test_multiple_phrase_matches_keep_reranker_order(self):
        q='Apa ketentuan waktu kerja lembur berdasarkan PP Nomor 35 Tahun 2021?'
        p1=engine.Passage('Waktu kerja lembur dan perintah tertulis', 'PP 35 Tahun 2021.pdf', 18)
        p2=engine.Passage('Masa kerja', 'PP 35 Tahun 2021.pdf', 24)
        p3=engine.Passage('Pasal 26. Waktu Kerja Lembur 4 jam sehari', 'PP 35 Tahun 2021.pdf', 17)
        self.assertEqual(engine.select_context_passages(q,[p1,p2,p3]),[p1,p3])

if __name__=='__main__': unittest.main()
