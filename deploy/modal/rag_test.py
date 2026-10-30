"""Uji Legal RAG lengkap di Modal: indeks CPU dahulu, inferensi GPU kemudian.

Lokasi yang diharapkan dalam repository:
    deploy/modal/rag_test.py

Perintah dari root repository:
    modal run deploy/modal/rag_test.py --mode index
    modal run deploy/modal/rag_test.py --mode answer

Tidak membuat endpoint publik. Sumber engine dari deploy/space/legal_engine.py
sehingga aplikasi HF ZeroGPU yang sudah ada tetap tidak berubah.
"""

from pathlib import Path

import modal

# Modal loads this module in two different layouts:
#   Local:  <repo>/deploy/modal/rag_test.py
#   Remote: /root/rag_test.py
# Resolve repository assets ONLY on the local machine; in Modal containers
# those assets have already been bundled into the Image under /app.
SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_REPO = (
    SCRIPT_PATH.parent.name == "modal"
    and SCRIPT_PATH.parent.parent.name == "deploy"
)

if IS_LOCAL_REPO:
    ROOT = SCRIPT_PATH.parents[2]
    ENGINE_FILE = ROOT / "deploy" / "space" / "legal_engine.py"
    DOCUMENT_DIR = ROOT / "Document RAG"

    if not ENGINE_FILE.is_file():
        raise FileNotFoundError(f"Tidak ditemukan: {ENGINE_FILE}")
    if not DOCUMENT_DIR.is_dir() or not list(DOCUMENT_DIR.glob("*.pdf")):
        raise FileNotFoundError(f"PDF hukum tidak ditemukan di: {DOCUMENT_DIR}")

app = modal.App("legal-rag-integration-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers>=5,<6",
        "accelerate>=1.3,<2",
        "bitsandbytes>=0.45,<1",
        "safetensors>=0.5,<1",
        "numpy>=1.26,<3",
        "pypdf>=5,<7",
        "rank-bm25>=0.2.2,<1",
        "sentence-transformers>=6.2,<7",
    )
    .env(
        {
            "PYTHONPATH": "/app",
            "HF_HOME": "/model-cache",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "LEGAL_MODEL_ID": "Saranjana/Llama-3-8b-Legal-RAG",
            "LEGAL_PDF_DIR": "/app/Document RAG",
            "LEGAL_CACHE_DIR": "/model-cache/rag-cache",
            "ENABLE_HYDE": "false",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# These paths exist only on the developer's computer. The same Python module
# is re-imported inside Modal containers, where source lives at /root/rag_test.py.
if IS_LOCAL_REPO:
    image = image.add_local_file(
        str(ENGINE_FILE), remote_path="/app/legal_engine.py", copy=True
    ).add_local_dir(
        str(DOCUMENT_DIR), remote_path="/app/Document RAG", copy=True
    )

# Reuse volume from first successful GPU test to reuse model downloads.
cache = modal.Volume.from_name("legal-rag-model-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/model-cache": cache},
    timeout=1800,
)
def prepare_rag_index() -> int:
    """Create embeddings + passages cache once on CPU, without allocating GPU."""
    from legal_engine import LegalRetriever, load_passages

    passages = load_passages()
    print(f"Jumlah potongan dokumen: {len(passages)}")
    retriever = LegalRetriever(passages)
    print(f"Embedding matrix: {retriever.embeddings.shape}")
    cache.commit()
    print("Cache indeks RAG berhasil disimpan ke Modal Volume.")
    return len(passages)


@app.cls(
    image=image,
    gpu=["L4", "A10"],
    cpu=2,
    memory=16384,
    volumes={"/model-cache": cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=15,
    timeout=900,
    startup_timeout=1200,
)
class LegalRAGWorker:
    @modal.enter()
    def load(self):
        import torch
        from legal_engine import get_rag

        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA tidak tersedia.")
        print("GPU:", torch.cuda.get_device_name(0))
        self.rag = get_rag()  # Generator loads fine-tuned SFT checkpoint on GPU.
        # Warm up CPU retrieval and load embeddings cached in prepare_rag_index().
        self.rag.retriever
        cache.commit()
        print("Legal RAG siap menerima pertanyaan pengujian.")

    @modal.method()
    def answer(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            return "Pertanyaan tidak boleh kosong."
        return self.rag.answer(question.strip())


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    volumes={"/model-cache": cache},
    timeout=900,
)
def inspect_retrieval(question: str):
    from legal_engine import LegalRetriever, load_passages, regulation_ids

    retriever = LegalRetriever(load_passages())
    allowed = regulation_ids(question) or None

    results = retriever.search(
        question,
        k=10,
        allowed_regulations=allowed,
    )

    print("\n=== TOP 10 RETRIEVAL ===")

    for rank, passage in enumerate(results, 1):
        preview = " ".join(passage.text.split())[:300]
        print(
            f"\nRank {rank}"
            f"\nDokumen: {passage.source}"
            f"\nHalaman PDF: {passage.page}"
            f"\nCuplikan: {preview}"
        )

@app.local_entrypoint()
def main(mode: str = "index"):
    if mode == "index":
        count = prepare_rag_index.remote()
        print(f"\nINDEX BERHASIL: {count} passages. Berikutnya --mode answer")
    elif mode == "answer":
        question = "Apa ketentuan umum mengenai waktu kerja lembur berdasarkan PP Nomor 35 Tahun 2021?"
        print("\nPERTANYAAN:", question)
        result = LegalRAGWorker().answer.remote(question)
        print("\n=== JAWABAN LEGAL RAG ===\n", result)
    elif mode == "debug":
        question = (
            "Apa ketentuan waktu kerja lembur "
            "berdasarkan PP Nomor 35 Tahun 2021?"
        )
        inspect_retrieval.remote(question)
    else:
        raise ValueError("Mode harus 'index' atau 'answer'.")