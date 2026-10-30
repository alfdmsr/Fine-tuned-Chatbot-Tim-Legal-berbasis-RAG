"""Pratinjau Gradio Legal RAG di Modal (frontend CPU terpisah dari backend GPU).

Persiapan dari root repo:
    modal deploy deploy/modal/rag_test.py

Simpan file ini sebagai deploy/modal/public_preview.py, lalu jalankan:
    modal serve deploy/modal/public_preview.py

PERINGATAN: URL -dev ini dapat diakses tanpa login. Gunakan hanya untuk uji
terbatas; perlindungan sesi berbasis memori ini bukan rate limit produksi.
"""

import modal

app = modal.App("legal-rag-public-preview")

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gradio>=5,<7",
    "fastapi>=0.115,<1",
)

BACKEND_APP = "legal-rag-integration-test"
BACKEND_CLASS = "LegalRAGWorker"


@app.function(
    image=web_image,
    cpu=0.5,
    memory=1024,
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
    timeout=900,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    """Publikasikan antarmuka Gradio; model GPU ada di App Modal terpisah."""
    from collections import defaultdict
    from threading import Lock
    from time import monotonic

    from fastapi import FastAPI
    import gradio as gr

    sessions = defaultdict(list)
    lock = Lock()

    def answer(message: str, history: list, request: gr.Request) -> str:
        question = (message or "").strip()
        if not question:
            return "Silakan tulis pertanyaan tentang peraturan Indonesia."
        if len(question) > 1000:
            return "Pertanyaan maksimal 1.000 karakter."

        # Preview-only: dapat direset saat container CPU restart.
        session_id = getattr(request, "session_hash", None) or "anonymous"
        current_time = monotonic()
        with lock:
            recent = [t for t in sessions[session_id] if current_time - t < 3600]
            if len(recent) >= 2:
                return "Batas uji sementara: 2 pertanyaan per sesi setiap jam."
            sessions[session_id] = recent + [current_time]

        try:
            # Hanya kelas dari deployment internal yang sudah di-deploy.
            Worker = modal.Cls.from_name(BACKEND_APP, BACKEND_CLASS)
            return Worker().answer.remote(question)
        except Exception as exc:
            # Tidak menampilkan traceback / identitas internal kepada pengunjung.
            print(f"Gagal menghubungi backend Legal RAG: {type(exc).__name__}")
            return (
                "Maaf, layanan belum dapat menjawab. Periksa log backend "
                "Modal atau ketersediaan GPU, lalu coba lagi."
            )

    with gr.Blocks(title="Legal RAG — Pratinjau Modal", analytics_enabled=False) as demo:
        gr.Markdown("# ⚖️ Legal RAG — Fine-tuned Llama-3-8B")
        gr.Markdown(
            "Pratinjau akademis berdasarkan PP 35/2021, PP 5/2021, "
            "PP 51/2023, dan UU 6/2023. **Bukan nasihat hukum.** "
            "Jangan memasukkan informasi pribadi atau rahasia."
        )
        gr.ChatInterface(
            fn=answer,
            examples=[
                "Apa ketentuan waktu kerja lembur berdasarkan PP Nomor 35 Tahun 2021?",
                "Apa yang dimaksud perizinan berusaha berbasis risiko?",
            ],
            concurrency_limit=1,
        )
        gr.Markdown(
            "*Demo pengujian sementara. Jawaban pertama mungkin lambat karena "
            "cold start GPU. Sitasi masih perlu diperiksa sebelum rilis publik.*"
        )
    demo.queue(default_concurrency_limit=1, max_size=5)

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.get("/health", include_in_schema=False)
    def health():
        return {"status": "ok", "service": "legal-rag-public-preview", "gpu_started": False}

    return gr.mount_gradio_app(api, demo, path="/")