"""Hugging Face ZeroGPU / Gradio public legal RAG demo."""

# ZeroGPU patches CUDA at import; import it before torch or CUDA-touching packages.
import spaces
import logging
import os

import gradio as gr

from legal_engine import get_rag

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ZeroGPU requires eager CUDA-model registration at module import time.
# Do NOT move this into a decorated function: expensive load every cold request.
RAG = get_rag()

EXAMPLES = [
    "Apa ketentuan umum mengenai waktu kerja lembur?",
    "Apa itu perizinan berusaha berbasis risiko?",
    "Apa yang diatur dalam PP Nomor 51 Tahun 2023?",
]


@spaces.GPU(duration=120)
def chat(message: str, history: list[dict]) -> str:
    question = (message or "").strip()
    if not question:
        return "Silakan masukkan pertanyaan tentang peraturan yang tersedia."
    try:
        return RAG.answer(question)
    except Exception as exc:
        logging.exception("Gagal memproses pertanyaan")
        # Never expose local credentials, file paths, traceback or exception text to visitors.
        return (
            "Maaf, sistem belum dapat menjawab. Pastikan Space menggunakan ZeroGPU "
            "dan berkas model serta dokumen hukumnya berhasil dimuat. "
            "Pengelola dapat melihat log aplikasi untuk detail masalah."
        )


@spaces.GPU(duration=15)
def test_gpu():
    import torch

    x = torch.ones((4, 4), device="cuda")
    result = x @ x

    return (
        f"GPU: {torch.cuda.get_device_name(0)}\n"
        f"Calculation: {result[0, 0].item()}\n"
        "STATUS: GPU TEST PASSED"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Chatbot Tim Legal | RAG", analytics_enabled=False) as demo:
        gr.Markdown("# ⚖️ Chatbot Tim Legal — Fine-tuned Llama 3 + RAG")
        gr.Markdown(
            "Tanyakan isi dokumen hukum Indonesia. Jawaban dibantu pencarian dokumen dan "
            "menampilkan halaman PDF yang diambil sebagai konteks. **Bukan pengganti penasihat hukum.**"
        )

        gr.Markdown("### Diagnostik GPU")

        gpu_output = gr.Textbox(
            label="GPU Status",
            interactive=False,
        )

        gr.Button("Test GPU").click(
            fn=test_gpu,
            inputs=[],
            outputs=gpu_output,
        )

        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=480),
            examples=EXAMPLES,
            concurrency_limit=1,
        )
        gr.Markdown(
            "**Cakupan dokumen:** PP 35/2021, PP 5/2021, PP 51/2023, UU 6/2023. "
            "Informasi mungkin tidak mutakhir. Jangan masukkan informasi pribadi atau dokumen rahasia. "
            "Model yang dipublikasikan saat ini adalah checkpoint SFT, bukan bobot GRPO yang belum diunggah. "
            "ZeroGPU memiliki kuota harian dan antrean bersama."
        )
    return demo


if __name__ == "__main__":
    build_demo().queue(default_concurrency_limit=1, max_size=10).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
    )
