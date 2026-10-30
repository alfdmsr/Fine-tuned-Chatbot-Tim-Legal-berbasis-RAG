
import modal

app = modal.App("legal-rag-gradio-smoke-test")

web_image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install(
    "fastapi>=0.115,<1",
    "gradio>=5,<7",
)


@app.function(
    image=web_image,
    cpu=0.5,
    memory=1024,
    max_containers=1,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    import gradio as gr

    with gr.Blocks(title="Legal RAG Test") as demo:
        gr.Markdown("# Legal RAG - Modal Test")
        gr.Markdown("Pengujian antarmuka tanpa GPU.")

        question = gr.Textbox(
            label="Pertanyaan",
            placeholder="Masukkan pertanyaan..."
        )
        answer = gr.Textbox(label="Jawaban")

        gr.Button("Kirim").click(
            fn=lambda q: f"Pertanyaan diterima: {q}",
            inputs=question,
            outputs=answer,
        )

    return gr.mount_gradio_app(
        FastAPI(),
        demo,
        path="/",
    )
