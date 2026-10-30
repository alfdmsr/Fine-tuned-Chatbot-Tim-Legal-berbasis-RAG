
import modal

app = modal.App("legal-rag-web-smoke-test")

image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install("fastapi>=0.115,<1")


@app.function(image=image, cpu=0.25, memory=512)
@modal.fastapi_endpoint(method="GET")
def hello():
    return {
        "status": "success",
        "message": "Modal Web Endpoint berhasil!",
    }
