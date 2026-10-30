"""Uji inferensi fine-tuned Llama-3-8B di Modal (belum API publik/RAG)."""

import modal

MODEL_ID = "Saranjana/Llama-3-8b-Legal-RAG"
app = modal.App("legal-rag-model-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers>=5,<6",
        "accelerate>=1.3,<2",
        "bitsandbytes>=0.45,<1",
        "safetensors>=0.5,<1",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)

cache = modal.Volume.from_name("legal-rag-model-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=["L4", "A10"],
    cpu=2,
    memory=16384,
    volumes={"/model-cache": cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=15,
    timeout=600,
    startup_timeout=1200,
)
class LegalModel:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA tidak tersedia di container Modal.")

        print("GPU:", torch.cuda.get_device_name(0))
        print("Memuat model:", MODEL_ID)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        self.model.eval()
        cache.commit()  # Simpan unduhan model untuk cold start berikutnya.
        print("Model berhasil dimuat.")

    @modal.method()
    def generate(self, question: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": "Jawab singkat dalam Bahasa Indonesia."},
            {"role": "user", "content": question},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


@app.local_entrypoint()
def main():
    answer = LegalModel().generate.remote("Perkenalkan dirimu dalam satu kalimat.")
    print("\n=== HASIL INFERENSI ===\n", answer)
