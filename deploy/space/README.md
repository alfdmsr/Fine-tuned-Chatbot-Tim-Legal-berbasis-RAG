---
title: Chatbot Tim Legal - Fine-tuned Llama 3 RAG
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12.12"
pinned: false
---

# ⚖️ Chatbot Tim Legal

Demo publik berbasis **model SFT** [`Saranjana/Llama-3-8b-Legal-RAG`](https://huggingface.co/Saranjana/Llama-3-8b-Legal-RAG) dengan retrieval hibrida (BM25 + multilingual embeddings + cross-encoder reranking) atas empat dokumen peraturan Indonesia. Dokumen ada di folder `Document RAG` yang disalin saat proses deploy.

**Gunakan Settings → Hardware → ZeroGPU (gratis untuk akun yang memenuhi syarat).** Aplikasi mengimpor `spaces` sebelum PyTorch, memuat Llama 8B secara eager pada module scope, lalu melakukan inferensi dalam `@spaces.GPU(duration=120)`. Pembatasan kuota dan antrian tetap berlaku. Kompatibilitas model & bitsandbytes di ZeroGPU harus dibuktikan dengan pengujian langsung, bukan hanya sintaks.

HyDE tersedia sebagai opsi dengan variable `ENABLE_HYDE=true` (lebih lambat). Web search fallback pada notebook **tidak digunakan** dalam demo publik ini agar tidak menyajikan konten web yang belum divalidasi sebagai referensi hukum resmi. Halaman PDF yang ditampilkan adalah hasil retrieval, **bukan validasi otomatis atas isi jawaban**.

Jangan kirim data pribadi/rahasia. Verifikasi status keberlakuan peraturan dan isi pasal pada sumber hukum resmi. Ini demo, bukan nasihat hukum.
