# Penggunaan file tambahan ZeroGPU

Paket ini menggantikan arsip sebelumnya `legal-rag-public-demo.zip` yang masih ditujukan untuk GPU dedicated T4/L4. Gunakan **paket ZeroGPU ini**, jangan gabungkan dua versi file `app.py` dan `legal_engine.py`.

Salin seluruh isi ZIP ke root folder repository GitHub `Fine-tuned-Chatbot-Tim-Legal-berbasis-RAG`, lalu ikuti langkah di `DEPLOYMENT.md`.

**Perlu diketahui**: konfigurasi ini memerlukan pengujian end-to-end di HF ZeroGPU. Sukses sintaks/unit test bukan bukti berhasil memuat model di GPU. Koneksi GitHub yang tersedia tidak dapat melakukan commit ke repo (403), jadi perlu `git push` lewat akun GitHub pemilik.
