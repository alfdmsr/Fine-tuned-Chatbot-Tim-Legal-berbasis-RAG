# Deploy Legal RAG ke Hugging Face ZeroGPU (target Rp0)

**Status:** paket ini disiapkan untuk ZeroGPU, **belum tervalidasi pada GPU Hugging Face milik pengguna**. Selain itu, koneksi GitHub yang tersedia tidak dapat menulis langsung ke repo (403). Tidak ada URL demo yang sudah aktif.

## 1. Persyaratan

- Akun Hugging Face personal gratis, email terverifikasi, umur >30 hari, good standing; maksimum 2 ZeroGPU Space pada akun gratis. Lihat https://huggingface.co/docs/hub/spaces-zerogpu
- Model SFT publik: https://huggingface.co/Saranjana/Llama-3-8b-Legal-RAG (bukan bobot GRPO yang hanya disimpan lokal). Tidak perlu training ulang.
- Pakai GPU **ZeroGPU** saja; **jangan pilih GPU T4/L4 dedicated berbayar** kalau anggaran Rp0.
- Pengunjung tanpa login mendapat kuota GPU ~2 menit per hari, akun Free ~5 menit per hari, dengan antrean bersama. Kuota ini milik pengunjung dan dapat membatasi jumlah pertanyaan.
- Dokumen berisi peraturan dari beberapa tahun; verifikasi status hukum di JDIH/peraturan.bpk.go.id. Hindari data personal/rahasia pada demo publik.

## 2. Memasukkan kode tambahan ke GitHub

1. Clone repository `https://github.com/alfdmsr/Fine-tuned-Chatbot-Tim-Legal-berbasis-RAG.git` secara lokal.
2. Ekstrak isi ZIP ini ke root repo (sehingga ada `.github/workflows/`, `deploy/space/`, dan `DEPLOYMENT.md`). **Jangan menimpa atau menghapus notebook asli**.
3. Commit dan push:

```bash
git checkout -b feat/zerogpu-public-demo
git add .github/workflows/deploy-huggingface-space.yml deploy/space DEPLOYMENT.md INSTALL_PATCH.md .gitignore tests
git commit -m "feat: ZeroGPU Gradio public demo for Legal RAG"
git push -u origin feat/zerogpu-public-demo
```

4. Buat pull request ke `main` lalu merge. Jangan taruh `HF_TOKEN` di git atau di dokumen publik.

## 3. Membuat Space gratis

1. Masuk ke https://huggingface.co/new-space.
2. Beri nama misal `legal-rag-chatbot`, visibility **Public**, SDK **Gradio**.
3. Pilih **ZeroGPU** sebagai hardware (saat pembuatan atau Settings → Hardware). Jika ZeroGPU tidak tersedia, periksa status akun dan email, bukan memilih GPU dedicated berbayar.
4. Ingat ID Space, misalnya `NAMA_AKUN_HF/legal-rag-chatbot` (ini bukan ID model).

## 4. Aktifkan deploy otomatis dari repo GitHub

Di GitHub → repository Settings → Secrets and variables → Actions:

- **Secret** `HF_TOKEN`: token fine-grained yang memiliki izin **write** ke Space tujuan. Buat di https://huggingface.co/settings/tokens
- **Variable** `HF_SPACE_ID`: `NAMA_AKUN_HF/legal-rag-chatbot`.

Di GitHub → **Actions**, jalankan `Deploy Legal RAG demo to Hugging Face Spaces`. Workflow menyalin app dan empat PDF dari folder repo `Document RAG` ke Space, termasuk file PDF besar. Pastikan git repo sumber tetap dalam kondisi publik/izin dokumen sesuai penggunaan.

Kunjungi `https://huggingface.co/spaces/NAMA_AKUN_HF/legal-rag-chatbot`, lihat tab **Logs**. URL ini template, bukan tautan publik yang telah aktif.

## 5. Uji kelayakan **sebelum membagikan URL**

1. Status Space `Running` dan model `Saranjana/Llama-3-8b-Legal-RAG` berhasil terunduh.
2. Ajukan satu pertanyaan: `Apa ketentuan lembur dalam PP 35 Tahun 2021?`.
3. Pastikan jawaban tidak error, muncul nama dan halaman PDF, dan **validasi isi jawaban terhadap PDF** secara manual.
4. Ajukan pertanyaan di luar dokumen, pastikan bot tidak membuat kepastian hukum tanpa dukungan.
5. Buka URL dari browser incognito dan uji pengunjung tanpa login (kuota ~2 menit/hari).

## 6. Masalah yang mungkin muncul

- **Error CUDA di awal**: `import spaces` harus paling awal sebelum import `torch`/library CUDA. Model disiapkan eager pada module root.
- **OOM RAM / model checkpoint tak muat**: 8B BF16 perlu file model besar saat download, meski runtime 4-bit. Lihat log memori dan pertimbangkan mengecilkan model/korpus untuk demo tahap awal.
- **bitsandbytes tidak kompatibel**: jalur NF4 4-bit ZeroGPU perlu pengujian pada Space nyata. Jika gagal, jangan memaksakan GPU berbayar; ubah metode kuantisasi / model kecil setelah diagnosis error yang spesifik.
- **Startup sangat lambat**: PDF ~100 MB dan 1.949 halaman membutuhkan pemrosesan embedding. Cache bertahan hanya selama penyimpanan sementara Space belum direset; upayakan membangun indeks sekali per versi ke berkas kecil.
- **Tidak ada ZeroGPU pada halaman create**: periksa email, syarat usia akun dan batas 2 Space; ketentuan UI dapat berubah.
- **Jangan mengklaim GPU sudah berjalan** hanya karena tampilan Gradio terbuka. Uji inferensi + sitasi dulu.

## 7. Perbedaan dari notebook

Demo menjalankan **model SFT**, retrieval BM25+dense menggunakan RRF, reranking, sitasi PDF. HyDE default nonaktif agar GPU kuota hemat. Web-search fallback tidak aktif untuk menghindari informasi hukum internet yang belum diverifikasi. Paket menggunakan satu transaksi inferensi per pertanyaan melalui `@spaces.GPU`.

Referensi resmi: https://huggingface.co/docs/hub/spaces-zerogpu
