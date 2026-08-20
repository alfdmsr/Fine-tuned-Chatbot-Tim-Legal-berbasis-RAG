# Chatbot Tim Legal [Fine-tuned LLM + Advanced RAG]

Chatbot AI yang dirancang untuk menjawab pertanyaan seputar **hukum ketenagakerjaan Indonesia**, dibangun di atas **Llama-3-8B** yang di-*fine-tune* khusus, dilengkapi arsitektur **Advanced RAG (Retrieval-Augmented Generation)** dengan *hybrid search*, *reranking*, dan *reasoning* ala GRPO.

🤗 **Model tersedia di Hugging Face**: [Saranjana/Llama-3-8b-Legal-RAG](https://huggingface.co/Saranjana/Llama-3-8b-Legal-RAG)

## Tentang Project

Project ini membangun chatbot berbasis LLM yang dikhususkan untuk membantu tim Legal/HR menjawab pertanyaan terkait hukum ketenagakerjaan, dengan jawaban yang **berdasar pada dokumen resmi** (bukan sekadar halusinasi model) dan dilengkapi sumber referensi (halaman & nama dokumen) di setiap jawabannya.

Project ini terdiri dari **3 tahap pengembangan model**, masing-masing di notebook terpisah:

```
1. Fine-tuning (SFT)  →  2. GRPO (Reasoning)  →  3. RAG (Retrieval + Generation)
   Melatih model dasar        Melatih model untuk         Menyusun pipeline chatbot
   agar paham gaya bahasa     "berpikir" step-by-step      lengkap: retrieval dokumen,
   hukum & format jawaban     sebelum menjawab (reward-    reranking, hingga fallback
                               based reasoning)              ke pencarian web
```

## Tahap 1: Fine-Tuning (`fine_tuning_model.ipynb`)

Melatih model dasar **`unsloth/llama-3-8b-Instruct-bnb-4bit`** menggunakan teknik **LoRA (Low-Rank Adaptation)** via **Unsloth**, agar lebih fasih menjawab dalam Bahasa Indonesia dan mengikuti format instruksi.

- **Dataset**: [`Ichsan2895/alpaca-gpt4-indonesian`](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian) (split 90% train / 10% validation)
- **Konfigurasi LoRA**: rank (`r`) = 16, `lora_alpha` = 16, target modules mencakup seluruh proyeksi attention & MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Training**: `SFTTrainer` dari library `trl`, kuantisasi 4-bit untuk efisiensi memori GPU

Dua eksperimen hyperparameter dijalankan dan dibandingkan:

| | Eksperimen 1 | Eksperimen 2 |
|---|---|---|
| Learning Rate | `2e-4` | `1e-4` (lebih pelan, belajar lebih teliti) |
| LR Scheduler | `linear` | `cosine` (pengereman lebih halus) |
| Max Steps | 800 | 800 |
| Batch Size | 1 (grad. accumulation 8) | 1 (grad. accumulation 8) |

Model hasil fine-tuning terbaik lalu diunggah ke Hugging Face Hub sebagai basis untuk tahap selanjutnya.

## Tahap 2: GRPO — Reasoning & Reward Modeling (`GRPO_model.ipynb`)

Melatih model lebih lanjut menggunakan **GRPO (Group Relative Policy Optimization)** — teknik *reinforcement learning* untuk LLM — agar model bisa **"berpikir" (`<think>...</think>`) sebelum menjawab**, mirip pendekatan reasoning model modern.

Model dilatih dengan **4 fungsi reward custom** yang dirancang khusus untuk kasus hukum ketenagakerjaan:

| Fungsi Reward | Tujuan |
|---|---|
| **Format Reward** | Memberi poin jika model menggunakan tag `<think>...</think>` dengan benar |
| **Reasoning Length Reward** | Mendorong model berpikir lebih panjang & mendalam (bukan menjawab asal-asalan) |
| **Correctness Reward** | Memvalidasi kesimpulan hukum yang benar (mis. hak atas uang lembur) |
| **Language Reward** | Menghukum model jika jawabannya tercampur Bahasa Inggris (mencegah *code-switching* tak diinginkan) |

## Tahap 3: Advanced RAG Pipeline (`RAG_model.ipynb`)

Membangun sistem **Retrieval-Augmented Generation** lengkap agar chatbot menjawab berdasarkan dokumen hukum yang sebenarnya, bukan hanya dari memori model. Alur pipeline:

1. **Document Loading & Chunking** : memuat dokumen PDF hukum ketenagakerjaan, dipecah menjadi chunk menggunakan `RecursiveCharacterTextSplitter`.
2. **Vector Database** : teks di-*embed* menggunakan model *multilingual embedding* dan disimpan ke **ChromaDB**.
3. **Hybrid (Ensemble) Retriever** : menggabungkan dua metode pencarian dengan bobot 50:50:
   - **BM25** (pencarian berbasis kata kunci)
   - **Vector Search** (pencarian berbasis makna/semantik)
4. **HyDE (Hypothetical Document Embeddings)** : model menghasilkan 2 "jawaban hipotesis" terlebih dahulu untuk memperluas & mempertajam pencarian dokumen relevan.
5. **Reranking** : kandidat dokumen hasil retrieval diurutkan ulang menggunakan model **Cross-Encoder** (`BAAI/bge-reranker-base`) untuk memastikan relevansi tertinggi berada di urutan teratas.
6. **Threshold Fallback ke Web Search** : jika skor relevansi dokumen lokal terlalu rendah (di bawah threshold), sistem otomatis beralih mencari jawaban lewat **DuckDuckGo Search** secara real-time.
7. **Sitasi Otomatis** : setiap jawaban chatbot disertai sumber referensi (nama dokumen & nomor halaman) yang dipakai sebagai dasar jawaban.

## Struktur Project

```
Fine-tuned-Chatbot-Tim-Legal-berbasis-RAG/
├── fine_tuning_model.ipynb     # Tahap 1: Fine-tuning Llama-3-8B dengan LoRA
├── GRPO_model.ipynb            # Tahap 2: Reasoning training dengan GRPO + custom reward
├── RAG_model.ipynb             # Tahap 3: Pipeline RAG lengkap (hybrid search, reranking, HyDE)
├── requirements.txt            # Daftar dependency Python
├── link_huggingface.txt        # Tautan model hasil fine-tuning di Hugging Face
└── README.md
```

## Tech Stack

- **Model dasar**: Llama-3-8B (`unsloth/llama-3-8b-Instruct-bnb-4bit`)
- **Unsloth** -> fine-tuning & inferensi LLM secara efisien (kuantisasi 4-bit)
- **TRL** (`SFTTrainer`, `GRPOTrainer`) & **PEFT** (LoRA) -> teknik fine-tuning & reinforcement learning
- **LangChain** (`langchain`, `langchain-community`, `langchain-text-splitters`) -> orkestrasi pipeline RAG
- **ChromaDB** -> vector database
- **Sentence Transformers** -> model embedding & Cross-Encoder reranker
- **rank_bm25** -> algoritma pencarian BM25
- **DuckDuckGo Search** -> fallback pencarian web real-time
- **PyPDF** -> parsing dokumen PDF hukum
- **Accelerate & Bitsandbytes** -> efisiensi training/inferensi model besar di GPU terbatas

## Cara Menjalankan

> Ketiga notebook ini didesain untuk berjalan di **Google Colab** dengan GPU (memanfaatkan `google.colab.drive` dan `google.colab.userdata`), karena membutuhkan resource GPU untuk memuat model 8B parameter.

1. **Clone repository**
   ```sh
   git clone https://github.com/alfdmsr/Fine-tuned-Chatbot-Tim-Legal-berbasis-RAG.git
   cd Fine-tuned-Chatbot-Tim-Legal-berbasis-RAG
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

3. **(Opsional) Latih ulang model dari awal**
   - Jalankan `fine_tuning_model.ipynb` untuk melatih model dasar dari nol.
   - Lanjutkan dengan `GRPO_model.ipynb` untuk melatih kemampuan reasoning.
   - Butuh `HF_TOKEN` (Hugging Face access token) untuk mengunggah model hasil training.

4. **Langsung mencoba chatbot (tanpa training ulang)**

   Model hasil fine-tuning sudah tersedia publik di Hugging Face, jadi kamu bisa langsung menjalankan `RAG_model.ipynb` tanpa training ulang:
   - Siapkan folder dokumen PDF hukum di Google Drive (`/content/drive/MyDrive/Legal-RAG-Dataset`).
   - Notebook akan otomatis memuat model `Saranjana/Llama-3-8b-Legal-RAG` dari Hugging Face.
   - Jalankan sel Advanced RAG untuk mulai bertanya-jawab dengan chatbot lewat terminal interaktif.
