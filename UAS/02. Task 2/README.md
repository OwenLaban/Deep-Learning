# Fine-tuning T5 for Question Answering Using SQuAD Dataset

**Nama:** Wilhelmina Arlene S  
**NIM:** 1103223046

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem Question Answering generatif menggunakan fine-tuning model T5-base (Text-to-Text Transfer Transformer) pada dataset SQuAD. Tujuan dari proyek ini adalah membangun model yang mampu menghasilkan jawaban berdasarkan pertanyaan dan konteks yang diberikan.

## Tujuan

Membangun model sequence-to-sequence yang mampu menjawab pertanyaan berdasarkan konteks dengan akurasi tinggi menggunakan teknik fine-tuning pada model pre-trained T5-base dengan dataset SQuAD.

## Dataset

Dataset yang digunakan adalah SQuAD (Stanford Question Answering Dataset) dengan struktur sebagai berikut:

- **Training Set:** 1,000 samples (dari total 87,599 samples)
- **Validation Set:** 200 samples (dari total 10,570 samples)
- **Format Input:** `"question: [Q] context: [C]"`
- **Format Output:** Teks jawaban
- **Sumber:** Hugging Face Datasets Library

## Metodologi

### 1. Data Loading & Exploration

- Loading dataset SQuAD dari Hugging Face
- Eksplorasi struktur dataset (context, question, answers)
- Analisis sampel data untuk memahami format

### 2. Data Preprocessing

- Formatting input sesuai template T5: `"question: [Q] context: [C]"`
- Tokenisasi menggunakan T5Tokenizer
- Pengaturan max length:
  - Input: 384 tokens
  - Target: 64 tokens
- Padding dan truncation untuk konsistensi panjang sequence

### 3. Model Architecture

**T5-base Model:**

- Model pre-trained Text-to-Text Transfer Transformer
- Parameters: ~220 juta
- Architecture: Encoder-Decoder Transformer
- Pre-trained pada C4 (Colossal Clean Crawled Corpus)
- Gradient Checkpointing: Enabled untuk efisiensi memory

### 4. Training Configuration

- **Optimizer:** Adafactor (memory-efficient)
- **Learning Rate:** 3e-4
- **Batch Size:** 4 per device
- **Gradient Accumulation Steps:** 4 (effective batch size = 16)
- **Epochs:** 2
- **Warmup Steps:** 100
- **Weight Decay:** 0.01
- **Mixed Precision (FP16):** Enabled jika GPU tersedia
- **Callbacks:**
  - Model checkpoint setiap epoch
  - Load best model at end

### 5. Model Evaluation

- Evaluasi pada validation set
- Testing dengan contoh dari SQuAD validation set
- Testing dengan custom questions dan contexts
- Metrik: Loss, Runtime

## Hasil

Model berhasil di-fine-tune dengan performa yang baik dalam menghasilkan jawaban berdasarkan konteks. Optimasi memory memungkinkan training pada sistem dengan RAM terbatas.

## Tools & Libraries

- **Python** 3.8+
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face library untuk T5 model
- **Datasets** - Hugging Face library untuk loading SQuAD
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Accelerate** - Training optimization

## Struktur Notebook

1. **Setup & Installation** - Install dependencies
2. **Import Libraries** - Import required packages
3. **Part 1: Data Preprocessing**
   - Load SQuAD dataset
   - Explore dataset structure
   - Initialize tokenizer
   - Create preprocessing function
   - Apply preprocessing
   - Verify preprocessing
4. **Part 2: Model Training**
   - Load T5-base model
   - Setup data collator
   - Configure training arguments
   - Initialize Trainer
   - Start training
   - Save fine-tuned model
5. **Part 3: Model Evaluation**
   - Evaluate on validation set
   - Create inference function
   - Test with validation examples
   - Test with custom examples
   - Interactive testing
   - Performance summary

## Catatan Implementasi

- Proyek dirancang dengan optimasi memory untuk RAM terbatas (4-6 GB)
- Model menggunakan GPU acceleration jika tersedia
- Gradient checkpointing digunakan untuk trade-off speed vs memory
- Adafactor optimizer mengurangi penggunaan memory hingga 30-50%
- Random seed untuk reproducibility

## Kesimpulan

Proyek ini mendemonstrasikan implementasi lengkap fine-tuning model sequence-to-sequence untuk task generative question answering, mencakup data preprocessing, model training dengan optimasi memory, dan evaluasi model. Pendekatan ini dapat diaplikasikan untuk berbagai task text-to-text lainnya seperti summarization, translation, atau text generation.
