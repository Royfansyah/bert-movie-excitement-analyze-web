# 📦 Project Setup Guide

## 📁 Struktur Folder

- **Model/**
  - Digunakan sebagai direktori untuk menyimpan model (BERT)

## 🤖 Download Model

Silakan download file model melalui link berikut:
- [Download Model](https://drive.google.com/file/d/1rLStOzclTd25USwAG27gVWmTxIEGlcK7/view?usp=sharing)

> Setelah download, letakkan file model ke dalam folder `Model/`

## ⚙️ Setup Environment

Aktifkan virtual environment dan install dependencies:

```bash
# Membuat virtual environment
python -m venv venv

# Mengaktifkan environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install flask
pip install torch torchvision torchaudio
pip install transformers
```

## 🚀 Menjalankan Aplikasi

```bash
python app.py
```

## 📌 Catatan

- Pastikan Python sudah terinstall
- Gunakan versi Python yang kompatibel dengan PyTorch & Transformers
