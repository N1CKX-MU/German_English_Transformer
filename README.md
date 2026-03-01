# Transformer From Scratch (PyTorch)

A clean, modular implementation of a Transformer model built step-by-step from scratch using PyTorch.

This project focuses on deeply understanding the architecture behind the original paper:

> "Attention Is All You Need" (Vaswani et al., 2017)

---

## 🚀 Project Status

- ✅ Dataset pipeline  
- ✅ Tokenization (spaCy)  
- ✅ Vocabulary building  
- ✅ Numericalization  
- ✅ Padding & DataLoader  
- ✅ Token Embedding  
- ✅ Positional Encoding  
- ⬜ Multi-Head Attention  
- ⬜ Encoder & Decoder  
- ⬜ Full Transformer  
- ⬜ Training loop  

---

## 📂 Project Structure

```
project/
│
├── src/
│   ├── dataset.py
│   └── model/
│       ├── embedding.py
│       └── positional_encoding.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_embedding.py
│   ├── test_positional_encoding.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Architecture Overview (Current)

```
Raw Text
↓
Tokenization (spaCy)
↓
Vocabulary (torchtext)
↓
Numericalization
↓
Padding & Batching
↓
Embedding Layer
↓
Positional Encoding
↓
(Next: Multi-Head Attention)
```

---

## 📦 Dataset

We use the **Multi30k** German → English translation dataset via `torchtext`.

Each training sample looks like:

```
("Zwei Männer laufen.", "Two men are running.")
```

---

## 🔤 Vocabulary & Numericalization

Special tokens used:

- `<sos>` — Start of sentence
- `<eos>` — End of sentence
- `<pad>` — Padding
- `<unk>` — Unknown word

Each sentence is converted into integer indices before being fed into the model.

---

## 🧩 Embedding Layer

Implemented using `nn.Embedding`.

Converts:

```
[45, 89, 120]
```

into:

```
[
  [vector_128],
  [vector_128],
  [vector_128]
]
```

Each token becomes a dense representation of size `d_model`.

---

## 📍 Positional Encoding

Since Transformers do not use recurrence (no LSTM / RNN), we add positional information using sinusoidal positional encoding:

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

This allows the model to understand word order.

---

## 🐳 Docker Support

Build the container:

```bash
docker build -t transformer-project .
```

Run it:

```bash
docker run -it transformer-project
```

The container includes:

- PyTorch
- torchtext
- spaCy
- Language models
- All required dependencies

---

## 🧪 Running Tests

We use `pytest` for modular testing.

Install pytest:

```bash
pip install pytest
```

Run all tests:

```bash
pytest
```

---

## ⚙️ Setup (Without Docker)

1. Create virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download spaCy models:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## 🎯 Goal of This Project

This is a learning-focused implementation of the Transformer architecture with:

- Clear modular code
- Full test coverage
- Clean project structure
- Production-style practices

The goal is to deeply understand each building block instead of relying entirely on `nn.Transformer`.

---

## 📚 References

- Vaswani et al., 2017 — Attention Is All You Need
- PyTorch Documentation
- torchtext Documentation

---

## 🔜 Upcoming Work

- Multi-Head Attention implementation
- Encoder block
- Decoder block
- Full Transformer stack
- Training loop
- BLEU evaluation

---

## 👩‍💻 Author

Built as a step-by-step exploration of Transformer architecture in PyTorch.