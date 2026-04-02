# 🧠 Sentiment Analysis using BERT Transformer

> **Author:** Aarica Raj
> **GitHub:** [@Aaricacoding](https://github.com/Aaricacoding)
> **Tech Stack:** Python · PyTorch · HuggingFace Transformers · DistilBERT · Gradio

---

## 📌 What is this project?

This project builds an **AI-powered Sentiment Analysis System** that detects whether
any text is **Positive** or **Negative** — along with a confidence score.

It uses **DistilBERT** — a lighter, faster version of **BERT (Bidirectional Encoder
Representations from Transformers)** fine-tuned on the **SST-2 dataset** (Stanford
Sentiment Treebank) containing 67,000+ movie reviews.

Real-world use cases:
- 📦 Product review analysis
- 🐦 Social media monitoring
- 🎬 Movie/book review classification
- 💬 Customer feedback analysis

---

## 🧠 How Does It Work? (Architecture)

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│  Tokenizer (WordPiece)              │  ← Splits text into subword tokens
│  e.g. "playing" → ["play", "##ing"] │
└──────────────┬──────────────────────┘
               │  Token IDs + Attention Mask
               ▼
┌─────────────────────────────────────┐
│  DistilBERT Encoder                 │  ← 6 Transformer layers
│  (Bidirectional Self-Attention)     │     reads context from BOTH directions
└──────────────┬──────────────────────┘
               │  Contextual Embeddings [CLS] token
               ▼
┌─────────────────────────────────────┐
│  Classification Head                │  ← Linear layer → Softmax
│  (Fine-tuned on SST-2)              │
└──────────────┬──────────────────────┘
               │
               ▼
        POSITIVE (94.3%) / NEGATIVE (5.7%)
```

### Key Concepts:
| Concept | Explanation |
|---|---|
| **BERT** | Reads text in both directions (left→right AND right→left) |
| **DistilBERT** | 40% smaller, 60% faster than BERT, keeps 97% accuracy |
| **Fine-tuning** | Pre-trained model trained further on sentiment data |
| **SST-2** | Stanford Sentiment Treebank — 67K labeled movie reviews |
| **[CLS] Token** | Special token BERT uses for classification tasks |
| **Softmax** | Converts raw scores to probabilities (sum = 100%) |

---

## 📁 Project Structure

```
sentiment-analysis-bert/
│
├── app.py               ← Main Gradio application (run this!)
├── requirements.txt     ← All Python dependencies
└── README.md            ← This documentation file
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Aaricacoding/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the App
```bash
python app.py
```

### Step 5 — Open in Browser
```
http://localhost:7861
```

> 💡 First run downloads DistilBERT model (~250MB). Much faster than BLIP!

---

## 🚀 How to Use

1. Open the app at `http://localhost:7861`
2. **Type any text** in the input box
3. Click **"🔍 Analyze Sentiment"** or press **Enter**
4. See:
   - ✅ Sentiment label (POSITIVE / NEGATIVE)
   - ✅ Confidence percentage
   - ✅ Score bar chart for both labels

---

## 🧪 Example Outputs

| Text | Sentiment | Confidence |
|---|---|---|
| "I love this product!" | 😊 POSITIVE | 99.8% |
| "This is terrible." | 😞 NEGATIVE | 99.5% |
| "It was okay I guess." | 😊 POSITIVE | 62.1% |
| "Worst experience ever!" | 😞 NEGATIVE | 99.9% |
| "Not bad, could be better." | 😞 NEGATIVE | 71.3% |

---

## 🔧 Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **PyTorch** | Deep learning framework |
| **HuggingFace Transformers** | Pre-trained DistilBERT model |
| **DistilBERT (SST-2)** | Fine-tuned sentiment classification model |
| **Gradio** | Web UI for ML demos |

---

## 🌐 Model Details

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Source:** [HuggingFace Model Hub](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- **Parameters:** ~66 million
- **Training Data:** SST-2 (Stanford Sentiment Treebank)
- **Accuracy:** ~91.3% on SST-2 benchmark
- **License:** Apache 2.0

---

## 💡 What I Learned Building This

- How **BERT** reads text bidirectionally using self-attention
- Difference between **BERT vs DistilBERT** (size vs speed tradeoff)
- How **fine-tuning** adapts pre-trained models to specific tasks
- How **tokenization** works in NLP (WordPiece tokenizer)
- How to build NLP apps using **HuggingFace Pipelines**
- How to deploy NLP models using **Gradio**

---

## 🔮 Future Improvements

- [ ] Fine-tune on custom dataset (e.g. product reviews, tweets)
- [ ] Add multi-class sentiment (Very Positive, Neutral, Very Negative)
- [ ] Support multiple languages
- [ ] Add batch processing for multiple texts at once
- [ ] Deploy on HuggingFace Spaces
- [ ] Add REST API using FastAPI

---

## 📄 License

MIT License — free to use, modify, and share!

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) — for Transformers library and model hub
- [Stanford NLP](https://nlp.stanford.edu/sentiment/) — for SST-2 dataset
- [Gradio Team](https://gradio.app) — for the UI framework
