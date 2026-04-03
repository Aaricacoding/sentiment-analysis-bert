---
title: Sentiment Analysis Bert
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🧠 Full Emotion & Sentiment Analyzer using BERT

> **Author:** Aarica Raj
> **GitHub:** [@Aaricacoding](https://github.com/Aaricacoding)
> **Tech Stack:** Python · PyTorch · HuggingFace Transformers · DistilRoBERTa · DistilBERT · Gradio

---

## 📌 What is this project?

This project builds an **AI-powered Full Emotion & Sentiment Analysis System** that detects:
- **7 Emotions** — Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
- **Overall Sentiment** — Positive or Negative

It uses **two models together:**
- **DistilRoBERTa** — fine-tuned on emotion dataset for 7-emotion detection
- **DistilBERT** — fine-tuned on SST-2 for Positive/Negative sentiment

Real-world use cases:
- 📦 Product review analysis
- 🐦 Social media emotion monitoring
- 🎬 Movie/book review classification
- 💬 Customer feedback analysis
- 🏥 Mental health text monitoring

---

## 🧠 How Does It Work? (Architecture)

```
Input Text
    │
    ├──────────────────────┬─────────────────────────┐
    ▼                      ▼                         │
┌──────────────┐   ┌───────────────────┐             │
│  Tokenizer   │   │   Tokenizer       │             │
│  (RoBERTa)   │   │   (BERT)          │             │
└──────┬───────┘   └────────┬──────────┘             │
       │                    │                         │
       ▼                    ▼                         │
┌──────────────┐   ┌───────────────────┐             │
│ DistilRoBERTa│   │ DistilBERT        │             │
│ Emotion Model│   │ Sentiment Model   │             │
└──────┬───────┘   └────────┬──────────┘             │
       │                    │                         │
       ▼                    ▼                         │
┌──────────────┐   ┌───────────────────┐             │
│ 7 Emotions   │   │ Positive/Negative │             │
│ with scores  │   │ with scores       │             │
└──────┬───────┘   └────────┬──────────┘             │
       └────────────────────┘                         │
                    │                                 │
                    ▼                                 │
           Full Breakdown Output ────────────────────┘
```

### Key Concepts:
| Concept | Explanation |
|---|---|
| **BERT** | Reads text in both directions for better understanding |
| **DistilBERT** | 40% smaller, 60% faster than BERT, keeps 97% accuracy |
| **RoBERTa** | Improved BERT with better training — more accurate |
| **DistilRoBERTa** | Lighter version of RoBERTa for faster inference |
| **Fine-tuning** | Pre-trained model trained further on specific data |
| **SST-2** | Stanford Sentiment Treebank — 67K labeled movie reviews |
| **Softmax** | Converts raw scores to probabilities that sum to 100% |

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
https://aaricacoding-sentiment-analysis-bert.hf.space
```

> 💡 First run downloads both models (~550MB total). Only happens once!

---

## 🚀 How to Use

1. Open the app at `https://aaricacoding-sentiment-analysis-bert.hf.space`
2. **Type any text** in the input box
3. Click **"🔍 Analyze"** or press **Enter**
4. See the full breakdown:
   - ✅ Primary emotion detected
   - ✅ All 7 emotion scores with percentages
   - ✅ Positive/Negative sentiment scores

---

## 🧪 Example Outputs

| Text | Primary Emotion | Sentiment |
|---|---|---|
| "I love this so much!" | 😊 Joy (98%) | 👍 Positive (99%) |
| "I am so angry right now!" | 😠 Anger (94%) | 👎 Negative (97%) |
| "The movie was okay, nothing special." | 😐 Neutral (68%) | 👍 Positive (54%) |
| "I am scared about tomorrow." | 😨 Fear (91%) | 👎 Negative (89%) |
| "Oh wow! What a surprise!" | 😲 Surprise (87%) | 👍 Positive (76%) |
| "This smells absolutely disgusting." | 🤢 Disgust (93%) | 👎 Negative (98%) |
| "I went to the store today." | 😐 Neutral (82%) | 👍 Positive (61%) |

---

## 🔧 Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **PyTorch** | Deep learning framework |
| **HuggingFace Transformers** | Pre-trained models library |
| **DistilRoBERTa** | 7-emotion detection model |
| **DistilBERT (SST-2)** | Positive/Negative sentiment model |
| **Gradio** | Web UI for ML demos |

---

## 🌐 Models Used

| Model | Task | Parameters | Accuracy |
|---|---|---|---|
| `j-hartmann/emotion-english-distilroberta-base` | 7 Emotions | ~82M | 66% (6-class) |
| `distilbert-base-uncased-finetuned-sst-2-english` | Pos/Neg | ~66M | 91.3% |

---

## 💡 What I Learned Building This

- How **BERT and RoBERTa** read text bidirectionally
- Difference between **DistilBERT vs DistilRoBERTa**
- How to run **multiple AI models** together in one app
- How **fine-tuning** adapts pre-trained models to specific tasks
- How **tokenization** works in NLP (WordPiece tokenizer)
- How to build and deploy NLP apps using **HuggingFace + Gradio**

---

## 🔮 Future Improvements

- [ ] Fine-tune model on custom emotions (jealousy, excitement, love)
- [ ] Add multilingual support (Hindi, Spanish, French)
- [ ] Add batch processing for multiple texts at once
- [ ] Deploy permanently on HuggingFace Spaces (free)
- [ ] Add REST API using FastAPI + Docker
- [ ] Add real-time Twitter/social media analysis

---

## 📄 License

MIT License — free to use, modify, and share!

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) — for Transformers library and model hub
- [j-hartmann](https://huggingface.co/j-hartmann) — for the emotion detection model
- [Stanford NLP](https://nlp.stanford.edu/sentiment/) — for SST-2 dataset
- [Gradio Team](https://gradio.app) — for the UI framework
