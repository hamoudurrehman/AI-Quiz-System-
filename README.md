# RACE Reading Comprehension & Quiz Generation System
### Intelligent AI Project — BS(CS) Spring 2026 | FAST-NUCES Islamabad

---

## 📌 Project Overview

An AI-powered Reading Comprehension and Quiz Generation System built on the **RACE dataset**.  
The system integrates two classical ML pipelines:

| Model | Role |
|-------|------|
| **Model A** | Answer Verifier (Logistic Regression, SVM, Naive Bayes, Ensemble) + K-Means + Label Propagation |
| **Model B** | Distractor Generator + Hint Extractor (Logistic Regression ranker) |

The UI is built with **Streamlit** and provides 4 screens:  
Article Input → Quiz View → Hint Panel → Analytics Dashboard

---

## 🗂️ Project Structure

```
race_rc_project/
├── data/
│   ├── raw/          ← place dev.csv here
│   └── processed/    ← auto-generated feature matrices
├── models/
│   ├── model_a/traditional/   ← trained sklearn models + vectorizer
│   └── model_b/traditional/   ← distractor ranker + hint scorer
├── src/
│   ├── preprocessing.py       ← data loading, 80-10-10 split, TF-IDF
│   ├── model_a_train.py       ← LR, SVM, NB, K-Means, Label Prop, Ensemble
│   ├── model_b_train.py       ← Distractor Ranker, Hint Scorer
│   ├── inference.py           ← Unified inference API
│   └── evaluate.py            ← Metric utilities
├── ui/
│   └── app.py                 ← Streamlit 4-screen application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone / Download the project
```bash
cd race_rc_project
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add the dataset
Place `dev.csv` inside `data/raw/`:
```
data/raw/dev.csv
```
> **Note:** The project uses ONLY `dev.csv` and performs its own 80-10-10 split automatically.

---

## 🚀 Running the Project

### Step 5 — Train Model A
```bash
python src/model_a_train.py
```
Trains: Logistic Regression, SVM (calibrated), Naive Bayes, K-Means, Label Propagation, Soft-Voting Ensemble.  
Saves all models to `models/model_a/traditional/`.

> ⏱️ Default: trains on 20,000 rows (fast). To use all ~88k rows, set `sample_n=None` in `main()`.

### Step 6 — Train Model B
```bash
python src/model_b_train.py
```
Trains: Distractor Ranker + Hint Scorer.  
Saves to `models/model_b/traditional/`.

### Step 7 — Launch the Streamlit App
```bash
streamlit run ui/app.py
```
Opens at: **http://localhost:8501**

---

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Source file | `dev.csv` (from RACE Kaggle download) |
| Total rows | ~87,866 |
| Split | 80% train / 10% val / 10% test |
| Columns | id, article, question, A, B, C, D, answer |
| Answer types | Multiple choice (A/B/C/D) |

---

## 🔬 Models & Features

### Model A — Answer Verifier
| Feature | Description |
|---------|-------------|
| TF-IDF | Combined (article × 2 + question + option), 10K features, bigrams |
| Lexical | Word overlap (article↔option, question↔option), length ratios |
| Models | LR, LinearSVM (calibrated), Naive Bayes, K-Means, Label Propagation |
| Ensemble | Soft-voting: 0.4×LR + 0.4×SVM + 0.2×NB |

### Model B — Distractor + Hint Generator
| Feature | Description |
|---------|-------------|
| Distractor | Cosine sim (OHE proxy), char-match ratio, frequency score, length ratio |
| Hint | Keyword overlap, sentence position, length, Wh-word presence |
| Models | Logistic Regression ranker for both tasks |

---

## 📏 Evaluation Metrics

### Model A
- Accuracy, Macro F1, Precision, Recall
- Confusion Matrix
- K-Means: Silhouette Score + Purity
- Label Propagation: Semi-supervised F1

### Model B
- Distractor Ranker: Accuracy, Precision, Recall, F1, Confusion Matrix
- Hint Scorer: Accuracy, F1, R² Score

---

## 🖥️ UI Screens

| Screen | Description |
|--------|-------------|
| 1. Article Input | Paste text or load random RACE sample; trigger inference |
| 2. Quiz View | Question + 4 options; check answer; see model prediction + confidence |
| 3. Hint Panel | Reveal graduated hints (Hint 1 → 3); reveal answer after all hints |
| 4. Analytics | Model accuracy, latency chart, session log, model file status |

---

## ⚠️ Notes

- `fit_transform()` is ONLY called on training data (no data leakage).
- All models are saved with `joblib` and reloaded at inference time.
- Inference for a single article+question completes well under 10 seconds.
- TF-IDF is used as the primary feature method (One-Hot Encoding cosine is approximated via word-set overlap for classical features).
