"""
preprocessing.py
================
Loads the RACE dataset from a SINGLE CSV file (dev.csv) and splits it
into train / val / test using an 80-10-10 ratio.

Feature Engineering:
  - Text cleaning  (lower-case, punctuation removal)
  - One-Hot Encoding of answer labels
  - TF-IDF vectorisation of (article + question + option) combinations
  - Cosine similarity features between article and each option
  - Handcrafted lexical features (word overlap, length ratios, …)
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
CSV_PATH   = "/home/aaiz_ikram/ml_env/bin/AI_Lab/race_rc_project/race_rc_project/data/raw/dev.csv"
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_A_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")

CSV_PATH   = os.path.join(RAW_DIR, "dev.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Lower-case, remove punctuation and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD & SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def load_and_split(csv_path: str = CSV_PATH,
                   train_ratio: float = 0.80,
                   val_ratio: float   = 0.10,
                   test_ratio: float  = 0.10,
                   random_state: int  = 42,
                   sample_n: int      = None):
    """
    Load the RACE CSV and perform an 80-10-10 split.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    print(f"[preprocessing] Loading {csv_path} …")
    df = pd.read_csv(csv_path, index_col=0)

    # Optional: work on a smaller sample for fast iteration
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    print(f"[preprocessing] Total rows: {len(df):,}")

    # Clean text columns
    for col in ["article", "question", "A", "B", "C", "D"]:
        df[col] = df[col].apply(clean_text)

    # --- 80 / 10 / 10 split ---
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1."

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=random_state, shuffle=True
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), random_state=random_state
    )

    print(f"[preprocessing] Split sizes  →  "
          f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD VERIFICATION FEATURES  (Model A — Answer Verifier)
# ═══════════════════════════════════════════════════════════════════════════

def _build_texts_and_labels(df: pd.DataFrame):
    """
    Expand each row into 4 (article, question, option, label) triples.
    label = 1 if this option is the correct answer, else 0.
    """
    rows, labels = [], []
    for _, r in df.iterrows():
        correct = r["answer"]          # 'A', 'B', 'C', or 'D'
        for opt_letter in ["A", "B", "C", "D"]:
            combined = (
                r["article"] + " " + r["article"] + " "   # article twice for weight
                + r["question"] + " " + r[opt_letter]
            )
            rows.append(combined)
            labels.append(1 if opt_letter == correct else 0)
    return rows, labels


def _lexical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Handcrafted features for each (article, question, option) triple:
      - Word-overlap ratio between article and option
      - Word-overlap ratio between question and option
      - Option length (normalised)
      - Article length (normalised)
    Returns shape (4*N, 4)
    """
    feats = []
    for _, r in df.iterrows():
        art_words = set(r["article"].split())
        q_words   = set(r["question"].split())
        for opt_letter in ["A", "B", "C", "D"]:
            opt_words = set(r[opt_letter].split()) if r[opt_letter] else set()
            art_overlap = (len(art_words & opt_words) / (len(art_words) + 1))
            q_overlap   = (len(q_words   & opt_words) / (len(q_words)   + 1))
            opt_len     = len(opt_words) / 50.0
            art_len     = len(art_words) / 500.0
            feats.append([art_overlap, q_overlap, opt_len, art_len])
    return np.array(feats, dtype=np.float32)


def build_verification_features(train_df, val_df, test_df,
                                  max_features: int = 10_000,
                                  save: bool = True):
    """
    Build TF-IDF + lexical feature matrices for the answer-verification task.

    Returns
    -------
    X_train, X_val, X_test : sparse / dense feature matrices
    y_train, y_val, y_test : np.ndarray of binary labels (1=correct)
    vectorizer              : fitted TfidfVectorizer
    """
    print("[preprocessing] Building TF-IDF verification features …")

    train_texts, y_train = _build_texts_and_labels(train_df)
    val_texts,   y_val   = _build_texts_and_labels(val_df)
    test_texts,  y_test  = _build_texts_and_labels(test_df)

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        stop_words   = "english",
        sublinear_tf = True,
        ngram_range  = (1, 2),
        min_df       = 2,
        max_df       = 0.95,
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts)   # fit ONLY on train
    X_val_tfidf   = vectorizer.transform(val_texts)
    X_test_tfidf  = vectorizer.transform(test_texts)

    # --- Lexical features ---
    lx_train = csr_matrix(_lexical_features(train_df))
    lx_val   = csr_matrix(_lexical_features(val_df))
    lx_test  = csr_matrix(_lexical_features(test_df))

    X_train = hstack([X_train_tfidf, lx_train])
    X_val   = hstack([X_val_tfidf,   lx_val])
    X_test  = hstack([X_test_tfidf,  lx_test])

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    print(f"[preprocessing] Feature matrix shapes → "
          f"train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    print(f"[preprocessing] Label balance (train): "
          f"{y_train.mean():.3f} positive (expected ≈0.25)")

    if save:
        os.makedirs(PROC_DIR,    exist_ok=True)
        os.makedirs(MODEL_A_DIR, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(MODEL_A_DIR, "tfidf_vectorizer.pkl"))
        joblib.dump((X_train, y_train), os.path.join(PROC_DIR, "train_feats.pkl"))
        joblib.dump((X_val,   y_val),   os.path.join(PROC_DIR, "val_feats.pkl"))
        joblib.dump((X_test,  y_test),  os.path.join(PROC_DIR, "test_feats.pkl"))
        print(f"[preprocessing] Saved features to {PROC_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


# ═══════════════════════════════════════════════════════════════════════════
# 4. QUESTION GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

WH_TEMPLATES = {
    "who":   "Who {rest}?",
    "what":  "What {rest}?",
    "where": "Where {rest}?",
    "when":  "When {rest}?",
    "why":   "Why {rest}?",
    "how":   "How {rest}?",
}

def extract_candidate_sentences(article: str, answer: str, top_k: int = 5):
    """
    Return the top-k sentences from the article ranked by word overlap
    with the correct answer (classical, no NLP tools).
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 10]
    ans_words = set(clean_text(answer).split())
    scored = []
    for sent in sentences:
        sent_words = set(clean_text(sent).split())
        overlap = len(sent_words & ans_words) / (len(sent_words) + 1)
        scored.append((overlap, sent))
    scored.sort(reverse=True)
    return [s for _, s in scored[:top_k]]


def apply_wh_template(sentence: str) -> str:
    """Transform a sentence into a simple Wh-question."""
    words = sentence.lower().split()
    for wh, tmpl in WH_TEMPLATES.items():
        if wh in words:
            idx  = words.index(wh)
            rest = " ".join(words[idx + 1:])
            return tmpl.format(rest=rest).capitalize()
    # Fallback: prepend "What happened when"
    return f"What can be inferred from: '{sentence[:60]}…'?"


# ═══════════════════════════════════════════════════════════════════════════
# 5. DISTRACTOR HELPERS  (Model B)
# ═══════════════════════════════════════════════════════════════════════════

def get_distractor_candidates(article: str,
                               correct_answer: str,
                               n_candidates: int = 10):
    """
    Extract candidate distractor phrases from the article using
    frequency-based word selection (no NLP tools required).

    Strategy:
      1. Tokenise the article; count word frequencies.
      2. Collect all sentences.
      3. Score each sentence by:
           - high word frequency (common in the article)
           - LOW overlap with the correct answer  ← key: wrong but plausible
      4. Return top-n_candidates as distractor strings.
    """
    from collections import Counter

    art_clean   = clean_text(article)
    words       = art_clean.split()
    freq        = Counter(words)

    # Remove very short / stopword-like tokens
    stopwords   = {"the", "a", "an", "is", "was", "are", "were", "it",
                   "he", "she", "they", "i", "in", "on", "at", "to",
                   "of", "and", "or", "but", "for", "with", "his", "her"}
    top_words   = {w for w, _ in freq.most_common(50) if w not in stopwords}

    ans_clean   = clean_text(correct_answer)
    ans_words   = set(ans_clean.split())

    sentences   = [s.strip() for s in re.split(r'[.!?]', art_clean)
                   if len(s.strip()) > 5]

    scored = []
    for sent in sentences:
        sent_words  = set(sent.split())
        freq_score  = sum(freq[w] for w in sent_words) / (len(sent_words) + 1)
        ans_overlap = len(sent_words & ans_words) / (len(ans_words) + 1)
        diversity   = len(sent_words & top_words) / (len(top_words) + 1)
        # We want high frequency, low answer-overlap, reasonable diversity
        score = freq_score * diversity / (ans_overlap + 0.1)
        scored.append((score, sent))

    scored.sort(reverse=True)
    candidates = []
    seen = set()
    for _, sent in scored:
        # Deduplicate by first 3 words
        key = " ".join(sent.split()[:3])
        if key not in seen and sent != ans_clean:
            seen.add(key)
            candidates.append(sent[:120])   # cap length for UI
        if len(candidates) >= n_candidates:
            break
    return candidates


def get_hint_sentences(article: str, question: str, top_k: int = 3):
    """
    Rank article sentences by keyword overlap with the question.
    Returns [Hint1_most_general, Hint2_medium, Hint3_near_explicit].
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 10]
    q_words   = set(clean_text(question).split())

    scored = []
    for i, sent in enumerate(sentences):
        s_words = set(clean_text(sent).split())
        overlap = len(s_words & q_words) / (len(q_words) + 1)
        # also reward sentences near the start (position heuristic)
        pos_bonus = 1.0 / (i + 1)
        scored.append((overlap + 0.1 * pos_bonus, sent))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:top_k]]
    # Hint 1 = least explicit (last of top), Hint 3 = most explicit (first of top)
    return list(reversed(top)) if len(top) >= 2 else top


# ═══════════════════════════════════════════════════════════════════════════
# CLI helper
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split(sample_n=20_000)
    build_verification_features(train_df, val_df, test_df)
    print("[preprocessing] Done.")
