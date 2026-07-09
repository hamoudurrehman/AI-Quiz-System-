"""
model_b_train.py
================
Trains and evaluates Model B: Distractor Ranker + Hint Extractor.

Pipeline
--------
  1. Candidate extraction from article (frequency-based, no NLP tools)
  2. Feature engineering with One-Hot Encoding + cosine similarity
  3. Logistic Regression ranker to score each candidate
  4. Extractive Hint Scorer (Logistic Regression on sentence features)
  5. Evaluation: Precision, Recall, F1, Accuracy, R²

Run:
    python src/model_b_train.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, r2_score)
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, issparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (load_and_split, clean_text,
                                 get_distractor_candidates,
                                 get_hint_sentences, CSV_PATH)

MODEL_B_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "model_b", "traditional")


# ═══════════════════════════════════════════════════════════════════════════
# PART A — DISTRACTOR RANKER
# ═══════════════════════════════════════════════════════════════════════════

def _candidate_features(candidate: str, correct_answer: str, article: str) -> list:
    """
    Compute feature vector for a single (candidate, answer) pair:
      [0] cosine_sim(candidate_words, answer_words)   ← One-Hot proxy
      [1] char_level_match_ratio
      [2] passage_frequency_score
      [3] length_ratio
    """
    art_words  = Counter(clean_text(article).split())
    ans_words  = set(clean_text(correct_answer).split())
    cand_words = set(clean_text(candidate).split())

    # Feature 0: word-overlap (One-Hot cosine proxy)
    dot   = len(cand_words & ans_words)
    norm  = (len(cand_words)**0.5 * len(ans_words)**0.5) + 1e-9
    cos_sim = dot / norm

    # Feature 1: character overlap ratio
    common_chars = sum(min(candidate.count(c), correct_answer.count(c))
                       for c in set(candidate))
    char_ratio = common_chars / (max(len(candidate), len(correct_answer)) + 1)

    # Feature 2: passage frequency (how common are candidate words)
    freq_score = sum(art_words[w] for w in cand_words) / (len(cand_words) + 1)

    # Feature 3: length ratio (similar length → more plausible)
    len_ratio = len(candidate.split()) / (len(correct_answer.split()) + 1)

    return [cos_sim, char_ratio, min(freq_score / 10.0, 1.0), min(len_ratio, 2.0)]


def build_distractor_training_data(df: pd.DataFrame, max_rows: int = 5_000):
    """
    For each row, treat the 3 WRONG options as positive distractors (label=1)
    and randomly chosen article sentences as negative candidates (label=0).
    """
    X_rows, y_labels = [], []
    for _, r in df.head(max_rows).iterrows():
        correct_letter = r["answer"]
        correct_text   = r[correct_letter]
        wrong_letters  = [l for l in ["A","B","C","D"] if l != correct_letter]

        # Positive examples: the actual wrong options
        for wl in wrong_letters:
            feats = _candidate_features(r[wl], correct_text, r["article"])
            X_rows.append(feats)
            y_labels.append(1)   # good distractor

        # Negative examples: random sentences from article (bad distractors)
        import re
        sents = [s.strip() for s in re.split(r'[.!?]', r["article"]) if len(s.strip()) > 5]
        for sent in sents[:3]:
            feats = _candidate_features(sent[:80], correct_text, r["article"])
            X_rows.append(feats)
            y_labels.append(0)   # poor distractor

    return np.array(X_rows, dtype=np.float32), np.array(y_labels)


def train_distractor_ranker(X_train, y_train):
    print("\n[Model B] Training Distractor Ranker (Logistic Regression) …")
    t0 = time.time()
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced",
                              random_state=42)
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return clf


def rank_distractors(article: str, correct_answer: str, ranker,
                      n_distractors: int = 3) -> list:
    """
    Return the top-3 ranked distractor candidates for a given (article, answer).
    """
    candidates = get_distractor_candidates(article, correct_answer, n_candidates=15)
    if not candidates:
        return ["[No distractor found]"] * n_distractors

    feat_matrix = np.array(
        [_candidate_features(c, correct_answer, article) for c in candidates],
        dtype=np.float32
    )
    probs   = ranker.predict_proba(feat_matrix)[:, 1]
    ranked  = sorted(zip(probs, candidates), reverse=True)

    # Apply diversity penalty: avoid candidates sharing too many words
    selected = []
    selected_words = set()
    for prob, cand in ranked:
        cand_words = set(cand.split())
        overlap    = len(cand_words & selected_words) / (len(cand_words) + 1)
        if overlap < 0.5:   # diverse enough
            selected.append(cand)
            selected_words |= cand_words
        if len(selected) >= n_distractors:
            break

    # Pad if needed
    while len(selected) < n_distractors:
        selected.append(candidates[len(selected)] if len(candidates) > len(selected)
                        else "[No distractor found]")
    return selected[:n_distractors]


# ═══════════════════════════════════════════════════════════════════════════
# PART B — HINT SCORER
# ═══════════════════════════════════════════════════════════════════════════

def _hint_sentence_features(sentence: str, question: str,
                              position: int, n_sentences: int) -> list:
    """
    Features for scoring a sentence as a hint:
      [0] keyword overlap with question
      [1] sentence position (normalised, early = higher)
      [2] sentence length (normalised)
      [3] question word presence (any of who/what/where/when/why/how)
    """
    import re
    s_words = set(clean_text(sentence).split())
    q_words = set(clean_text(question).split())
    wh_words = {"who", "what", "where", "when", "why", "how"}

    kw_overlap   = len(s_words & q_words) / (len(q_words) + 1)
    pos_norm     = 1.0 - (position / (n_sentences + 1))
    len_norm     = min(len(s_words) / 30.0, 1.0)
    has_wh       = float(bool(q_words & wh_words))

    return [kw_overlap, pos_norm, len_norm, has_wh]


def build_hint_training_data(df: pd.DataFrame, max_rows: int = 3_000):
    """
    For each row, score each article sentence.
    The sentence with the highest keyword-overlap with the question is
    labelled as the key hint sentence (label=1); others are label=0.
    """
    import re
    X_rows, y_labels = [], []

    # Re-read raw data to get uncleaned articles for sentence splitting
    raw_df = df.copy()
    # Try to get raw article from the original CSV if available
    raw_csv = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "raw", "dev.csv")

    for idx, r in df.head(max_rows).iterrows():
        article  = str(r["article"])   # already cleaned (no punctuation)
        question = str(r["question"])

        # Split on whitespace runs (punctuation removed, so use word boundaries)
        # Approximate sentences: split on 'and' / conjunctions as rough boundaries
        # Better: reconstruct from words using fixed window
        words = article.split()
        if len(words) < 6:
            continue
        # Create overlapping sentence windows of ~15 words
        window, step = 15, 10
        sents = []
        for start in range(0, max(1, len(words) - window + 1), step):
            chunk = " ".join(words[start:start + window])
            if chunk.strip():
                sents.append(chunk)

        if len(sents) < 2:
            continue

        q_words  = set(question.split())
        overlaps = []
        for sent in sents:
            s_words = set(sent.split())
            ov = len(s_words & q_words) / (len(q_words) + 1)
            overlaps.append(ov)

        best_idx = int(np.argmax(overlaps))
        n_sents  = len(sents)

        for i, sent in enumerate(sents):
            feats = _hint_sentence_features(sent, question, i, n_sents)
            X_rows.append(feats)
            y_labels.append(1 if i == best_idx else 0)

    if not X_rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=int)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=int)
    return X, y


def train_hint_scorer(X_train, y_train):
    print("\n[Model B] Training Hint Scorer (Logistic Regression) …")
    t0 = time.time()
    clf = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced",
                              random_state=42)
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return clf


def score_hints(article: str, question: str, hint_scorer,
                top_k: int = 3) -> list:
    """
    Return top_k hints ranked from most general (Hint 1) to
    most specific (Hint 3).
    """
    import re
    sents = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 10]
    if not sents:
        return ["Read the passage carefully."] * top_k

    n   = len(sents)
    X   = np.array([_hint_sentence_features(s, question, i, n)
                    for i, s in enumerate(sents)], dtype=np.float32)
    probs = hint_scorer.predict_proba(X)[:, 1]

    ranked = sorted(zip(probs, sents), reverse=True)
    top    = [s for _, s in ranked[:top_k]]
    # Hint 1 = least explicit → reverse order
    return list(reversed(top)) if len(top) >= 2 else top


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_distractor_ranker(ranker, X_val, y_val):
    preds = ranker.predict(X_val)
    print("\n[Model B — Distractor Ranker]")
    print(f"  Accuracy : {accuracy_score(y_val, preds):.4f}")
    print(f"  Precision: {precision_score(y_val, preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_val, preds, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_val, preds, zero_division=0):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, preds)}")


def evaluate_hint_scorer(scorer, X_val, y_val):
    preds_prob = scorer.predict_proba(X_val)[:, 1]
    preds      = scorer.predict(X_val)
    r2         = r2_score(y_val, preds_prob)
    print("\n[Model B — Hint Scorer]")
    print(f"  Accuracy : {accuracy_score(y_val, preds):.4f}")
    print(f"  F1       : {f1_score(y_val, preds, zero_division=0):.4f}")
    print(f"  R² Score : {r2:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, preds)}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(sample_n: int = 15_000):
    print("=" * 60)
    print("  MODEL B — TRAINING PIPELINE")
    print("=" * 60)

    train_df, val_df, test_df = load_and_split(sample_n=sample_n)

    # ── Distractor Ranker ──
    print("\n[Model B] Building distractor training data …")
    X_dist_tr, y_dist_tr = build_distractor_training_data(train_df, max_rows=5_000)
    X_dist_val, y_dist_val = build_distractor_training_data(val_df, max_rows=1_000)

    distractor_ranker = train_distractor_ranker(X_dist_tr, y_dist_tr)
    evaluate_distractor_ranker(distractor_ranker, X_dist_val, y_dist_val)

    # ── Hint Scorer ──
    print("\n[Model B] Building hint training data …")
    X_hint_tr,  y_hint_tr  = build_hint_training_data(train_df, max_rows=3_000)
    X_hint_val, y_hint_val = build_hint_training_data(val_df,   max_rows=500)

    hint_scorer = train_hint_scorer(X_hint_tr, y_hint_tr)
    evaluate_hint_scorer(hint_scorer, X_hint_val, y_hint_val)

    # ── Save ──
    os.makedirs(MODEL_B_DIR, exist_ok=True)
    joblib.dump(distractor_ranker, os.path.join(MODEL_B_DIR, "distractor_ranker.pkl"))
    joblib.dump(hint_scorer,       os.path.join(MODEL_B_DIR, "hint_scorer.pkl"))
    print(f"\n[Model B] Models saved to {MODEL_B_DIR}")

    return distractor_ranker, hint_scorer


if __name__ == "__main__":
    main(sample_n=15_000)
