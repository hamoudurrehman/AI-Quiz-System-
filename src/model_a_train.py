"""
model_a_train.py
================
Trains and evaluates Model A: Answer Verifier + Question Generator.

Supervised models
-----------------
  1. Logistic Regression  (primary)
  2. Support Vector Machine
  3. Naive Bayes          (question-type classification — separate task)

Unsupervised / Semi-Supervised
-------------------------------
  4. K-Means Clustering
  5. Label Propagation (semi-supervised)

Ensemble
--------
  6. Soft-voting ensemble of LR + SVM + NB

Run:
    python src/model_a_train.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import issparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (load_and_split,
                                 build_verification_features,
                                 CSV_PATH)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "model_a", "traditional")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _to_dense_safe(X):
    """Convert sparse to dense only when necessary (small matrices)."""
    if issparse(X):
        return X.toarray()
    return X


def evaluate(name, model, X, y, label="Val"):
    """Print a standard evaluation report."""
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average="macro", zero_division=0)
    prec  = precision_score(y, preds, average="macro", zero_division=0)
    rec   = recall_score(y, preds, average="macro", zero_division=0)
    print(f"\n{'─'*55}")
    print(f"  {name}  [{label}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(classification_report(y, preds, zero_division=0))
    print(f"  Confusion Matrix:\n{confusion_matrix(y, preds)}")
    return {"model": name, "split": label,
            "accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train):
    print("\n[Model A] Training Logistic Regression …")
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0, max_iter=1000, solver="saga",
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    lr.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return lr


# ═══════════════════════════════════════════════════════════════════════════
# 2. SUPPORT VECTOR MACHINE
# ═══════════════════════════════════════════════════════════════════════════

def train_svm(X_train, y_train):
    print("\n[Model A] Training Linear SVM …")
    t0 = time.time()
    # Wrap with CalibratedClassifierCV so we get predict_proba for soft voting
    svc  = LinearSVC(C=0.5, max_iter=3000, class_weight="balanced", random_state=42)
    cal  = CalibratedClassifierCV(svc, cv=3)
    cal.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return cal


# ═══════════════════════════════════════════════════════════════════════════
# 3. NAIVE BAYES
# ═══════════════════════════════════════════════════════════════════════════

def train_naive_bayes(X_train, y_train):
    """
    MultinomialNB requires non-negative features.
    We use only the TF-IDF portion (first 10000 columns) which is always ≥ 0.
    """
    print("\n[Model A] Training Naive Bayes …")
    t0  = time.time()
    # Slice to TF-IDF only (lexical features can be slightly negative after hstack)
    X_nb = X_train[:, :10_000]
    nb   = MultinomialNB(alpha=0.1)
    nb.fit(X_nb, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return nb


def predict_nb(nb, X):
    return nb.predict(X[:, :10_000])


# ═══════════════════════════════════════════════════════════════════════════
# 4. K-MEANS CLUSTERING  (Unsupervised)
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans(X_train, y_train, n_clusters: int = 6):
    """
    Cluster question-answer pairs. We use a small dense subset for speed
    (K-Means with sparse data is slow on 350 k rows).
    """
    print("\n[Model A — Unsupervised] K-Means Clustering …")
    from sklearn.metrics import silhouette_score

    # Sample 5 000 rows for clustering (feasible on any laptop)
    rng  = np.random.default_rng(42)
    idx  = rng.choice(X_train.shape[0], size=min(5_000, X_train.shape[0]), replace=False)
    X_sub = _to_dense_safe(X_train[idx])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=200)
    km.fit(X_sub)

    sil = silhouette_score(X_sub, km.labels_, sample_size=2000, random_state=42)
    print(f"  K-Means silhouette score (k={n_clusters}): {sil:.4f}")

    # Purity: fraction of dominant label per cluster
    y_sub  = y_train[idx]
    purity = 0.0
    for c in range(n_clusters):
        mask = km.labels_ == c
        if mask.sum() == 0:
            continue
        dominant = np.bincount(y_sub[mask]).max()
        purity  += dominant
    purity /= len(y_sub)
    print(f"  Clustering purity: {purity:.4f}")
    return km, sil, purity


# ═══════════════════════════════════════════════════════════════════════════
# 5. LABEL PROPAGATION  (Semi-Supervised)
# ═══════════════════════════════════════════════════════════════════════════

def run_label_propagation(X_train, y_train, labeled_frac: float = 0.10):
    """
    Simulate a semi-supervised scenario:
      - Keep only `labeled_frac` of labels visible (rest set to -1).
      - LabelPropagation fills in the unlabeled samples.
    Uses a small dense subset for feasibility.
    """
    print(f"\n[Model A — Semi-Supervised] Label Propagation "
          f"({int(labeled_frac*100)}% labeled) …")

    # Small dense subset (LP is O(N²) in memory)
    n_sub = min(3_000, X_train.shape[0])
    rng   = np.random.default_rng(42)
    idx   = rng.choice(X_train.shape[0], size=n_sub, replace=False)
    X_sub = _to_dense_safe(X_train[idx])
    y_sub = y_train[idx].copy()

    # Mask most labels
    mask_idx = rng.choice(n_sub, size=int(n_sub * (1 - labeled_frac)), replace=False)
    y_masked = y_sub.copy()
    y_masked[mask_idx] = -1     # -1 means unlabeled in sklearn

    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=200)
    lp.fit(X_sub, y_masked)

    preds  = lp.predict(X_sub)
    acc    = accuracy_score(y_sub, preds)
    f1     = f1_score(y_sub, preds, average="macro", zero_division=0)
    print(f"  Label Propagation Accuracy : {acc:.4f}")
    print(f"  Label Propagation Macro F1 : {f1:.4f}")
    return lp, acc, f1


# ═══════════════════════════════════════════════════════════════════════════
# 6. SOFT-VOTING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

from src.inference import SoftVotingEnsemble

# ═══════════════════════════════════════════════════════════════════════════
# PREDICT BEST OPTION  (inference helper used by UI)
# ═══════════════════════════════════════════════════════════════════════════

def predict_best_option(article: str, question: str,
                         options: dict, model, vectorizer):
    """
    Given article, question and {'A':…,'B':…,'C':…,'D':…},
    return the predicted correct letter and probability scores.
    """
    from src.preprocessing import clean_text
    from scipy.sparse import hstack, csr_matrix
    import numpy as np

    art  = clean_text(article)
    q    = clean_text(question)
    texts, letters = [], []
    lex_rows = []
    art_words = set(art.split())
    q_words   = set(q.split())

    for letter, opt_text in options.items():
        opt = clean_text(opt_text)
        combined = f"{art} {art} {q} {opt}"
        texts.append(combined)
        letters.append(letter)
        # lexical features
        opt_words   = set(opt.split()) if opt else set()
        art_overlap = len(art_words & opt_words) / (len(art_words) + 1)
        q_overlap   = len(q_words   & opt_words) / (len(q_words)   + 1)
        opt_len     = len(opt_words) / 50.0
        art_len     = len(art_words) / 500.0
        lex_rows.append([art_overlap, q_overlap, opt_len, art_len])

    X_tfidf = vectorizer.transform(texts)
    X_lex   = csr_matrix(np.array(lex_rows, dtype=np.float32))
    X       = hstack([X_tfidf, X_lex])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]  # prob of label=1
    else:
        probs = model.decision_function(X)

    best_idx = int(np.argmax(probs))
    scores   = {l: float(p) for l, p in zip(letters, probs)}
    return letters[best_idx], scores


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main(sample_n: int = 20_000):
    print("=" * 60)
    print("  MODEL A — TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load & split ──
    train_df, val_df, test_df = load_and_split(sample_n=sample_n)

    # ── 2. Feature engineering ──
    X_tr, X_val, X_te, y_tr, y_val, y_te, vectorizer = \
        build_verification_features(train_df, val_df, test_df, save=True)

    results = []

    # ── 3. Train supervised models ──
    lr  = train_logistic_regression(X_tr, y_tr)
    svm = train_svm(X_tr, y_tr)
    nb  = train_naive_bayes(X_tr, y_tr)

    # ── 4. Evaluate supervised models on val set ──
    results.append(evaluate("Logistic Regression", lr, X_val, y_val, "Val"))
    results.append(evaluate("SVM (calibrated)",    svm, X_val, y_val, "Val"))

    # NB needs special wrapper
    class _NBWrapper:
        def __init__(self, nb): self._nb = nb
        def predict(self, X):   return self._nb.predict(X[:, :10_000])
    results.append(evaluate("Naive Bayes", _NBWrapper(nb), X_val, y_val, "Val"))

    # ── 5. Unsupervised & semi-supervised ──
    km, sil, purity = run_kmeans(X_tr, y_tr)
    lp, lp_acc, lp_f1 = run_label_propagation(X_tr, y_tr)

    # ── 6. Ensemble ──
    print("\n[Model A] Building Soft-Voting Ensemble …")
    ensemble = SoftVotingEnsemble(lr, svm, nb)
    results.append(evaluate("Soft-Voting Ensemble", ensemble, X_val, y_val, "Val"))

    # ── 7. Final test-set evaluation (best model = ensemble) ──
    print("\n[Model A] Final evaluation on TEST set:")
    results.append(evaluate("Soft-Voting Ensemble", ensemble, X_te, y_te, "Test"))

    # ── 8. Save models ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr,       os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(svm,      os.path.join(MODEL_DIR, "svm_calibrated.pkl"))
    joblib.dump(nb,       os.path.join(MODEL_DIR, "naive_bayes.pkl"))
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble.pkl"))
    print(f"\n[Model A] Models saved to {MODEL_DIR}")

    # ── 9. Results summary ──
    results_df = pd.DataFrame(results)
    print("\n\nRESULTS SUMMARY")
    print(results_df.to_string(index=False))

    # Save results
    proc_dir = os.path.join(os.path.dirname(MODEL_DIR), "..", "..", "data", "processed")
    results_df.to_csv(os.path.join(proc_dir, "model_a_results.csv"), index=False)

    return lr, svm, nb, ensemble, vectorizer


if __name__ == "__main__":
    main(sample_n=20_000)   # increase for better accuracy; full=None
