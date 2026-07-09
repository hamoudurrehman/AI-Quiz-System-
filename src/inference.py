"""
inference.py
============
Unified inference API — loads both trained models and exposes
simple functions that the UI layer calls.

Usage
-----
    from src.inference import RACEInferenceEngine

    engine = RACEInferenceEngine()
    result = engine.run(article, question, options)
    # result keys: predicted_answer, confidence_scores,
    #              distractors, hints
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_A_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
MODEL_B_DIR = os.path.join(BASE_DIR, "models", "model_b", "traditional")

# ── SoftVotingEnsemble defined HERE so pickle can always find it ─────────────
class SoftVotingEnsemble:
    """Average probability from LR, SVM (calibrated), and NB."""

    def __init__(self, lr, svm, nb, weights=(0.4, 0.4, 0.2)):
        self.lr  = lr
        self.svm = svm
        self.nb  = nb
        self.w   = np.array(weights)

    def predict_proba(self, X):
        p_lr  = self.lr.predict_proba(X)
        p_svm = self.svm.predict_proba(X)
        p_nb  = self.nb.predict_proba(X[:, :10_000])
        return self.w[0]*p_lr + self.w[1]*p_svm + self.w[2]*p_nb

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
class RACEInferenceEngine:
    """Loads both models once and runs inference on demand."""

    def __init__(self):
        self._model_a    = None
        self._vectorizer = None
        self._distractor = None
        self._hint_scorer = None
        self._ready      = False
        self._load_error = None

    # ── Loading ─────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load all saved models. Returns True on success."""
        try:
            vec_path = os.path.join(MODEL_A_DIR, "tfidf_vectorizer.pkl")
            ens_path = os.path.join(MODEL_A_DIR, "ensemble.pkl")
            dis_path = os.path.join(MODEL_B_DIR, "distractor_ranker.pkl")
            hin_path = os.path.join(MODEL_B_DIR, "hint_scorer.pkl")

            for p in [vec_path, ens_path, dis_path, hin_path]:
                if not os.path.exists(p):
                    self._load_error = f"Model file not found: {p}"
                    return False

            self._vectorizer  = joblib.load(vec_path)
            self._model_a     = joblib.load(ens_path)
            self._distractor  = joblib.load(dis_path)
            self._hint_scorer = joblib.load(hin_path)
            self._ready       = True
            return True

        except Exception as e:
            self._load_error = str(e)
            return False

    @property
    def is_ready(self):
        return self._ready

    @property
    def load_error(self):
        return self._load_error

    # ── Core inference ───────────────────────────────────────────────────

    def predict_answer(self, article: str, question: str, options: dict) -> dict:
        """
        Predict which option is correct.

        Parameters
        ----------
        article  : str
        question : str
        options  : dict  {'A': text, 'B': text, 'C': text, 'D': text}

        Returns
        -------
        dict with keys:
          predicted_letter, confidence_scores, latency_ms
        """
        if not self._ready:
            raise RuntimeError("Models not loaded. Call engine.load() first.")

        from src.model_a_train import predict_best_option

        t0 = time.time()
        letter, scores = predict_best_option(
            article, question, options, self._model_a, self._vectorizer
        )
        latency = (time.time() - t0) * 1000

        # Normalise scores to [0,1]
        vals  = np.array(list(scores.values()))
        shift = vals - vals.min()
        total = shift.sum() + 1e-9
        norm  = {k: float(shift[i] / total) for i, k in enumerate(scores)}

        return {
            "predicted_letter":  letter,
            "confidence_scores": norm,
            "latency_ms":        round(latency, 1),
        }

    def generate_distractors(self, article: str, correct_answer: str) -> list:
        """Return 3 plausible distractor strings."""
        if not self._ready:
            raise RuntimeError("Models not loaded.")
        from src.model_b_train import rank_distractors
        return rank_distractors(article, correct_answer, self._distractor)

    def generate_hints(self, article: str, question: str) -> list:
        """Return [hint1_general, hint2_medium, hint3_specific]."""
        if not self._ready:
            raise RuntimeError("Models not loaded.")
        from src.model_b_train import score_hints
        return score_hints(article, question, self._hint_scorer, top_k=3)

    # ── Full pipeline ────────────────────────────────────────────────────

    def run(self, article: str, question: str, options: dict = None) -> dict:
        """
        Full pipeline for a single article + question.

        If `options` is None (question generated from article), we build
        the option set using distractors from Model B.

        Returns
        -------
        dict with all keys needed by the UI.
        """
        if not self._ready:
            raise RuntimeError("Models not loaded.")

        t_total = time.time()

        # Hints first (always available)
        hints = self.generate_hints(article, question)

        if options is not None:
            # RACE-style: 4 options already known, just verify
            verification = self.predict_answer(article, question, options)
            return {
                "mode":             "verification",
                "predicted_letter": verification["predicted_letter"],
                "confidence_scores":verification["confidence_scores"],
                "hints":            hints,
                "latency_ms":       round((time.time() - t_total) * 1000, 1),
            }
        else:
            raise ValueError("options dict is required in current implementation.")

    # ── Session logging ───────────────────────────────────────────────────

    def log_session(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct columns for the UI log."""
        return pd.DataFrame(columns=[
            "timestamp", "question_snippet",
            "predicted", "correct", "is_correct",
            "confidence", "latency_ms"
        ])


# ── Singleton for Streamlit ─────────────────────────────────────────────

_engine_instance = None

def get_engine() -> RACEInferenceEngine:
    """Return a cached singleton (avoids reloading models on every rerun)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RACEInferenceEngine()
    return _engine_instance
