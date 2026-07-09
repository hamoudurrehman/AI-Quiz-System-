"""
app.py
======
Streamlit UI — Intelligent Reading Comprehension & Quiz Generation System

Screens
-------
  Screen 1 — Article Input
  Screen 2 — Question & Answer Quiz View
  Screen 3 — Hint Panel
  Screen 4 — Developer / Analytics Dashboard

Run:
    streamlit run ui/app.py
"""

import os
import sys
import time
import random
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title  = "RACE Quiz System",
    page_icon   = "📚",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.2rem; font-weight:800; color:#1a237e; }
    .section-hdr  { font-size:1.3rem; font-weight:700; color:#283593; margin-top:1rem; }
    .hint-box     { background:#e8f5e9; border-left:4px solid #43a047;
                    padding:10px 14px; border-radius:6px; margin:6px 0; }
    .correct-box  { background:#e8f5e9; border:2px solid #43a047;
                    padding:12px; border-radius:8px; }
    .wrong-box    { background:#ffebee; border:2px solid #e53935;
                    padding:12px; border-radius:8px; }
    .metric-card  { background:#f5f5f5; border-radius:8px;
                    padding:12px; text-align:center; }
    div[data-testid="stRadio"] > label { font-size:1.0rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ML models …")
def load_engine():
    """Load the inference engine once for all sessions."""
    from src.inference import RACEInferenceEngine
    engine = RACEInferenceEngine()
    success = engine.load()
    return engine, success


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset(csv_path: str, nrows: int = 5_000):
    df = pd.read_csv(csv_path, index_col=0, nrows=nrows)
    for col in ["article", "question", "A", "B", "C", "D"]:
        df[col] = df[col].fillna("").astype(str)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "article":          "",
        "question":         "",
        "options":          {"A": "", "B": "", "C": "", "D": ""},
        "correct_answer":   "",
        "distractors":      [],
        "hints":            [],
        "hints_revealed":   0,
        "user_answer":      None,
        "answer_checked":   False,
        "session_log":      [],
        "current_screen":   "Article Input",
        "model_a_metrics":  [],
        "model_b_metrics":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📚 RACE Quiz System")
    st.markdown("*AI-powered Reading Comprehension*")
    st.divider()

    screen = st.radio(
        "Navigate to:",
        ["Article Input", "Quiz View", "Hint Panel", "Analytics Dashboard"],
        key="nav_radio"
    )
    st.session_state["current_screen"] = screen

    st.divider()
    st.markdown("**Dataset path**")
    csv_path = st.text_input(
        "CSV file path",
        value=os.path.join(ROOT, "data", "raw", "dev.csv"),
        label_visibility="collapsed"
    )
    st.caption("Uses the first 5,000 rows for random sampling")
    st.divider()
    st.info("📌 Models must be trained first.\nRun `python src/model_a_train.py`\nand `python src/model_b_train.py`")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ENGINE
# ═══════════════════════════════════════════════════════════════════════════

engine, model_ready = load_engine()

if not model_ready:
    st.warning(
        f"⚠️ Models not found. Please train them first.  \n"
        f"Error: {engine.load_error}  \n\n"
        f"```bash\n"
        f"cd race_rc_project\n"
        f"python src/model_a_train.py\n"
        f"python src/model_b_train.py\n"
        f"```"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 1 — ARTICLE INPUT
# ═══════════════════════════════════════════════════════════════════════════


if screen == "Article Input":
    st.markdown('<p class="main-title">📖 Screen 1 — Article Input</p>',
                unsafe_allow_html=True)
    st.caption("Paste a reading passage or load a random sample from RACE.")

    col_left, col_right = st.columns([3, 1])

    with col_right:
        if st.button("🎲 Load Random RACE Sample", use_container_width=True,
                     disabled=not os.path.exists(csv_path)):
            try:
                df = load_dataset(csv_path)
                row = df.sample(1).iloc[0]
                st.session_state["article"]        = row["article"]
                st.session_state["question"]        = row["question"]
                st.session_state["options"]         = {
                    "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]
                }
                st.session_state["correct_answer"]  = row["answer"]
                st.session_state["answer_checked"]  = False
                st.session_state["user_answer"]     = None
                st.session_state["hints_revealed"]  = 0
                st.session_state["hints"]           = []
                st.session_state["distractors"]     = []
                st.success("Sample loaded ✓")
            except Exception as e:
                st.error(f"Could not load dataset: {e}")

        st.markdown("**Or enter manually:**")

    with col_left:
        article_input = st.text_area(
            "Reading Passage",
            value=st.session_state["article"],
            height=250,
            placeholder="Paste or type a reading passage here …",
        )
        st.session_state["article"] = article_input

    st.divider()

    col_q, col_ans = st.columns([3, 1])
    with col_q:
        q_input = st.text_input(
            "Question",
            value=st.session_state["question"],
            placeholder="Enter the question …"
        )
        st.session_state["question"] = q_input

    with col_ans:
        ans_input = st.selectbox(
            "Correct Answer (for RACE rows)",
            ["A", "B", "C", "D"],
            index=["A","B","C","D"].index(
                st.session_state["correct_answer"]
                if st.session_state["correct_answer"] in ["A","B","C","D"] else "A"
            )
        )
        st.session_state["correct_answer"] = ans_input

    # Option inputs
    st.markdown('<p class="section-hdr">Answer Options</p>', unsafe_allow_html=True)
    o_cols = st.columns(4)
    for i, letter in enumerate(["A","B","C","D"]):
        with o_cols[i]:
            val = st.text_area(
                f"Option {letter}",
                value=st.session_state["options"].get(letter, ""),
                height=80
            )
            st.session_state["options"][letter] = val

    st.divider()

    submit_col, _ = st.columns([1, 3])
    with submit_col:
        submit_clicked = st.button(
            "🚀 Submit — Run Model A & B",
            type="primary",
            use_container_width=True,
            disabled=not model_ready
        )

    if submit_clicked:
        article  = st.session_state["article"].strip()
        question = st.session_state["question"].strip()
        options  = st.session_state["options"]

        if not article:
            st.error("Please provide a reading passage.")
        elif not question:
            st.error("Please provide a question.")
        elif not all(options[l].strip() for l in "ABCD"):
            st.error("Please fill in all four answer options.")
        else:
            with st.spinner("Running Model A (Answer Verifier) …"):
                try:
                    result = engine.predict_answer(article, question, options)
                    st.session_state["model_a_metrics"].append({
                        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
                        "question":   question[:60],
                        "predicted":  result["predicted_letter"],
                        "correct":    st.session_state["correct_answer"],
                        "is_correct": result["predicted_letter"] == st.session_state["correct_answer"],
                        "latency_ms": result["latency_ms"],
                    })
                    st.session_state["predicted_answer"]   = result["predicted_letter"]
                    st.session_state["confidence_scores"]  = result["confidence_scores"]
                except Exception as e:
                    st.error(f"Model A error: {e}")

            with st.spinner("Running Model B (Distractors & Hints) …"):
                try:
                    correct_text = options[st.session_state["correct_answer"]]
                    distractors  = engine.generate_distractors(article, correct_text)
                    hints        = engine.generate_hints(article, question)
                    st.session_state["distractors"]    = distractors
                    st.session_state["hints"]          = hints
                    st.session_state["hints_revealed"] = 0
                    st.session_state["answer_checked"] = False
                    st.session_state["user_answer"]    = None
                except Exception as e:
                    st.error(f"Model B error: {e}")

            st.success("✓ Both models ran successfully. Go to **Quiz View** →")


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 2 — QUIZ VIEW
# ═══════════════════════════════════════════════════════════════════════════

elif screen == "Quiz View":
    st.markdown('<p class="main-title">🧠 Screen 2 — Question & Answer Quiz</p>',
                unsafe_allow_html=True)

    article  = st.session_state.get("article", "")
    question = st.session_state.get("question", "")
    options  = st.session_state.get("options", {})
    correct  = st.session_state.get("correct_answer", "")

    if not article or not question:
        st.info("No article loaded yet. Go to **Article Input** first.")
        st.stop()

    # Show article
    with st.expander("📄 Reading Passage", expanded=True):
        st.write(article)

    st.divider()
    st.markdown(f"### ❓ {question}")
    st.divider()

    # Radio buttons for answer selection
    option_labels = [f"**{l}.**  {options.get(l,'')}" for l in ["A","B","C","D"]]
    user_pick = st.radio(
        "Choose your answer:",
        ["A", "B", "C", "D"],
        format_func=lambda l: f"{l}.  {options.get(l,'')}",
        index=None,
        key="quiz_radio"
    )
    st.session_state["user_answer"] = user_pick

    check_col, _ = st.columns([1, 4])
    with check_col:
        check_btn = st.button("✅ Check Answer", type="primary",
                              use_container_width=True,
                              disabled=(user_pick is None))

    if check_btn and user_pick:
        st.session_state["answer_checked"] = True
        # Log result
        log_entry = {
            "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
            "question":   question[:60],
            "user_pick":  user_pick,
            "correct":    correct,
            "is_correct": user_pick == correct,
        }
        st.session_state["session_log"].append(log_entry)

    if st.session_state.get("answer_checked"):
        user_pick = st.session_state.get("user_answer", "")
        is_correct = (user_pick == correct)

        if is_correct:
            st.markdown(
                f'<div class="correct-box">✅ <b>Correct!</b> '
                f'The answer is <b>{correct}</b>: {options.get(correct,"")}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="wrong-box">❌ <b>Incorrect.</b> '
                f'You chose <b>{user_pick}</b>.  '
                f'The correct answer is <b>{correct}</b>: {options.get(correct,"")}</div>',
                unsafe_allow_html=True
            )

        # Model prediction
        pred = st.session_state.get("predicted_answer", "—")
        conf = st.session_state.get("confidence_scores", {})
        st.divider()
        st.markdown("**🤖 Model A Prediction**")
        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            st.metric("Predicted Answer", pred,
                      delta="✓ Matches correct" if pred == correct else "✗ Incorrect",
                      delta_color="normal")
        with pcol2:
            if conf:
                conf_df = pd.DataFrame({
                    "Option": list(conf.keys()),
                    "Confidence": list(conf.values())
                })
                fig = px.bar(conf_df, x="Option", y="Confidence",
                             color="Option",
                             title="Answer Confidence Scores",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(fig, use_container_width=True)

        # Model B distractors
        distractors = st.session_state.get("distractors", [])
        if distractors:
            st.divider()
            st.markdown("**🎯 Model B — Generated Distractors**")
            st.caption("These are AI-generated plausible wrong options:")
            for i, d in enumerate(distractors, 1):
                st.markdown(f"  `{i}.` {d}")


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 3 — HINT PANEL
# ═══════════════════════════════════════════════════════════════════════════

elif screen == "Hint Panel":
    st.markdown('<p class="main-title">💡 Screen 3 — Hint Panel</p>',
                unsafe_allow_html=True)

    hints    = st.session_state.get("hints", [])
    question = st.session_state.get("question", "")
    correct  = st.session_state.get("correct_answer", "")
    options  = st.session_state.get("options", {})

    if not hints:
        st.info("No hints yet. Submit an article on the **Article Input** screen first.")
        st.stop()

    st.markdown(f"**Question:** {question}")
    st.divider()

    revealed = st.session_state.get("hints_revealed", 0)

    hint_labels = ["🌱 Hint 1 — General Clue",
                   "🌿 Hint 2 — More Specific",
                   "🌳 Hint 3 — Near-Explicit"]

    for i in range(3):
        if i < revealed:
            st.markdown(
                f'<div class="hint-box"><b>{hint_labels[i]}</b><br>'
                f'{hints[i] if i < len(hints) else "—"}</div>',
                unsafe_allow_html=True
            )
        else:
            st.button(
                f"Reveal {hint_labels[i]}",
                key=f"hint_btn_{i}",
                on_click=lambda: st.session_state.update(
                    {"hints_revealed": st.session_state["hints_revealed"] + 1}
                ),
                disabled=(i > revealed)
            )

    # Re-check state after button press
    if st.session_state.get("hints_revealed", 0) >= 3:
        st.divider()
        if st.button("🔓 Reveal Answer", type="primary"):
            st.success(f"✅ The correct answer is **{correct}**: {options.get(correct,'')}")


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 4 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

elif screen == "Analytics Dashboard":
    st.markdown('<p class="main-title">📊 Screen 4 — Analytics Dashboard</p>',
                unsafe_allow_html=True)

    # ── Model A metrics ──
    st.markdown('<p class="section-hdr">Model A — Answer Verifier Performance</p>',
                unsafe_allow_html=True)

    model_a_log = st.session_state.get("model_a_metrics", [])

    if model_a_log:
        log_df = pd.DataFrame(model_a_log)

        # Summary metrics
        total     = len(log_df)
        correct_n = log_df["is_correct"].sum()
        accuracy  = correct_n / total
        avg_lat   = log_df["latency_ms"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Inferences",        total)
        c2.metric("Correct Predictions", int(correct_n))
        c3.metric("Accuracy",          f"{accuracy:.1%}")
        c4.metric("Avg Latency",       f"{avg_lat:.0f} ms")

        # Confusion-like bar
        pred_counts = log_df["predicted"].value_counts().reset_index()
        pred_counts.columns = ["Letter", "Count"]
        fig_pred = px.bar(pred_counts, x="Letter", y="Count",
                          title="Predicted Answer Distribution",
                          color="Letter",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pred.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Raw log table
        with st.expander("📋 Session Log"):
            st.dataframe(log_df, use_container_width=True)
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv,
                               "model_a_session_log.csv", "text/csv")
    else:
        st.info("No Model A inferences yet. Submit an article on **Article Input**.")

    st.divider()

    # ── Session quiz log ──
    st.markdown('<p class="section-hdr">User Quiz Session Log</p>',
                unsafe_allow_html=True)

    session_log = st.session_state.get("session_log", [])
    if session_log:
        sess_df = pd.DataFrame(session_log)
        total_q  = len(sess_df)
        user_acc = sess_df["is_correct"].mean()

        q1, q2 = st.columns(2)
        q1.metric("Questions Attempted", total_q)
        q2.metric("User Accuracy",       f"{user_acc:.1%}")

        st.dataframe(sess_df, use_container_width=True)
        csv2 = sess_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Quiz Log", csv2,
                           "quiz_session_log.csv", "text/csv")
    else:
        st.info("No quiz attempts yet. Try answering questions in **Quiz View**.")

    st.divider()

    # ── Latency tracking ──
    if model_a_log:
        st.markdown('<p class="section-hdr">Inference Latency (ms)</p>',
                    unsafe_allow_html=True)
        lat_df = pd.DataFrame({"Inference #": range(1, len(model_a_log)+1),
                                "Latency (ms)": [m["latency_ms"] for m in model_a_log]})
        fig_lat = px.line(lat_df, x="Inference #", y="Latency (ms)",
                          title="Per-Request Latency", markers=True)
        fig_lat.add_hline(y=10_000, line_dash="dash",
                          annotation_text="10s limit", line_color="red")
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Model files status ──
    st.divider()
    st.markdown('<p class="section-hdr">Model Files Status</p>',
                unsafe_allow_html=True)
    model_files = {
        "TF-IDF Vectorizer":    os.path.join(ROOT, "models","model_a","traditional","tfidf_vectorizer.pkl"),
        "LR Classifier":        os.path.join(ROOT, "models","model_a","traditional","logistic_regression.pkl"),
        "SVM Classifier":       os.path.join(ROOT, "models","model_a","traditional","svm_calibrated.pkl"),
        "Naive Bayes":          os.path.join(ROOT, "models","model_a","traditional","naive_bayes.pkl"),
        "Ensemble":             os.path.join(ROOT, "models","model_a","traditional","ensemble.pkl"),
        "Distractor Ranker":    os.path.join(ROOT, "models","model_b","traditional","distractor_ranker.pkl"),
        "Hint Scorer":          os.path.join(ROOT, "models","model_b","traditional","hint_scorer.pkl"),
    }
    status_rows = []
    for name, path in model_files.items():
        exists = os.path.exists(path)
        size   = f"{os.path.getsize(path)/1024:.1f} KB" if exists else "—"
        status_rows.append({"Model File": name, "Status": "✅ Found" if exists else "❌ Missing", "Size": size})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True)
