# Sentiment Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit sentiment analysis and opinion mining dashboard for the Week 10 lab.

**Architecture:** Keep model and batch-analysis logic in `sentiment_utils.py`, with the Streamlit interface in `app.py`. Use Plotly for the confidence gauge and opinion-mining charts, and provide a keyword fallback if the Hugging Face pipeline is unavailable.

**Tech Stack:** Python, Streamlit, Hugging Face Transformers, PyTorch, Plotly, Pandas, Pytest.

---

### Task 1: Deterministic Sentiment Utilities

**Files:**
- Create: `sentiment_utils.py`
- Create: `tests/test_sentiment_utils.py`

- [ ] Write tests for label normalization, fallback classification, sample data generation, and batch summary.
- [ ] Run `python3 -m pytest tests/test_sentiment_utils.py -q` and confirm the missing module failure.
- [ ] Implement utility functions in `sentiment_utils.py`.
- [ ] Run `python3 -m pytest tests/test_sentiment_utils.py -q` and confirm all tests pass.

### Task 2: Streamlit Dashboard UI

**Files:**
- Create: `app.py`
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] Implement a dark technology-themed Streamlit page with three tabs.
- [ ] Add a single-review analyzer with a Plotly gauge.
- [ ] Add explicit and implicit sentiment comparison inputs.
- [ ] Add the batch opinion mining dashboard with generated reviews, metrics, pie chart, and results table.
- [ ] Add dependency metadata and ignore generated caches.

### Task 3: Verification

**Files:**
- Verify: `app.py`
- Verify: `sentiment_utils.py`
- Verify: `tests/test_sentiment_utils.py`

- [ ] Run `python3 -m pytest -q`.
- [ ] Run `python3 -m py_compile app.py sentiment_utils.py`.
- [ ] Try starting `streamlit run app.py` if Streamlit is installed.
- [ ] Report any missing dependencies or network/model-download constraints clearly.
