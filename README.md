# Chinese Sentiment Analysis Dashboard

A Streamlit app for Chinese e-commerce review sentiment analysis and opinion mining.

## Features

- Single-review sentiment classification
- Explicit vs implicit sentiment comparison
- Batch opinion-mining dashboard
- Hugging Face model inference with local keyword fallback

## Project structure

- `app.py`: Streamlit entrypoint
- `sentiment_utils.py`: sentiment analysis helpers
- `requirements.txt`: Python dependencies
- `tests/test_sentiment_utils.py`: unit tests
- `.streamlit/config.toml`: local Streamlit config

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

This app is ready to deploy from GitHub.

1. Push this folder to a standalone GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the entrypoint file to `app.py`.
4. In Advanced settings, choose Python `3.12` for best package compatibility.

The app will still run if the Hugging Face model can't be loaded at startup. In that case, it automatically falls back to local keyword rules.
