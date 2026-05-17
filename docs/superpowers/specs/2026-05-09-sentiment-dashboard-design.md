# Sentiment Dashboard Design

## Goal

Build a Streamlit web platform for the Week 10 lab: fine-grained sentiment analysis and opinion mining for Chinese e-commerce reviews.

## Confirmed Direction

Use the B direction: a modern data-screen style with dark dashboard surfaces, metric cards, Plotly charts, and three tabs matching the lab modules.

## Features

1. Single-review sentiment analysis
   - Accept one Chinese product review.
   - Run a Hugging Face sentiment-analysis pipeline.
   - Display Positive, Neutral, or Negative.
   - Display model confidence using a Plotly semicircle gauge.

2. Explicit vs implicit sentiment comparison
   - Provide one input for explicit emotional wording.
   - Provide one input for implicit objective description.
   - Analyze both inputs independently.
   - Explain explicit sentiment and implicit sentiment in concise Chinese.

3. Opinion mining dashboard
   - Generate 10-15 simulated product reviews.
   - Analyze reviews in batch.
   - Count Positive, Neutral, and Negative reviews.
   - Show sentiment distribution with a Plotly pie chart.
   - Show the analyzed review table and summary metrics.

## Technical Approach

The app will use `app.py` for Streamlit UI and `sentiment_utils.py` for deterministic sentiment helpers and model wrapping. Hugging Face will be the primary classifier with `lxyuan/distilbert-base-multilingual-cased-sentiments-student`; if loading or inference fails, a small Chinese keyword fallback keeps the classroom demo usable.

## Error Handling

Model loading and inference failures should not crash the app. The UI should show a warning when fallback mode is active.

## Testing

Unit tests will cover deterministic helper behavior: label normalization, keyword fallback, confidence bounds, sample data shape, and batch aggregation.
