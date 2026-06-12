# AlphaForge AI

**A Retrieval-Augmented Framework for Explainable Financial Forecasting and Portfolio Intelligence**

AlphaForge AI integrates historical market data, machine learning forecasting (XGBoost & LSTM), a RAG knowledge base built from financial news, and a lightweight multi-agent reasoning system to generate **explainable BUY/HOLD/SELL recommendations** backed by quantitative signals and retrieved evidence.

---

## Table of Contents

- [Project Overview](#project-overview)
- [High-Level Architecture](#high-level-architecture)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Notebook 1: Data Engineering & Feature Pipeline](#notebook-1-data-engineering--feature-pipeline)
  - [Notebook 2: Financial Forecasting Models](#notebook-2-financial-forecasting-models)
  - [Notebook 3: RAG & Multi-Agent Reasoning](#notebook-3-rag--multi-agent-reasoning)
- [Results & Example Output](#results--example-output)
- [Research Questions Addressed](#research-questions-addressed)
- [Optional Extensions](#optional-extensions)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

Financial markets are driven by both **quantitative signals** (price, volume, technical indicators) and **qualitative information** (news, earnings reports, macroeconomic events). Traditional machine learning models often ignore textual context, while large language models struggle with factual accuracy and hallucination.

AlphaForge AI addresses both gaps by:

1. Forecasting future price direction with tree-based and deep learning models.
2. Retrieving the most relevant financial news using a vector similarity search (FAISS).
3. Using a **multi-agent reasoning layer** to combine the forecast, news sentiment, and market conditions into an **explainable recommendation** with supporting citations.

The system is designed to be **modular, reproducible, and extensible** -- all code is organized in three Jupyter notebooks that run end-to-end after a simple package installation.

---

## High-Level Architecture

```text
Historical Market Data (yfinance)
             +
Financial News Corpus (yfinance news)
             |
             v
Data Cleaning & Feature Engineering
             |
    +-------------------+
    |                   |
    v                   v
Forecasting Models   RAG Pipeline
 (XGBoost / LSTM) (Embeddings + FAISS)
    |                   |
    +---------+---------+
              |
     AI Reasoning Layer
  Agent 1: Quantitative Analyst
  Agent 2: News Intelligence
  Agent 3: Investment Advisor
              |
              v
     Forecast + Explanation
        Confidence Score
       Supporting Evidence
```

---

## Key Features

- **End-to-end pipeline** -- from raw market data to explainable recommendation.
- **Dual forecasting models** -- XGBoost (with hyperparameter tuning) and LSTM (PyTorch) to compare performance.
- **Comprehensive evaluation** -- RMSE, MAE, Directional Accuracy, and Sharpe Ratio.
- **RAG knowledge base** -- Financial news titles are embedded (`all-MiniLM-L6-v2`), indexed with FAISS, and retrieved based on the current market state.
- **FinBERT sentiment analysis** -- Each retrieved article is scored for positive, negative, and neutral sentiment.
- **Three-agent reasoning**:
  - **Quantitative Analyst** -- reads the forecast summary and model metrics.
  - **News Intelligence Agent** -- retrieves relevant news and computes aggregated sentiment.
  - **Investment Advisor** -- combines all signals to output BUY/HOLD/SELL with a confidence score and evidence.
- **Interactive Q&A (optional)** -- A simple chat assistant answers questions about the recommendation.
- **Fully modular** -- Change the ticker symbol in the configuration cell and rerun the notebooks to analyze any asset.

---

## Repository Structure

```text
AlphaForge-AI/
├── Notebook1_Data_Engineering.ipynb  # Data collection, cleaning, feature engineering
├── Notebook2_Forecasting_Models.ipynb # XGBoost and LSTM training, evaluation
├── Notebook3_RAG_Pipeline.ipynb       # RAG build, three agents, final recommendation
├── data/                              # Saved CSV files (created by Notebook 1)
│   ├── raw_prices.csv
│   └── processed_features.csv
├── results/                           # Model outputs (created by Notebook 2 & 3)
│   ├── forecast_predictions.csv
│   ├── model_comparison.csv
│   ├── forecast_summary.json
│   ├── news_index.faiss
│   └── news_metadata.json
├── README.md
└── .gitignore
```

---

## Technologies Used

| Area | Tools |
|------|-------|
| **Finance** | `yfinance`, `pandas`, `numpy` |
| **ML & DL** | `scikit-learn`, `XGBoost`, `PyTorch` |
| **NLP & Sentiment** | `FinBERT` (ProsusAI/finbert), `Sentence-Transformers` |
| **RAG** | `FAISS`, `langchain` (optional) |
| **Visualization** | `matplotlib`, `seaborn` |

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/AlphaForge-AI.git](https://github.com/your-username/AlphaForge-AI.git)
   cd AlphaForge-AI
   ```

2. **Create and activate a Python environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install yfinance pandas numpy matplotlib seaborn scikit-learn ta xgboost torch faiss-cpu sentence-transformers transformers
   ```
   > **Note:** If you have a GPU and want to use `faiss-gpu`, replace `faiss-cpu` with `faiss-gpu`.

4. **Launch Jupyter Notebook (or JupyterLab / VS Code):**
   ```bash
   jupyter notebook
   ```

---

## Usage

Run the three notebooks in order. They automatically save required files that the next notebook loads.

### Notebook 1: Data Engineering & Feature Pipeline
* Downloads 10 years of historical OHLCV data for AAPL (change the `TICKER` variable to any valid symbol).
* Cleans missing values, removes anomalies (|z| > 6), and engineers over 25 technical features (SMA, EMA, MACD, RSI, Bollinger Bands, OBV, ATR, etc.).
* Outputs `data/processed_features.csv`.
* **Time:** ~30 seconds.

### Notebook 2: Financial Forecasting Models
* Loads `data/processed_features.csv`, splits chronologically (80/20).
* Trains and tunes an XGBoost regressor and a 2-layer LSTM (with 30-day lookback).
* Evaluates both models on RMSE, MAE, Directional Accuracy, and Sharpe Ratio.
* **Saves:**
  * `results/forecast_predictions.csv` (test-period predictions)
  * `results/model_comparison.csv`
  * `results/forecast_summary.json` (most recent forecast + metrics)
* **Time:** ~3-5 minutes (LSTM training may take longer without GPU).

### Notebook 3: RAG & Multi-Agent Reasoning
* Fetches the latest financial news for the chosen ticker via `yfinance`.
* Embeds news headlines and creates a FAISS vector index.
* **Agent 1** reads `forecast_summary.json` to produce a quantitative analysis.
* **Agent 2** constructs a query, retrieves the top-k news articles, and runs FinBERT sentiment.
* **Agent 3** combines quantitative and sentiment signals, applies a rule-based logic, and outputs:

```text
Recommendation: BUY / HOLD / SELL
Confidence: XX.X%
Reasons: [list]
Supporting Evidence: [article titles + links]
```

* An optional chat assistant can answer questions like:
  * *"Why is the model recommending BUY?"*
  * *"What news influenced the prediction?"*
  * *"What are the biggest risks?"*
* **Time:** ~2-4 minutes (first run downloads FinBERT and the embedding model).

---

## Results & Example Output

**Forecast Summary (latest day):**
```text
Ticker: AAPL
Date: 2024-12-31
Last Close: $250.42
Predicted Next Close: $252.15 (+0.69%)
Direction: UP
Primary Model: XGBoost
Model Confidence: 72.35% (directional accuracy)
Recent Volatility (20d): 0.015
RSI(14): 58.4
```

**Final Recommendation:**
```text
Recommendation: BUY
Confidence: 78.4%
Reasons:
  - Quantitative model predicts a 0.69% increase.
  - News sentiment is predominantly positive.
  - RSI indicates neutral momentum.
Supporting Evidence:
  [1] Apple reports record holiday quarter revenue -- Reuters
  [2] Tech sector rallies on AI optimism -- Bloomberg
```

---

## Research Questions Addressed

* **Does adding textual information improve prediction accuracy?**
  The RAG pipeline provides evidence that can explain model decisions; quantitative improvement is not directly measured, but the framework allows easy A/B testing with/without news.
* **Which forecasting model generalizes best?**
  The comparison between XGBoost and LSTM (RMSE, directional accuracy, Sharpe ratio) answers this for the chosen ticker.
* **Does retrieval reduce hallucination?**
  By grounding the explanation in real, retrieved articles (with citations), the system reduces unverified claims.
* **Can the AI justify its recommendations?**
  Yes -- the final output includes explicit reasoning and links to source articles.
* **Are retrieved documents aligned with market behaviour?**
  The query construction uses current market state (direction, volatility, RSI), making the retrieved context relevant.

---

## Optional Extensions

The project includes an interactive Financial RAG Chat Assistant (Notebook 3, Section 5). Additionally, the architecture supports:
* **Portfolio Optimization** -- Use predicted returns to construct an efficient frontier.
* **Market Stress Analysis** -- Evaluate model performance during high-volatility periods.
* **Live Dashboard** -- Extend the chat assistant with a web interface.

---

## Future Work

* Incorporate full news article bodies (instead of only titles) for deeper context.
* Replace rule-based advisor with a fine-tuned LLM for natural language reasoning.
* Integrate alternative data sources (SEC filings, earnings call transcripts).
* Add backtesting engine and performance tracking.
* Dockerise the pipeline for easy deployment.

---

## License

This project is provided for educational and research purposes. See `LICENSE` file for details.
Built as part of the AlphaForge AI research framework.
