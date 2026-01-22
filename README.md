
# 📈 LSTM-Based Financial Time Series Analysis and Forecasting

This project implements a complete, end-to-end financial time series analysis and forecasting pipeline using **Long Short-Term Memory (LSTM)** networks.  
It covers data collection, preprocessing, exploratory analysis, feature engineering, deep learning modeling, and evaluation using real-world market data.

The project is structured as a **modular, testable, and reproducible ML system**, following industry and research best practices.

---

## 🎯 Project Objectives

- Analyze historical financial time series data
- Study stationarity, volatility, and risk metrics
- Prepare data for sequence-based deep learning models
- Train and evaluate an LSTM model for price forecasting
- Apply professional software engineering practices (testing, structure, reproducibility)

---

## 📊 Data Description

Financial market data is sourced using **Yahoo Finance** (`yfinance`) for the following assets:

| Ticker | Description |
|------|------------|
| TSLA | Tesla Inc. (Equity) |
| BND  | Vanguard Total Bond Market ETF |
| SPY  | S&P 500 ETF |

Data fields include:
- Open
- High
- Low
- Close
- Volume

---

## 🧠 Tasks Overview

### **Task 1 — Data Collection & Preprocessing**
- Download historical data
- Handle missing values
- Compute:
  - Daily returns
  - Rolling volatility
  - Risk metrics (VaR, Sharpe Ratio)
- Stationarity analysis using ADF test
- Unit tests for data integrity

### **Task 2 — Exploratory Data Analysis (EDA)**
- Price trend visualization
- Return distributions
- Volatility analysis
- Correlation insights
- Rolling statistics

### **Task 3 — Feature Engineering**
- Lag features
- Rolling statistics
- Normalization & scaling
- Sequence creation for LSTM

### **Task 4 — LSTM Modeling**
- Train-test split for time series
- LSTM architecture design
- Model training & checkpointing
- Loss visualization

### **Task 5 — Model Evaluation**
- Prediction vs actual comparison
- Error metrics (RMSE, MAE)
- Visual diagnostics
- Interpretation of results

---

## 🗂️ Project Structure

```

LSTM/
│
├── data/
│   ├── raw/            # Raw CSV files (ignored in git)
│   ├── processed/      # Cleaned data
│   └── features/       # LSTM-ready features
│
├── notebooks/
│   ├── 01_task1_preprocessing.ipynb
│   ├── 02_task2_eda.ipynb
│   ├── 03_task3_features.ipynb
│   ├── 04_task4_lstm.ipynb
│   └── 05_task5_evaluation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── test_data_loader.py
│   └── test_preprocessing.py
│
├── models/             # Saved models (ignored)
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 🧪 Testing

Unit tests ensure:
- Data is loaded correctly
- Expected columns exist
- Returns & volatility are computed
- Statistical tests run without errors

Run tests:
```bash
python -m unittest discover -s tests
````

---

## ⚙️ Setup Instructions

```bash
git clone <repo-url>
cd LSTM
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

```bash
# Fetch data
python -m src.data_loader

# Preprocess data
python -m src.preprocessing TSLA
python -m src.preprocessing BND
python -m src.preprocessing SPY
```

Notebook-based analysis:

```bash
jupyter notebook
```

---

## 📌 Notes

* Raw data and trained models are intentionally excluded from version control
* The project is designed for **educational, research, and portfolio use**
* Structure supports easy CI/CD integration

