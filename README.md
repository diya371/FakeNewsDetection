# Fake News Detection on ISOT Dataset (Classical ML + DistilBERT + Hybrid)

This repository contains a minimal implementation of my fake news detection project.
It compares classical machine learning models, a transformer-based model, and a hybrid model on the ISOT Fake News Dataset.

## 🔍 Overview

**Goal:** Detect fake news articles using NLP and compare different model families:

- Classical ML models on TF–IDF features:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM
  - Random Forest
  - XGBoost
- Transformer-based model:
  - Frozen DistilBERT + Logistic Regression
- Hybrid model:
  - DistilBERT CLS embeddings + TF–IDF (reduced with TruncatedSVD) + Logistic Regression

The hybrid model combines lexical features (TF–IDF) and contextual features (DistilBERT) and typically gives the best performance on the ISOT dataset.

## 📂 Dataset

This project uses the **ISOT Fake News Dataset**.

You can download it from the official source and place the CSV files in a `data/` folder:

- `data/Fake.csv`
- `data/True.csv`

Each file should contain at least:
- `title`
- `text`

The script will assign:
- `label = 1` for fake news
- `label = 0` for real news

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt
