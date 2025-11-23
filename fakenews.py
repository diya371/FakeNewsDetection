import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from transformers import AutoTokenizer, AutoModel


# ---------- CONFIG ----------

DATA_DIR = "data"  # folder where ISOT CSVs are stored
FAKE_FILE = "Fake.csv"
REAL_FILE = "True.csv"

DISTILBERT_MODEL = "distilbert-base-uncased"
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
SVD_COMPONENTS = 100
RANDOM_STATE = 42


# ---------- DATA LOADING & PREPROCESSING ----------

def load_isot(data_dir: str) -> pd.DataFrame:
    """
    Expects ISOT dataset CSVs:
      - Fake.csv
      - True.csv
    with at least: 'title', 'text'
    """
    fake_path = os.path.join(data_dir, FAKE_FILE)
    real_path = os.path.join(data_dir, REAL_FILE)

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = 1  # 1 = fake
    real_df["label"] = 0  # 0 = real

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """
    Very basic preprocessing:
    - lowercase
    - remove URLs
    - remove non-alphabetic chars
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(df: pd.DataFrame):
    """
    Merge title + text, clean, and split into train/test.
    """
    df["combined"] = (df["title"].fillna("") + " " +
                      df["text"].fillna(""))

    df["cleaned"] = df["combined"].apply(clean_text)

    X = df["cleaned"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------- TF–IDF + CLASSICAL MODELS ----------

def build_tfidf_features(X_train, X_test):
    """
    Fit TF–IDF on training data and transform both train and test.
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Fit model and print accuracy & F1-score.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n[{name}]")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    return acc, f1


def run_classical_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Train and evaluate classical ML models on TF–IDF features.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="liblinear"
        ),
        "Multinomial NB": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist"
        ),
    }

    results = {}
    for name, clf in models.items():
        acc, f1 = evaluate_model(name, clf, X_train_tfidf, y_train, X_test_tfidf, y_test)
        results[name] = {"accuracy": acc, "f1": f1}
    return results


# ---------- DISTILBERT EMBEDDINGS ----------

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_distilbert_embeddings(texts, tokenizer, model, batch_size=16, max_length=128):
    """
    Convert list of texts to CLS embeddings using DistilBERT.
    """
    device = get_device()
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i+batch_size])

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # DistilBERT: last_hidden_state[:,0,:] is [CLS]-like embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings


def build_distilbert_features(X_train, X_test):
    """
    Load DistilBERT and compute CLS embeddings for train and test.
    """
    print("\nLoading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    model = AutoModel.from_pretrained(DISTILBERT_MODEL)

    print("Extracting DistilBERT embeddings for training set...")
    X_train_embed = get_distilbert_embeddings(
        X_train,
        tokenizer,
        model,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LEN
    )

    print("Extracting DistilBERT embeddings for test set...")
    X_test_embed = get_distilbert_embeddings(
        X_test,
        tokenizer,
        model,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LEN
    )

    return X_train_embed, X_test_embed


def run_distilbert_classifier(X_train_embed, X_test_embed, y_train, y_test):
    """
    Train Logistic Regression on DistilBERT embeddings.
    """
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    )
    print("\nTraining Logistic Regression on DistilBERT embeddings...")
    clf.fit(X_train_embed, y_train)
    y_pred = clf.predict(X_test_embed)
    print("\n[DistilBERT + LR]")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    return clf


# ---------- HYBRID MODEL (DistilBERT + TF–IDF SVD) ----------

def build_hybrid_features(X_train_tfidf, X_test_tfidf, X_train_embed, X_test_embed):
    """
    Reduce TF–IDF via SVD and concatenate with DistilBERT embeddings.
    """
    print("\nRunning TruncatedSVD on TF–IDF features...")
    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=RANDOM_STATE
    )
    X_train_tfidf_svd = svd.fit_transform(X_train_tfidf)
    X_test_tfidf_svd = svd.transform(X_test_tfidf)

    X_train_hybrid = np.hstack([X_train_embed, X_train_tfidf_svd])
    X_test_hybrid = np.hstack([X_test_embed, X_test_tfidf_svd])

    return X_train_hybrid, X_test_hybrid, svd


def run_hybrid_classifier(X_train_hybrid, X_test_hybrid, y_train, y_test):
    """
    Train Logistic Regression on hybrid features.
    """
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    )
    print("\nTraining Logistic Regression on Hybrid (DistilBERT + TF–IDF) features...")
    clf.fit(X_train_hybrid, y_train)
    y_pred = clf.predict(X_test_hybrid)
    print("\n[Hybrid: DistilBERT + TF–IDF SVD + LR]")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    return clf


# ---------- MAIN ----------

def main():
    print("Loading ISOT dataset...")
    df = load_isot(DATA_DIR)
    print(f"Total samples: {len(df)}")

    print("Preparing data (train/test split)...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Optional: use a smaller subset to avoid memory issues on small GPUs
    # idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=12000, replace=False)
    # X_train = X_train[idx]
    # y_train = y_train[idx]

    print("\nBuilding TF–IDF features...")
    X_train_tfidf, X_test_tfidf, _ = build_tfidf_features(X_train, X_test)

    print("\n=== Classical ML models on TF–IDF ===")
    _ = run_classical_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

    print("\n=== DistilBERT embeddings ===")
    X_train_embed, X_test_embed = build_distilbert_features(X_train, X_test)
    _ = run_distilbert_classifier(X_train_embed, X_test_embed, y_train, y_test)

    print("\n=== Hybrid model (DistilBERT + TF–IDF SVD) ===")
    X_train_hybrid, X_test_hybrid, _ = build_hybrid_features(
        X_train_tfidf, X_test_tfidf, X_train_embed, X_test_embed
    )
    _ = run_hybrid_classifier(X_train_hybrid, X_test_hybrid, y_train, y_test)


if __name__ == "__main__":
    main()
