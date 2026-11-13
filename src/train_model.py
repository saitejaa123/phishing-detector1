"""
Model Trainer
=============
Trains Random Forest, XGBoost, Logistic Regression and a soft-voting
Ensemble on the phishing URL dataset.

Key practices to avoid overfitting / inflated accuracy:
  - Strict train/test split (80/20) with stratification
  - 5-fold stratified cross-validation
  - No data leakage (scaler fitted on train only)
  - Hyperparameters tuned conservatively
  - Reports: accuracy, AUC-ROC, precision, recall, F1, confusion matrix
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "phishing_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import FEATURE_NAMES


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    if not DATA_PATH.exists():
        print("[*] Dataset not found — generating …")
        from dataset_generator import generate_dataset
        generate_dataset(500, 500, str(DATA_PATH))

    df = pd.read_csv(DATA_PATH)
    print(f"[+] Dataset loaded: {len(df)} rows | "
          f"{(df['label']==1).sum()} phishing | {(df['label']==0).sum()} legit")

    X = df[FEATURE_NAMES].values
    y = df["label"].values
    return X, y


# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate(name: str, model, X_test, y_test, X_train, y_train) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else np.zeros(len(y_test))
    )

    # Core metrics
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    rep  = classification_report(y_test, y_pred, output_dict=True)

    # Train accuracy (to detect overfitting gap)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    overfit_gap = round(train_acc - acc, 4)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"  Train Acc : {train_acc:.4f}  |  Test Acc : {acc:.4f}  "
          f"|  Gap (overfit) : {overfit_gap:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}  |  Precision: {prec:.4f}  "
          f"|  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    print(f"  Confusion Matrix: {cm}")

    if overfit_gap > 0.05:
        print(f"  ⚠️  Overfit gap {overfit_gap:.4f} > 0.05 — model may be memorising training data")
    else:
        print(f"  ✓  Overfit gap {overfit_gap:.4f} ≤ 0.05 — generalisation is acceptable")

    return {
        "accuracy":                round(acc,        4),
        "train_accuracy":          round(train_acc,  4),
        "overfit_gap":             round(overfit_gap,4),
        "auc_roc":                 round(auc,        4),
        "precision":               round(prec,       4),
        "recall":                  round(rec,        4),
        "f1_score":                round(f1,         4),
        "confusion_matrix":        cm,
        "classification_report":   rep,
    }


# ── Main training ─────────────────────────────────────────────────────────────
def train():
    print("[+] Loading dataset …")
    X, y = load_data()

    # ── Strict 80/20 split — stratified ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train: {len(X_train)} | Test: {len(X_test)}")

    metrics = {}

    # ── 1. Random Forest ──────────────────────────────────────────────────────
    # Conservative depth to prevent overfitting
    print("\n[+] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,            # capped — prevents memorisation
        min_samples_split=10,   # requires more samples to split
        min_samples_leaf=5,     # prevents tiny leaf nodes
        max_features="sqrt",    # feature subsampling
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    metrics["random_forest"] = evaluate(
        "Random Forest", rf, X_test, y_test, X_train, y_train
    )
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    # ── 2. XGBoost ────────────────────────────────────────────────────────────
    print("\n[+] Training XGBoost …")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,            # shallow trees — less overfit
        learning_rate=0.05,     # slow learning
        subsample=0.7,          # row subsampling
        colsample_bytree=0.7,   # feature subsampling
        reg_alpha=0.1,          # L1 regularisation
        reg_lambda=1.0,         # L2 regularisation
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    metrics["xgboost"] = evaluate(
        "XGBoost", xgb, X_test, y_test, X_train, y_train
    )
    joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")

    # ── 3. Logistic Regression ────────────────────────────────────────────────
    print("\n[+] Training Logistic Regression …")
    # Scaler fitted on TRAIN only — no leakage
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000, C=0.5,   # C < 1 = stronger regularisation
        random_state=42,
    )
    lr.fit(X_train_sc, y_train)
    metrics["logistic_regression"] = evaluate(
        "Logistic Regression", lr, X_test_sc, y_test, X_train_sc, y_train
    )
    joblib.dump(lr,     MODEL_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # ── 4. Ensemble (soft voting) ─────────────────────────────────────────────
    print("\n[+] Training Ensemble (soft-voting) …")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])
    ensemble = VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(
                n_estimators=100, max_depth=8,
                min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=42, n_jobs=-1,
            )),
            ("xgb", XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )),
            ("lr",  lr_pipe),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    metrics["ensemble"] = evaluate(
        "Ensemble", ensemble, X_test, y_test, X_train, y_train
    )
    joblib.dump(ensemble, MODEL_DIR / "ensemble.pkl")

    # ── 5. Stratified 5-fold CV on Ensemble ───────────────────────────────────
    print("\n[+] 5-fold Stratified Cross-Validation on Ensemble …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X, y, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    metrics["ensemble"]["cv_mean"] = round(cv_scores.mean(), 4)
    metrics["ensemble"]["cv_std"]  = round(cv_scores.std(),  4)
    print(f"    CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Per-fold:    {[round(s,4) for s in cv_scores]}")

    # ── 6. Feature importances ────────────────────────────────────────────────
    metrics["feature_importances"] = dict(
        zip(FEATURE_NAMES, rf.feature_importances_.tolist())
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n[+] All models saved → models/")
    print("[+] Metrics saved    → models/metrics.json")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<25} {'Test Acc':>9} {'AUC-ROC':>9} {'F1':>7}")
    print("  " + "-"*53)
    for key, label in [
        ("random_forest",       "Random Forest"),
        ("xgboost",             "XGBoost"),
        ("logistic_regression", "Logistic Regression"),
        ("ensemble",            "Ensemble"),
    ]:
        m = metrics[key]
        print(f"  {label:<25} {m['accuracy']:>9.4f} "
              f"{m['auc_roc']:>9.4f} {m['f1_score']:>7.4f}")
    print("=" * 55)
    print("\n✓ Training complete.")
    return metrics


if __name__ == "__main__":
    train()
