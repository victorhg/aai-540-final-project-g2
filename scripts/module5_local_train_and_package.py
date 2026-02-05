#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ART_DIR = Path("artifacts/module5")
OUT_DIR = ART_DIR / "model_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    p = ART_DIR / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run module5_prepare_training_data.py first.")
    return pd.read_csv(p)


def detect_label_column(df: pd.DataFrame) -> str:
    # Common label names we might have
    for c in ["emotion", "label", "target", "y", "class"]:
        if c in df.columns:
            return c

    # Otherwise: choose a likely categorical label column
    # (object dtype, low-ish cardinality, not obviously metadata)
    exclude = {
        "filepath", "path", "file", "wav_path",
        "speaker_id", "sentence_id", "intensity", "dataset", "language", "split", "record_id", "event_time"
    }
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == "object":
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 50:
                candidates.append((c, nunique))

    if not candidates:
        raise SystemExit(
            "Could not auto-detect label column. "
            "Please open artifacts/module5/train.csv and verify the label column name."
        )

    # pick the smallest cardinality categorical column (usually the class label)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def main():
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    label_col = detect_label_column(train_df)
    print(f"✅ Using label column: {label_col}")

    # Drop metadata + label from features (only keep numeric feature columns)
    drop_cols = {
        label_col,
        "filepath", "path", "file", "wav_path",
        "speaker_id", "sentence_id", "intensity", "dataset", "language", "split",
        "record_id", "event_time",
    }

    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Keep only numeric columns (protect against strings sneaking in)
    numeric_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            numeric_cols.append(c)

    if not numeric_cols:
        raise SystemExit(
            "No numeric feature columns found after filtering. "
            "Check your train.csv format."
        )

    X_train, y_train = train_df[numeric_cols], train_df[label_col]
    X_val, y_val = val_df[numeric_cols], val_df[label_col]
    X_test, y_test = test_df[numeric_cols], test_df[label_col]

    print(f"Features: {len(numeric_cols)} | Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, n_jobs=-1)),  # no deprecated params
        ]
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    metrics = {
        "label_col": label_col,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_f1_weighted": float(f1_score(y_val, val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted")),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(numeric_cols)),
        "feature_cols_preview": numeric_cols[:10],
    }

    joblib.dump(model, OUT_DIR / "model.joblib")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    (OUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, test_pred, digits=4)
    )

    cm = confusion_matrix(y_test, test_pred)
    pd.DataFrame(cm).to_csv(OUT_DIR / "confusion_matrix.csv", index=False)

    summary = (
        "Module 5 local package created.\n"
        f"- Label column: {label_col}\n"
        f"- Features: {len(numeric_cols)}\n"
        f"- Val Acc: {metrics['val_accuracy']:.4f} | Val F1(w): {metrics['val_f1_weighted']:.4f}\n"
        f"- Test Acc: {metrics['test_accuracy']:.4f} | Test F1(w): {metrics['test_f1_weighted']:.4f}\n"
        "Saved:\n"
        f"- {OUT_DIR / 'model.joblib'}\n"
        f"- {OUT_DIR / 'metrics.json'}\n"
        f"- {OUT_DIR / 'classification_report.txt'}\n"
        f"- {OUT_DIR / 'confusion_matrix.csv'}\n"
    )
    (OUT_DIR / "module5_local_summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
