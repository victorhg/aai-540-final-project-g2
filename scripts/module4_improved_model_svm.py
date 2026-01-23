#!/usr/bin/env python3
"""
Module 4 - Improved Model (SVM) for CREMA-D MFCC features.

Reads:
  artifacts/cremad_mfcc_features.parquet
  artifacts/cremad_splits.csv

Writes:
  artifacts/module4_svm_metrics.txt

Fix:
- Handles column-name collisions (e.g., 'split' exists in both sources) by using merge suffixes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

FEATURES_PATH = ART / "cremad_mfcc_features.parquet"
SPLITS_PATH = ART / "cremad_splits.csv"
OUT_PATH = ART / "module4_svm_metrics.txt"


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}. Found: {df.columns.tolist()}")


def _resolve_merged_col(df: pd.DataFrame, base: str) -> str:
    """
    After merge with suffixes=('_feat','_split'), the right-hand column becomes base+'_split'
    if there was a collision. If no collision, it stays as base.
    """
    if base in df.columns:
        return base
    alt = f"{base}_split"
    if alt in df.columns:
        return alt
    # fallback: sometimes user changed suffixes
    for c in df.columns:
        if c.startswith(base) and ("_split" in c or "_y" in c or "_right" in c):
            return c
    raise KeyError(f"Could not resolve merged column for '{base}'. Found: {df.columns.tolist()}")


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features: {FEATURES_PATH}")
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Missing splits: {SPLITS_PATH}")

    feats = pd.read_parquet(FEATURES_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    feat_path_col = _pick_col(feats, ["filepath", "path", "wav_path", "file_path", "file"])
    split_path_col = _pick_col(splits, ["filepath", "path", "wav_path", "file_path", "file"])
    label_col_base = _pick_col(splits, ["emotion", "label", "y"])
    split_col_base = _pick_col(splits, ["split", "set", "fold"])

    # Merge with explicit suffixes so we can reliably select the splits.csv columns
    df = feats.merge(
        splits[[split_path_col, split_col_base, label_col_base]],
        left_on=feat_path_col,
        right_on=split_path_col,
        how="inner",
        suffixes=("_feat", "_split"),
    )

    # Drop duplicate right-side path column
    if split_path_col in df.columns:
        df = df.drop(columns=[split_path_col])

    # Resolve correct split/label columns after merge (handles collisions)
    split_col = _resolve_merged_col(df, split_col_base)
    label_col = _resolve_merged_col(df, label_col_base)

    # Keep only train/test
    df = df[df[split_col].isin(["train", "test"])].copy()

    # Feature columns = numeric columns excluding known metadata
    drop_cols = {
        feat_path_col,
        split_col,
        label_col,
        "speaker_id",
        "sentence_id",
        "intensity",
        "dataset",
        "language",
    }
    X_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not X_cols:
        raise ValueError("No numeric feature columns found after merge. Check parquet schema.")

    train_df = df[df[split_col] == "train"].copy()
    test_df = df[df[split_col] == "test"].copy()

    X_train = train_df[X_cols].to_numpy(dtype=np.float32)
    y_train = train_df[label_col].astype(str).to_numpy()
    X_test = test_df[X_cols].to_numpy(dtype=np.float32)
    y_test = test_df[label_col].astype(str).to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=2)

    ART.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("Module 4 - Improved Model (LinearSVC)\n")
        f.write(f"Features: {FEATURES_PATH.name}\n")
        f.write(f"Splits: {SPLITS_PATH.name}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1w:.4f}\n\n")
        f.write(report)
        f.write("\n")

    print(f"Saved metrics → {OUT_PATH}")
    print(f"Accuracy: {acc:.4f} | Weighted F1: {f1w:.4f}")


if __name__ == "__main__":
    main()
