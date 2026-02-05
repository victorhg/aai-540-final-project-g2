#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


FEATS_PATH = Path("artifacts/cremad_mfcc_features.parquet")
OUT_DIR = Path("artifacts/module5")

SPLIT_COL = "split"
LABEL_COL = "emotion"          # source label in parquet
OUT_LABEL_COL = "label"        # numeric label we create

# Only MFCC feature columns (mfcc_0..mfcc_*)
FEATURE_PREFIX = "mfcc_"


def main():
    if not FEATS_PATH.exists():
        raise SystemExit(f"Missing {FEATS_PATH}. Make sure artifacts/cremad_mfcc_features.parquet exists.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATS_PATH)

    # Basic checks
    for c in ["filepath", LABEL_COL, SPLIT_COL]:
        if c not in df.columns:
            raise SystemExit(f"Expected column '{c}' not found in parquet. Columns: {df.columns.tolist()[:30]}")

    # Keep only train/validation/test
    df = df[df[SPLIT_COL].isin(["train", "validation", "test"])].copy()

    # Pick MFCC columns
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise SystemExit("No MFCC feature columns found (expected columns starting with 'mfcc_').")

    # Build numeric labels from emotions (stable ordering)
    emotions = sorted(df[LABEL_COL].dropna().unique().tolist())
    label_map = {emo: i for i, emo in enumerate(emotions)}
    df[OUT_LABEL_COL] = df[LABEL_COL].map(label_map).astype("int64")

    # Final columns to export
    out_cols = [OUT_LABEL_COL] + feature_cols

    # Ensure all feature cols numeric + no NaNs
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=out_cols).copy()

    # Split + write CSVs WITH HEADER
    splits = {
        "train": df[df[SPLIT_COL] == "train"][out_cols],
        "validation": df[df[SPLIT_COL] == "validation"][out_cols],
        "test": df[df[SPLIT_COL] == "test"][out_cols],
    }

    # Split + write CSVs WITH HEADER
    train_df = df[df[SPLIT_COL] == "train"][out_cols].copy()
    val_df = df[df[SPLIT_COL] == "validation"][out_cols].copy()
    test_df = df[df[SPLIT_COL] == "test"][out_cols].copy()

    # If no validation split exists, carve one out of train (stratified-ish by label)
    if len(val_df) == 0:
        print("⚠️ No validation rows found in source splits. Creating validation split from train (10%).")
        # sample 10% per class
        val_parts = []
        train_parts = []
        for lbl, grp in train_df.groupby(OUT_LABEL_COL):
            n_val = max(1, int(round(0.10 * len(grp))))
            val_part = grp.sample(n=n_val, random_state=42)
            train_part = grp.drop(index=val_part.index)
            val_parts.append(val_part)
            train_parts.append(train_part)
        val_df = pd.concat(val_parts, ignore_index=True)
        train_df = pd.concat(train_parts, ignore_index=True)

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    for name, sdf in splits.items():
        out_path = OUT_DIR / f"{name}.csv"
        sdf.to_csv(out_path, index=False)  # header=True by default
        print(f"✅ wrote {out_path} rows={len(sdf)} cols={len(sdf.columns)}")

    # Save label mapping
    mapping_path = OUT_DIR / "label_mapping.csv"
    pd.DataFrame({"emotion": emotions, "label": [label_map[e] for e in emotions]}).to_csv(mapping_path, index=False)

    # Summary
    summary_path = OUT_DIR / "prep_summary.txt"
    summary = (
        f"Source: {FEATS_PATH}\n"
        f"Total rows used: {len(df)}\n"
        f"Features: {len(feature_cols)} ({feature_cols[:5]} ...)\n"
        f"Labels: {label_map}\n"
        f"Train/Val/Test: {len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])}\n"
    )
    summary_path.write_text(summary)
    print(f"✅ wrote {mapping_path}")
    print(f"✅ wrote {summary_path}")


if __name__ == "__main__":
    main()

