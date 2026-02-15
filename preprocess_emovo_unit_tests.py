import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "rms_mean",
    "pitch_mean",
    "pitch_std",
    "spectral_centroid_mean",
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "tempo",
]

def fail(msg: str):
    raise RuntimeError(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--missing-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_csv = os.path.join(args.input_dir, "emovo_features.csv")
    if not os.path.exists(in_csv):
        cands = [f for f in os.listdir(args.input_dir) if f.endswith(".csv")]
        if not cands:
            fail(f"No CSV found in {args.input_dir}")
        in_csv = os.path.join(args.input_dir, cands[0])

    df = pd.read_csv(in_csv)

    # schema unit test
    if "emotion" not in df.columns:
        fail("Unit test failed: missing required column 'emotion'.")

    missing_feats = [c for c in FEATURES if c not in df.columns]
    if missing_feats:
        fail(f"Unit test failed: missing required feature columns: {missing_feats}")

    # tempo
    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")

    # types unit test
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # missingness checks
    miss = df[FEATURES].isna().mean().to_dict()
    offenders = {k: v for k, v in miss.items() if v > args.missing_threshold}
    if offenders:
        fail(f"Unit test failed: missingness above threshold {args.missing_threshold:.2%}: {offenders}")

    # remove NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["emotion"]).copy()

    if len(df) < 200:
        fail(f"Unit test failed: too few rows after cleaning ({len(df)}).")

    # label encoding
    le = LabelEncoder()
    df["emotion_encoded"] = le.fit_transform(df["emotion"].astype(str))

    # classes unit test
    n_classes = df["emotion_encoded"].nunique()
    if n_classes < 2:
        fail(f"Unit test failed: need >=2 classes, found {n_classes}.")

    os.makedirs(args.output_dir, exist_ok=True)

    # save mapping
    mapping = {cls: int(idx) for cls, idx in zip(le.classes_, le.transform(le.classes_))}
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # Use existing split if present, otherwise create split
    use_existing_split = "split" in df.columns and df["split"].isin(["train", "val", "test"]).all()

    if use_existing_split:
        train_df = df[df["split"] == "train"].copy()
        val_df   = df[df["split"] == "val"].copy()
        test_df  = df[df["split"] == "test"].copy()
        if min(len(train_df), len(val_df), len(test_df)) == 0:
            use_existing_split = False
    # use 80/10/10
    if not use_existing_split:
        tmp_train, test_df = train_test_split(
            df, test_size=0.10, random_state=args.seed, stratify=df["emotion_encoded"]
        )
        train_df, val_df = train_test_split(
            tmp_train, test_size=0.1111, random_state=args.seed, stratify=tmp_train["emotion_encoded"]
        )

    # XGBoost format
    def to_xgb(dfin: pd.DataFrame) -> pd.DataFrame:
        out = dfin[["emotion_encoded"] + FEATURES].copy()
        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        return out

    train_xgb = to_xgb(train_df)
    val_xgb   = to_xgb(val_df)
    test_xgb  = to_xgb(test_df)

    # write outputs
    train_dir = os.path.join(args.output_dir, "train")
    val_dir   = os.path.join(args.output_dir, "validation")
    test_dir  = os.path.join(args.output_dir, "test")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    train_xgb.to_csv(os.path.join(train_dir, "train.csv"), index=False, header=False)
    val_xgb.to_csv(os.path.join(val_dir, "validation.csv"), index=False, header=False)
    test_xgb.to_csv(os.path.join(test_dir, "test.csv"), index=False, header=False)

    # small report for debugging
    report = {
        "rows_in": int(len(df)),
        "rows_train": int(len(train_xgb)),
        "rows_val": int(len(val_xgb)),
        "rows_test": int(len(test_xgb)),
        "num_classes": int(n_classes),
        "missingness": miss,
        "labels": mapping,
    }
    with open(os.path.join(args.output_dir, "preprocess_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Preprocess + unit tests passed.")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
