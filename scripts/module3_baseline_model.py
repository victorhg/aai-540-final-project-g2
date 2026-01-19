"""
Module 3 – Baseline Model (CREMA-D)
- Uses MFCC features parquet + speaker-safe split CSV
- Joins on the correct key for your files (filepath)
- Avoids column collisions (split, emotion, etc.)
- Trains Logistic Regression baseline (no sklearn warning)
- Saves metrics to artifacts/baseline_metrics.txt
"""

from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


ARTIFACTS_DIR = Path("artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "cremad_mfcc_features.parquet"
SPLITS_PATH = ARTIFACTS_DIR / "cremad_splits.csv"
METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.txt"


def main():
    features_df = pd.read_parquet(FEATURES_PATH)
    splits_df = pd.read_csv(SPLITS_PATH)

    # --- Your splits file uses "filepath"
    if "filepath" not in splits_df.columns:
        raise KeyError(f"'filepath' not found in splits. columns={splits_df.columns.tolist()}")

    # --- Your features file often uses wav_path or filepath; normalize to filepath
    # If features has wav_path, copy it to filepath for the merge
    if "filepath" not in features_df.columns:
        if "wav_path" in features_df.columns:
            features_df = features_df.copy()
            features_df["filepath"] = features_df["wav_path"]
        elif "path" in features_df.columns:
            features_df = features_df.copy()
            features_df["filepath"] = features_df["path"]
        else:
            raise KeyError(
                "Features parquet has no filepath/wav_path/path column to join on. "
                f"columns={features_df.columns.tolist()}"
            )

    # --- Prevent column collisions by renaming splits columns before merge
    # We only need filepath + split for training
    splits_min = splits_df[["filepath", "split"]].copy()
    splits_min.rename(columns={"split": "data_split"}, inplace=True)

    # --- Merge
    df = features_df.merge(splits_min, on="filepath", how="inner")

    # --- Now the split column is guaranteed to be "data_split"
    if "data_split" not in df.columns:
        raise KeyError(f"data_split missing after merge. columns={df.columns.tolist()}")

    # --- Train / Test
    train_df = df[df["data_split"] == "train"].copy()
    test_df = df[df["data_split"] == "test"].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"After merge, train={len(train_df)} test={len(test_df)}. "
            "Check split labels (expected train/test)."
        )

    # --- Target
    if "emotion" not in df.columns:
        raise KeyError(f"'emotion' not found in merged df. columns={df.columns.tolist()}")

    y_train = train_df["emotion"]
    y_test = test_df["emotion"]

    # --- Feature columns: drop non-features + keep numeric only
    drop_cols = {
        "emotion", "data_split",
        "filepath", "wav_path", "path",
        "utterance_id", "speaker_id", "sentence_id",
        "intensity", "dataset", "language"
    }

    X_train = train_df.drop(columns=[c for c in train_df.columns if c in drop_cols], errors="ignore")
    X_test = test_df.drop(columns=[c for c in test_df.columns if c in drop_cols], errors="ignore")

    X_train = X_train.select_dtypes(include="number")
    X_test = X_test.select_dtypes(include="number")

    if X_train.shape[1] == 0:
        raise ValueError("No numeric feature columns left after cleaning. Check parquet schema.")

    # --- Model (no deprecated multi_class arg)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    print(f"\nBaseline Accuracy: {acc:.4f}")
    print(f"Baseline F1 (weighted): {f1:.4f}\n")
    print(report)

    METRICS_PATH.parent.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        f.write("Join key: filepath\n")
        f.write("Split column: data_split\n\n")
        f.write(f"Baseline Accuracy: {acc:.4f}\n")
        f.write(f"Baseline F1 (weighted): {f1:.4f}\n\n")
        f.write(report)

    print(f"\nSaved metrics → {METRICS_PATH}")


if __name__ == "__main__":
    main()
