from pathlib import Path
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score


ARTIFACTS_DIR = Path("artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "cremad_mfcc_features.parquet"
OUT_METRICS = ARTIFACTS_DIR / "module4_svc_rbf_metrics.txt"


def main():
    df = pd.read_parquet(FEATURES_PATH)

    # Safety checks
    required = {"split", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in features parquet: {missing}. Found: {df.columns.tolist()}")

    # Keep train/test only
    df = df[df["split"].isin(["train", "test"])].copy()

    # Confirm we actually have rows
    counts = df["split"].value_counts().to_dict()
    if counts.get("train", 0) == 0 or counts.get("test", 0) == 0:
        raise ValueError(f"Bad split distribution (need both train and test). Got: {counts}")

    # Use only MFCC vector columns
    feature_cols = [c for c in df.columns if c.startswith("mfcc_")]
    if not feature_cols:
        raise ValueError("No MFCC feature columns found (expected columns like mfcc_0, mfcc_1, ...).")

    X_train = df.loc[df["split"] == "train", feature_cols]
    y_train = df.loc[df["split"] == "train", "emotion"]

    X_test = df.loc[df["split"] == "test", feature_cols]
    y_test = df.loc[df["split"] == "test", "emotion"]

    # RBF SVC (nonlinear)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
            )),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    OUT_METRICS.parent.mkdir(exist_ok=True)
    with open(OUT_METRICS, "w") as f:
        f.write("Model: SVC (RBF)\n")
        f.write(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n\n")
        f.write(report)

    print(f"Saved metrics → {OUT_METRICS}")
    print(f"Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")


if __name__ == "__main__":
    main()

