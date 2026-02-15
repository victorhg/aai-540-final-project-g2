import os
import json
import tarfile
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def ensure_xgboost():
    try:
        import xgboost
        return
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost==1.7.6"])

def find_model_file(extract_dir: str) -> str:
    candidates = [
        os.path.join(extract_dir, "xgboost-model"),
        os.path.join(extract_dir, "model.bin"),
        os.path.join(extract_dir, "model"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f in ("xgboost-model", "model.bin") or f.endswith(".bin"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not locate model file under {extract_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    ensure_xgboost()
    import xgboost as xgb

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract model
    extract_dir = "/tmp/model_artifact"
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(args.model_artifact, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    model_file = find_model_file(extract_dir)
    booster = xgb.Booster()
    booster.load_model(model_file)

    # test csv: label first
    test = pd.read_csv(args.test_path, header=None)
    y_true = test.iloc[:, 0].astype(int).values
    X = test.iloc[:, 1:].values

    dtest = xgb.DMatrix(X)
    y_pred = booster.predict(dtest)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))

    evaluation = {"classification_metrics": {"accuracy": {"value": acc}}}

    out_path = os.path.join(args.output_dir, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print("Saved:", out_path)
    print(json.dumps(evaluation, indent=2))

if __name__ == "__main__":
    main()
