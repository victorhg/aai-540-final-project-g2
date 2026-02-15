import os
import json
import tarfile
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

def find_model_file(extract_dir: str) -> str:
    preferred = {"xgboost-model", "model.bin", "model"}
    found = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            full = os.path.join(root, f)
            found.append(full)
            if f in preferred:
                return full

    if len(found) == 1:
        return found[0]

    for full in found:
        if os.path.getsize(full) > 0:
            return full

    raise FileNotFoundError(f"Could not locate model file. Extracted files: {found[:25]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

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
