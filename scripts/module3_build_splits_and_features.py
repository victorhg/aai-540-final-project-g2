from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
from sklearn.model_selection import train_test_split


def parse_cremad_filename(p: Path) -> dict | None:
    # Example: 1085_IWW_ANG_XX.wav
    parts = p.stem.split("_")
    if len(parts) < 4:
        return None
    speaker_id, sentence_id, emotion, intensity = parts[0], parts[1], parts[2], parts[3]
    return {
        "filepath": str(p),
        "speaker_id": speaker_id,
        "sentence_id": sentence_id,
        "emotion": emotion,
        "intensity": intensity,
        "dataset": "CREMA-D",
        "language": "en",
    }


def mfcc_mean(path: str, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/cremad", help="CREMA-D root folder")
    ap.add_argument("--out_dir", default="artifacts", help="Artifacts output folder")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mfcc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list(raw_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wav files found under: {raw_dir}")

    # Build metadata
    rows = []
    for p in wavs:
        rec = parse_cremad_filename(p)
        if rec:
            rows.append(rec)
    df = pd.DataFrame(rows)

    # Speaker-safe split
    speakers = df["speaker_id"].unique()
    train_spk, temp_spk = train_test_split(speakers, test_size=0.30, random_state=args.seed)
    val_spk, test_spk = train_test_split(temp_spk, test_size=0.50, random_state=args.seed)

    def split_from_speaker(s: str) -> str:
        if s in train_spk:
            return "train"
        if s in val_spk:
            return "val"
        return "test"

    df["split"] = df["speaker_id"].apply(split_from_speaker)

    splits_csv = out_dir / "cremad_splits.csv"
    df.to_csv(splits_csv, index=False)

    # Feature extraction (MFCC mean)
    feats = []
    for fp in tqdm(df["filepath"].tolist(), desc="Extracting MFCC"):
        feats.append(mfcc_mean(fp, sr=args.sr, n_mfcc=args.n_mfcc))

    X = np.vstack(feats)
    feat_cols = [f"mfcc_{i}" for i in range(X.shape[1])]
    feat_df = pd.DataFrame(X, columns=feat_cols)

    full_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out_parquet = out_dir / "cremad_mfcc_features.parquet"
    full_df.to_parquet(out_parquet, index=False)

    # Small summary text for the report
    summary_txt = out_dir / "module3_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Module 3 Artifacts Summary\n")
        f.write("==========================\n\n")
        f.write(f"WAV files: {len(df)}\n")
        f.write(f"Speakers: {df['speaker_id'].nunique()}\n\n")
        f.write("Split counts:\n")
        f.write(df["split"].value_counts().to_string())
        f.write("\n\nEmotion counts:\n")
        f.write(df["emotion"].value_counts().to_string())
        f.write("\n\nArtifacts:\n")
        f.write(f"- {splits_csv}\n")
        f.write(f"- {out_parquet}\n")
        f.write(f"- {summary_txt}\n")

    print("\n✅ Done.")
    print("Wrote:", splits_csv)
    print("Wrote:", out_parquet)
    print("Wrote:", summary_txt)


if __name__ == "__main__":
    main()
