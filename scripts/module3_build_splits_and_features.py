from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
from sklearn.model_selection import train_test_split


def extract_comprehensive_features(file_path):
    """Extract comprehensive audio features for emotion recognition"""
    y, sr = librosa.load(file_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    features = {}
    
    # Basic features
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    features['rms_mean'] = np.mean(librosa.feature.rms(y=y_trimmed))
    features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(y_trimmed))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    
    # MFCCs (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    # Tempo and rhythm
    try:
        tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr)
        features['tempo'] = float(tempo)
    except:
        features['tempo'] = 0
    
    # Pitch features
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y_trimmed, 
                                                    fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'))
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            features['pitch_mean'] = np.mean(f0_voiced)
            features['pitch_std'] = np.std(f0_voiced)
            features['pitch_range'] = np.max(f0_voiced) - np.min(f0_voiced)
        else:
            features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = 0
        features['voicing_rate'] = np.sum(voiced_flag) / len(voiced_flag)
    except:
        features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = features['voicing_rate'] = 0
    
    return features


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
        "language": "en"
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
    for p in tqdm(wavs, desc="Processing WAV files"):
        rec = parse_cremad_filename(p)
        if rec:
            # Extract comprehensive features
            song_features = extract_comprehensive_features(p)
            rec.update(song_features)

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

    splits_csv = out_dir / "cremad_processed.csv"
    df.to_csv(splits_csv, index=False)

   
    out_parquet = out_dir / "cremad_processed.parquet"
    df.to_parquet(out_parquet, index=False)

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
