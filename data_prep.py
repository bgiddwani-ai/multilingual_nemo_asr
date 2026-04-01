#!/usr/bin/env python3
"""
data_prep.py — Prepare Hindi (IndicVoices) and English (Svarah) ASR datasets.

Usage examples
--------------
# Using a pre-cloned local dataset directory:
python data_prep.py --lang hi --split train --data_path ./dataset/IndicVoices --dataname indicvoices

# Using a Hugging Face token (dataset downloaded automatically):
python data_prep.py --lang hi --split train --hf_token hf_xxxx --dataname indicvoices

# English dataset (Svarah only has a 'test' split, mapped to 'train' here) from local dataset directory:
python data_prep.py --lang en --split train --data_path ./dataset/Svarah --dataname svarah

# Override output directory and manifest path:
python data_prep.py --lang hi --split valid \
    --data_path ./dataset/IndicVoices \
    --output_dir ./out/hi_val_audio \
    --dataname indicvoices \
    --manifest ./out/hi_val_manifest.json

Default output layout
---------------------
  dataset/<dataname>/<lang>_audio/         ← WAV files
  dataset/<dataname>/<lang>_manifest.json  ← JSONL manifest
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "hi": {
        "hf_repo": "ai4bharat/IndicVoices",
        "hf_config": "hindi",          # subset/config name
        "split_map": {                 # CLI split -> HF split name
            "train": "train",
            "valid": "valid",
        },
        "lang_tag": "hi",
    },
    "en": {
        "hf_repo": "ai4bharat/Svarah",
        "hf_config": None,             # no subset
        "split_map": {
            "train": "test",           # Svarah only has a 'test' split
            "valid": "test",
        },
        "lang_tag": "en",
    },
}


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def process_dataset(dataset, output_dir: Path, manifest_path: Path, lang_tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, sample in enumerate(dataset):
            raw_bytes = sample["audio_filepath"]["bytes"]
            array, sr = sf.read(io.BytesIO(raw_bytes))
            array = array.astype(np.float32)

            wav_path = output_dir / f"{i:06d}.wav"
            sf.write(wav_path, array, sr, subtype="PCM_16")

            record = {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(len(array) / sr, 4),
                "text": sample["text"],
                "lang": lang_tag,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            total = i + 1

            if total % 500 == 0:
                print(f"  processed {total} samples ...", flush=True)

    print(f"Done: {total} files -> {output_dir}")
    print(f"      manifest -> {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert IndicVoices / Svarah HF datasets to WAV + JSONL manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--lang",
        choices=["hi", "en"],
        required=True,
        help="Language to prepare: 'hi' (IndicVoices) or 'en' (Svarah).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        required=True,
        help="Dataset split to prepare.",
    )
    parser.add_argument(
        "--dataname",
        type=str,
        required=True,
        help="Dataset folder name, e.g. 'IndicVoices' or 'Svarah'. "
             "Used to build default output paths: dataset/<dataname>/.",
    )

    # Source — pick one
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to a pre-cloned local dataset directory.",
    )
    src.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token; dataset is downloaded automatically.",
    )

    # Output overrides (optional)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where WAV files are saved. "
             "Defaults to ./dataset/<dataname>/<lang>_audio/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path for the JSONL manifest file. "
             "Defaults to ./dataset/<dataname>/<lang>_manifest.json.",
    )

    return parser.parse_args()


def resolve_defaults(args):
    """Fill in output_dir and manifest using dataset/<dataname>/ layout."""
    base = Path("./dataset") / args.dataname / args.split

    if args.output_dir is None:
        args.output_dir = base / f"{args.lang}_audio"
    if args.manifest is None:
        args.manifest = base / f"{args.lang}_manifest.json"


def load_hf_dataset(args, cfg):
    hf_split = cfg["split_map"][args.split]
    load_kwargs = dict(
        path=str(args.data_path) if args.data_path else cfg["hf_repo"],
        split=hf_split,
        trust_remote_code=True,
    )
    if cfg["hf_config"]:
        load_kwargs["name"] = cfg["hf_config"]
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    print(f"Loading dataset  : {load_kwargs['path']}")
    print(f"  config/subset  : {cfg['hf_config'] or '(none)'}")
    print(f"  split          : {hf_split}")

    dataset = load_dataset(**load_kwargs)
    dataset = dataset.cast_column("audio_filepath", Audio(decode=False))
    return dataset


def main():
    args = parse_args()

    if args.data_path is None and args.hf_token is None:
        print(
            "ERROR: provide either --data_path (local clone) or --hf_token (HF download).",
            file=sys.stderr,
        )
        sys.exit(1)

    resolve_defaults(args)
    cfg = DATASET_CONFIGS[args.lang]

    print(f"\n{'='*55}")
    print(f"  lang={args.lang}  split={args.split}  dataname={args.dataname}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  manifest   : {args.manifest}")
    print(f"{'='*55}\n")

    dataset = load_hf_dataset(args, cfg)
    process_dataset(dataset, args.output_dir, args.manifest, cfg["lang_tag"])


if __name__ == "__main__":
    main()