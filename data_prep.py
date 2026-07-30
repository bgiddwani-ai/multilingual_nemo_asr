#!/usr/bin/env python3
"""
data_prep.py — Prepare Indian-language + English ASR datasets.

Supported corpora
-----------------
  indicvoices       ai4bharat/IndicVoices          (22 languages)
  svarah            ai4bharat/Svarah               (English, test split only)
  kathbath          ai4bharat/Kathbath             (12 languages)
  shrutilipi        ai4bharat/Shrutilipi           (12 languages)
  lahaja            ai4bharat/Lahaja               (Hindi, accent-diverse)
  tamil_asr_corpus  parambharat/tamil_asr_corpus   (Tamil)

Audio conversion is parallelized using Hugging Face Dataset.map(num_proc=...).

Usage examples
--------------
# --split may be omitted; both train and valid are then prepared.
python data_prep.py --lang ta --hf_token hf_xxxx \
    --dataname kathbath --num_workers 8

python data_prep.py --lang ta --split train \
    --hf_token hf_xxxx --dataname kathbath --num_workers 8

python data_prep.py --lang bn --split train \
    --data_path /data/dataset/Shrutilipi --dataname shrutilipi

python data_prep.py --lang hi --split train --hf_token hf_xxxx \
    --dataname lahaja

python data_prep.py --lang ta --split train \
    --hf_token hf_xxxx --dataname tamil_asr

# Legacy behaviour (auto-routes en -> Svarah, others -> IndicVoices):
python data_prep.py --lang hi --split train --data_path ./IndicVoices \
    --dataname indicvoices

# Discovery:
python data_prep.py --list_langs
python data_prep.py --list_datasets

Default output layout
---------------------
/data/dataset/<dataname>/<split>/<lang>_audio/
/data/dataset/<dataname>/<split>/<lang>_manifest.json
"""

import argparse
import gc
import io
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import soundfile as sf
from datasets import Audio, Dataset, VerificationMode, load_dataset


# ---------------------------------------------------------------------------
# Language tables
# ---------------------------------------------------------------------------

# ISO 639-1/639-3 code -> Hugging Face config (subset) name.
INDICVOICES_LANGUAGES: Dict[str, str] = {
    "as": "assamese",
    "bn": "bengali",
    "brx": "bodo",
    "doi": "dogri",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "ks": "kashmiri",
    "kok": "konkani",
    "mai": "maithili",
    "ml": "malayalam",
    "mni": "manipuri",
    "mr": "marathi",
    "ne": "nepali",
    "or": "odia",
    "pa": "punjabi",
    "sa": "sanskrit",
    "sat": "santali",
    "sd": "sindhi",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
}

KATHBATH_LANGUAGES: Dict[str, str] = {
    "bn": "bengali",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "ml": "malayalam",
    "mr": "marathi",
    "or": "odia",
    "pa": "punjabi",
    "sa": "sanskrit",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
}

SHRUTILIPI_LANGUAGES: Dict[str, str] = {
    "bn": "bengali",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "ml": "malayalam",
    "mr": "marathi",
    "or": "odia",
    "pa": "punjabi",
    "sa": "sanskrit",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
#
# Each entry provides:
#   hf_repo         Hugging Face repo id (or a local path via --data_path)
#   languages       {lang_code: hf_config} or None for single-config repos
#   fixed_langs     allowed codes when `languages` is None
#   split_map       requested split -> upstream split name
#   column_aliases  upstream column name -> canonical name, tried in order
#
# Canonical columns consumed downstream are "audio_filepath" and "text".

DATASETS: Dict[str, Dict[str, Any]] = {
    "indicvoices": {
        "hf_repo": "ai4bharat/IndicVoices",
        "languages": INDICVOICES_LANGUAGES,
        "split_map": {"train": "train", "valid": "valid"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript"],
        },
    },
    "svarah": {
        "hf_repo": "ai4bharat/Svarah",
        "languages": None,
        "fixed_langs": ["en"],
        # Svarah exposes only a test split.
        "split_map": {"train": "test", "valid": "test"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript"],
        },
    },
    "kathbath": {
        "hf_repo": "ai4bharat/Kathbath",
        "languages": KATHBATH_LANGUAGES,
        "split_map": {"train": "train", "valid": "valid"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript", "sentence"],
        },
    },
    "shrutilipi": {
        "hf_repo": "ai4bharat/Shrutilipi",
        "languages": SHRUTILIPI_LANGUAGES,
        # Shrutilipi is a mined corpus published as a single train split.
        "split_map": {"train": "train", "valid": "train"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript", "sentence"],
        },
    },
    "lahaja": {
        "hf_repo": "ai4bharat/Lahaja",
        "languages": None,
        "fixed_langs": ["hi"],
        "split_map": {"train": "train", "valid": "train"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript", "sentence"],
        },
    },
    "tamil_asr_corpus": {
        "hf_repo": "parambharat/tamil_asr_corpus",
        "languages": None,
        "fixed_langs": ["ta"],
        "split_map": {"train": "train", "valid": "test"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript", "sentence"],
        },
    },
    "springinx": {
        "hf_repo": "SPRINGLab/SPRING_INX_Marathi_R2",
        "languages": None,
        "fixed_langs": ["mr", "gu"],
        "split_map": {"train": "train", "valid": "validation"},
        "column_aliases": {
            "audio_filepath": ["audio_filepath", "audio"],
            "text": ["text", "transcript", "sentence"],
        },
    },
}

# Used when --dataset is omitted, preserving the original CLI behaviour.
AUTO_ROUTING = {"en": "svarah"}
AUTO_DEFAULT = "indicvoices"


def dataset_languages(dataset_name: str) -> List[str]:
    spec = DATASETS[dataset_name]

    if spec["languages"] is None:
        return list(spec["fixed_langs"])

    return sorted(spec["languages"])


def supported_languages() -> List[str]:
    codes: set = set()

    for name in DATASETS:
        codes.update(dataset_languages(name))

    return sorted(codes)


def resolve_dataset_name(dataset_arg: Optional[str], lang: str) -> str:
    """
    Map --dataset (possibly 'auto' or None) to a concrete registry key.
    """
    if dataset_arg in (None, "auto"):
        return AUTO_ROUTING.get(lang, AUTO_DEFAULT)

    if dataset_arg not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unsupported dataset: {dataset_arg}. Available: {available}"
        )

    return dataset_arg


def get_dataset_config(dataset_name: str, lang: str) -> Dict[str, Any]:
    """
    Resolve the concrete loading configuration for a dataset/language pair.
    """
    spec = DATASETS[dataset_name]
    allowed = dataset_languages(dataset_name)

    if lang not in allowed:
        raise ValueError(
            f"Language '{lang}' is not available in dataset "
            f"'{dataset_name}'. Available: {', '.join(allowed)}"
        )

    hf_config = (
        spec["languages"][lang]
        if spec["languages"] is not None
        else None
    )

    return {
        "dataset_name": dataset_name,
        "hf_repo": spec["hf_repo"],
        "hf_config": hf_config,
        "split_map": dict(spec["split_map"]),
        "column_aliases": {
            key: list(value)
            for key, value in spec["column_aliases"].items()
        },
        "lang_tag": lang,
    }


# ---------------------------------------------------------------------------
# Parallel audio conversion
# ---------------------------------------------------------------------------

def open_audio_source(
    audio_reference: Union[Dict[str, Any], str, Path],
) -> Union[io.BytesIO, str]:
    """
    Convert a Hugging Face Audio(decode=False) value into a source accepted
    by soundfile.read().

    Audio(decode=False) usually returns:

        {
            "bytes": b"...",  # or None
            "path": "/path/to/audio.wav"
        }
    """
    if isinstance(audio_reference, dict):
        raw_bytes = audio_reference.get("bytes")
        audio_path = audio_reference.get("path")

        if raw_bytes is not None:
            return io.BytesIO(raw_bytes)

        if audio_path:
            return str(audio_path)

        raise ValueError(
            "Audio entry contains neither embedded bytes nor a file path."
        )

    if isinstance(audio_reference, (str, Path)):
        return str(audio_reference)

    raise TypeError(
        f"Unsupported audio entry type: {type(audio_reference).__name__}"
    )


def convert_batch(
    batch: Dict[str, List[Any]],
    indices: List[int],
    output_dir: str,
    lang_tag: str,
) -> Dict[str, List[str]]:
    """
    Convert a batch of source audio files to PCM-16 WAV.

    This function is defined at module level so that it can be serialized and
    executed by Dataset.map multiprocessing workers.
    """
    destination = Path(output_dir)
    manifest_lines: List[str] = []

    audio_entries = batch["audio_filepath"]
    transcripts = batch["text"]

    for index, audio_reference, transcript in zip(
        indices,
        audio_entries,
        transcripts,
    ):
        try:
            source = open_audio_source(audio_reference)

            # Decode directly to float32 to avoid a separate NumPy conversion.
            audio, sample_rate = sf.read(
                source,
                dtype="float32",
                always_2d=False,
            )

            if sample_rate <= 0:
                raise ValueError(
                    f"Invalid sample rate: {sample_rate}"
                )

            if audio.ndim not in (1, 2):
                raise ValueError(
                    f"Unexpected audio shape: {audio.shape}"
                )

            # Each original dataset index receives a unique filename.
            wav_path = destination / f"{index:09d}.wav"

            sf.write(
                file=str(wav_path),
                data=audio,
                samplerate=sample_rate,
                format="WAV",
                subtype="PCM_16",
            )

            # For mono, shape is [frames].
            # For multi-channel audio, shape is [frames, channels].
            number_of_frames = audio.shape[0]
            duration = round(number_of_frames / sample_rate, 4)

            record = {
                "audio_filepath": str(wav_path.resolve()),
                "duration": duration,
                "text": transcript,
                "lang": lang_tag,
            }

            manifest_lines.append(
                json.dumps(record, ensure_ascii=False)
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to convert dataset sample {index}: {error}"
            ) from error

    return {"manifest_line": manifest_lines}


def determine_worker_count(
    requested_workers: int,
    dataset_size: int,
) -> int:
    """
    Determine the number of conversion worker processes.

    A requested value of zero enables automatic selection. The automatic
    value is capped at eight because audio conversion is often limited by
    storage throughput rather than CPU throughput.
    """
    if requested_workers < 0:
        raise ValueError("num_workers must be greater than or equal to zero.")

    if dataset_size <= 0:
        return 1

    if requested_workers == 0:
        requested_workers = min(8, os.cpu_count() or 1)

    return max(1, min(requested_workers, dataset_size))


def write_manifest(
    converted_dataset: Dataset,
    manifest_path: Path,
    total_samples: int,
    chunk_size: int = 10_000,
) -> None:
    """
    Write manifest records incrementally instead of loading every record into
    one large Python list.
    """
    with manifest_path.open(
        mode="w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as manifest_file:
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)

            manifest_lines = converted_dataset[start:end][
                "manifest_line"
            ]

            manifest_file.write("\n".join(manifest_lines))
            manifest_file.write("\n")


def process_dataset(
    dataset: Dataset,
    output_dir: Path,
    manifest_path: Path,
    lang_tag: str,
    num_workers: int,
    batch_size: int,
) -> None:
    """
    Convert the complete dataset in parallel and create the JSONL manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = len(dataset)
    worker_count = determine_worker_count(
        requested_workers=num_workers,
        dataset_size=total_samples,
    )

    print(
        f"Converting {total_samples} samples "
        f"with {worker_count} worker process(es)..."
    )

    if total_samples == 0:
        manifest_path.write_text("", encoding="utf-8")

        print(f"Done: 0 files -> {output_dir}")
        print(f"Manifest       -> {manifest_path}")
        return

    # Dataset.map writes temporary Arrow result shards. Keeping them in a
    # temporary directory prevents repeated runs from filling the HF cache.
    with TemporaryDirectory(prefix="asr_data_prep_") as temp_directory:
        cache_file = Path(temp_directory) / "converted.arrow"

        converted_dataset = dataset.map(
            convert_batch,
            batched=True,
            batch_size=batch_size,
            with_indices=True,
            num_proc=worker_count,
            fn_kwargs={
                "output_dir": str(output_dir.resolve()),
                "lang_tag": lang_tag,
            },
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            cache_file_name=str(cache_file),
            desc="Converting audio",
        )

        # Dataset.map preserves the original dataset row order.
        write_manifest(
            converted_dataset=converted_dataset,
            manifest_path=manifest_path,
            total_samples=total_samples,
        )

        # Release memory-mapped Arrow files before deleting the temporary
        # directory. This is especially important on Windows.
        del converted_dataset
        gc.collect()

    print(f"Done: {total_samples} files -> {output_dir}")
    print(f"Manifest                  -> {manifest_path}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def normalize_columns(
    dataset: Dataset,
    column_aliases: Dict[str, List[str]],
) -> Dataset:
    """
    Rename dataset-specific column names to the canonical
    'audio_filepath' / 'text' names used by the conversion worker.
    """
    for canonical, candidates in column_aliases.items():
        if canonical in dataset.column_names:
            continue

        match = next(
            (
                name
                for name in candidates
                if name in dataset.column_names
            ),
            None,
        )

        if match is None:
            available = ", ".join(dataset.column_names)
            expected = ", ".join(candidates)

            raise ValueError(
                f"Could not locate a '{canonical}' column. Tried: "
                f"{expected}. Available columns: {available}"
            )

        dataset = dataset.rename_column(match, canonical)

    return dataset


def load_hf_dataset(
    args: argparse.Namespace,
    config: Dict[str, Any],
    split: str,
) -> Dataset:
    """
    Load either a local Hugging Face dataset directory or a remote dataset.
    """
    if split not in config["split_map"]:
        raise ValueError(
            f"Split '{split}' is not defined for dataset "
            f"'{config['dataset_name']}'."
        )

    hf_split = config["split_map"][split]

    dataset_source = (
        str(args.data_path)
        if args.data_path is not None
        else config["hf_repo"]
    )

    # An explicit --hf_config always wins over the built-in mapping.
    hf_config: Optional[str] = (
        args.hf_config
        if args.hf_config is not None
        else config["hf_config"]
    )

    load_kwargs: Dict[str, Any] = {
        "path": dataset_source,
        "split": hf_split,
        "verification_mode": VerificationMode.NO_CHECKS,
    }

    if hf_config is not None:
        load_kwargs["name"] = hf_config

    if args.hf_token is not None:
        load_kwargs["token"] = args.hf_token

    print(f"Loading dataset : {dataset_source}")
    print(f"Config/subset   : {hf_config or '(none)'}")
    print(f"Source split    : {hf_split}")

    dataset = load_dataset(**load_kwargs)

    dataset = normalize_columns(
        dataset=dataset,
        column_aliases=config["column_aliases"],
    )

    # Keep the original compressed bytes or path. Audio is decoded in worker
    # processes instead of in the main process.
    dataset = dataset.cast_column(
        "audio_filepath",
        Audio(decode=False),
    )

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_datasets() -> None:
    print("Supported datasets:")

    for name in sorted(DATASETS):
        spec = DATASETS[name]
        langs = ", ".join(dataset_languages(name))
        splits = ", ".join(sorted(set(spec["split_map"].values())))

        print(f"  {name}")
        print(f"      repo   : {spec['hf_repo']}")
        print(f"      splits : {splits}")
        print(f"      langs  : {langs}")


def print_languages() -> None:
    print("Supported language codes (dataset: subset):")

    for code in supported_languages():
        entries = []

        for name in sorted(DATASETS):
            if code not in dataset_languages(name):
                continue

            subset = DATASETS[name]["languages"]
            label = subset[code] if subset is not None else "-"
            entries.append(f"{name}:{label}")

        print(f"  {code:<4} -> {', '.join(entries)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert IndicVoices, Svarah, Kathbath, Shrutilipi, Lahaja or "
            "tamil_asr_corpus datasets to PCM-16 WAV files and a JSONL "
            "ASR manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--list_langs",
        action="store_true",
        help="Print the supported language codes and exit.",
    )

    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="Print the supported datasets and exit.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        metavar="CODE",
        help=(
            "Language code to prepare, for example 'hi', 'bn', 'ta'. "
            "Use 'en' for Svarah. Run --list_langs to see all codes."
        ),
    )

    parser.add_argument(
        "--split",
        choices=["train", "valid", "both"],
        default="both",
        help=(
            "Dataset split to prepare. When omitted, train and valid are "
            "both prepared into their respective directories."
        ),
    )

    parser.add_argument(
        "--dataname",
        type=str,
        default=None,
        help=(
            "Dataset output folder name, for example "
            "'IndicVoices' or 'Kathbath'."
        ),
    )

    parser.add_argument(
        "--hf_config",
        type=str,
        default=None,
        help=(
            "Override the Hugging Face config/subset name. Defaults to the "
            "subset mapped from --dataset and --lang."
        ),
    )

    source_group = parser.add_mutually_exclusive_group()

    source_group.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to a pre-cloned local Hugging Face dataset directory.",
    )

    source_group.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token. The configured dataset is "
            "downloaded automatically."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for converted WAV files. Defaults to "
            "/data/dataset/<dataname>/<split>/<lang>_audio/. "
            "Requires an explicit --split."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Output JSONL manifest path. Defaults to "
            "/data/dataset/<dataname>/<split>/<lang>_manifest.json. "
            "Requires an explicit --split."
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. Use zero for automatic "
            "selection, capped at eight workers."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Number of samples handled by each worker invocation. "
            "Use a lower value for very long audio files."
        ),
    )

    args = parser.parse_args()

    if args.list_langs or args.list_datasets:
        return args

    for name in ("lang", "dataname"):
        if getattr(args, name) is None:
            parser.error(
                f"--{name} is required "
                f"(or use --list_langs / --list_datasets)."
            )

    args.lang = args.lang.strip().lower()

    try:
        args.dataset = resolve_dataset_name(args.dataname, args.lang)
    except ValueError as error:
        parser.error(str(error))

    allowed = dataset_languages(args.dataset)

    if args.lang not in allowed:
        parser.error(
            f"unsupported --lang '{args.lang}' for dataset "
            f"'{args.dataset}'. Supported: {', '.join(allowed)}"
        )

    if args.num_workers < 0:
        parser.error("--num_workers must be greater than or equal to 0.")

    if args.batch_size < 1:
        parser.error("--batch_size must be greater than or equal to 1.")

    if args.split == "both" and (
        args.output_dir is not None or args.manifest is not None
    ):
        parser.error(
            "--output_dir/--manifest cannot be used when both splits are "
            "prepared; pass --split train or --split valid."
        )

    return args


def resolve_defaults(
    args: argparse.Namespace,
    split: str,
    output_dir: Optional[Path],
    manifest: Optional[Path],
) -> None:
    """
    Fill output paths for a single split using:

        /data/dataset/<dataname>/<split>/
    """
    base_directory = (
        Path("/data/dataset")
        / args.dataname
        / split
    )

    args.output_dir = (
        output_dir
        if output_dir is not None
        else base_directory / f"{args.lang}_audio"
    )

    args.manifest = (
        manifest
        if manifest is not None
        else base_directory / f"{args.lang}_manifest.json"
    )


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        print_datasets()
        return

    if args.list_langs:
        print_languages()
        return

    if args.data_path is None and args.hf_token is None:
        print(
            "ERROR: provide either --data_path for a local dataset or "
            "--hf_token for a Hugging Face download.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        dataset_config = get_dataset_config(args.dataset, args.lang)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    requested_split = args.split

    splits = (
        ["train", "valid"]
        if requested_split == "both"
        else [requested_split]
    )

    explicit_output_dir = args.output_dir
    explicit_manifest = args.manifest

    failed_splits: List[str] = []

    for split in splits:
        args.split = split

        resolve_defaults(
            args=args,
            split=split,
            output_dir=explicit_output_dir,
            manifest=explicit_manifest,
        )

        subset_name = args.hf_config or dataset_config["hf_config"]
        upstream_split = dataset_config["split_map"].get(split)

        print()
        print("=" * 64)
        print(f"Dataset        : {args.dataset}")
        print(f"Language       : {args.lang}")
        print(f"Source repo    : {dataset_config['hf_repo']}")
        print(f"Subset         : {subset_name or '(none)'}")
        print(f"Requested split: {split} -> {upstream_split}")
        print(f"Output name    : {args.dataname}")
        print(f"Output dir     : {args.output_dir}")
        print(f"Manifest       : {args.manifest}")
        print(
            f"Workers        : "
            f"{'auto' if args.num_workers == 0 else args.num_workers}"
        )
        print(f"Batch size     : {args.batch_size}")
        print("=" * 64)
        print()

        if split == "valid" and upstream_split != "valid":
            print(
                f"NOTE: dataset '{args.dataset}' has no dedicated validation "
                f"split; falling back to '{upstream_split}'. Hold out a "
                f"subset of the manifest yourself if you need a clean "
                f"split.\n",
                file=sys.stderr,
            )

        try:
            dataset = load_hf_dataset(
                args=args,
                config=dataset_config,
                split=split,
            )

            process_dataset(
                dataset=dataset,
                output_dir=args.output_dir,
                manifest_path=args.manifest,
                lang_tag=dataset_config["lang_tag"],
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )

        except KeyboardInterrupt:
            print(
                "\nConversion interrupted by user.",
                file=sys.stderr,
            )
            sys.exit(130)

        except Exception as error:
            print(
                f"\nERROR [{split}]: {error}",
                file=sys.stderr,
            )

            if requested_split != "both":
                sys.exit(1)

            failed_splits.append(split)

            print(
                "Continuing with the remaining split(s)...\n",
                file=sys.stderr,
            )

    if failed_splits:
        print(
            f"\nFinished with failures in: {', '.join(failed_splits)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
