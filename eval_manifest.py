#!/usr/bin/env python3
"""Transcribe one NeMo manifest with Nemotron and compute corpus WER and CER.

WER (word error rate) and CER (character error rate) are both reported. CER is
especially informative for Indic scripts, where a single word can carry several
combining characters and word segmentation is less reliable.
"""

import argparse
import json
from pathlib import Path

from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt


def edit_distance(reference, hypothesis):
    """Levenshtein distance over any two sequences (word lists or char strings)."""
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def normalize(text):
    return " ".join(text.strip().split())


def force_prompt_index(target_lang, allow_fallback=False):
    """Pin the transcription prompt to ``target_lang`` for every cut.

    ``model.transcribe()`` builds its dataloader without setting
    ``default_prompt_mode``, so the prompt-index dataset falls back to
    ``'unified'`` and tries to read each cut's language. Transcribing bare audio
    files leaves the language unset, which raises "Unknown prompt key: 'None'".
    We override the per-cut lookup to return a fixed index, making evaluation
    deterministic and consistent with training.

    The transcribe path resolves keys against ``test_ds.prompt_dictionary``,
    which is locale-keyed (``mr-IN``, ``gu-IN``) rather than using the bare
    codes (``mr``, ``gu``) that the training dataloaders declare. A bare code
    therefore silently degrades to ``auto``, so an unresolved key is an error
    unless the caller opts into the fallback.
    """
    from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
        LhotseSpeechToTextBpeDatasetWithPromptIndex as _PromptDataset,
    )

    def _get_prompt_index_for_cut(self, cut):
        if target_lang in self.prompt_dict:
            return self.prompt_dict[target_lang]
        near = sorted(k for k in self.prompt_dict if k.lower().startswith(target_lang.lower()))
        if not allow_fallback:
            raise KeyError(
                f"Prompt key {target_lang!r} is not in the model prompt dictionary. "
                f"Did you mean one of {near}? "
                f"Pass --allow-prompt-fallback to transcribe with auto "
                f"(index {self.auto_index}) instead."
            )
        if not getattr(self, "_prompt_fallback_warned", False):
            self._prompt_fallback_warned = True
            print(
                f"[warn] prompt key '{target_lang}' not in model prompt dictionary; "
                f"falling back to auto index {self.auto_index}. Close matches: {near}"
            )
        return self.auto_index

    _PromptDataset._get_prompt_index_for_cut = _get_prompt_index_for_cut


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-lang", default="auto", help="Prompt key, e.g. mr, gu, mr-IN, or auto")
    parser.add_argument(
        "--lang-label",
        help="Language label for reporting (defaults to --target-lang). Lets several "
        "languages share one prompt key, e.g. auto, while staying distinct in summaries.",
    )
    parser.add_argument("--output", type=Path, help="Per-utterance predictions JSONL")
    parser.add_argument("--metrics-json", type=Path, help="Write corpus metrics + counts as JSON")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full manifest")
    parser.add_argument(
        "--allow-prompt-fallback",
        action="store_true",
        help="Transcribe with the auto prompt when --target-lang is not in the "
        "model prompt dictionary, instead of failing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
                if args.limit and len(records) >= args.limit:
                    break
    if not records:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    model = EncDecRNNTBPEModelWithPrompt.restore_from(str(args.model))
    force_prompt_index(args.target_lang, allow_fallback=args.allow_prompt_fallback)
    hypotheses = model.transcribe(
        [record["audio_filepath"] for record in records],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_lang=args.target_lang,
    )
    predictions = [item.text if hasattr(item, "text") else str(item) for item in hypotheses]
    if len(predictions) != len(records):
        raise RuntimeError(
            f"Transcription returned {len(predictions)} predictions for {len(records)} records"
        )

    word_errors = 0
    reference_words = 0
    char_errors = 0
    reference_chars = 0
    exact_matches = 0
    output = args.output or args.manifest.with_name(
        f"{args.manifest.stem}.{args.target_lang}.predictions.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record, prediction in zip(records, predictions):
            reference = normalize(str(record["text"]))
            prediction = normalize(prediction)
            reference_tokens = reference.split()
            word_errors += edit_distance(reference_tokens, prediction.split())
            reference_words += len(reference_tokens)
            char_errors += edit_distance(reference, prediction)
            reference_chars += len(reference)
            exact_matches += reference == prediction
            result = dict(record)
            result["pred_text"] = prediction
            result["target_lang"] = args.target_lang
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    wer = word_errors / reference_words if reference_words else 0.0
    cer = char_errors / reference_chars if reference_chars else 0.0
    exact = exact_matches / len(records) if records else 0.0
    print(f"Samples:       {len(records):,}")
    print(f"Target prompt: {args.target_lang}")
    print(f"WER:           {wer:.4%}")
    print(f"CER:           {cer:.4%}")
    print(f"Exact match:   {exact:.4%}")
    print(f"Predictions:   {output}")

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "manifest": str(args.manifest),
            "model": str(args.model),
            "lang_label": args.lang_label or args.target_lang,
            "target_lang": args.target_lang,
            "samples": len(records),
            "wer": wer,
            "cer": cer,
            "exact_match": exact,
            "word_errors": word_errors,
            "reference_words": reference_words,
            "char_errors": char_errors,
            "reference_chars": reference_chars,
            "exact_matches": exact_matches,
        }
        with args.metrics_json.open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
        print(f"Metrics:       {args.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())