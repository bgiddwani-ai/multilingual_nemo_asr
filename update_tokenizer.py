#!/usr/bin/env python3
"""Replace Nemotron's BPE tokenizer only when coverage checks prove it necessary.

NeMo's change_vocabulary() reinitializes the complete RNNT decoder and joint
network. This preserves the pretrained encoder, but discards pretrained decoder
and joint weights. Run check_tokenizer_coverage.py before using this script.
"""

import argparse
from pathlib import Path

from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="NeMo BPE tokenizer directory containing tokenizer.model",
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--confirm-reinitialize-decoder",
        action="store_true",
        help="Required acknowledgement that decoder and joint weights are replaced",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.confirm_reinitialize_decoder:
        raise SystemExit(
            "Refusing to replace the tokenizer: this reinitializes Nemotron's RNNT "
            "decoder and joint. First run check_tokenizer_coverage.py, then repeat "
            "with --confirm-reinitialize-decoder if replacement is necessary."
        )
    if not args.input_model.is_file():
        raise SystemExit(f"Input model does not exist: {args.input_model}")
    if not (args.tokenizer_dir / "tokenizer.model").is_file():
        raise SystemExit(f"Missing tokenizer.model in: {args.tokenizer_dir}")
    if args.output_model.exists():
        raise SystemExit(f"Refusing to overwrite existing model: {args.output_model}")

    model = EncDecRNNTBPEModelWithPrompt.restore_from(str(args.input_model), map_location="cpu")
    old_vocab_size = getattr(model.tokenizer, "vocab_size", None)
    model.change_vocabulary(
        new_tokenizer_dir=str(args.tokenizer_dir),
        new_tokenizer_type="bpe",
    )
    new_vocab_size = getattr(model.tokenizer, "vocab_size", None)
    print(f"Vocabulary size: {old_vocab_size} -> {new_vocab_size} (decoder + joint reinitialized)")
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(args.output_model))
    print(f"Saved model with replacement tokenizer to {args.output_model}")


if __name__ == "__main__":
    main()
