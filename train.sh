#!/bin/bash
# Update parameters from here if necessary
python Speech/examples/asr/speech_to_text_finetune.py \
    --config-path=/home/multilingual_nemo_asr/conf \
    --config-name=nemotron_fastconformer_transducer_bpe_streaming_prompt_mr_gu \
    model.train_ds.num_workers=2 \
    model.train_ds.max_duration=30.0 \
    model.optim.name="adamw" \
    model.optim.lr=2 \
    model.optim.weight_decay=1e-3
