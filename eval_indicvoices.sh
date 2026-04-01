#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python NeMo/examples/asr/transcribe_speech_parallel.py \
    model='./models/ckpts/commotion/run1/epoch5/parakeet-multilingual_epoch5.nemo' \
    predict_ds.manifest_filepath='dataset/indicvoices/valid/hi_manifest.json' \
    predict_ds.batch_size=32 \
    output_path="results/indivoices/commotion_run1_epoch5/hi"