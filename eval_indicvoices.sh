#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python Speech/examples/asr/transcribe_speech_parallel.py \
    model='/data/results/indicvoices/new_final_model.nemo' \
    predict_ds.manifest_filepath='dataset/indicvoices/valid/hi_manifest.json' \
    predict_ds.batch_size=32 \
    output_path="/data/results/indivoices/commotion_run1_epoch5/hi"
