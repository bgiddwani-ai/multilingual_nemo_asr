#!/bin/bash

python NeMo/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=/home/asr/conf \
    --config-name=parakeet_0_6v2_tdt_bpe \
    model.train_ds.is_tarred=true \
    ++model.train_ds.manifest_filepath=[["/home/asr/dataset/indicvoices/train/hi_tarred/tarred_audio_manifest.json"],["/home/asr/dataset/svarah/train/en_tarred/tarred_audio_manifest.json"]] \
    model.train_ds.tarred_audio_filepaths=[["/home/asr/dataset/indicvoices/train/hi_tarred/audio__OP_0..255_CL_.tar"],["/home/asr/dataset/svarah/train/en_tarred/audio__OP_0..255_CL_.tar"]] \
    model.train_ds.num_workers=2 \
    model.train_ds.batch_size=32 \
    model.train_ds.max_duration=30.0 \
    model.optim.name="adamw" \
    model.optim.lr=5e-4 \
    trainer.devices=8 \
    trainer.accelerator="gpu" \
    trainer.precision="bf16" \
    trainer.strategy="ddp" \
    trainer.max_epochs=30 \
    trainer.val_check_interval=0.5 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="train_en_hi_tarred" \
    exp_manager.wandb_logger_kwargs.project="multi_asr_tdt" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true
