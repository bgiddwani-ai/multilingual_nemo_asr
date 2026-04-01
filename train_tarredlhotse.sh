#!/bin/bash

# Manifest+Tarred Dataset
python NeMo/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=/home/asr/conf \
    --config-name=parakeet_0_6v2_tdt_bpe \
    +model.train_ds.use_lhotse=true \
    +model.train_ds.input_cfg="/home/asr/dataset/input_cfg.yaml" \
    +model.train_ds.seed=42 \
    +model.train_ds.shard_seed="trng" \
    +model.train_ds.bucket_duration_bins="[2.186,3.616,4.895,5.631,6.292,6.896,7.552,8.223,8.894,10.238,10.913,12.464,13.345,14.291,15.374,16.66,18.161]" \
    +model.train_ds.bucket_batch_size="[1024,512,512,256,256,256,256,128,128,128,128,128,128,64,64,64,64,64]" \
    +model.train_ds.bucket_buffer_size=10000 \
    +model.train_ds.shuffle_buffer_size=10000 \
    model.train_ds.num_workers=2 \
    model.train_ds.max_duration=30.0 \
    model.optim.name="adamw" \
    model.optim.lr=5e-4 \
    trainer.devices=8 \
    trainer.accelerator="gpu" \
    trainer.strategy=ddp \
    trainer.max_steps=100000 \
    trainer.val_check_interval=1000 \
    ++trainer.limit_train_batches=1000 \
    ++trainer.use_distributed_sampler=false \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="train_en_hi_tarredlhotse" \
    exp_manager.wandb_logger_kwargs.project="multi_asr_tdt" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true

#[1024,512,512,256,256,256,256,128,128,128,128,128,128,64,64,64,64,64]
#[1323,756,526,468,394,351,312,250,243,216,182,162,140,136,124,120,101,92]