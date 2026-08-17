### 👨‍💻 Author

**Bharat Giddwani**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/bgiddwani-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/bharat3012)

</div>

# Multilingual ASR Fine-Tuning with NVIDIA NeMo

Training NVIDIA’s **[Nemotron 3.5 Multilingual ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)** model by taking encoder as base and decoder/joint from scratch for multilingual Automatic Speech Recognition (ASR) using the NeMo framework. This pipeline is designed for scalable, high-performance training with tarred datasets, Lhotse-based bucketing, and custom multilingual tokenization.

---

## Table of Contents

* [Overview](#overview)
* [Requirements](#requirements)
* [Installation](#installation)
* [Pipeline Overview](#pipeline-overview)
* [Step 1 — Data Preparation](#step-1--data-preparation)
* [Step 2 — Data Processing](#step-2--data-processing)
* [Step 3 — Model Setup](#step-3--model-setup)
* [Step 4 — Tokenizer Training](#step-4--building-tokenizer)
* [Step 5 — Training](#step-5--training)
* [Step 6 — Evaluation](#step-6--evaluation)
* [Step 7 — Visualization](#step-7--visualization)

---

## Overview

This pipeline enables fine-tuning of the Parakeet RNNT model on a **weighted multilingual dataset** (e.g., Hindi + English).

### Key Features

* **Tarred datasets** for high-throughput, streaming-based I/O
* **Lhotse bucketing** for efficient batching of variable-length audio - makes dataloader an infinite dataloader
* **Weighted multilingual sampling** during training
* **Custom SentencePiece tokenizer** built from domain-specific data
* **Code-Switching with Data changes**
* Etc..

---

## Requirements

* Python 3.10+
* CUDA 12.x compatible GPU (A100 / H100 / H200 recommended)
* `git`, `git-lfs`

---

## Installation

Start with base environment - I prefer [NVIDIA NGC containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

```bash
docker run --gpus all -it -v $PWD:/home -v /path/to/data:/data --ipc=host -p 8001:8001 nvcr.io/nvidia/pytorch:25.11-py3
cd /home
git clone https://github.com/bgiddwani-ai/multilingual_nemo_asr.git
cd multilingual_nemo_asr
git clone https://github.com/NVIDIA-NeMo/Speech.git
cd Speech
pip install '.[all]'
pip install -r tools/speech_data_explorer/requirements.txt
```

---

## Pipeline Overview

```
Raw Audio + Transcripts
        │
        ▼
NeMo manifest + audio extraction
        │
        ▼
Weighted multi-language configuration
        │
        ▼
Bucket estimation + batch sizing
        │
        ▼
Base model initialization
        │
        ▼
Multilingual tokenizer construction
        │
        ▼
Fine-tuned ASR model
```

---

## Step 1 — Data Preparation

### 1. Create NeMo Manifests

NeMo manifests are JSON files that map audio files to transcriptions:

```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 3.45, "text": "नमस्कार जग", "lang": "hi"}
```

Use the provided script for sample:

```bash
python data_prep.py \
    --lang mr \
    --split valid \
    --data_path /data/dataset/IndicVoices \
    --dataname indicvoices \
    --output_dir /data/dataset/indicvoices/valid/hi_audio \
    --manifest /data/dataset/indicvoices/valid/hi_manifest.json \
    --num_workers 8 \
    --batch_size 8
```

Datasets used in this pipeline:

* IndicVoices (Marathi)
* IndiVoices (Gujarati)

### Samples Expected Directory Structure

```
dataset/
├── indicvoices/
│   ├── train/
│   │   ├── mr_audio
│   │   └── mr_manifest.json
│   └── valid/
│   │   ├── mr_audio
│   │   └── mr_manifest.json
│   ├── train/
│   │   ├── gu_audio
│   │   └── gu_manifest.json
│   └── valid/
│       ├── gu_audio
│       └── gu_manifest.json
```

---

### 2. Convert to Tarred Dataset for faster data i/o

Convert raw audio + manifests into sharded tar datasets:

```bash
python Speech/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path='<path/to/manifest.json>' \
  --target_dir='<path/to/manifest.json>' \
  --num_shards=256 \ #512 or 1024
  --max_duration=30.0 \ #Provide based on data
  --min_duration=0.025 \ #Provide based on data
  --shuffle \
  --workers=16
```

Example:

```bash
bash tarred_datasets.sh
```

### Expected Output

```
/data/dataset/
├── indicvoices/
│   ├── train/
│   │   └── mr_tarred/
│   │       ├── audio__OP_0..255_CL_.tar
│   │       └── sharded_manifests/
│   │           └── manifest__OP_0..255_CL_.json
│   └── valid/
│   │   ├── mr_audio
│   │   └── mr_manifest.json
|   │
│   └── train/
│   │   └── gu_tarred/
│   │       ├── audio__OP_0..255_CL_.tar
│   │       └── sharded_manifests/
│   │           └── manifest__OP_0..255_CL_.json
│   └── valid/
│       ├── gu_audio
│       └── gu_manifest.json
```

> **Note:** The pattern `__OP_0..255_CL_` is NeMo’s glob syntax representing shard indices from 0 to 255.

---

## Step 2 — Data Processing

Efficient training requires optimal batching based on audio duration distribution.

### Dataset Configuration

Create `/data/dataset/input_cfg.yaml`:

```yaml
- type: nemo_tarred
  manifest_filepath: /data/dataset/indicvoices/train/mr_tarred/sharded_manifests/manifest__OP_0..255_CL_.json
  tarred_audio_filepaths: /data/dataset/indicvoices/train/mr_tarred/audio__OP_0..255_CL_.tar
  weight: 0.8
  tags:
    lang: mr

- type: nemo_tarred
  manifest_filepath: /data/dataset/indicvoices/train/gu_tarred/sharded_manifests/manifest__OP_0..255_CL_.json
  tarred_audio_filepaths: /data/dataset/indicvoices/train/gu_tarred/audio__OP_0..255_CL_.tar
  weight: 0.8
  tags:
    lang: gu
```

### Estimate Duration Buckets (Optional - To improve Training Time) 
Read more about it here: [NeMo Speech Datasets](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#pushing-gpu-utilization-to-the-limits-with-bucketing-and-oomptimizer)

```bash
python3 Speech/scripts/speech_recognition/estimate_duration_bins.py -b 20 /home/dataset/input_cfg.yaml
```

### Optimize Batch Sizes (Extended - Optional)

```bash
CUDA_VISIBLE_DEVICES=0 python3 Speech/scripts/speech_recognition/oomptimizer.py \
--config-path /home/multilingual_nemo_asr/conf/parakeet_0_6v2_tdt_bpe.yaml \
--module-name nemo.collections.asr.models.EncDecRNNTBPEModel \
--memory-fraction 0.9 \
--buckets '[2.186,3.616,4.895,5.631,6.292,6.896,7.552,8.223,8.894,10.238,10.913,12.464,13.345,14.291,15.374,16.66,18.161]'
```

> Adjust dataset weights to control language sampling ratios. Values are normalized automatically.

---

## Step 3 — Model Setup

Download and extract the base model:

```bash
cd models
git clone https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
cd nemotron-3.5-asr-streaming-0.6b
tar -xvf nemotron-3.5-asr-streaming-0.6b.nemo
cd ..
```

The `.nemo` archive contains:

* Model weights
* Configuration (model_config.yaml) - Read through to understand LR, context size etc that was used.
* Tokenizer

---

## Step 4 — Building Indic BPE Tokenizer

### Train Single Unified Tokenizer

Suggestion: Collect almost equal quantity of data and use 

```bash
python3 Speech/scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest "/data/dataset/indicvoices/train/mr_manifest.json,/data/dataset/indicvoices/train/gu_manifest.json" \
  --data_root models/mr_gu_nemotron_tokenizer \
  --vocab_size 1024 \
  --tokenizer spe \
  --spe_type bpe \
  --spe_character_coverage 0.99
```


## Step 5 — Training

Launch training using:

```bash
Speech/examples/asr/speech_to_text_finetune.py
```

Configuration file:

Inspiration: 
```
Speech/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml
```

Updated for our use-case and inputs from model_config.yaml (from .nemo)

```
conf/nemotron_fastconformer_transducer_bpe_streaming_prompt_mr_gu.yaml
```

### Training 

* `train.sh` → Tarred dataset + Lhotse (optionally can use-bucketing for faster training)

---

## Step 6 — Evaluation

Evaluate model performance (WER/CER):

```bash
bash eval.sh
```

---

## Step 7 — Visualization

Launch the Speech Data Explorer:

```bash
python Speech/tools/speech_data_explorer/data_explorer.py /home/multilingual_nemo_asr/results/indivoices/commotion_run1_epoch5/hi/predictions_all.json --port 8001
```

---

## Summary

This pipeline provides a **production-ready framework** for multilingual ASR fine-tuning with:

* Scalable data loading via tarred datasets
* Efficient batching using Lhotse
* Flexible multilingual training via weighted sampling
* Custom tokenizer integration

It is optimized for large-scale GPU training and adaptable to additional languages and datasets.

---
