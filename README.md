### 👨‍💻 Author

**Bharat Giddwani**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/bgiddwani-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/bharat3012)

</div>

# Multilingual ASR Fine-Tuning with NVIDIA NeMo

Fine-tune NVIDIA’s **[Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** model for multilingual Automatic Speech Recognition (ASR) using the NeMo framework. This pipeline is designed for scalable, high-performance training with tarred datasets, Lhotse-based bucketing, and custom multilingual tokenization.

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
* **Lhotse bucketing** for efficient batching of variable-length audio
* **Weighted multilingual sampling** during training
* **Custom SentencePiece tokenizer** built from domain-specific data

---

## Requirements

* Python 3.10+
* CUDA 12.x compatible GPU (A100 / H100 / H200 recommended)
* `git`, `git-lfs`, Jupyter Notebook (for `.ipynb` workflows)

---

## Installation

Start with base environment - I prefer [NVIDIA NGC containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

```bash
docker run --gpus all -it -v $PWD:/home --ipc=host -p 8001:8001 nvcr.io/nvidia/pytorch:25.11-py3
cd /home
git clone https://github.com/bgiddwani-ai/multilingual_nemo_asr.git
cd multilingual_nemo_asr
git clone https://github.com/NVIDIA-NeMo/NeMo.git
cd NeMo
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
        │
        ▼
Eval ASR model
        │
        ▼
Visualize the results
```

---

## Step 1 — Data Preparation

### 1. Create NeMo Manifests

NeMo manifests are JSON files that map audio files to transcriptions:

```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 3.45, "text": "नमस्ते दुनिया", "lang": "hi"}
```

Use the provided script:

```bash
python data_prep.py --lang hi --split train --data_path ./dataset/IndicVoices --dataname indicvoices
```

Datasets used in this pipeline:

* IndicVoices (Hindi)
* Svarah (English)

### Expected Directory Structure

```
dataset/
├── indicvoices/
│   ├── train/
│   │   ├── hi_audio
│   │   └── hi_manifest.json
│   └── valid/
│       ├── hi_audio
│       └── hi_manifest.json
├── svarah/
│   └── train/
│       ├── hi_audio
│       └── hi_manifest.json
```

---

### 2. Convert to Tarred Dataset

Convert raw audio + manifests into sharded tar datasets:

```bash
python NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path='<path/to/manifest.json>' \
  --target_dir='<path/to/manifest.json>' \
  --num_shards=256 \ #512 or 1024
  --max_duration=20.0 \ #Provide based on data
  --min_duration=0.1 \ #Provide based on data
  --shuffle \
  --workers=16
```

Example:

```bash
bash tarred_svarah.sh
bash tarred_indicvoices.sh
```

### Expected Output

```
dataset/
├── indicvoices/
│   ├── train/
│   │   └── hi_tarred/
│   │       ├── audio__OP_0..255_CL_.tar
│   │       └── sharded_manifests/
│   │           └── manifest__OP_0..255_CL_.json
│   └── valid/
│       ├── hi_audio
│       └── hi_manifest.json
├── svarah/
│   └── train/
│       └── en_tarred/
│           ├── audio__OP_0..255_CL_.tar
│           └── sharded_manifests/
│               └── manifest__OP_0..255_CL_.json
```

> **Note:** The pattern `__OP_0..255_CL_` is NeMo’s glob syntax representing shard indices from 0 to 255.

---

## Step 2 — Data Processing

Efficient training requires optimal batching based on audio duration distribution.

### Dataset Configuration

Create `dataset/input_cfg.yaml`:

```yaml
- type: nemo_tarred
  manifest_filepath: /home/asr/dataset/indicvoices/train/hi_tarred/sharded_manifests/manifest__OP_0..255_CL_.json
  tarred_audio_filepaths: /home/asr/dataset/indicvoices/train/hi_tarred/audio__OP_0..255_CL_.tar
  weight: 0.8
  tags:
    lang: hi

- type: nemo_tarred
  manifest_filepath: /home/asr/dataset/svarah/train/en_tarred/sharded_manifests/manifest__OP_0..255_CL_.json
  tarred_audio_filepaths: /home/asr/dataset/svarah/train/en_tarred/audio__OP_0..255_CL_.tar
  weight: 0.2
  tags:
    lang: en
```

### Estimate Duration Buckets

```bash
python3 NeMo/scripts/speech_recognition/estimate_duration_bins.py -b 20 /home/asr/dataset/input_cfg.yaml
```

### Optimize Batch Sizes

```bash
CUDA_VISIBLE_DEVICES=0 python3 NeMo/scripts/speech_recognition/oomptimizer.py \
--config-path /home/asr/conf/parakeet_0_6v2_tdt_bpe.yaml \
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
git clone https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
cd parakeet-tdt-0.6b-v2
tar -xvf parakeet-tdt-0.6b-v2.nemo
cd ..
```

The `.nemo` archive contains:

* Model weights
* Configuration
* Tokenizer

---

## Step 4 — Building Tokenizer

### Train Hindi Tokenizer

```bash
python3 NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest dataset/indicvoices/train/hi_manifest.json \
  --data_root models/hi_tokenizer_v256 \
  --vocab_size 256 \
  --tokenizer spe \
  --spe_type bpe \
  --spe_character_coverage 0.99
```

### Copy English Tokenizer

```bash
mkdir -p models/en_tokenizer_tdt0_6b_v2/tokenizer_spe_bpe_v1024
cp models/parakeet-tdt-0.6b-v2/*.vocab models/en_tokenizer_tdt0_6b_v2/tokenizer_spe_bpe_v1024/tokenizer.vocab
cp models/parakeet-tdt-0.6b-v2/*.model models/en_tokenizer_tdt0_6b_v2/tokenizer_spe_bpe_v1024/tokenizer.model
cp models/parakeet-tdt-0.6b-v2/*.txt models/en_tokenizer_tdt0_6b_v2/tokenizer_spe_bpe_v1024/vocab.txt
```

### Update Model Tokenizer

```bash
python3 update_tokenizer.py
```

---

## Step 5 — Training

Launch training using:

```bash
speech_to_text_rnnt_bpe.py
```

Configuration file:

```
conf/parakeet_0_6v2_tdt_bpe.yaml
```

### Training Modes

* `train_manifest.sh` → Direct manifest-based training
* `train_tarred.sh` → Tarred dataset training (high throughput)
* `train_tarredlhotse.sh` → Tarred dataset + Lhotse bucketing (recommended)

---

## Step 6 — Evaluation

Evaluate model performance (WER/CER):

```bash
bash eval_indicvoices.sh
```

---

## Step 7 — Visualization

Launch the Speech Data Explorer:

```bash
python NeMo/tools/speech_data_explorer/data_explorer.py /home/asr/results/indivoices/commotion_run1_epoch5/hi/predictions_all.json --port 8001
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
