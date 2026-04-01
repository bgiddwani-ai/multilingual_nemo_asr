# 🎙️ NVIDIA NeMo Conformer-Based ASR Models (Hugging Face)

A curated list of **Conformer / FastConformer-based ASR models** from NVIDIA NeMo available on Hugging Face.

---

## 📌 Overview

This list includes models built on:

* ⚡ **FastConformer Encoder (primary focus)**
* 🧠 **Conformer Encoder (baseline variants)**
* 🔁 Architectures: **CTC, RNNT, TDT, Encoder-Decoder**

> ⚠️ Most modern NeMo SOTA models (Parakeet, Canary) are based on **FastConformer**, which is ~2–3× faster than vanilla Conformer with similar accuracy ([NVIDIA Docs][1])

---

## 🚀 Model Families

---

# 🐦 Parakeet (FastConformer-based)

> FastConformer encoder + CTC / RNNT / TDT decoders ([NVIDIA Docs][2])

### 🔹 Core Models

| Model                | Type | Description                | HF Link                                                                                              |
| -------------------- | ---- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `parakeet-ctc-0.6b`  | CTC  | Lightweight high-speed ASR | [https://huggingface.co/nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b)   |
| `parakeet-ctc-1.1b`  | CTC  | Larger, better accuracy    | [https://huggingface.co/nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b)   |
| `parakeet-rnnt-0.6b` | RNNT | Streaming-friendly ASR     | [https://huggingface.co/nvidia/parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) |
| `parakeet-rnnt-1.1b` | RNNT | High-accuracy RNNT         | [https://huggingface.co/nvidia/parakeet-rnnt-1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) |

### 🔹 TDT (State-of-the-Art)

| Model                  | Type | Notes                      |
| ---------------------- | ---- | -------------------------- |
| `parakeet-tdt-0.6b-v2` | TDT  | Top of OpenASR leaderboard |
| `parakeet-tdt-1.1b`    | TDT  | Larger variant             |

---

# 🐤 Canary (FastConformer + Transformer Decoder)

> Encoder-decoder architecture with FastConformer encoder ([NVIDIA Developer][3])

### 🔹 Core Models

| Model               | Params | Description                       | HF Link |
| ------------------- | ------ | --------------------------------- | ------- |
| `canary-1b`         | 1B     | Multilingual ASR + translation    |         |
| `canary-1b-v2`      | 1B     | Improved multilingual performance |         |
| `canary-1b-flash`   | 1B     | Faster inference                  |         |
| `canary-180m-flash` | 180M   | Lightweight fast model            |         |

### 🔹 Advanced / Hybrid

| Model              | Description                           |
| ------------------ | ------------------------------------- |
| `canary-qwen-2.5b` | ASR + LLM hybrid (speech + reasoning) |

---

# ⚡ FastConformer (Generic Models)

> Optimized Conformer variant (~2.4× faster) ([NVIDIA Docs][1])

### 🔹 CTC Models

| Model                            | Description              | HF Link                                                                                                                      |
| -------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `stt_en_fastconformer_ctc_large` | Fast inference CTC model | [https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) |

### 🔹 RNNT Models

| Model                                   | Description          | HF Link                                                                                                                                    |
| --------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `stt_en_fastconformer_transducer_large` | Streaming RNNT model | [https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large) |

### 🔹 Hybrid Models

| Model                                  | Description                |
| -------------------------------------- | -------------------------- |
| `stt_en_fastconformer_hybrid_large_pc` | CTC + RNNT hybrid decoding |

---

# 🧠 Architecture Summary

| Family        | Encoder       | Decoder          | Strength                      |
| ------------- | ------------- | ---------------- | ----------------------------- |
| Parakeet      | FastConformer | CTC / RNNT / TDT | Best speed + accuracy balance |
| Canary        | FastConformer | Transformer      | Multilingual + translation    |
| FastConformer | FastConformer | CTC / RNNT       | Production-ready baseline     |

---

#### Note: This list is built by AI (It can have some bugs)
---
