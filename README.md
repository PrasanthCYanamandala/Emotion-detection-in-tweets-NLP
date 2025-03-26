# 😃 Emotion Detection – Fall 2024 Kaggle Competition  
**BUAN 6359: Natural Language Processing Final Project**

This repository contains our end-to-end solution for the [Kaggle Emotion Detection Fall 2024 competition](https://www.kaggle.com/competitions/emotion-detection-fall-2024/overview), which challenges participants to build robust models that detect multiple emotional states in tweets. The task involves **multi-label classification** across 11 emotions such as joy, sadness, fear, anger, trust, and more.

## 📦 Repository Contents

- `Comprehensive Report.pdf` – Summarizes all experiments (HW5–HW8), models, metrics, and future plans  
- `HW6_Report.pdf` – Detailed comparison of encoder-only models (RoBERTa, DistilBERT, DistilRoBERTa)  
- Model weights, logs, and experiments tracked via [Weights & Biases](#-experiment-tracking)

---

## 🔍 Project Summary

### 🧠 HW5: Feedforward Neural Network (Scratch Model)
- Built a custom multilabel classifier using PyTorch
- Input: Tokenized tweet embeddings
- Performance:  
  - **F1 Score**: 0.1361  
  - **Accuracy**: 0.0569  
- Limitation: Poor semantic capture and low capacity

### 🤖 HW6: Transformer-Based Models (Encoder-Only)
- Compared 3 pretrained encoder-only models:
  - `RoBERTa Base` – Best F1 score (0.5897)
  - `DistilBERT` – Efficient and accurate (F1: 0.5806)
  - `DistilRoBERTa` – Lighter RoBERTa alternative (F1: 0.5703)
- Applied class weights for handling imbalance
- All models evaluated using Macro F1 and Accuracy

### 💬 HW7: Sentence Similarity via Embeddings
- Used transformer-based embedding models:
  - `meta-llama/Llama-3.2-1B`
  - `google/gemma-2-2b`
  - `intfloat/e5-mistral-7b-instruct (MTEB)`
- Purpose: Measure sentence-level emotional similarity
- Notable Performance:
  - Gemma: F1 = 0.5377
  - LLaMA: F1 = 0.4806
  - MTEB: F1 = 0.2663

### 🪄 HW8: Zero-Shot & Instruction-Tuned Models
- Compared:
  - Base (untuned) embedding model
  - Instruction-tuned model
  - Zero-shot classification with `BART-large-mnli`
- F1 Scores:
  - Instruction-tuned: **0.5271**
  - Base: 0.4797
  - Zero-shot: Poor performance without fine-tuning

---

## ⚖️ Evaluation Metrics

- **Macro F1 Score** (primary metric)
- **Validation Accuracy**
- **W&B Logging** for all experiments

---

## 📈 Experiment Tracking (W&B)

### 🔹 HW6
- RoBERTa: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/Exp1)
- DistilBERT: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/Exp2)
- DistilRoBERTa: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/Exp3)

### 🔹 HW7
- Gemma: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/gemma)
- LLaMA: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/LLama)
- MTEB: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/MTEB)

### 🔹 HW8
- Base Model: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/Base)
- Instruction-Tuned: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/Instruction-tuned)
- Zero-Shot: [View Run](https://wandb.ai/pxy230011-the-university-of-texas-at-dallas/zero-shot)

---

## 🛠️ Key Techniques

- Multi-label classification with sigmoid outputs
- Weighted loss functions to address imbalance
- Tokenizer-specific preprocessing (e.g., RoBERTaTokenizer)
- F1 score optimization during training
- Use of transformer embedding models for similarity tasks

---

## 🚀 Future Work

- Incorporate data augmentation to boost minority class representation
- Explore T5 and GPT models for better zero-/few-shot performance
- Fine-tune decoder models for contextual emotion generation
- Optimize hybrid architectures combining encoders and decoders

---

## 👨‍💻 Author

**Prasanth Chowdary Yanamandala**  
University of Texas at Dallas | Fall 2024  
Kaggle Username: `pxy230011`

---

📌 *This project reflects deep experimentation with transformer-based NLP models for multi-label emotion detection in tweets, with model selection driven by interpretability, accuracy, and efficiency.*
