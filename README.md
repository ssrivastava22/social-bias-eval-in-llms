# social-bias-eval-in-llms
# Evaluating and Debiasing LLMs on Social Bias Benchmarks

This repository contains our research and implementation for evaluating and mitigating social bias in large language models (LLMs) using the [BBQ (Bias Benchmark for QA)](https://github.com/nyu-mll/BBQ) dataset. The project benchmarks multiple LLMs, fine-tunes Falcon-7B using LoRA, and introduces an adversarial learning setup to improve fairness without degrading accuracy.

---

## Project Highlights

- Benchmarking social bias in Falcon-7B and LLaMA-2 models
- Structured QA format using the BBQ dataset (Race/Ethnicity and Sexual Orientation subsets)
- Parameter-efficient fine-tuning using LoRA adapters
- Debiasing via an adversarial training head + gradient reversal
- Evaluation with accuracy and uncertainty metrics on QA

---

## Reproducing Experiments
Follow these steps to reproduce the results across all three phases of the project:

### 1. Falcon-7B Zero-shot Evaluation
Run the following notebook to evaluate the off-the-shelf Falcon-7B model on the BBQ dataset:
```
notebooks/initial_evaluations/Falcon_BBQ.ipynb
```
### 2. LLama-3B Zero-shot Evaluation
Run the following notebook to evaluate the off-the-shelf Llama-3B model on the BBQ dataset:
```
notebooks/initial_evaluations/llama_BBQ.ipynb
```
### 3. LoRA Fine-tuning (QA-only)
Fine-tune Falcon-7B using parameter-efficient LoRA adapters and a lightweight QA head:
```
notebooks/model_training/Falcon_LoRA_QA.ipynb
```
### 4. Adversarial Debiasing
Train the `DebiasFalcon` model using both the QA and adversarial loss components. This setup applies a gradient reversal layer to penalize group-specific feature encoding:
```
notebooks/model_training/Adversarial_learning.ipynb
```



