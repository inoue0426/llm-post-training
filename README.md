# LLM Tuning Playground

A hands-on playground for learning LLM fine-tuning and alignment with Hugging Face.

## Goals

- Learn LoRA / QLoRA with a small, reproducible experiment.
- Compare a base model with an SFT model.
- Add preference tuning with DPO next.
- Keep experiments runnable in Google Colab.

## Current experiment

`notebooks/01_lora.ipynb` is the first experiment. It trains a small causal language model with LoRA on a tiny instruction dataset and then compares generations before and after tuning.

## Colab

Open the notebook in Colab after cloning this repository, or use the GitHub `Open in Colab` flow when available.

The first experiment is intentionally small so that it can be run on a typical Colab GPU. Exact GPU availability and memory vary by Colab session.

## Repository layout

```text
llm-tuning-playground/
├── README.md
├── requirements.txt
└── notebooks/
    └── 01_lora.ipynb
```

## Next steps

1. LoRA / QLoRA SFT
2. DPO with chosen/rejected preference pairs
3. Base vs SFT vs DPO evaluation
4. Optional PPO-style RLHF experiment

## Disclaimer

This repository is for experimentation and learning. Model availability, library APIs, and recommended settings can change over time; pin or update dependencies when reproducing an experiment.
