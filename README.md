# LLM Tuning Playground

A hands-on collection of **LLM fine-tuning, preference optimization, and reliability experiments** built with Hugging Face, PEFT, and TRL.

This repository is intentionally notebook-first. I use it to prototype training objectives, diagnose model behavior, and run controlled experiments before promoting ideas into larger research codebases.

## What this repository demonstrates

- **Parameter-efficient fine-tuning:** LoRA / QLoRA with PEFT
- **Supervised fine-tuning (SFT):** instruction tuning and task-specific adaptation
- **Preference optimization:** DPO and IPO, including prompt/completion boundary checks
- **Reinforcement-learning-style tuning:** GRPO experiments
- **Reliability evaluation:** distractors, counterfactuals, abstention, and no-evidence cases
- **Experimental design:** multi-seed runs, entity-disjoint splits, controlled perturbations, and explicit baselines
- **Large-model evaluation:** BF16 inference and zero-shot sanity checks on GPU/HPC environments
- **Biomedical applications:** drug-target retrieval and evidence-grounded Chemical -> Gene -> Disease reasoning using CTD-derived relations

## Representative notebooks

| Notebook | Focus |
| --- | --- |
| [`01_lora.ipynb`](notebooks/01_lora.ipynb) | LoRA fine-tuning from a minimal instruction-tuning setup |
| [`02_dpo.ipynb`](notebooks/02_dpo.ipynb) | Introductory DPO / preference tuning |
| [`03_pharma_dpo_fixed.ipynb`](notebooks/03_pharma_dpo_fixed.ipynb) | Preference optimization in a pharmacology-oriented setting |
| [`05_grpo_drug_target_retrieval_local_fixed.ipynb`](notebooks/05_grpo_drug_target_retrieval_local_fixed.ipynb) | GRPO for drug-target retrieval |
| [`06_ctd_reasoning_sft_v2.ipynb`](notebooks/06_ctd_reasoning_sft_v2.ipynb) | SFT for controlled biomedical evidence reasoning |
| [`08_ctd_distractor_scaling_robust_sft_fast.ipynb`](notebooks/08_ctd_distractor_scaling_robust_sft_fast.ipynb) | Distractor-aware robust SFT |
| [`09_ctd_abstention_robust_sft.ipynb`](notebooks/09_ctd_abstention_robust_sft.ipynb) | Abstention-aware training |
| [`11_ctd_multiseed_split_runner_v2.ipynb`](notebooks/11_ctd_multiseed_split_runner_v2.ipynb) | Multi-seed, entity-disjoint evaluation |
| [`16_ctd_dpo_boundary_fixed.ipynb`](notebooks/16_ctd_dpo_boundary_fixed.ipynb) | DPO with explicit tokenizer-boundary validation |
| [`18_ctd_rewarded_soup_sft_interpolation.ipynb`](notebooks/18_ctd_rewarded_soup_sft_interpolation.ipynb) | Weight interpolation across specialized SFT adapters |
| [`20_ctd_qwen38_27b_zero_shot_sanity_biowulf.ipynb`](notebooks/20_ctd_qwen38_27b_zero_shot_sanity_biowulf.ipynb) | Strong-model zero-shot sanity check on an A100-class HPC setup |

The numbering reflects the chronological development of the experiments rather than a polished tutorial sequence.

## A recurring testbed: controlled biomedical evidence reasoning

A substantial part of the repository uses a deliberately simple and verifiable task constructed from CTD-style relations:

```text
Chemical -> Gene -> Disease
```

Given a queried chemical and gene plus supplied gene-disease evidence, the model must either return the disease supported by the supplied path or abstain when no supported path exists.

The simplicity is useful: it makes it possible to isolate specific behaviors such as **evidence selection**, **distractor robustness**, **evidence sufficiency**, and **abstention** without relying on an opaque judge model.

Experiments progressively add:

1. vanilla SFT,
2. distractor-aware SFT,
3. no-path / abstention supervision,
4. counterfactual evaluation,
5. multi-seed and entity-disjoint splits,
6. DPO / IPO preference optimization,
7. mixture and confidence-based mitigation baselines,
8. adapter interpolation, and
9. strong-model zero-shot checks.

This is an exploratory research sandbox, not a claim that the benchmark itself represents the full complexity of biomedical or clinical reasoning.

## Stack

- PyTorch
- Transformers
- Datasets
- PEFT
- TRL
- Accelerate
- bitsandbytes
- pandas / NumPy

Install the common dependencies with:

```bash
pip install -r requirements.txt
```

Some later notebooks require newer model-specific `transformers` versions or additional lightweight packages; those notebooks document their environment assumptions locally.

## Credentials and model cache

Do not hard-code Hugging Face tokens in notebooks. A local `.env` file can be used when authentication is required:

```bash
cp .env.example .env
```

The repository ignores `.env`, Hugging Face caches, downloaded CTD data, and local experiment artifacts that should not be committed.

## Running the notebooks

Most early experiments are designed for Google Colab or a single-GPU environment. Larger-model notebooks may require substantially more VRAM and are intended for HPC / A100-class hardware.

Paths in environment-specific notebooks are examples; adjust `ROOT`, cache directories, and output directories for your system.

## Repository layout

```text
llm-tuning-playground/
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/      # training and evaluation experiments
└── results/        # selected compact experiment summaries
```

## Notes on reproducibility

These notebooks are research prototypes rather than a versioned software package. Model APIs and TRL/Transformers interfaces evolve quickly, so exact reproduction may require matching the package versions used by an individual notebook. Where an implementation issue was found (for example, a prompt-completion token-boundary problem in preference tuning), corrected experiments are kept explicitly rather than silently overwriting the experimental history.

## Disclaimer

This repository contains experimental research code for LLM tuning and evaluation. Biomedical tasks are used as controlled machine-learning testbeds and are **not** intended for clinical decision-making.
