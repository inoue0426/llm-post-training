# LLM Post-Training

Hands-on implementations of **LLM fine-tuning, preference optimization, alignment, and reliability evaluation** using Hugging Face, PEFT, and TRL.

This repository is a notebook-first collection of post-training experiments. I use it to prototype training objectives, inspect failure modes, and run controlled evaluations across supervised fine-tuning, preference optimization, reinforcement-learning-style tuning, and large-model inference.

## What is covered

- **Parameter-efficient fine-tuning:** LoRA / QLoRA with PEFT
- **Supervised fine-tuning (SFT):** instruction tuning and task-specific adaptation
- **Preference optimization:** DPO and IPO, including prompt/completion boundary validation
- **Reinforcement-learning-style tuning:** GRPO experiments
- **Reliability evaluation:** distractors, counterfactuals, abstention, no-evidence cases, and prompt-sensitivity controls
- **Experimental design:** multi-seed runs, entity-disjoint splits, controlled perturbations, explicit baselines, and follow-up tests for alternative explanations
- **Large-model evaluation:** BF16 inference and zero-shot sanity checks on GPU/HPC environments
- **Biomedical applications:** drug-target retrieval and evidence-grounded Chemical -> Gene -> Disease reasoning using CTD-derived relations

## Representative notebooks

| Notebook | Focus |
| --- | --- |
| [`01_lora.ipynb`](notebooks/01_lora.ipynb) | Minimal LoRA fine-tuning and before/after generation comparison |
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

## Controlled biomedical evidence reasoning

A recurring testbed in this repository uses a deliberately simple and verifiable relation chain:

```text
Chemical -> Gene -> Disease
```

Given a queried chemical and gene plus supplied gene-disease evidence, the model must either return the disease supported by the supplied path or abstain when no supported path exists.

The task is intentionally simple so that specific behaviors can be isolated without relying on an opaque judge model. The experiments examine:

- evidence selection under distractors,
- evidence sufficiency and abstention,
- counterfactual robustness,
- entity-disjoint generalization,
- supervision-induced trade-offs,
- behavior under stronger zero-shot models, and
- whether apparent failures survive prompt and position controls.

The progression includes vanilla SFT, distractor-aware SFT, no-path supervision, DPO / IPO, mixture and confidence baselines, adapter interpolation, and strong-model sanity checks.

### Example of a negative result

One exploratory Qwen3.8-27B experiment initially showed a large false-abstention effect when irrelevant evidence was added under the original task prompt. With the correct edge fixed in the first position, accuracy changed from **0.94 at zero distractors** to **0.44 at one distractor** and **0.32 at five distractors**.

Follow-up controls showed that this was not a robust evidence-selection failure. Rewriting the instruction as an explicit exact-match lookup eliminated the effect completely: accuracy was **1.00 at 0, 1, 5, and 20 distractors**, with zero false abstentions. An algorithmic step-by-step formulation showed the same behavior in the tested conditions.

This is kept as a useful diagnostic result: a striking reliability failure can disappear once the task specification is made explicit. The broader lesson is to test prompt sensitivity and alternative explanations before treating an observed failure mode as a model capability limit.

This is an exploratory machine-learning testbed, not a claim that the benchmark captures the full complexity of biomedical or clinical reasoning.

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

Some later notebooks require newer model-specific `transformers` versions or additional lightweight packages; environment assumptions are documented in the relevant notebooks.

## Credentials and model cache

Do not hard-code Hugging Face tokens in notebooks. When authentication is needed, create a local `.env` file from the example:

```bash
cp .env.example .env
```

The repository ignores `.env`, Hugging Face caches, downloaded CTD data, model checkpoints, and other local experiment artifacts.

## Running the notebooks

Early experiments are designed for Google Colab or a single-GPU environment. Larger-model notebooks may require substantially more VRAM and are intended for HPC / A100-class hardware.

Environment-specific paths are examples; adjust `ROOT`, cache directories, and output directories for your system.

## Repository layout

```text
llm-post-training/
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/      # training and evaluation experiments
└── results/        # selected compact experiment summaries
```

## Notes on reproducibility

These notebooks are research prototypes rather than a versioned software package. Model APIs and TRL/Transformers interfaces evolve quickly, so exact reproduction may require matching the package versions used by an individual notebook.

Where an implementation issue was found—for example, a prompt-completion token-boundary problem in preference tuning—corrected experiments are kept explicitly rather than silently overwriting the experimental history. The same principle applies to behavioral findings: follow-up controls that weaken an initial interpretation are documented rather than omitted.

## Disclaimer

This repository contains experimental research code for LLM post-training and evaluation. Biomedical tasks are used as controlled machine-learning testbeds and are **not** intended for clinical decision-making.
