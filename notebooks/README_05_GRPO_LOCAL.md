# GRPO local-data workflow

Use `05_grpo_drug_target_retrieval_local.ipynb` for the current workflow.

1. Keep `drug_bank.csv` outside the public repository.
2. Upload it directly to the Colab runtime.
3. The notebook builds the drug-to-target dictionary locally.
4. GRPO uses exact target matching as the verifiable reward.
5. Before/after accuracy is measured on unseen drugs.