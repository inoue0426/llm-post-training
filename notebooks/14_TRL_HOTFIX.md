# Experiment 14 TRL compatibility hotfix

The canonical Experiment 14 notebook is being updated to avoid TRL API-version failures. The training cell now inspects `SFTConfig` / `SFTTrainer` signatures at runtime, uses `warmup_steps` when supported, and only passes arguments supported by the installed TRL version.
