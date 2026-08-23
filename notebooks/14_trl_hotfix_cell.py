# Run this cell after the imports/model setup cell in Experiment 14.
# It replaces train_model() with a TRL-version-compatible implementation.

import inspect


def _supported_kwargs(callable_obj, candidates):
    params = inspect.signature(callable_obj).parameters
    return {k: v for k, v in candidates.items() if k in params}


def train_model(ds, outdir, seed):
    set_seed(seed)
    m = load_model(True)

    cfg_candidates = {
        "output_dir": outdir,
        "dataset_text_field": "text",
        "max_length": 768,
        "max_seq_length": 768,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": MAX_STEPS,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "warmup_steps": max(1, int(MAX_STEPS * 0.05)),
        "logging_steps": 20,
        "save_strategy": "no",
        "report_to": "none",
        "packing": False,
        "gradient_checkpointing": True,
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
    }

    # Keep only arguments accepted by the installed TRL version.
    cfg_kwargs = _supported_kwargs(SFTConfig, cfg_candidates)

    # Prefer warmup_ratio when available; otherwise use warmup_steps.
    if "warmup_ratio" in cfg_kwargs:
        cfg_kwargs.pop("warmup_steps", None)
    elif "warmup_steps" in cfg_kwargs:
        cfg_kwargs.pop("warmup_ratio", None)

    # Prefer max_length when available; otherwise max_seq_length.
    if "max_length" in cfg_kwargs:
        cfg_kwargs.pop("max_seq_length", None)
    elif "max_seq_length" in cfg_kwargs:
        cfg_kwargs.pop("max_length", None)

    print("SFTConfig kwargs:", sorted(cfg_kwargs))
    args = SFTConfig(**cfg_kwargs)

    trainer_candidates = {
        "model": m,
        "args": args,
        "train_dataset": ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": lora,
    }
    trainer_kwargs = _supported_kwargs(SFTTrainer, trainer_candidates)

    # Current TRL uses processing_class; older TRL used tokenizer.
    if "processing_class" in trainer_kwargs:
        trainer_kwargs.pop("tokenizer", None)
    elif "tokenizer" in trainer_kwargs:
        trainer_kwargs.pop("processing_class", None)

    print("SFTTrainer kwargs:", sorted(trainer_kwargs))
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.model.config.use_cache = True
    return trainer.model
