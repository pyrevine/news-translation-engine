"""Stage 1 QLoRA fine-tuning runner.

Reads a YAML config, loads a JSONL ko/en dataset, and fine-tunes a HF causal
LM via QLoRA (4bit base + LoRA adapter). Saves the adapter only.

The prompt template is shared with evaluation/prompts.py::build_messages so
that training and eval see the same format (fair baseline/adapter comparison).

Usage
    uv run python -m training.finetune_lora \\
        --config training/configs/stage1_qwen2.5_7b.yaml

Requires Linux + NVIDIA GPU (bitsandbytes is CUDA-only). Designed for Runpod.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow `python training/finetune_lora.py ...` invocation as well as `-m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def build_formatted_dataset(rows: list[dict[str, Any]], direction: str):
    """Turn {ko, en, ...} rows into conversational {messages: [...]} rows.

    Modern trl (>=0.14-ish) with `assistant_only_loss=True` expects a
    conversational dataset — i.e. each row has a `messages` list of
    role/content dicts. SFTTrainer handles chat-template application and
    label masking internally using the tokenizer passed via processing_class.
    """
    from datasets import Dataset

    from evaluation.prompts import build_messages

    def to_messages(row: dict[str, Any]) -> dict[str, Any]:
        if direction == "ko2en":
            src, tgt = row["ko"], row["en"]
        elif direction == "en2ko":
            src, tgt = row["en"], row["ko"]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        messages = build_messages(src, direction) + [
            {"role": "assistant", "content": tgt}
        ]
        return {"messages": messages}

    ds = Dataset.from_list(rows)
    return ds.map(to_messages, remove_columns=ds.column_names)


def build_model_and_tokenizer(cfg: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_name = cfg["model"]["base"]
    quant_name = cfg["model"].get("quantization", "4bit")

    if quant_name == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_name == "none":
        quant_cfg = None
    else:
        raise ValueError(f"Unsupported quantization: {quant_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    attn_impl = cfg["model"].get("attn_implementation", "sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    use_gc = cfg["train"].get("gradient_checkpointing", True)
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gc
        )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def build_sft_config(cfg: dict[str, Any]):
    """Build an SFTConfig tolerant of trl's rapid API churn.

    We assemble the union of kwargs we'd like to pass, resolve a couple of
    known renames against the installed SFTConfig signature (e.g. the recent
    `max_seq_length` → `max_length` rename), then drop any remaining kwargs
    the installed version doesn't accept. That way new renames degrade to
    a warning instead of a crash.
    """
    import inspect

    from trl import SFTConfig

    tr = cfg["train"]
    desired: dict[str, Any] = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["grad_accum"],
        learning_rate=float(tr["lr"]),
        num_train_epochs=tr["epochs"],
        warmup_ratio=tr.get("warmup_ratio", 0.03),
        lr_scheduler_type=tr.get("lr_scheduler", "cosine"),
        logging_steps=tr.get("logging_steps", 25),
        eval_strategy="steps",
        eval_steps=tr.get("eval_steps", 500),
        save_steps=tr.get("save_steps", 500),
        save_total_limit=tr.get("save_total_limit", 3),
        bf16=tr.get("bf16", True),
        optim=tr.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=tr.get("gradient_checkpointing", True),
        packing=False,
        report_to=cfg.get("report_to", "none") or "none",
        seed=cfg.get("seed", 42),
        assistant_only_loss=True,
    )

    params = inspect.signature(SFTConfig.__init__).parameters
    # trl renamed max_seq_length → max_length somewhere around 0.14
    max_len = tr["max_seq_len"]
    if "max_length" in params:
        desired["max_length"] = max_len
    elif "max_seq_length" in params:
        desired["max_seq_length"] = max_len

    accepted: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in desired.items():
        if k in params:
            accepted[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(
            f"WARN: dropping SFTConfig kwargs not in installed trl: {dropped}",
            file=sys.stderr,
        )
    return SFTConfig(**accepted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}", file=sys.stderr)

    if cfg.get("report_to") == "wandb" and not os.environ.get("WANDB_API_KEY"):
        print("WARN: report_to=wandb but WANDB_API_KEY not set", file=sys.stderr)

    model, tokenizer = build_model_and_tokenizer(cfg)

    data_cfg = cfg["data"]
    direction = data_cfg["direction"]
    train_rows = load_jsonl(data_cfg["train"], limit=data_cfg.get("max_train_samples"))
    val_rows = load_jsonl(data_cfg["val"], limit=data_cfg.get("max_val_samples"))
    print(f"Loaded train={len(train_rows):,} val={len(val_rows):,}", file=sys.stderr)

    train_ds = build_formatted_dataset(train_rows, direction)
    val_ds = build_formatted_dataset(val_rows, direction)

    import inspect as _inspect

    from trl import SFTTrainer

    sft_cfg = build_sft_config(cfg)
    trainer_params = _inspect.signature(SFTTrainer.__init__).parameters
    # HF/trl renamed `tokenizer` → `processing_class` around transformers 4.46
    tokenizer_kwarg = (
        "processing_class" if "processing_class" in trainer_params else "tokenizer"
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **{tokenizer_kwarg: tokenizer},
    )

    trainer.train()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nSaved LoRA adapter + tokenizer to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
