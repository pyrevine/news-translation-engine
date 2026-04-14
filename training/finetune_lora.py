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


def build_formatted_dataset(
    rows: list[dict[str, Any]], tokenizer, direction: str
):
    """Turn {ko, en, ...} rows into {text} rows with full chat-formatted string."""
    from datasets import Dataset

    from evaluation.prompts import build_messages

    def to_text(row: dict[str, Any]) -> dict[str, str]:
        if direction == "ko2en":
            src, tgt = row["ko"], row["en"]
        elif direction == "en2ko":
            src, tgt = row["en"], row["ko"]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        messages = build_messages(src, direction) + [
            {"role": "assistant", "content": tgt}
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(rows)
    return ds.map(to_text, remove_columns=ds.column_names)


def resolve_response_template(tokenizer) -> str:
    """Auto-detect the literal string that begins the assistant turn.

    We use this with DataCollatorForCompletionOnlyLM so loss is computed
    only on the target translation, not on the prompt.
    """
    probe = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ],
        tokenize=False,
    )
    candidates = [
        "<|im_start|>assistant\n",                                     # Qwen / ChatML
        "<|start_header_id|>assistant<|end_header_id|>\n\n",           # Llama 3
        "<start_of_turn>model\n",                                      # Gemma
    ]
    for cand in candidates:
        if cand in probe:
            return cand
    raise RuntimeError(
        "Could not auto-detect an assistant-turn marker in this tokenizer's "
        "chat template. Override resolve_response_template() for your model."
    )


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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
    from trl import SFTConfig

    tr = cfg["train"]
    return SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["grad_accum"],
        learning_rate=float(tr["lr"]),
        num_train_epochs=tr["epochs"],
        max_seq_length=tr["max_seq_len"],
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
        dataset_text_field="text",
        packing=False,
        report_to=cfg.get("report_to", "none") or "none",
        seed=cfg.get("seed", 42),
    )


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

    train_ds = build_formatted_dataset(train_rows, tokenizer, direction)
    val_ds = build_formatted_dataset(val_rows, tokenizer, direction)

    response_template = resolve_response_template(tokenizer)
    print(f"Response template: {response_template!r}", file=sys.stderr)

    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        from trl.trainer import DataCollatorForCompletionOnlyLM

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    from trl import SFTTrainer

    sft_cfg = build_sft_config(cfg)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
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
