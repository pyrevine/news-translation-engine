"""End-to-end translation evaluation: generate predictions, compute BLEU/COMET.

Run example:
    uv run python -m evaluation.eval_bleu_comet \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset flores --split devtest --direction ko2en \\
        --output evaluation/results/qwen2.5-7b_flores_ko2en_baseline.json

For fine-tuned adapters:
    uv run python -m evaluation.eval_bleu_comet \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --adapter outputs/stage1_qwen2.5_7b_r16 \\
        ...

GPU required for model inference and COMET. Use --skip-comet to run BLEU only
(e.g., for a CPU-only sanity pass).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evaluation.datasets.flores import Pair, load_flores_ko_en
from evaluation.prompts import build_messages


def load_dataset_pairs(name: str, split: str) -> list[Pair]:
    if name == "flores":
        return load_flores_ko_en(split)
    raise ValueError(f"Unknown dataset: {name}")


def translate_batch(
    model_name: str,
    adapter_path: str | None,
    sources: list[str],
    direction: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """Generate translations with a HF transformers model.

    Lazy import of torch/transformers so that CPU-only environments can still
    run BLEU on pre-computed predictions via --predictions-file.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    preds: list[str] = []
    for i in range(0, len(sources), batch_size):
        chunk = sources[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(s, direction), tokenize=False, add_generation_prompt=True
            )
            for s in chunk
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        preds.extend(s.strip() for s in decoded)
        print(f"  [{i + len(chunk)}/{len(sources)}]", file=sys.stderr)
    return preds


def compute_bleu(preds: list[str], refs: list[str], target_lang: str) -> dict:
    import sacrebleu

    tokenize = {"en": "13a", "ko": "char", "ja": "ja-mecab"}.get(target_lang, "13a")
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize=tokenize)
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return {
        "bleu": round(bleu.score, 2),
        "bleu_tokenize": tokenize,
        "chrf": round(chrf.score, 2),
    }


def compute_comet(
    sources: list[str], preds: list[str], refs: list[str], model_name: str
) -> float:
    from comet import download_model, load_from_checkpoint

    ckpt_path = download_model(model_name)
    model = load_from_checkpoint(ckpt_path)
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, preds, refs, strict=True)]
    output = model.predict(data, batch_size=8, gpus=1)
    return round(float(output.system_score), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--adapter", default=None, help="PEFT adapter path (optional)")
    ap.add_argument("--dataset", default="flores")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--direction", default="ko2en", choices=["ko2en", "en2ko"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="limit N pairs (debug)")
    ap.add_argument("--skip-comet", action="store_true")
    ap.add_argument(
        "--comet-model", default="Unbabel/wmt22-comet-da", help="COMET checkpoint name"
    )
    ap.add_argument("--output", required=True, help="path to results JSON")
    args = ap.parse_args()

    pairs = load_dataset_pairs(args.dataset, args.split)
    if args.limit:
        pairs = pairs[: args.limit]

    if args.direction == "ko2en":
        sources = [p.ko for p in pairs]
        refs = [p.en for p in pairs]
        target_lang = "en"
    else:
        sources = [p.en for p in pairs]
        refs = [p.ko for p in pairs]
        target_lang = "ko"

    print(f"Loaded {len(pairs)} pairs from {args.dataset}/{args.split}", file=sys.stderr)
    print(f"Translating with {args.model}" + (f" + {args.adapter}" if args.adapter else ""),
          file=sys.stderr)

    t0 = time.time()
    preds = translate_batch(
        model_name=args.model,
        adapter_path=args.adapter,
        sources=sources,
        direction=args.direction,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    translate_sec = time.time() - t0

    metrics = compute_bleu(preds, refs, target_lang)
    if not args.skip_comet:
        metrics["comet"] = compute_comet(sources, preds, refs, args.comet_model)
        metrics["comet_model"] = args.comet_model

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": args.split,
        "direction": args.direction,
        "n_pairs": len(pairs),
        "translate_seconds": round(translate_sec, 1),
        "metrics": metrics,
        "samples": [
            {"src": s, "ref": r, "pred": p}
            for s, r, p in list(zip(sources, refs, preds, strict=True))[:20]
        ],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # Raw predictions — separate file, gitignored.
    preds_path = output_path.with_suffix(".preds.jsonl")
    with preds_path.open("w") as f:
        for s, r, p in zip(sources, refs, preds, strict=True):
            f.write(json.dumps({"src": s, "ref": r, "pred": p}, ensure_ascii=False) + "\n")

    print(f"\nDone. metrics={metrics}", file=sys.stderr)
    print(f"Wrote {output_path} and {preds_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
