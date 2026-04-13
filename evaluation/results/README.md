# Evaluation Results

각 실험 단위로 결과 JSON을 `{model}_{dataset}_{direction}_{label}.json` 형식으로 저장합니다. 원시 예측(`*.preds.jsonl`)은 `.gitignore` 처리.

## Summary Table

| Model | Adapter | Dataset | Direction | BLEU | chrF | COMET | n | Date | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | — | FLORES devtest | ko→en | TBD | TBD | TBD | 1012 | TBD | baseline, zero-shot |
| Qwen/Qwen2.5-7B-Instruct | stage1_r16 | FLORES devtest | ko→en | TBD | TBD | TBD | 1012 | TBD | OPUS QLoRA |
| Qwen/Qwen2.5-7B-Instruct | stage2_r16 | FLORES devtest | ko→en | TBD | TBD | TBD | 1012 | TBD | + silver labels |
| Qwen/Qwen2.5-7B-Instruct | stage2_r16 | News testset v1 | ko→en | TBD | TBD | TBD | 100 | TBD | in-domain |

## How to add a row

1. 실행:
   ```bash
   uv run python -m evaluation.eval_bleu_comet \
     --model <model_id> [--adapter <path>] \
     --dataset flores --split devtest --direction ko2en \
     --output evaluation/results/<model>_flores_ko2en_<label>.json
   ```
2. 결과 JSON의 `metrics` 값을 위 테이블에 추가하고 커밋.

## Conventions

- **BLEU tokenizer**: en reference → `13a` (sacrebleu default), ko → `char`, ja → `ja-mecab`.
- **COMET**: `Unbabel/wmt22-comet-da` (기본). 다른 checkpoint 쓰면 Notes에 명시.
- **Decoding**: greedy (`do_sample=False, num_beams=1`) — 재현성 우선. beam search 실험은 Notes에 명시.
