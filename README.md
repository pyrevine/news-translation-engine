# news-translation-engine

## 개요

- 한국어 뉴스를 영어로 번역하는 도메인 특화 번역기
- 오픈소스 LLM(Qwen / Llama 계열) QLoRA 파인튜닝으로 구축 
- 대용량 텍스트 전처리·정제 파이프라인과 평가 기반 실험 설계

## Why this project

- **비용 효율**: 상용 API(GPT-4, DeepL) 대비 자체 호스팅 오픈모델의 손익분기점 실측
- **도메인 특화**: 범용 번역기 대비 뉴스 도메인에서의 BLEU/COMET 개선 폭을 측정
- **데이터 파이프라인**: 학습 코드보다 전처리·silver label 생성·정제 파이프라인이 더 많은 공수를 차지

## Architecture overview

```
raw text  →  preprocessing  →  silver label gen  →  filtering  →  QLoRA SFT  →  eval (BLEU/COMET)  →  vLLM serve
(crawl)     (kss / length)    (Qwen2.5-32B)       (BT + COMET-QE)                                    (optional)
```

- **Base model**: Qwen2.5-7B-Instruct (기본) / Llama-3.1-8B-Instruct (비교군)
- **Silver teacher**: Qwen2.5-32B-Instruct (Apache-2.0 계열 라이선스 — OpenAI/Anthropic 출력은 학습에 사용하지 않음)
- **Adapter**: QLoRA r=16, 4bit quantization

## Current status

- [x] M0: Project scaffolding
- [ ] M1: Evaluation set (FLORES-200 + custom news testset 100 pairs)
- [ ] M2: Stage 1 baseline — public parallel corpus QLoRA
- [ ] M3: Stage 2 — news-domain silver label pipeline
- [ ] M4 (optional): vLLM serving + cost analysis
- [ ] M5 (optional): ko→ja expansion

## Quick start
```bash
# 평가 전용 (로컬 맥에서도 가능, COMET 제외 시)
uv sync --extra eval

# 학습 (GPU 환경 권장 — Linux/CUDA 필요)
uv sync --extra train

# 데이터 크롤링·전처리
uv sync --extra data

# 서빙
uv sync --extra serve
```

환경변수는 `.env.example`을 복사해 `.env`로 설정합니다:

```bash
cp .env.example .env
# HF_TOKEN 등 값 채우기
```

## Datasets

| Stage | Dataset | Size | License | Used for |
|---|---|---|---|---|
| Stage 1 | [OPUS News-Commentary ko-en](https://opus.nlpl.eu/News-Commentary.php) | TBD | CC-BY-SA | 베이스라인 SFT |
| Stage 1 | [OPUS MultiParaCrawl ko-en](https://opus.nlpl.eu/MultiParaCrawl.php) | TBD | CC0 | 확장 SFT |
| Stage 2 | News silver labels (Qwen2.5-32B 생성) | TBD | 로컬 전용, 배포 불가 | 도메인 특화 SFT |
| Eval | [FLORES-200](https://huggingface.co/datasets/facebook/flores) | 1012 (devtest) | CC-BY-SA | 범용 번역 평가 |
| Eval | Custom news testset v1 | 100 | 로컬 전용 | 뉴스 도메인 평가 |

> 뉴스 원문 및 silver label은 저작권 이슈로 레포에 커밋하지 않음  
> 통계·메타데이터만 공개

## Results

| Model | Dataset | Direction | BLEU | COMET | Notes |
|---|---|---|---|---|---|
| _baseline_ | FLORES devtest | ko→en | TBD | TBD | 파인튜닝 없음 |
| _Stage 1_ | FLORES devtest | ko→en | TBD | TBD | OPUS QLoRA |
| _Stage 2_ | News testset v1 | ko→en | TBD | TBD | Silver label 추가 |

실험 단위 결과: [evaluation/results/](evaluation/results/)

## Repository structure

```
news-translation-engine/
├── data/
│   ├── scripts/          # 크롤링, 정제, silver label 생성 코드
│   ├── processed/        # gitignored — 로컬 전용
│   └── raw/              # gitignored — 로컬 전용
├── training/
│   ├── finetune_lora.py
│   └── configs/          # YAML (모델/실험별)
├── evaluation/
│   ├── datasets/         # 평가셋 로더
│   ├── eval_bleu_comet.py
│   └── results/          # 결과 JSON + summary md
├── serving/              # vLLM 서빙
├── docs/                 # 비용 분석, 설계 문서
├── plan_by_claude_bootstrap.md  # 구현 계획
├── pyproject.toml
└── README.md
```

## License

MIT. 코드는 MIT로 공개하되, 뉴스 원문 및 silver label은 저작권·API ToS 이슈로 **재배포하지 않습니다**. 공개 데이터셋(OPUS, FLORES)은 각 소스의 원 라이선스를 따릅니다.
