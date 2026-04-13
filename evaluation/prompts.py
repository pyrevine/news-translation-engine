"""Translation prompt templates.

Kept intentionally simple — the goal is to measure model/fine-tuning effects,
not prompt engineering. Use the same template across baseline and fine-tuned
evals for fair comparison.
"""

from __future__ import annotations

TRANSLATE_KO_EN_SYSTEM = (
    "You are a professional translator. Translate the given Korean news sentence "
    "into natural, fluent English. Preserve proper nouns, numbers, and dates exactly. "
    "Output only the translation, with no preamble or explanation."
)

TRANSLATE_EN_KO_SYSTEM = (
    "당신은 전문 번역가입니다. 주어진 영어 뉴스 문장을 자연스러운 한국어로 번역하세요. "
    "고유명사, 숫자, 날짜는 정확히 보존하세요. 번역문만 출력하고 부연 설명은 하지 마세요."
)


def build_messages(source: str, direction: str) -> list[dict[str, str]]:
    """Build chat-format messages for a translation request.

    Args:
        source: source sentence.
        direction: "ko2en" or "en2ko".

    Returns:
        Chat messages list suitable for tokenizer.apply_chat_template().
    """
    if direction == "ko2en":
        system = TRANSLATE_KO_EN_SYSTEM
    elif direction == "en2ko":
        system = TRANSLATE_EN_KO_SYSTEM
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": source},
    ]
