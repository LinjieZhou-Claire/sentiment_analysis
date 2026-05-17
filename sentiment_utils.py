from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    source: str = "model"


POSITIVE_KEYWORDS = (
    "好",
    "满意",
    "喜欢",
    "清晰",
    "流畅",
    "耐用",
    "很快",
    "惊喜",
    "优秀",
    "值得",
    "舒服",
    "漂亮",
    "推荐",
)

NEGATIVE_KEYWORDS = (
    "差",
    "垃圾",
    "失望",
    "卡",
    "慢",
    "坏",
    "退货",
    "看不清",
    "没电",
    "发热",
    "难用",
    "刺眼",
    "不值",
    "根本",
)


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    mapping = {
        "positive": "Positive",
        "pos": "Positive",
        "label_2": "Positive",
        "5 stars": "Positive",
        "4 stars": "Positive",
        "neutral": "Neutral",
        "neu": "Neutral",
        "label_1": "Neutral",
        "3 stars": "Neutral",
        "negative": "Negative",
        "neg": "Negative",
        "label_0": "Negative",
        "1 star": "Negative",
        "2 stars": "Negative",
    }
    return mapping.get(value, "Neutral")


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def fallback_analyze(text: str) -> SentimentResult:
    content = text.strip()
    if not content:
        return SentimentResult("Neutral", 0.5, "fallback")

    positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in content)
    negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in content)

    if positive_hits > negative_hits:
        score = 0.62 + min(positive_hits, 4) * 0.07
        return SentimentResult("Positive", clamp_score(score), "fallback")
    if negative_hits > positive_hits:
        score = 0.62 + min(negative_hits, 4) * 0.07
        return SentimentResult("Negative", clamp_score(score), "fallback")
    return SentimentResult("Neutral", 0.58, "fallback")


def model_analyze(text: str, classifier: Callable[[str], list[dict]]) -> SentimentResult:
    output = classifier(text)
    if isinstance(output, list) and output:
        first = output[0]
        return SentimentResult(
            label=normalize_label(first.get("label", "Neutral")),
            score=clamp_score(first.get("score", 0.0)),
            source="model",
        )
    return fallback_analyze(text)


def generate_sample_reviews() -> list[dict[str, object]]:
    reviews = [
        "物流很快，手机外观漂亮，屏幕显示也很清晰。",
        "续航太差了，玩游戏半小时就没电，真的失望。",
        "包装完整，今天刚收到，还没有开始深度使用。",
        "客服回复很耐心，耳机音质比预期好。",
        "在太阳底下根本看不清屏幕上的字。",
        "价格还可以，功能基本够用，没有特别惊喜。",
        "系统非常流畅，拍照清晰，老人用也很方便。",
        "充电器发热明显，用起来有点担心。",
        "物流比预计慢了一天，但商品没有损坏。",
        "这块屏幕画质太垃圾了，色彩很刺眼。",
        "手感舒服，做工扎实，整体值得推荐。",
        "用了两天就卡顿，还出现自动重启。",
    ]
    return [{"id": index, "review": review} for index, review in enumerate(reviews, start=1)]


def analyze_batch(
    reviews: Iterable[dict[str, object]],
    analyzer: Callable[[str], SentimentResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in reviews:
        review = str(item.get("review", ""))
        result = analyzer(review)
        rows.append(
            {
                "id": item.get("id", len(rows) + 1),
                "review": review,
                "sentiment": result.label,
                "confidence": round(result.score, 4),
                "source": result.source,
            }
        )
    return rows


def summarize_results(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    summary = {label: 0 for label in SENTIMENT_LABELS}
    for row in rows:
        label = normalize_label(str(row.get("sentiment", "Neutral")))
        summary[label] += 1
    return summary
