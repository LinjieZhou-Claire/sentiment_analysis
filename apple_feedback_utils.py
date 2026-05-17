from __future__ import annotations

from sentiment_utils import normalize_label


CHANNELS = ["微博", "小红书", "App Store", "知乎", "社区论坛", "客服工单"]
PRODUCTS = ["iPhone", "MacBook", "iPad", "Apple Watch", "AirPods", "Vision Pro"]
STATUSES = ["待处理", "高优先级", "已回复", "观察中", "已归档"]
RISK_STATUS = {"高优先级", "待处理"}

TOPIC_KEYWORDS = {
    "续航": ("电池", "续航", "掉电", "没电", "充电"),
    "发热": ("发热", "烫", "温度"),
    "系统性能": ("卡", "卡顿", "重启", "闪退", "慢", "系统"),
    "屏幕显示": ("屏幕", "显示", "刺眼", "看不清", "色彩", "亮度"),
    "连接": ("断连", "断一下", "连接", "蓝牙", "信号", "网络"),
    "服务": ("客服", "售后", "维修", "回复"),
    "价格": ("价格", "贵", "不值", "优惠"),
    "音质": ("音质", "降噪", "声音", "耳机"),
}


def infer_topic(text: str) -> str:
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return topic
    return "综合体验"


def risk_level(sentiment: str, confidence: float, status: str, engagement: int) -> str:
    label = normalize_label(sentiment)
    if label == "Negative" and (confidence >= 0.86 or status == "高优先级" or engagement >= 1800):
        return "红色预警"
    if label == "Negative" or status in RISK_STATUS or engagement >= 900:
        return "重点关注"
    return "常规"


def recommended_action(sentiment: str, risk: str) -> str:
    label = normalize_label(sentiment)
    if risk == "红色预警":
        return "转人工客服并同步产品团队"
    if label == "Negative":
        return "进入负面评论复核队列"
    if label == "Positive":
        return "沉淀为产品卖点语料"
    return "继续观察"


def split_messages(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]
