from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

from sentiment_utils import (
    MODEL_NAME,
    SENTIMENT_LABELS,
    SentimentResult,
    analyze_batch,
    fallback_analyze,
    generate_sample_reviews,
    model_analyze,
    summarize_results,
)


st.set_page_config(
    page_title="电商评论情感分析与意见挖掘平台",
    page_icon="📘",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 18% 12%, rgba(20, 184, 166, 0.16), transparent 30%),
            radial-gradient(circle at 82% 2%, rgba(59, 130, 246, 0.18), transparent 28%),
            #0b1020;
        color: #e5eefb;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.72);
    }
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div.element-container) {
        gap: 0.65rem;
    }
    .hero {
        border: 1px solid rgba(125, 211, 252, 0.28);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.72));
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 30px;
        letter-spacing: 0;
        color: #f8fafc;
    }
    .hero p {
        margin: 0;
        color: #b9c6d8;
        line-height: 1.65;
    }
    .panel {
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(15, 23, 42, 0.76);
        border-radius: 8px;
        padding: 18px;
    }
    .sentiment-positive { color: #6ee7b7; font-weight: 700; }
    .sentiment-neutral { color: #c4b5fd; font-weight: 700; }
    .sentiment-negative { color: #fda4af; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载 Hugging Face 中文情感分析模型...")
def load_classifier():
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def get_analyzer():
    try:
        classifier = load_classifier()
        return lambda text: model_analyze(text, classifier), None
    except Exception as exc:  # pragma: no cover - depends on local model/network state.
        return fallback_analyze, str(exc)


def sentiment_html(result: SentimentResult) -> str:
    css = {
        "Positive": "sentiment-positive",
        "Neutral": "sentiment-neutral",
        "Negative": "sentiment-negative",
    }[result.label]
    return f'<span class="{css}">{result.label}</span>'


def make_gauge(result: SentimentResult) -> go.Figure:
    color = {"Positive": "#34d399", "Neutral": "#a78bfa", "Negative": "#fb7185"}[result.label]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(result.score * 100, 2),
            number={"suffix": "%", "font": {"color": "#f8fafc", "size": 34}},
            title={"text": "Confidence Score", "font": {"color": "#cbd5e1", "size": 16}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.26},
                "bgcolor": "rgba(15,23,42,0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.35)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(248,113,113,0.22)"},
                    {"range": [45, 70], "color": "rgba(167,139,250,0.22)"},
                    {"range": [70, 100], "color": "rgba(52,211,153,0.22)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5eefb"},
    )
    return fig


def make_pie(summary: dict[str, int]) -> go.Figure:
    colors = ["#34d399", "#a78bfa", "#fb7185"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(summary.keys()),
                values=list(summary.values()),
                hole=0.46,
                marker={"colors": colors, "line": {"color": "#0b1020", "width": 2}},
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5eefb"},
        legend={"orientation": "h", "y": -0.05},
    )
    return fig


def render_result(result: SentimentResult):
    st.markdown(
        f"""
        <div class="panel">
            <div>情感极性：{sentiment_html(result)}</div>
            <div>置信度：<strong>{result.score:.2%}</strong></div>
            <div>分析来源：<strong>{'Hugging Face 模型' if result.source == 'model' else '本地关键词兜底'}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


analyzer, load_error = get_analyzer()

st.markdown(
    """
    <div class="hero">
        <h1>电商评论情感分析与意见挖掘平台</h1>
        <p>面向中文商品评论的细粒度情感分析实验：从单句极性判断，到显式/隐式表达对比，再到批量舆情统计与商业洞察。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.warning(
        "Hugging Face 模型暂时不可用，当前使用本地关键词规则进行演示。"
        "联网并安装依赖后会优先使用模型。"
    )

tab_single, tab_expression, tab_dashboard = st.tabs(
    ["模块 1：单文本情感分类", "模块 2：显式 vs 隐式情感", "模块 3：舆情挖掘看板"]
)

with tab_single:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("单条中文评论分析")
        review = st.text_area(
            "输入中文商品评论",
            value="这款耳机音质很好，佩戴舒服，物流也很快。",
            height=150,
        )
        if st.button("分析情感极性", type="primary", width="stretch"):
            st.session_state["single_result"] = analyzer(review)

        result = st.session_state.get("single_result")
        if result:
            render_result(result)
        else:
            st.info("输入一段明显好评或差评，观察置信度仪表盘如何变化。")

    with right:
        result = st.session_state.get("single_result", analyzer("这款耳机音质很好，佩戴舒服，物流也很快。"))
        st.plotly_chart(make_gauge(result), width="stretch")

with tab_expression:
    st.subheader("显式情感与隐式情感对比")
    st.markdown(
        """
        显式情感通常包含直接褒贬词，例如“太棒了”“太垃圾了”。
        隐式情感表面上像客观描述，但事实本身暗含态度，例如“手机玩游戏半小时就没电了”。
        """
    )
    col_explicit, col_implicit = st.columns(2, gap="large")
    with col_explicit:
        explicit_text = st.text_area(
            "显式情感评价",
            value="这屏幕画质太垃圾了，看着很刺眼。",
            height=130,
        )
        explicit_result = analyzer(explicit_text)
        render_result(explicit_result)
        st.plotly_chart(make_gauge(explicit_result), width="stretch")

    with col_implicit:
        implicit_text = st.text_area(
            "隐式客观描述",
            value="在太阳底下根本看不清屏幕上的字。",
            height=130,
        )
        implicit_result = analyzer(implicit_text)
        render_result(implicit_result)
        st.plotly_chart(make_gauge(implicit_result), width="stretch")

with tab_dashboard:
    st.subheader("Opinion Mining Dashboard")
    if "sample_reviews" not in st.session_state:
        st.session_state["sample_reviews"] = generate_sample_reviews()

    col_action, col_hint = st.columns([0.32, 0.68])
    with col_action:
        if st.button("生成测试舆情数据", type="primary", width="stretch"):
            st.session_state["sample_reviews"] = generate_sample_reviews()
            st.session_state["batch_rows"] = analyze_batch(st.session_state["sample_reviews"], analyzer)
    with col_hint:
        st.caption("点击按钮后，系统会生成 10-15 条模拟商品评价并批量分析情感极性。")

    if "batch_rows" not in st.session_state:
        st.session_state["batch_rows"] = analyze_batch(st.session_state["sample_reviews"], analyzer)

    rows = st.session_state["batch_rows"]
    summary = summarize_results(rows)
    total = sum(summary.values())
    positive_rate = summary["Positive"] / total if total else 0
    negative_rate = summary["Negative"] / total if total else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("评论总量", total)
    metric_cols[1].metric("Positive", summary["Positive"], f"{positive_rate:.0%}")
    metric_cols[2].metric("Neutral", summary["Neutral"])
    metric_cols[3].metric("Negative", summary["Negative"], f"{negative_rate:.0%}")

    chart_col, table_col = st.columns([0.42, 0.58], gap="large")
    with chart_col:
        st.plotly_chart(make_pie(summary), width="stretch")
    with table_col:
        df = pd.DataFrame(rows)
        df["confidence"] = (df["confidence"] * 100).round(2).astype(str) + "%"
        st.dataframe(
            df.rename(
                columns={
                    "id": "编号",
                    "review": "评论文本",
                    "sentiment": "情感极性",
                    "confidence": "置信度",
                    "source": "来源",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        """
        <div class="panel">
            商业解读：当 Negative 占比升高时，可优先排查续航、屏幕、发热、物流等高频问题；
            当 Positive 占比稳定时，可将好评关键词用于卖点提炼与商品详情页优化。
        </div>
        """,
        unsafe_allow_html=True,
    )
