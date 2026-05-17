from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

from apple_feedback_utils import (
    CHANNELS,
    PRODUCTS,
    STATUSES,
    infer_topic,
    recommended_action,
    risk_level,
    split_messages,
)
from sentiment_utils import (
    MODEL_NAME,
    SENTIMENT_LABELS,
    SentimentResult,
    fallback_analyze,
    model_analyze,
    normalize_label,
)


st.set_page_config(
    page_title="Apple 社区与运营监控",
    page_icon="🍎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f7fb 0%, #edf1f7 44%, #f8fafc 100%);
        color: #111827;
    }
    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.88);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
    }
    .hero, .panel, .post {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 8px;
    }
    .hero {
        padding: 22px 24px;
        margin-bottom: 16px;
    }
    .hero h1 {
        margin: 0 0 7px 0;
        font-size: 30px;
        letter-spacing: 0;
        color: #0f172a;
    }
    .hero p {
        margin: 0;
        color: #475569;
        line-height: 1.58;
    }
    .panel {
        padding: 16px;
    }
    .post {
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .post-meta {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .post-title {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .post-body {
        color: #334155;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    .tag {
        display: inline-block;
        border: 1px solid rgba(37, 99, 235, 0.14);
        background: rgba(37, 99, 235, 0.08);
        color: #1d4ed8;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 12px;
        margin-right: 6px;
    }
    .status-ok { color: #047857; font-weight: 700; }
    .status-watch { color: #b45309; font-weight: 700; }
    .status-risk { color: #be123c; font-weight: 700; }
    .source-note {
        color: #64748b;
        font-size: 13px;
        line-height: 1.55;
    }
    section[data-testid="stSidebar"] button {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载中文情感分析模型...")
def load_classifier():
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def get_analyzer():
    try:
        classifier = load_classifier()
        return lambda text: model_analyze(text, classifier), None
    except Exception as exc:  # pragma: no cover - depends on local model/network state.
        return fallback_analyze, str(exc)


def seed_feedback() -> list[dict[str, object]]:
    base_time = datetime(2026, 5, 18, 9, 0)
    messages = [
        ("微博", "iPhone", "待处理", "iPhone 更新后掉电太快，早上满电中午就没电。", "数码观察员", 2480),
        ("小红书", "MacBook", "已回复", "MacBook 屏幕显示很细腻，剪视频调色很舒服。", "设计师Luna", 860),
        ("App Store", "iPad", "高优先级", "iPad 一边充电一边会议发热明显，真的有点担心。", "远程办公用户", 1760),
        ("社区论坛", "AirPods", "观察中", "AirPods 在地铁里偶尔断连，其他时候还算稳定。", "通勤党", 520),
        ("知乎", "Vision Pro", "待处理", "Vision Pro 佩戴半小时后压脸，价格也不便宜。", "硬件爱好者", 1320),
        ("客服工单", "Apple Watch", "已回复", "Apple Watch 记录运动很准，续航如果再长一点就完美。", "跑步用户", 390),
        ("微博", "iPhone", "高优先级", "相机打开会卡顿，错过好几次拍照瞬间，太失望了。", "摄影新手", 2110),
        ("小红书", "AirPods", "已归档", "降噪效果比预期好，办公室里终于能安静工作。", "产品经理C", 740),
        ("App Store", "MacBook", "观察中", "系统更新后偶尔重启，暂时没有找到规律。", "开发者Kai", 980),
        ("社区论坛", "iPad", "待处理", "阳光下看不清屏幕，做户外笔记有点难受。", "建筑学生", 1180),
    ]
    rows: list[dict[str, object]] = []
    for index, (channel, product, status, message, author, engagement) in enumerate(messages, start=1):
        rows.append(
            {
                "id": index,
                "channel": channel,
                "product": product,
                "status": status,
                "message": message,
                "author": author,
                "engagement": engagement,
                "time": (base_time + timedelta(minutes=index * 18)).strftime("%Y-%m-%d %H:%M"),
                "collected": status != "已归档",
            }
        )
    return rows


def enrich_feedback(row: dict[str, object], analyzer) -> dict[str, object]:
    message = str(row.get("message", ""))
    result: SentimentResult = analyzer(message)
    sentiment = normalize_label(result.label)
    confidence = round(result.score, 4)
    topic = str(row.get("topic") or infer_topic(message))
    risk = risk_level(
        sentiment,
        confidence,
        str(row.get("status", "待处理")),
        int(row.get("engagement", 0)),
    )
    return {
        **row,
        "topic": topic,
        "sentiment": sentiment,
        "confidence": confidence,
        "source": result.source,
        "risk": risk,
        "action": recommended_action(sentiment, risk),
    }


def initialize_rows(analyzer) -> None:
    if "feedback_rows" not in st.session_state:
        st.session_state["feedback_rows"] = [enrich_feedback(row, analyzer) for row in seed_feedback()]


def rows_to_corpus_csv(rows: list[dict[str, object]]) -> str:
    output = StringIO()
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    columns = ["id", "time", "channel", "product", "topic", "sentiment", "confidence", "message", "source"]
    df.loc[:, columns].to_csv(output, index=False)
    return output.getvalue()


def make_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 15, "color": "#64748b"})
    fig.update_layout(
        height=310,
        margin={"l": 8, "r": 8, "t": 12, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def make_sentiment_donut(summary: dict[str, int]) -> go.Figure:
    if not any(summary.values()):
        return make_empty_figure("当前筛选无评论")
    colors = {"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#e11d48"}
    fig = go.Figure(
        go.Pie(
            labels=list(summary.keys()),
            values=list(summary.values()),
            hole=0.58,
            marker={"colors": [colors[label] for label in summary], "line": {"color": "#f8fafc", "width": 2}},
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        height=310,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.08},
    )
    return fig


def make_topic_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return make_empty_figure("当前筛选无主题数据")
    topic_counts = df["topic"].value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=topic_counts.values, y=topic_counts.index, orientation="h", marker_color="#2563eb"))
    fig.update_layout(
        height=310,
        margin={"l": 8, "r": 8, "t": 12, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="评论量",
        yaxis_title="",
    )
    return fig


def make_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return make_empty_figure("当前筛选无趋势数据")
    data = df.copy()
    data["hour"] = pd.to_datetime(data["time"]).dt.strftime("%H:%M")
    pivot = (
        data.pivot_table(index="hour", columns="sentiment", values="id", aggfunc="count")
        .reindex(columns=SENTIMENT_LABELS, fill_value=0)
        .fillna(0)
    )
    colors = {"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#e11d48"}
    fig = go.Figure()
    for label in SENTIMENT_LABELS:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[label],
                mode="lines+markers",
                name=label,
                line={"color": colors[label], "width": 3},
            )
        )
    fig.update_layout(
        height=310,
        margin={"l": 8, "r": 8, "t": 12, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="评论量",
        xaxis_title="时间",
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def make_gauge(result: SentimentResult) -> go.Figure:
    color = {"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#e11d48"}[result.label]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(result.score * 100, 2),
            number={"suffix": "%", "font": {"size": 30, "color": "#0f172a"}},
            title={"text": f"情感极性：{result.label}", "font": {"size": 16, "color": "#334155"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(248,250,252,0.7)",
                "bordercolor": "rgba(15,23,42,0.12)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(225,29,72,0.15)"},
                    {"range": [45, 70], "color": "rgba(100,116,139,0.14)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(normalize_label(str(row.get("sentiment", "Neutral"))) for row in rows)
    return {label: counts.get(label, 0) for label in SENTIMENT_LABELS}


def render_health_panel(rows: list[dict[str, object]]) -> None:
    total = len(rows) or 1
    negative = sum(1 for row in rows if row.get("sentiment") == "Negative")
    red_alerts = sum(1 for row in rows if row.get("risk") == "红色预警")
    average_confidence = sum(float(row.get("confidence", 0.0)) for row in rows) / total
    negative_rate = negative / total
    if red_alerts:
        status_class = "status-risk"
        status = "需要立即处理"
        advice = "红色预警评论已经出现，应优先查看高互动负面内容并建立人工复核记录。"
    elif negative_rate >= 0.35:
        status_class = "status-watch"
        status = "负面趋势上升"
        advice = "负面占比偏高，建议按产品和主题拆解问题来源，补充对应语料。"
    else:
        status_class = "status-ok"
        status = "舆情稳定"
        advice = "当前评论结构较稳定，可继续扩大语料采集并观察低置信度样本。"

    st.markdown(
        f"""
        <div class="panel">
            <div>监控状态：<span class="{status_class}">{status}</span></div>
            <div>负面占比：<strong>{negative_rate:.1%}</strong></div>
            <div>平均模型置信度：<strong>{average_confidence:.1%}</strong></div>
            <div style="margin-top:8px;color:#475569;">{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_feedback_from_community(message: str, product: str, analyzer) -> None:
    rows = st.session_state["feedback_rows"]
    next_id = max(int(row["id"]) for row in rows) + 1 if rows else 1
    row = {
        "id": next_id,
        "channel": "Apple 社区",
        "product": product,
        "status": "待处理",
        "message": message,
        "author": "社区用户",
        "engagement": 0,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "collected": True,
    }
    st.session_state["feedback_rows"] = [enrich_feedback(row, analyzer)] + rows


def render_post(row: dict[str, object]) -> None:
    sentiment = normalize_label(str(row.get("sentiment", "Neutral")))
    sentiment_text = {"Positive": "体验分享", "Neutral": "讨论", "Negative": "求助反馈"}[sentiment]
    st.markdown(
        f"""
        <div class="post">
            <div class="post-meta">{row.get("author", "社区用户")} · {row.get("time", "")} · {row.get("product", "")}</div>
            <div class="post-title">{row.get("topic", "综合体验")}：{sentiment_text}</div>
            <div class="post-body">{row.get("message", "")}</div>
            <span class="tag">{row.get("channel", "Apple 社区")}</span>
            <span class="tag">{row.get("topic", "综合体验")}</span>
            <span class="tag">互动 {row.get("engagement", 0)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_community(analyzer) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Apple 社区</h1>
            <p>面向 Apple 用户的体验交流社区。用户可以发布产品使用体验、求助问题和建议，评论会同步进入运营后台进行语料采集和情感极性监控。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = st.session_state["feedback_rows"]
    df = pd.DataFrame(rows)
    visible_df = df[df["channel"].isin(["Apple 社区", "微博", "小红书", "社区论坛", "App Store"])].copy()
    visible_df = visible_df.sort_values("id", ascending=False)

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        product_filter = st.segmented_control("社区分区", PRODUCTS, default="iPhone")
        for row in visible_df[visible_df["product"].eq(product_filter)].head(6).to_dict("records"):
            render_post(row)
        if visible_df[visible_df["product"].eq(product_filter)].empty:
            st.info("这个分区暂时还没有帖子，可以在右侧发布第一条。")

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("发布体验")
        with st.form("community_post_form", clear_on_submit=True):
            product = st.selectbox("产品", PRODUCTS)
            message = st.text_area(
                "内容",
                placeholder="分享你的使用体验、问题或建议",
                height=130,
            )
            submitted = st.form_submit_button("发布到 Apple 社区", type="primary", width="stretch")
        if submitted:
            if message.strip():
                add_feedback_from_community(message.strip(), product, analyzer)
                st.success("已发布，并同步进入运营后台监控。")
                st.rerun()
            else:
                st.warning("请输入内容后再发布。")
        st.markdown("</div>", unsafe_allow_html=True)

        community_rows = [row for row in rows if row.get("channel") == "Apple 社区"]
        negative_count = sum(1 for row in rows if row.get("sentiment") == "Negative")
        st.metric("社区新增语料", len(community_rows))
        st.metric("全站待关注负面", negative_count)


def render_admin_dashboard() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Apple 社区运营后台监控网站</h1>
            <p>后台围绕课程作业任务设计：从 Apple 社区采集评论语料，使用情感分析模型判断极性，并通过舆情看板监控负面趋势和高频体验问题。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_rows = st.session_state["feedback_rows"]
    df_all = pd.DataFrame(all_rows)

    filter_cols = st.columns([0.2, 0.22, 0.18, 0.22, 0.18])
    selected_channels = filter_cols[0].multiselect("社交渠道", CHANNELS + ["Apple 社区"], default=CHANNELS + ["Apple 社区"])
    selected_products = filter_cols[1].multiselect("产品线", PRODUCTS, default=PRODUCTS)
    selected_sentiments = filter_cols[2].multiselect("情感极性", list(SENTIMENT_LABELS), default=list(SENTIMENT_LABELS))
    selected_statuses = filter_cols[3].multiselect("处理状态", STATUSES, default=STATUSES)
    only_corpus = filter_cols[4].toggle("只看语料", value=False)

    mask = (
        df_all["channel"].isin(selected_channels)
        & df_all["product"].isin(selected_products)
        & df_all["sentiment"].isin(selected_sentiments)
        & df_all["status"].isin(selected_statuses)
    )
    if only_corpus:
        mask = mask & df_all["collected"].astype(bool)

    df_view = df_all.loc[mask].copy()
    view_rows = df_view.to_dict("records")
    summary = build_summary(view_rows)
    total = len(view_rows)
    negative_rate = summary["Negative"] / total if total else 0.0
    corpus_count = int(df_all["collected"].sum())
    red_count = int((df_view["risk"] == "红色预警").sum()) if not df_view.empty else 0
    avg_confidence = float(df_view["confidence"].mean()) if not df_view.empty else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("监控评论", total)
    k2.metric("负面占比", f"{negative_rate:.1%}")
    k3.metric("红色预警", red_count)
    k4.metric("语料库样本", corpus_count)
    k5.metric("平均置信度", f"{avg_confidence:.1%}")

    tab_single, tab_expression, tab_monitor, tab_ingest, tab_corpus = st.tabs(
        ["单文本情感分类", "显式/隐式情感", "舆情挖掘看板", "社区语料采集", "语料库"]
    )

    with tab_single:
        left, right = st.columns([0.58, 0.42], gap="large")
        with left:
            st.subheader("模块 1：单文本情感分类")
            review_text = st.text_area(
                "输入一条 Apple 社区评论",
                value="iPhone 更新后掉电太快，早上满电中午就没电。",
                height=130,
            )
            result = analyzer(review_text)
            st.markdown(
                f"""
                <div class="panel">
                    <div>情感极性：<strong>{result.label}</strong></div>
                    <div>置信度：<strong>{result.score:.2%}</strong></div>
                    <div>分析来源：<strong>{'Hugging Face 模型' if result.source == 'model' else '本地关键词兜底'}</strong></div>
                    <div class="source-note">该模块用于演示模型如何把一条社区评论分类为 Positive、Neutral 或 Negative。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.plotly_chart(make_gauge(result), width="stretch")

    with tab_expression:
        st.subheader("模块 2：显式情感 vs 隐式情感")
        explicit_text = "这次系统更新太差了，电池掉电很快，真的失望。"
        implicit_text = "早上满电出门，中午开会前只剩 18%。"
        col_explicit, col_implicit = st.columns(2, gap="large")
        with col_explicit:
            st.text_area("显式情感评论", value=explicit_text, height=110, disabled=True)
            explicit_result = analyzer(explicit_text)
            st.plotly_chart(make_gauge(explicit_result), width="stretch")
        with col_implicit:
            st.text_area("隐式体验描述", value=implicit_text, height=110, disabled=True)
            implicit_result = analyzer(implicit_text)
            st.plotly_chart(make_gauge(implicit_result), width="stretch")
        st.markdown(
            """
            <div class="panel">
                显式评论直接出现“差、失望”等情绪词，模型通常更容易判断；
                隐式评论依赖常识和场景理解，更适合放进语料库中作为后续模型优化样本。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_monitor:
        top_left, top_right = st.columns([0.62, 0.38], gap="large")
        with top_left:
            st.subheader("模块 3：舆情挖掘看板")
            st.plotly_chart(make_trend_chart(df_view), width="stretch")
        with top_right:
            st.subheader("模型健康")
            render_health_panel(view_rows)

        chart_a, chart_b = st.columns(2, gap="large")
        with chart_a:
            st.subheader("情感极性分布")
            st.plotly_chart(make_sentiment_donut(summary), width="stretch")
        with chart_b:
            st.subheader("高频体验主题")
            st.plotly_chart(make_topic_bar(df_view), width="stretch")

        st.subheader("评论监控列表")
        display_columns = {
            "time": "时间",
            "channel": "渠道",
            "product": "产品",
            "topic": "主题",
            "sentiment": "极性",
            "confidence": "置信度",
            "risk": "风险",
            "status": "状态",
            "engagement": "互动量",
            "message": "评论内容",
            "action": "建议动作",
            "source": "分析来源",
        }
        if df_view.empty:
            st.info("当前筛选条件下暂无评论。")
        else:
            st.dataframe(df_view.loc[:, display_columns.keys()].rename(columns=display_columns), width="stretch", hide_index=True)

    with tab_ingest:
        left, right = st.columns([0.45, 0.55], gap="large")
        with left:
            st.subheader("从 Apple 社区采集语料")
            raw_messages = st.text_area(
                "每行一条社区评论",
                value="iPhone 拍照很清晰，但是电池掉电还是有点快。\nAirPods 降噪很好，地铁通勤舒服多了。",
                height=150,
            )
            channel = st.selectbox("来源渠道", ["Apple 社区"] + CHANNELS, key="ingest_channel")
            product = st.selectbox("关联产品", PRODUCTS, key="ingest_product")
            status = st.selectbox("初始处理状态", STATUSES, index=0, key="ingest_status")
            engagement = st.number_input("互动量", min_value=0, max_value=100000, value=300, step=50)
            collect = st.checkbox("加入语料库", value=True)
            if st.button("采集到语料库并分析", type="primary", width="stretch"):
                messages = split_messages(raw_messages)
                next_id = max(int(row["id"]) for row in all_rows) + 1 if all_rows else 1
                now = datetime.now()
                new_rows = []
                for offset, message in enumerate(messages):
                    row = {
                        "id": next_id + offset,
                        "channel": channel,
                        "product": product,
                        "status": status,
                        "message": message,
                        "author": "后台采集",
                        "engagement": int(engagement),
                        "time": (now + timedelta(minutes=offset)).strftime("%Y-%m-%d %H:%M"),
                        "collected": collect,
                    }
                    new_rows.append(enrich_feedback(row, analyzer))
                st.session_state["feedback_rows"] = new_rows + all_rows
                st.success(f"已采集 {len(new_rows)} 条社区语料并完成极性分析。")
                st.rerun()

        with right:
            st.subheader("实时分析预览")
            preview_messages = split_messages(raw_messages)
            preview_rows = [
                enrich_feedback(
                    {
                        "id": index,
                        "channel": channel,
                        "product": product,
                        "status": status,
                        "message": message,
                        "author": "预览",
                        "engagement": int(engagement),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "collected": collect,
                    },
                    analyzer,
                )
                for index, message in enumerate(preview_messages, start=1)
            ]
            if preview_rows:
                st.dataframe(
                    pd.DataFrame(preview_rows)
                    .loc[:, ["message", "topic", "sentiment", "confidence", "risk", "action"]]
                    .rename(
                        columns={
                            "message": "评论内容",
                            "topic": "主题",
                            "sentiment": "极性",
                            "confidence": "置信度",
                            "risk": "风险",
                            "action": "建议动作",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with tab_corpus:
        corpus_df = df_all[df_all["collected"].astype(bool)].copy()
        st.subheader("已采集语料")
        c1, c2, c3 = st.columns(3)
        c1.metric("语料条数", len(corpus_df))
        c2.metric("负面语料", int((corpus_df["sentiment"] == "Negative").sum()))
        c3.metric("低置信度样本", int((corpus_df["confidence"] < 0.75).sum()))

        st.dataframe(
            corpus_df.loc[:, ["id", "time", "channel", "product", "topic", "sentiment", "confidence", "message"]].rename(
                columns={
                    "id": "编号",
                    "time": "时间",
                    "channel": "渠道",
                    "product": "产品",
                    "topic": "主题",
                    "sentiment": "极性",
                    "confidence": "置信度",
                    "message": "语料文本",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "下载语料 CSV",
            data=rows_to_corpus_csv(corpus_df.to_dict("records")),
            file_name="apple_feedback_corpus.csv",
            mime="text/csv",
            width="stretch",
        )


analyzer, load_error = get_analyzer()
initialize_rows(analyzer)

if load_error:
    st.warning("模型暂时不可用，当前使用本地关键词规则演示。联网且模型缓存可用后会自动切换为 Hugging Face 模型。")

with st.sidebar:
    st.title("Apple 社区")
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "社区"
    if st.button("社区", type="primary" if st.session_state["active_page"] == "社区" else "secondary", width="stretch"):
        st.session_state["active_page"] = "社区"
        st.rerun()
    if st.button(
        "Apple 社区运营后台监控网站",
        type="primary" if st.session_state["active_page"] == "Apple 社区运营后台监控网站" else "secondary",
        width="stretch",
    ):
        st.session_state["active_page"] = "Apple 社区运营后台监控网站"
        st.rerun()
    page = st.session_state["active_page"]

if page == "社区":
    render_community(analyzer)
else:
    render_admin_dashboard()
