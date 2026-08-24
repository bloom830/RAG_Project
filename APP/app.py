"""
政策智能问答助手 - Web 界面 v6（模块化）
架构：
  app.py             ← 主入口，组装各模块
  style.py           ← CSS 样式
  session_manager.py  ← 会话持久化 CRUD
  rag_engine.py      ← RAG 核心（文档加载 / 向量库 / Chain）
  chat_utils.py      ← 闲聊检测 / 欢迎语 / 来源卡片
  transfer_service.py ← 转人工客服
  feedback.py        ← 点赞 / 点踩反馈
  web_search.py      ← 联网搜索增强兜底
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ============= 路径与配置 =============
BASE_DIR = Path(__file__).resolve().parent.parent

# 固定 Streamlit 端口为 8502（避免 8501 被占用）
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR.parent / ".env"
load_dotenv(env_path)

# ============= Streamlit 初始化 =============
import streamlit as st

st.set_page_config(
    page_title="政策智能问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============= 注入样式 =============
from style import CSS
st.markdown(CSS, unsafe_allow_html=True)

# ============= 导入业务模块 =============
from session_manager import (
    list_sessions, load_session, save_session, delete_session,
    generate_session_title, switch_to_session, create_new_session,
)
from rag_engine import build_rag_system, build_history_text, generate_ai_response
from chat_utils import is_greeting, greeting_reply, build_welcome_html, format_source_card
from transfer_service import render_transfer_button
from feedback import render_feedback_buttons, get_feedback_summary
# 联网搜索功能已关闭，相关导入保留但不再使用
# from web_search import (
#     web_search, should_trigger_search, build_search_augmented_prompt,
#     render_search_status, get_provider as get_search_provider,
# )

# ============= 侧边栏：API Key 状态 =============
with st.sidebar:
    st.markdown("### ⚙️ 系统设置")
    api_key = os.getenv("ZHIPUAI_API_KEY", "")
    if api_key:
        st.success(f"✅ API Key 已配置（{api_key[:8]}...）", icon="🔑")
    else:
        st.error("❌ 未找到 ZHIPUAI_API_KEY", icon="⚠️")
        st.markdown("请在项目根目录创建 `.env` 文件，写入：")
        st.code("ZHIPUAI_API_KEY=你的智谱API密钥")

# ============= 加载 RAG 系统（懒加载，只一次） =============
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

if not st.session_state.rag_loaded:
    with st.spinner("🔄 正在加载知识库..."):
        chain, retriever, doc_count, load_errors = build_rag_system()
        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.session_state.doc_count = doc_count
        st.session_state.load_errors = load_errors
        st.session_state.rag_loaded = True

# ============= 显示加载错误 =============
if st.session_state.get("load_errors"):
    with st.expander("⚠️ 加载过程中存在以下问题（点击展开）",
                    expanded=bool(st.session_state.chain is None)):
        for err in st.session_state.load_errors:
            st.markdown(f"- {err}")

# 如果 RAG 未就绪，提供重新加载按钮
if st.session_state.chain is None:
    st.error("❌ 知识库未就绪，无法使用 RAG 问答。请修复上方错误后点击重试。")
    if st.button("🔄 重新加载知识库", use_container_width=True, type="primary"):
        st.session_state.rag_loaded = False
        st.rerun()

# ============= 初始化 session state =============
defaults = {
    "current_session_id": None,
    "messages": [],
    "pending_question": None,
    "first_load": True,
    "confirm_delete": None,
    "session_created_at": None,
    "search_enabled": False,  # 联网搜索已禁用
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============= 侧边栏：会话管理（常驻）+ 设置(expander折叠) + 转人工客服 =============
with st.sidebar:
    # --- 新建对话（浅蓝色按钮） ---
    if st.button("➕  新建对话", use_container_width=True, type="secondary"):
        if st.session_state.current_session_id and st.session_state.messages:
            save_session(
                st.session_state.current_session_id,
                generate_session_title(st.session_state.messages),
                st.session_state.messages,
                st.session_state.session_created_at,
            )
        new = create_new_session()
        st.session_state.current_session_id = new["session_id"]
        st.session_state.messages = new["messages"]
        st.session_state.first_load = True
        st.session_state.session_created_at = new["created_at"]
        st.rerun()

    # --- 历史会话列表（折叠显示） ---
    sessions = list_sessions()
    if sessions:
        with st.expander(f"📋 历史会话（{len(sessions)}）", expanded=False):
            for s in sessions:
                is_active = (s["id"] == st.session_state.current_session_id)
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"{'🟢' if is_active else '💬'} {s['title']}",
                        key=f"switch_{s['id']}",
                        use_container_width=True,
                        type="secondary" if not is_active else "primary",
                    ):
                        if st.session_state.current_session_id and st.session_state.messages:
                            save_session(
                                st.session_state.current_session_id,
                                generate_session_title(st.session_state.messages),
                                st.session_state.messages,
                                st.session_state.session_created_at,
                            )
                        data = switch_to_session(s["id"])
                        if data:
                            st.session_state.current_session_id = s["id"]
                            st.session_state.messages = data["messages"]
                            st.session_state.first_load = (len(data["messages"]) == 0)
                            st.session_state.session_created_at = data["created_at"]
                            st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{s['id']}", help="删除此会话"):
                        st.session_state.confirm_delete = s["id"]

            # 删除确认
            if st.session_state.confirm_delete:
                st.warning("确定删除会话？此操作不可恢复。")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 确认删除", type="primary", use_container_width=True, key="confirm_del"):
                        delete_session(st.session_state.confirm_delete)
                        if st.session_state.confirm_delete == st.session_state.current_session_id:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.session_state.confirm_delete = None
                        st.rerun()
                with c2:
                    if st.button("❌ 取消", use_container_width=True, key="cancel_del"):
                        st.session_state.confirm_delete = None
                        st.rerun()

    # --- 转人工客服（重要功能，常驻显示） ---
    render_transfer_button()

    # --- ⚡️ 设置区：所有次要功能折叠 ---
    with st.expander("⚙️ 设置", expanded=False):
        # 系统状态
        st.markdown("**📊 系统状态**")
        if st.session_state.chain:
            st.markdown(f"""
            <div class="stat-box">
                <h3>{st.session_state.doc_count}</h3>
                <p>已加载文档数</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("RAG 系统未就绪")

        st.markdown("---")

        # 联网搜索功能已关闭
        st.markdown("**🌐 联网搜索**")
        st.info("已禁用")

        st.markdown("---")

        # 反馈摘要
        fb = get_feedback_summary(days=7)
        st.markdown("**👍👎 反馈摘要**")
        if fb["total"] > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("👍 点赞", fb["likes"])
            with col_b:
                st.metric("👎 点踢", fb["dislikes"])
            st.caption(f"好评率 {fb['like_rate']:.0f}% | 共 {fb['total']} 条")
            if fb["recent_dislikes"]:
                with st.expander(f"查看最近 {len(fb['recent_dislikes'])} 条差评", expanded=False):
                    for rec in fb["recent_dislikes"]:
                        q = rec.get("question", "")[:60]
                        st.caption(f"❌ {q}")
        else:
            st.caption("暂无反馈数据")

        st.markdown("---")

        # 使用提示
        st.markdown("**💡 使用提示**")
        st.markdown("""
        - 📅 支持日期范围查询
        - 🔍 支持精确数字查询
        - ⚠️ 找不到答案时明确告知
        - 📎 每个答案标注信息来源
        - 💬 支持多轮追问，上下文自动记忆
        - 👍👎 对回答点赞/点踢帮助改进
        - 🌐 联网搜索兑底补充答案
        """)

        st.markdown("---")

        # AI 身份
        st.markdown("**🤖 AI 身份**")
        st.info("""
        **智策通**
        政策研究专家 + 行业分析师
        知识库：2026 年 AI 行业分析 + 政策文件
        """)

# ============= 主界面 =============
st.markdown('<div class="main-header">📚 智策通 · 政策智能问答助手</div>', unsafe_allow_html=True)
st.markdown("---")

# 当前会话指示器
if st.session_state.current_session_id:
    title = generate_session_title(st.session_state.messages) if st.session_state.messages else "新对话"
    st.caption(f"📍 当前会话：{title}")

# 欢迎语
if st.session_state.first_load and len(st.session_state.messages) == 0:
    st.markdown(build_welcome_html(), unsafe_allow_html=True)
    st.session_state.first_load = False

# ============= 显示历史消息（含点赞/点踩按钮） =============
for msg_idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📎 查看 {len(msg['sources'])} 个信息来源", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(format_source_card(i, src), unsafe_allow_html=True)

            # --- 点赞 / 点踩按钮 ---
            render_feedback_buttons(msg_idx, msg)

# ============= 处理问题的核心函数（含联网搜索兜底） =============
def process_question(question: str):
    if not question or not question.strip():
        return

    # 确保有 session_id
    if not st.session_state.current_session_id:
        new = create_new_session()
        st.session_state.current_session_id = new["session_id"]
        st.session_state.session_created_at = new["created_at"]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # 闲聊短路
    if is_greeting(question):
        answer = greeting_reply()
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
            render_feedback_buttons(len(st.session_state.messages), {"role": "assistant", "content": answer, "sources": []})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        save_session(
            st.session_state.current_session_id,
            generate_session_title(st.session_state.messages),
            st.session_state.messages,
            st.session_state.session_created_at,
        )
        return

    # RAG 问答
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 智策通正在思考..."):
            history = build_history_text(st.session_state.messages[:-1])
            if st.session_state.chain is not None:
                answer, sources = generate_ai_response(
                    st.session_state.chain,
                    st.session_state.retriever,
                    question,
                    history,
                )
            else:
                # RAG 未就绪：提示用户修复知识库
                answer = "⚠️ 知识库未就绪，请检查左侧加载错误并点击「重新加载知识库」。"
                sources = []

            st.markdown(answer)

            # 展示 RAG 来源
            if sources:
                with st.expander(f"📎 查看 {len(sources)} 个知识库来源", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(format_source_card(i, src), unsafe_allow_html=True)

            # 保存消息
            msg_idx = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "web_results": [],
            })
            save_session(
                st.session_state.current_session_id,
                generate_session_title(st.session_state.messages),
                st.session_state.messages,
                st.session_state.session_created_at,
            )

            # 渲染反馈按钮
            render_feedback_buttons(msg_idx, st.session_state.messages[-1])


# 处理 pending question
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    process_question(q)
    st.rerun()

# 输入框
if user_input := st.chat_input("💬 请输入你的问题..."):
    process_question(user_input)
    st.rerun()

# ============= 底部快捷问题 =============
st.markdown("---")
st.markdown("### 🚀 试试这些问题")

example_questions = [
    "📊 2026年小规模纳税人月销售额多少以下免征增值税？",
    "🤖 中国AI大模型Token调用量在哪一周首次超越美国？",
    "📚 深圳龙岗区对AI创业的最高补贴是多少？",
    "🎬 OpenAI什么时候正式宣布关停Sora服务？",
    "💰 注册OPC企业能享受多少所得税优惠？",
    "📈 2025年Q2中国学习平板出货量同比增长多少？",
]

cols = st.columns(3)
for i, q in enumerate(example_questions):
    with cols[i % 3]:
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

st.markdown("---")
st.caption("🤖 智策通 v6.0 | 模块化架构 | 转人工 · 点赞反馈")
