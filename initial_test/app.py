"""
政策智能问答助手 - Web 界面 v3（优化版）
修复：Python 3.12 兼容 / 依赖容错 / 启动健壮性
"""
import sys
import os
import re
import traceback
from pathlib import Path
from dotenv import load_dotenv

# ============= 路径配置 =============
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "Resources"

# 加载 .env（优先项目根目录，其次上级目录）
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR.parent / ".env"
load_dotenv(env_path)

# ============= 页面配置（必须放在最前面） =============
import streamlit as st

st.set_page_config(
    page_title="政策智能问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============= 核心样式 =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .source-card {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .welcome-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #bbdefb;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============= 依赖安全导入 =============
IMPORT_ERRORS = []

def safe_import(name, extras=None):
    try:
        if extras:
            for e in extras:
                __import__(e)
        return __import__(name)
    except ImportError as e:
        IMPORT_ERRORS.append(f"❌ 缺少依赖: {name} → {e}")
        return None

# 按需导入，避免启动即崩溃
PyPDFLoader = None
UnstructuredMarkdownLoader = None
TextLoader = None
FAISS = None
ZhipuAIEmbeddings = None
ChatZhipuAI = None
RecursiveCharacterTextSplitter = None
ChatPromptTemplate = None
RunnablePassthrough = None
StrOutputParser = None

# ============= 侧边栏：先显示导入状态 =============
with st.sidebar:
    st.markdown("### ⚙️ 系统设置")
    api_key = os.getenv("ZHIPUAI_API_KEY", "")
    if api_key:
        st.success(f"✅ API Key 已配置（{api_key[:8]}...）", icon="🔑")
    else:
        st.error("❌ 未找到 ZHIPUAI_API_KEY", icon="⚠️")
        st.markdown("请在项目根目录创建 `.env` 文件，写入：")
        st.code("ZHIPUAI_API_KEY=你的智谱API密钥")

# ============= RAG 核心（懒加载） =============
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """加载 RAG 系统，带完整容错"""
    errors = []

    # --- 动态导入 ---
    global PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
    global FAISS, ZhipuAIEmbeddings, ChatZhipuAI
    global RecursiveCharacterTextSplitter, ChatPromptTemplate
    global RunnablePassthrough, StrOutputParser

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as e:
        errors.append(f"PyPDFLoader: {e}")
        PyPDFLoader = None

    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
    except ImportError as e:
        errors.append(f"UnstructuredMarkdownLoader: {e}")
        UnstructuredMarkdownLoader = None

    try:
        from langchain_community.document_loaders import TextLoader
    except ImportError as e:
        errors.append(f"TextLoader: {e}")
        TextLoader = None

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        errors.append(f"FAISS: {e}")
        FAISS = None

    try:
        from rag_utils import ZhipuAIEmbeddings  # native SDK wrapper, auto-batches
    except ImportError:
        try:
            from langchain_community.embeddings import ZhipuAIEmbeddings  # fallback
        except ImportError as e:
            errors.append(f"ZhipuAIEmbeddings: {e}")
            ZhipuAIEmbeddings = None

    try:
        from langchain_community.chat_models import ChatZhipuAI
    except ImportError as e:
        errors.append(f"ChatZhipuAI: {e}")
        ChatZhipuAI = None

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        errors.append(f"RecursiveCharacterTextSplitter: {e}")
        RecursiveCharacterTextSplitter = None

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as e:
        errors.append(f"ChatPromptTemplate: {e}")
        ChatPromptTemplate = None

    try:
        from langchain_core.runnables import RunnablePassthrough
    except ImportError as e:
        errors.append(f"RunnablePassthrough: {e}")
        RunnablePassthrough = None

    try:
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as e:
        errors.append(f"StrOutputParser: {e}")
        StrOutputParser = None

    if errors:
        return None, None, 0, errors

    # --- 加载文档 ---
    all_docs = []

    if RESOURCES_DIR.exists():
        for pdf in RESOURCES_DIR.rglob("*.pdf"):
            try:
                if PyPDFLoader:
                    all_docs.extend(PyPDFLoader(str(pdf)).load())
            except Exception as e:
                errors.append(f"PDF加载失败 {pdf.name}: {e}")

        for md in sorted(RESOURCES_DIR.rglob("*.md")):
            try:
                if UnstructuredMarkdownLoader:
                    loader = UnstructuredMarkdownLoader(str(md), mode="single")
                    docs = loader.load()
                    for d in docs:
                        d.metadata["filename"] = md.name
                        m = re.match(r"(\d{4})(\d{2})(\d{2})", md.name)
                        if m:
                            d.metadata["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    all_docs.extend(docs)
            except Exception as e:
                errors.append(f"MD加载失败 {md.name}: {e}")

        for txt in RESOURCES_DIR.rglob("*.txt"):
            try:
                if TextLoader:
                    all_docs.extend(TextLoader(str(txt), encoding="utf-8").load())
            except Exception as e:
                errors.append(f"TXT加载失败 {txt.name}: {e}")
    else:
        errors.append(f"Resources 目录不存在: {RESOURCES_DIR}")

    if not all_docs:
        errors.append("未加载到任何文档，请检查 Resources 目录")
        return None, None, 0, errors

    # --- 构建向量库 ---
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
    except Exception as e:
        errors.append(f"文档切分失败: {e}")
        return None, None, 0, errors

    try:
        embeddings = ZhipuAIEmbeddings(model="embedding-2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        errors.append(f"向量库构建失败: {e}")
        return None, None, 0, errors

    # --- 构建 Chain ---
    try:
        llm = ChatZhipuAI(model="glm-4.7-flash", temperature=0)

        prompt = ChatPromptTemplate.from_template("""你是"政策与行业分析智能助手"，专注于回答中国 AI 产业、政策、企业相关的专业问题。

【人设】
- 名字：智策通
- 角色：政策研究专家 + 行业分析师
- 风格：专业、严谨、引用数据
- 边界：不编造、不推测、不补充常识

【铁律】
1. 只能基于参考资料回答，严禁推测、联想、常识补充
2. 不确定就拒答：必须回答"参考资料中未找到相关信息"
3. 数字必须原文一致
4. 引用要带出处，末尾标注信息来源文件名

参考资料：
{context}

用户问题：{question}""")

        def format_docs(docs):
            lines = []
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("filename", "?")
                date = doc.metadata.get("date", "")
                date_str = f"（{date}）" if date else ""
                lines.append(f"【来源 {i}{date_str}】{src}\n{doc.page_content}")
            return "\n\n".join(lines)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
    except Exception as e:
        errors.append(f"Chain 构建失败: {e}")
        return None, None, len(all_docs), errors

    return chain, retriever, len(all_docs), errors


def generate_ai_response(question):
    """生成 AI 回答"""
    if not st.session_state.chain:
        return "❌ RAG 系统未正确加载，请检查侧边栏错误信息", []

    try:
        answer = st.session_state.chain.invoke(question)
        retrieved = st.session_state.retriever.invoke(question)
        sources = []
        for doc in retrieved[:5]:
            sources.append({
                "filename": doc.metadata.get("filename", "未知"),
                "date": doc.metadata.get("date", "N/A"),
                "preview": doc.page_content[:200].replace("\n", " ") + "...",
            })
        return answer, sources
    except Exception as e:
        return f"❌ 出错了:\n\n{traceback.format_exc()}", []


# ============= 主界面 =============
st.markdown('<div class="main-header">📚 智策通 · 政策智能问答助手</div>', unsafe_allow_html=True)
st.markdown("---")

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "first_load" not in st.session_state:
    st.session_state.first_load = True
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

# 加载 RAG 系统
if not st.session_state.rag_loaded:
    with st.spinner("🔄 正在加载知识库..."):
        chain, retriever, doc_count, load_errors = load_rag_system()
        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.session_state.doc_count = doc_count
        st.session_state.load_errors = load_errors
        st.session_state.rag_loaded = True

# 显示加载错误（如果有）
if st.session_state.load_errors:
    with st.expander("⚠️ 加载过程中存在以下问题（点击展开）", expanded=bool(st.session_state.chain is None)):
        for err in st.session_state.load_errors:
            st.markdown(f"- {err}")

# 侧边栏补充状态信息
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 系统状态")
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
    st.markdown("### 💡 使用提示")
    st.markdown("""
    - 📅 支持日期范围查询
    - 🔍 支持精确数字查询
    - ⚠️ 找不到答案时明确告知
    - 📎 每个答案标注信息来源
    """)

    st.markdown("---")
    st.markdown("### 🤖 AI 身份")
    st.info("""
    **智策通**
    政策研究专家 + 行业分析师
    知识库：2026 年 AI 行业分析 + 政策文件
    """)

# 开屏欢迎语
if st.session_state.first_load and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-box">
        <h3>👋 你好！我是智策通</h3>
        <p>专注于 <b>AI 产业 / 政策 / 企业</b> 领域的智能问答助手。</p>
        <p>📚 我能帮你：</p>
        <ul>
            <li>查询最新 AI 政策与行业动态</li>
            <li>解读具体政策的核心条款与数字</li>
            <li>对比不同时间段的市场趋势</li>
            <li>严格基于参考资料回答，<b>不编造</b></li>
        </ul>
        <p>👇 点下方示例问题开始，或直接输入你的问题</p>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.first_load = False

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📎 查看 {len(msg['sources'])} 个信息来源", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <b>来源 {i}</b>: {src['filename']}<br>
                            📅 日期: {src['date']}<br>
                            📝 摘要: {src['preview']}
                        </div>
                        """, unsafe_allow_html=True)

# 处理问题的通用函数
def _is_greeting(text: str) -> bool:
    """Detect greetings and small-talk that don't need RAG retrieval."""
    t = text.strip().lower().rstrip("!.?。！？~,， ")
    greetings = {
        "你好", "您好", "hi", "hello", "hey", "哈喽", "嗨",
        "在吗", "在么", "在不在", "你是谁", "你叫什么", "你能做什么",
        "早上好", "下午好", "晚上好", "早安", "晚安", "thanks", "thank you", "谢谢",
    }
    if t in greetings:
        return True
    if len(t) <= 4 and any(g in t for g in ["你好", "您好", "hi", "hello", "嗨", "在吗", "谢谢"]):
        return True
    return False


def _greeting_reply() -> str:
    """Friendly reply for greetings, no RAG call needed."""
    return (
        "你好！我是**智策通**，专注于中国 AI 产业、政策、企业相关问题的政策智能问答助手。\n\n"
        "我可以帮你：\n"
        "- 查找政策原文与数据（如 OPC 政策、税收优惠）\n"
        "- 解读行业报告（AI 红包大战、Seedance 2.0、AI 学习机等）\n"
        "- 回答事实型 / 数据型 / 推理型问题\n\n"
        "你可以在下方点击快捷问题开始体验，或直接输入你的问题 🙂"
    )


def process_question(question):
    if not question or not question.strip():
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # Greeting short-circuit: skip RAG for small-talk
    if _is_greeting(question):
        answer = _greeting_reply()
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })
        return

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 智策通正在思考..."):
            answer, sources = generate_ai_response(question)
            st.markdown(answer)
            if sources:
                with st.expander(f"📎 查看 {len(sources)} 个信息来源", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <b>来源 {i}</b>: {src['filename']}<br>
                            📅 日期: {src['date']}<br>
                            📝 摘要: {src['preview']}
                        </div>
                        """, unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

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
st.caption("🤖 智策通 v1.0 | 基于 LangChain + 智谱 GLM-4 + FAISS 构建")