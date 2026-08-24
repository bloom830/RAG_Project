"""
RAG 引擎模块
职责：文档加载、向量库构建、Chain 构建、回答生成
"""

import re
import traceback
from pathlib import Path

# ============= 路径配置 =============
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# 限定加载范围：避免加载过多文档导致 embedding 失败（1210/1214 错误）
DATA_SOURCES = [
    BASE_DIR / "openaxo-main" / "2026" / "02",
    BASE_DIR / "openaxo-main" / "2026" / "03",
]

# ============= 检索配置 =============
# 混合检索：FAISS 召回 Top-N + BM25 召回 Top-N → RRF 融合 → Cross-Encoder 精排 → Top-K
RETRIEVE_TOP_K = 5            # 最终返回给 LLM 的文档数
RERANK_CANDIDATES = 10        # 混合检索召回的候选数（精排前）
HYBRID_WEIGHTS = (0.6, 0.4)   # (FAISS 权重, BM25 权重)，经网格搜索确定
RRF_K = 60                    # RRF 融合平滑常数，越大排名差异越平
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"  # Cross-Encoder 精排模型

# ============= 懒加载的模块引用 =============
PyPDFLoader = None
TextLoader = None
FAISS = None
ZhipuAIEmbeddings = None
ChatZhipuAI = None
RecursiveCharacterTextSplitter = None
ChatPromptTemplate = None
RunnablePassthrough = None
StrOutputParser = None
BM25Retriever = None  # 混合检索：关键词召回

LOADED = False
LOAD_ERRORS = []


def _import_all():
    """安全导入所有依赖，只执行一次"""
    global PyPDFLoader, TextLoader
    global FAISS, ZhipuAIEmbeddings, ChatZhipuAI
    global RecursiveCharacterTextSplitter, ChatPromptTemplate
    global RunnablePassthrough, StrOutputParser, BM25Retriever, LOADED, LOAD_ERRORS

    if LOADED:
        return

    errors = []

    try:
        from langchain_community.document_loaders import PyPDFLoader as _L1
        PyPDFLoader = _L1
    except ImportError as e:
        errors.append(f"PyPDFLoader: {e}")

    try:
        from langchain_community.document_loaders import TextLoader as _L3
        TextLoader = _L3
    except ImportError as e:
        errors.append(f"TextLoader: {e}")

    try:
        from langchain_community.vectorstores import FAISS as _V1
        FAISS = _V1
    except ImportError as e:
        errors.append(f"FAISS: {e}")

    try:
        from rag_utils import ZhipuAIEmbeddings as _E1
        ZhipuAIEmbeddings = _E1
    except ImportError:
        try:
            from langchain_community.embeddings import ZhipuAIEmbeddings as _E2
            ZhipuAIEmbeddings = _E2
        except ImportError as e:
            errors.append(f"ZhipuAIEmbeddings: {e}")

    try:
        from langchain_community.chat_models import ChatZhipuAI as _C1
        ChatZhipuAI = _C1
    except ImportError as e:
        errors.append(f"ChatZhipuAI: {e}")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter as _S1
        RecursiveCharacterTextSplitter = _S1
    except ImportError as e:
        errors.append(f"RecursiveCharacterTextSplitter: {e}")

    try:
        from langchain_core.prompts import ChatPromptTemplate as _P1
        ChatPromptTemplate = _P1
    except ImportError as e:
        errors.append(f"ChatPromptTemplate: {e}")

    try:
        from langchain_core.runnables import RunnablePassthrough as _R1
        RunnablePassthrough = _R1
    except ImportError as e:
        errors.append(f"RunnablePassthrough: {e}")

    try:
        from langchain_core.output_parsers import StrOutputParser as _O1
        StrOutputParser = _O1
    except ImportError as e:
        errors.append(f"StrOutputParser: {e}")

    try:
        from langchain_community.retrievers import BM25Retriever as _B1
        BM25Retriever = _B1
    except ImportError as e:
        errors.append(f"BM25Retriever: {e}")

    LOAD_ERRORS = errors
    LOADED = True


# ============= 混合检索 + 精排 =============

def _tokenize(text: str) -> set:
    """简易分词：中文按字、英文/数字按词，用于关键词重排序兜底。"""
    if not text:
        return set()
    return set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text))


def _keyword_score(query: str, doc_text: str) -> float:
    """关键词共现打分：归一化命中数，Cross-Encoder 不可用时的兜底精排。"""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = _tokenize(doc_text)
    hits = q_tokens & d_tokens
    return len(hits) / len(q_tokens) + len(hits) / max(len(d_tokens), 1)


def _hybrid_retrieve(question, faiss_retriever, bm25_retriever, top_k, weights, rrf_k=60):
    """RRF 融合 FAISS 与 BM25 检索结果。"""
    faiss_docs = faiss_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question) if bm25_retriever else []

    doc_map, scores = {}, {}
    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + weights[0] / (rank + rrf_k)
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + weights[1] / (rank + rrf_k)

    sorted_docs = sorted(doc_map.values(), key=lambda d: scores[d.page_content], reverse=True)
    return sorted_docs[:top_k]


def _rerank_cross_encoder(question, docs, top_k, cross_encoder):
    """Cross-Encoder 精排：问题+文档成对输入，按相关性分数排序。"""
    if not docs or cross_encoder is None:
        return docs[:top_k] if docs else []
    pairs = [(question, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


class _CrossEncoderReranker:
    """加载 BAAI/bge-reranker-base 做精排，分批推理避免 OOM。"""

    def __init__(self, model_name: str):
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, local_files_only=True)
        self.model.eval()
        self._torch = torch

    def predict(self, pairs, show_progress_bar=False):
        if not pairs:
            return []
        batch_size, all_scores = 8, []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                [p[0] for p in batch], [p[1] for p in batch],
                padding=True, truncation=True, return_tensors="pt", max_length=512,
            )
            with self._torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)
            all_scores.extend(scores.numpy().tolist())
        return all_scores


class HybridRerankRetriever:
    """包装检索器：对外暴露 invoke(question) → 内部走 混合检索 + Cross-Encoder 精排。
    Cross-Encoder 加载失败时自动降级为关键词重排序，保证系统可用。"""

    def __init__(self, faiss_retriever, bm25_retriever, cross_encoder=None):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder

    def invoke(self, question):
        # 1. 混合检索：FAISS + BM25 → RRF 融合，召回候选
        candidates = _hybrid_retrieve(
            question, self.faiss_retriever, self.bm25_retriever,
            top_k=RERANK_CANDIDATES, weights=HYBRID_WEIGHTS, rrf_k=RRF_K,
        )
        # 2. 精排：Cross-Encoder 优先，不可用则关键词重排序兜底
        if self.cross_encoder is not None:
            return _rerank_cross_encoder(question, candidates, RETRIEVE_TOP_K, self.cross_encoder)
        # 兜底：按关键词匹配度重排
        scored = sorted(
            candidates,
            key=lambda d: _keyword_score(question, d.page_content),
            reverse=True,
        )
        return scored[:RETRIEVE_TOP_K]


def build_rag_system():
    """
    构建 RAG 系统，返回 (chain, retriever, doc_count, errors)
    调用方检查 chain 是否为 None
    """
    import rag_engine as _self
    _self._import_all()

    if _self.LOAD_ERRORS:
        return None, None, 0, _self.LOAD_ERRORS

    errors = []
    all_docs = []

    # --- 加载文档（仅限 DATA_SOURCES 指定的子目录） ---
    sources_to_load = []
    for src in _self.DATA_SOURCES:
        if src.exists():
            sources_to_load.append(src)
        else:
            errors.append(f"数据源目录不存在: {src}")

    if not sources_to_load:
        errors.append(f"所有数据源目录都不存在，请检查配置: {_self.DATA_SOURCES}")
        return None, None, 0, errors

    for src in sources_to_load:
        # PDF（使用 pypdf 的 PyPDFLoader，不需要 unstructured）
        for pdf in src.rglob("*.pdf"):
            try:
                if _self.PyPDFLoader:
                    docs = _self.PyPDFLoader(str(pdf)).load()
                    for d in docs:
                        d.metadata["filename"] = pdf.name
                    all_docs.extend(docs)
            except Exception as e:
                errors.append(f"PDF加载失败 {pdf.name}: {e}")

        # Markdown（使用 TextLoader，避免依赖 unstructured）
        for md in sorted(src.rglob("*.md")):
            try:
                if _self.TextLoader:
                    docs = _self.TextLoader(str(md), encoding="utf-8").load()
                    for d in docs:
                        d.metadata["filename"] = md.name
                        m = re.match(r"(\d{4})(\d{2})(\d{2})", md.name)
                        if m:
                            d.metadata["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    all_docs.extend(docs)
            except Exception as e:
                errors.append(f"MD加载失败 {md.name}: {e}")

        # TXT
        for txt in src.rglob("*.txt"):
            try:
                if _self.TextLoader:
                    docs = _self.TextLoader(str(txt), encoding="utf-8").load()
                    for d in docs:
                        d.metadata["filename"] = txt.name
                    all_docs.extend(docs)
            except Exception as e:
                errors.append(f"TXT加载失败 {txt.name}: {e}")

    if not all_docs:
        errors.append("未加载到任何文档，请检查 DATA_SOURCES 配置")
        return None, None, 0, errors

    # --- 构建向量库 ---
    try:
        splitter = _self.RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
    except Exception as e:
        errors.append(f"文档切分失败: {e}")
        return None, None, 0, errors

    try:
        embeddings = _self.ZhipuAIEmbeddings(model="embedding-2")
        vectorstore = _self.FAISS.from_documents(chunks, embeddings)
        # FAISS 召回候选数（精排前），混合检索 + 精排后取 Top-5
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RERANK_CANDIDATES})
    except Exception as e:
        errors.append(f"向量库构建失败: {e}")
        return None, None, 0, errors

    # --- BM25 检索器（关键词召回，与 FAISS 互补） ---
    bm25_retriever = None
    if _self.BM25Retriever:
        try:
            bm25_retriever = _self.BM25Retriever.from_documents(chunks, k=RERANK_CANDIDATES)
        except Exception as e:
            errors.append(f"BM25 检索器构建失败（降级为纯 FAISS）: {e}")

    # --- Cross-Encoder 精排模型（加载失败则降级为关键词重排序） ---
    cross_encoder = None
    try:
        cross_encoder = _CrossEncoderReranker(CROSS_ENCODER_MODEL)
    except Exception as e:
        errors.append(f"Cross-Encoder 加载失败（降级为关键词重排序）: {e}")

    # --- 混合检索 + 精排包装检索器 ---
    retriever = HybridRerankRetriever(faiss_retriever, bm25_retriever, cross_encoder)

    # --- 构建 Chain ---
    try:
        llm = _self.ChatZhipuAI(model="glm-4-flash", temperature=0)

        prompt = _self.ChatPromptTemplate.from_template("""你是"政策与行业分析智能助手"，专注于回答中国 AI 产业、政策、企业相关的专业问题。

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

【对话历史】
以下是之前你和用户的对话记录，请参考上下文理解用户当前问题（尤其是代词和省略）：
{history}

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

        # Chain: 字典并行映射，必须用 lambda 显式从输入 dict 中提取 question 给 retriever
        # 否则 retriever 收到的是整个 dict，导致 embed_query('dict') 报错
        chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "history": lambda x: x.get("history", ""),
            }
            | prompt | llm | _self.StrOutputParser()
        )
    except Exception as e:
        errors.append(f"Chain 构建失败: {e}")
        return None, None, len(all_docs), errors

    return chain, retriever, len(all_docs), errors


# ============= 历史消息格式化 =============

def build_history_text(messages, max_messages=20):
    """将历史消息格式化为文本，注入 prompt"""
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    lines = []
    for msg in recent:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
    return "\n".join(lines)


# ============= 回答生成 =============

def generate_ai_response(chain, retriever, question, history_text):
    """
    生成 AI 回答
    返回 (answer: str, sources: list[dict])
    """
    if chain is None:
        return "❌ RAG 系统未正确加载", []

    try:
        answer = chain.invoke({
            "question": question,
            "history": history_text,
        })
        retrieved = retriever.invoke(question)
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
