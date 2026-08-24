r"""
CRUD_RAG 数据集 RAG 评估脚本
复用现有 ragvenv + 智谱 API，实现图片中的 RAG 完整流程评测：
  1) 文档切分（chunk=500, overlap=50）
  2) Embedding 向量化 + FAISS 向量库
  3) 向量检索 Top-K + 重排序优化
  4) LLM 生成答案
  5) 评估：keyword_overlap / number_hit_rate / bad case 分布

数据源：G:\Lai\RAG\CRUD_RAG\data\crud_split\split_merged.json
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 把 APP 目录加入路径以复用 ZhipuAIEmbeddings
APP_DIR = Path(__file__).resolve().parent.parent / "APP"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rag_utils import ZhipuAIEmbeddings


# ============= 配置 =============
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CRUD_DATA_PATH = BASE_DIR / "CRUD_RAG" / "data" / "crud_split" / "split_merged.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_CANDIDATES = 10  # 重排序时先召回 10 条，再取前 5
MAX_TEST_SAMPLES = 50   # 评测样本数，可调大/调小
USE_RERANK = False      # 默认跑基础版，命令行 --rerank 跑优化版
USE_HYBRID = False      # 是否启用 BM25 + FAISS 混合检索
USE_CROSS_ENCODER = False  # 是否启用 Cross-Encoder 重排序
CROSS_ENCODER_MODEL = "G:/Lai/RAG/huggingface_cache/local/BAAI--bge-reranker-base"


# ============= 工具函数 =============
def load_crud_dataset(path: Path):
    """加载 CRUD_RAG split_merged.json，返回 event_summary 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # event_summary 类别同时包含 event（问题）、summary（答案）、text（原文）
    items = data.get("event_summary", [])
    # 过滤有 text 和 summary 的
    valid = [it for it in items if it.get("text") and it.get("summary")]
    return valid


def tokenize(text: str) -> set:
    """中文按字、英文按词抽取 token。"""
    return set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text or ""))


def extract_numbers(text: str) -> set:
    """抽取文本中的数字（含百分比、千分位）。"""
    return set(re.findall(r"\d+(?:\.\d+)?%?|\d+(?:,\d{3})*(?:\.\d+)?", text or ""))


def keyword_overlap(reference: str, prediction: str) -> float:
    """参考答案与预测答案的关键词重合率（Jaccard 覆盖度）。"""
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref:
        return 0.0
    return len(ref & pred) / len(ref)


def number_hit_rate(reference: str, prediction: str) -> tuple[float, set, set] | None:
    """数字命中率：参考答案中的数字在预测答案中的比例。"""
    ref_nums = extract_numbers(reference)
    pred_nums = extract_numbers(prediction)
    if not ref_nums:
        return None
    hit = ref_nums & pred_nums
    return len(hit) / len(ref_nums), hit, ref_nums - pred_nums


def keyword_score(query: str, doc_text: str) -> float:
    """用关键词共现数量给文档打分，用于重排序。"""
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = tokenize(doc_text)
    hits = q_tokens & d_tokens
    # 归一化：命中数 / 查询词数 + 命中数 / 文档词数（惩罚长文档）
    return len(hits) / len(q_tokens) + len(hits) / max(len(d_tokens), 1)


def hybrid_retrieve(question: str, faiss_retriever, bm25_retriever, top_k: int, weights=(0.6, 0.4), rrf_k: int = 60) -> list[Document]:
    """RRF 融合 FAISS 与 BM25 检索结果。"""
    faiss_docs = faiss_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    doc_map = {}
    scores = {}
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


def rerank_documents(question: str, docs: list[Document], top_k: int, cross_encoder=None) -> list[Document]:
    """按关键词匹配度或 Cross-Encoder 对检索结果重排序。"""
    if cross_encoder is not None and len(docs) > 0:
        pairs = [(question, doc.page_content) for doc in docs]
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    scored = []
    for doc in docs:
        score = keyword_score(question, doc.page_content)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


# ============= RAG 构建 =============
def build_rag(items: list[dict], use_rerank: bool = False, use_hybrid: bool = False, use_cross_encoder: bool = False, weights=(0.6, 0.4)):
    """根据 items 中的 text 构建 FAISS 向量库与 RAG Chain。
    weights: 混合检索时 (FAISS权重, BM25权重)，仅在 use_hybrid=True 时生效。
    """
    print(f"[INFO] 加载 {len(items)} 篇文档构建向量库...")

    # 每篇 text 作为一个 Document
    documents = []
    for it in items:
        text = it["text"]
        documents.append(Document(
            page_content=text,
            metadata={
                "id": it.get("ID", ""),
                "title": it.get("title", ""),
                "time": it.get("time", ""),
            },
        ))

    # Chunk 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] 切分后共 {len(chunks)} 个 chunk")

    # 向量化 + 向量库
    embeddings = ZhipuAIEmbeddings(model="embedding-2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # FAISS 检索器
    candidate_k = TOP_K if not (use_rerank or use_cross_encoder or use_hybrid) else RERANK_CANDIDATES
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": candidate_k})

    # 混合检索：FAISS + BM25
    bm25_retriever = None
    if use_hybrid:
        print("[INFO] 启用 BM25 + FAISS 混合检索")
        bm25_retriever = BM25Retriever.from_documents(chunks, k=RERANK_CANDIDATES)

    # Cross-Encoder 重排序模型（按需导入，避免 torch 安装未完成时阻塞 BM25 混合检索）
    cross_encoder = None
    if use_cross_encoder:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        import torch

        class CrossEncoderReranker:
            def __init__(self, model_name: str):
                print(f"[INFO] 加载 Cross-Encoder 模型: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                config = AutoConfig.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, local_files_only=True)
                self.model.eval()

            def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False):
                if not pairs:
                    return []
                # 分批处理，避免单 batch 太长导致 OOM
                batch_size = 8
                all_scores = []
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    inputs = self.tokenizer(
                        [p[0] for p in batch],
                        [p[1] for p in batch],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512,
                    )
                    with torch.no_grad():
                        scores = self.model(**inputs).logits.squeeze(-1)
                    all_scores.extend(scores.numpy().tolist())
                return all_scores

        cross_encoder = CrossEncoderReranker(CROSS_ENCODER_MODEL)

    def retrieve(question: str) -> list[Document]:
        if use_hybrid:
            docs = hybrid_retrieve(question, faiss_retriever, bm25_retriever, RERANK_CANDIDATES, weights=weights)
        else:
            docs = faiss_retriever.invoke(question)
        if use_cross_encoder:
            docs = rerank_documents(question, docs, TOP_K, cross_encoder=cross_encoder)
        elif use_rerank:
            docs = rerank_documents(question, docs, TOP_K)
        return docs

    llm = ChatZhipuAI(model="glm-4-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template("""你是新闻事件摘要问答助手。请根据参考资料生成完整、信息丰富的回答。

【要求】
1. 只能基于参考资料回答，严禁推测、联想、常识补充
2. 涵盖参考资料中的关键主体、时间、地点、事件经过、数字和结论
3. 数字必须原文一致，不得四舍五入或换算
4. 如果参考资料没有相关信息，回答"参考资料中未找到相关信息"
5. 回答末尾用括号标注信息来源标题

参考资料：
{context}

用户问题：{question}

请给出完整准确的回答：""")

    def format_docs(docs: list[Document]) -> str:
        lines = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "未知")
            lines.append(f"[来源{i}] {title}\n{doc.page_content}")
        return "\n\n".join(lines)

    chain = (
        {"context": lambda q: format_docs(retrieve(q)), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return chain, retrieve


# ============= 评估主流程 =============
def evaluate(items: list[dict], chain, retrieve_fn, use_rerank: bool = False, use_hybrid: bool = False, use_cross_encoder: bool = False, limit: int = MAX_TEST_SAMPLES):
    """对前 limit 条记录做评估。"""
    test_items = items[:limit]
    results = []
    total_time = 0.0

    scores = []
    num_rates = []
    bad_cases = []

    mode_label = "base"
    if use_hybrid and use_cross_encoder:
        mode_label = "hybrid_cross"
    elif use_hybrid:
        mode_label = "hybrid"
    elif use_cross_encoder:
        mode_label = "cross"
    elif use_rerank:
        mode_label = "rerank"

    print(f"\n{'=' * 70}")
    print(f"开始 CRUD_RAG 评测 | 样本数={len(test_items)} | mode={mode_label}")
    print(f"{'=' * 70}")

    for i, item in enumerate(test_items, 1):
        question = item["event"]
        reference = item["summary"]

        start = time.time()
        try:
            prediction = chain.invoke(question)
        except Exception as e:
            # 敏感内容等导致 LLM 失败时，fallback 为检索到的文本摘要
            err_text = str(e)
            if "400" in err_text or "content" in err_text.lower() or "sensitive" in err_text.lower():
                try:
                    docs = retrieve_fn(question)
                    prediction = "参考资料摘要：\n" + "\n".join(
                        f"- {d.page_content[:200]}...（{d.metadata.get('title', '未知')}）" for d in docs[:3]
                    )
                except Exception as e2:
                    prediction = f"[ERROR] {e}; fallback failed: {e2}"
            else:
                prediction = f"[ERROR] {e}"
        elapsed = time.time() - start
        total_time += elapsed

        overlap = keyword_overlap(reference, prediction)
        scores.append(overlap)

        num_info = number_hit_rate(reference, prediction)
        num_rate = None
        missing_nums = []
        if num_info:
            num_rate, _, missing_nums = num_info
            num_rates.append(num_rate)

        mark = "OK" if overlap >= 0.5 else "WARN" if overlap >= 0.3 else "BAD"
        extra = f" num-hit={num_rate * 100:.0f}%" if num_rate is not None else ""
        print(f"[{i:3d}/{len(test_items)}] {mark} overlap={overlap * 100:5.1f}%{extra} time={elapsed:.1f}s | Q: {question[:40]}...")

        record = {
            "id": item.get("ID", i),
            "question": question,
            "reference_answer": reference,
            "rag_answer": prediction,
            "keyword_overlap": round(overlap, 3),
            "response_time_sec": round(elapsed, 2),
        }
        if num_rate is not None:
            record["number_hit_rate"] = round(num_rate, 3)
            record["missing_numbers"] = list(missing_nums)[:5]
        results.append(record)

        # 记录 bad case
        if overlap < 0.5:
            bad_cases.append({
                "id": item.get("ID", i),
                "question": question,
                "reference_answer": reference,
                "rag_answer": prediction,
                "keyword_overlap": round(overlap, 3),
                "missing_numbers": list(missing_nums)[:5],
            })

    # 汇总
    avg_overlap = float(np.mean(scores)) if scores else 0.0
    avg_num_rate = float(np.mean(num_rates)) if num_rates else 0.0
    avg_time = total_time / len(test_items) if test_items else 0.0
    pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores) * 100 if scores else 0.0

    # bad case 分类统计
    bad_categories = {
        "数字缺失": 0,
        "答案不完整": 0,
        "找回错误文档": 0,
        "其他": 0,
    }
    for bc in bad_cases:
        ref_nums = extract_numbers(bc["reference_answer"])
        pred_nums = extract_numbers(bc["rag_answer"])
        if ref_nums and not (ref_nums & pred_nums):
            bad_categories["数字缺失"] += 1
        elif bc["keyword_overlap"] < 0.2:
            bad_categories["找回错误文档"] += 1
        elif bc["keyword_overlap"] < 0.5:
            bad_categories["答案不完整"] += 1
        else:
            bad_categories["其他"] += 1

    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "rerank_candidates": RERANK_CANDIDATES if (use_rerank or use_hybrid or use_cross_encoder) else None,
            "use_rerank": use_rerank,
            "use_hybrid": use_hybrid,
            "use_cross_encoder": use_cross_encoder,
            "cross_encoder_model": CROSS_ENCODER_MODEL if use_cross_encoder else None,
            "test_samples": len(test_items),
        },
        "summary": {
            "avg_keyword_overlap": round(avg_overlap, 3),
            "avg_number_hit_rate": round(avg_num_rate, 3),
            "pass_rate": round(pass_rate, 1),
            "avg_response_time_sec": round(avg_time, 2),
            "bad_case_count": len(bad_cases),
            "bad_case_categories": bad_categories,
        },
        "details": results,
        "bad_cases": bad_cases[:20],  # 只保存前 20 条 bad case
    }

    return report


def print_report(report: dict):
    """打印并保存报告。"""
    s = report["summary"]
    c = report["config"]

    if c.get("use_hybrid") and c.get("use_cross_encoder"):
        mode_label = "hybrid_cross"
    elif c.get("use_hybrid"):
        mode_label = "hybrid"
    elif c.get("use_cross_encoder"):
        mode_label = "cross"
    elif c.get("use_rerank"):
        mode_label = "rerank"
    else:
        mode_label = "base"

    lines = []
    def emit(line=""):
        print(line)
        lines.append(line)

    emit(f"\n{'=' * 70}")
    emit(f"CRUD_RAG Evaluation Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    emit(f"{'=' * 70}")
    emit(f"Config: chunk_size={c['chunk_size']}, top_k={c['top_k']}, mode={mode_label}")
    emit(f"Total samples: {c['test_samples']}")
    emit()
    emit(f"[Overall]")
    emit(f"  平均 keyword overlap: {s['avg_keyword_overlap'] * 100:.1f}%")
    emit(f"  平均 number hit rate: {s['avg_number_hit_rate'] * 100:.1f}%")
    emit(f"  通过率 (overlap>=0.5): {s['pass_rate']:.1f}%")
    emit(f"  平均响应时间:          {s['avg_response_time_sec']:.2f} s/q")
    emit()
    emit(f"[Bad case 分布]")
    emit(f"  总 bad case 数: {s['bad_case_count']}")
    for cat, cnt in s["bad_case_categories"].items():
        emit(f"  - {cat}: {cnt}")
    emit()
    emit(f"[Visualization]")
    val = s["avg_keyword_overlap"] * 100
    bar = "#" * int(val / 5) + "." * (20 - int(val / 5))
    emit(f"  Keyword Overlap {bar} {val:.1f}%")
    val = s["pass_rate"]
    bar = "#" * int(val / 5) + "." * (20 - int(val / 5))
    emit(f"  Pass Rate       {bar} {val:.1f}%")

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"crud_eval_{mode_label}_{timestamp}.json"
    txt_path = RESULTS_DIR / f"crud_eval_{mode_label}_{timestamp}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\n报告已保存：")
    emit(f"  JSON: {json_path}")
    emit(f"  TXT:  {txt_path}")


# ============= 入口 =============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CRUD_RAG RAG evaluation")
    parser.add_argument("--rerank", action="store_true", help="启用关键词重排序优化")
    parser.add_argument("--hybrid", action="store_true", help="启用 BM25 + FAISS 混合检索")
    parser.add_argument("--cross-encoder", action="store_true", help="启用 Cross-Encoder 重排序")
    parser.add_argument("--samples", type=int, default=MAX_TEST_SAMPLES, help=f"评测样本数，默认 {MAX_TEST_SAMPLES}")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Top-K，默认 {TOP_K}")
    args = parser.parse_args()

    if not os.getenv("ZHIPUAI_API_KEY"):
        print("[ERROR] 未配置 ZHIPUAI_API_KEY，请在 .env 中设置")
        sys.exit(1)

    if not CRUD_DATA_PATH.exists():
        print(f"[ERROR] 数据集不存在: {CRUD_DATA_PATH}")
        sys.exit(1)

    items = load_crud_dataset(CRUD_DATA_PATH)
    print(f"[INFO] 加载数据集：共 {len(items)} 条有效 event_summary 记录")

    # 用全部文档建向量库
    chain, retrieve_fn = build_rag(
        items,
        use_rerank=args.rerank,
        use_hybrid=args.hybrid,
        use_cross_encoder=args.cross_encoder,
    )

    # 评测
    report = evaluate(
        items, chain, retrieve_fn,
        use_rerank=args.rerank,
        use_hybrid=args.hybrid,
        use_cross_encoder=args.cross_encoder,
        limit=args.samples,
    )
    print_report(report)
