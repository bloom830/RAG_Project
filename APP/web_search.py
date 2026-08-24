"""
联网搜索增强模块
职责：
  - 当 RAG 检索结果质量差或为空时，自动触发联网搜索兜底
  - 支持两种模式：
    1. DuckDuckGo（免费，无需 API Key，默认）
    2. SerpAPI（需 SERPAPI_KEY 环境变量）
  - 搜索结果注入 prompt 作为补充上下文

依赖（可选）：
  pip install duckduckgo-search   # 免费方案
  pip install google-search-results  # SerpAPI 方案
"""

import os
import warnings
from typing import Literal


# ============= 配置 =============

# 当 RAG 检索到的文档平均分低于此阈值时触发联网
RELEVANCE_THRESHOLD: float = 0.3

# 最大搜索结果数
MAX_SEARCH_RESULTS: int = 5

# 搜索策略
SearchProvider = Literal["duckduckgo", "serpapi", "none"]

_provider: SearchProvider = "none"
_search_client = None

# 自动检测可用引擎
if os.getenv("SERPAPI_KEY"):
    _provider = "serpapi"
else:
    # 优先使用新的 ddgs 包（duckduckgo_search 已弃用）
    try:
        from ddgs import DDGS as _DDGS
        _provider = "duckduckgo"
        _search_client = _DDGS
        _search_client_name = "ddgs"
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS
            _provider = "duckduckgo"
            _search_client = _DDGS
            _search_client_name = "duckduckgo_search"
        except ImportError:
            _provider = "none"
            _search_client = None
            _search_client_name = "none"


def get_provider() -> SearchProvider:
    return _provider


def _search_duckduckgo(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """使用 DuckDuckGo 免费搜索（兼容 ddgs 9.x 和 duckduckgo_search 8.x）"""
    if _search_client is None:
        return []
    try:
        # ddgs 9.x 推荐作为上下文管理器使用；老版 duckduckgo_search 也支持 with
        with _search_client() as ddgs:
            raw_results = ddgs.text(
                query,
                max_results=max_results,
                region="cn-zh" if any('\u4e00' <= c <= '\u9fff' for c in query) else "wt-wt",
            )
            results = list(raw_results)
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": (r.get("body", "") or "")[:300],
            })
        return formatted
    except Exception as e:
        warnings.warn(f"DuckDuckGo 搜索失败: {e}")
        return []


def _search_serpapi(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """使用 SerpAPI 搜索"""
    try:
        from serpapi import GoogleSearch
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": max_results,
            "hl": "zh-cn",
        }
        results = GoogleSearch(params).get_dict()
        formatted = []
        for r in results.get("organic_results", [])[:max_results]:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", "")[:300],
            })
        return formatted
    except Exception as e:
        warnings.warn(f"SerpAPI 搜索失败: {e}")
        return []


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    统一搜索入口
    返回 [{"title": str, "url": str, "snippet": str}, ...]
    """
    if _provider == "duckduckgo":
        return _search_duckduckgo(query, max_results)
    elif _provider == "serpapi":
        return _search_serpapi(query, max_results)
    else:
        warnings.warn("未配置联网搜索，请安装 duckduckgo-search 或配置 SERPAPI_KEY")
        return []


def format_search_results(results: list[dict]) -> str:
    """将搜索结果格式化为可注入 prompt 的文本"""
    if not results:
        return ""
    lines = ["【联网搜索补充信息】"]
    for i, r in enumerate(results, 1):
        lines.append(f"[网络来源 {i}] {r['title']}")
        lines.append(f"  URL: {r['url']}")
        lines.append(f"  {r['snippet']}")
    return "\n".join(lines)


def should_trigger_search(sources: list[dict], answer: str) -> bool:
    """
    判断是否应该触发联网搜索兜底
    触发条件：
    1. RAG 未检索到任何文档
    2. 答案包含"未找到""未提及"等拒答信号
    3. 检索到的文档数 < 2
    """
    if not sources or len(sources) == 0:
        return True

    refusal_signals = [
        "未找到", "未提及", "未涉及", "未包含",
        "无法回答", "没有找到", "cannot answer", "not found",
        "参考资料中未", "不在参考资料",
    ]
    answer_lower = answer.lower()
    for sig in refusal_signals:
        if sig in answer_lower:
            return True

    return False


def build_search_augmented_prompt(
    original_prompt: str,
    question: str,
    search_results: list[dict],
) -> str:
    """将搜索结果注入 prompt"""
    search_text = format_search_results(search_results)
    if not search_text:
        return original_prompt

    # 在"参考资料"区块后追加搜索结果
    return original_prompt + f"\n\n{search_text}\n\n请结合以上联网信息补充回答，并标注网络来源 URL。"


# ============= 侧边栏状态展示 =============

def render_search_status():
    """在侧边栏展示联网搜索状态"""
    import streamlit as st

    st.markdown("---")
    st.markdown("### 🌐 联网搜索")

    provider = get_provider()
    if provider == "duckduckgo":
        st.success(f"✅ { _search_client_name } 已就绪", icon="🔍")
    elif provider == "serpapi":
        st.success("✅ SerpAPI 已配置", icon="🔍")
    else:
        st.warning("⚠️ 未配置联网搜索", icon="🌐")
        st.caption("安装 ddgs 即可免费启用")
        if st.button("📦 安装指令", key="show_install_search", use_container_width=True):
            st.code("pip install ddgs")
