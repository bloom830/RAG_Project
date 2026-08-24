"""
点赞 / 点踩反馈模块
职责：
  - 在每条 AI 回答下方渲染 👍👎 按钮
  - 记录反馈到本地 feedback/ 目录（JSONL 格式）
  - 提供查看反馈摘要的辅助函数

反馈文件格式：feedback/feedback_YYYYMMDD.jsonl
每条记录：{"timestamp": "...", "session_id": "...", "question": "...", "answer": "...", "rating": "like"|"dislike", "sources": [...]}
"""

import json
import uuid
from pathlib import Path
from datetime import datetime


def _feedback_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "feedback"
    d.mkdir(exist_ok=True)
    return d


def _feedback_path() -> Path:
    today = datetime.now().strftime("%Y%m%d")
    return _feedback_dir() / f"feedback_{today}.jsonl"


def render_feedback_buttons(msg_index: int, message: dict):
    """
    在 AI 消息下方渲染点赞 / 点踩按钮
    msg_index: st.session_state.messages 中的索引，用于去重
    message: 当前 assistant 消息字典
    """
    import streamlit as st

    # 检查是否已反馈
    feedback_key = f"feedback_{msg_index}"
    if feedback_key in st.session_state:
        status = st.session_state[feedback_key]
        icon = "✅ 已赞" if status == "like" else "✅ 已踩"
        st.caption(f"{icon}  感谢你的反馈！")
        return

    col_like, col_dislike, col_fill = st.columns([1, 1, 6])
    with col_like:
        if st.button("👍", key=f"like_{msg_index}", help="回答有帮助"):
            _save_feedback(msg_index, message, "like")
            st.session_state[feedback_key] = "like"
            st.toast("感谢你的点赞 🙏", icon="👍")
            st.rerun()
    with col_dislike:
        if st.button("👎", key=f"dislike_{msg_index}", help="回答质量差"):
            _save_feedback(msg_index, message, "dislike")
            st.session_state[feedback_key] = "dislike"
            st.toast("感谢反馈，我们会改进 🛠️", icon="👎")
            st.rerun()


def _save_feedback(msg_index: int, message: dict, rating: str):
    """持久化反馈记录"""
    import streamlit as st

    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state.get("current_session_id", "unknown"),
        "msg_index": msg_index,
        "question": "",
        "answer": message.get("content", "")[:500],
        "rating": rating,
        "sources": message.get("sources", []),
    }

    # 尝试找到对应的 user question
    msgs = st.session_state.messages
    if msg_index > 0 and msgs[msg_index - 1].get("role") == "user":
        record["question"] = msgs[msg_index - 1].get("content", "")[:300]

    with open(_feedback_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_feedback_summary(days: int = 7) -> dict:
    """
    汇总最近 N 天的反馈统计
    返回 {"total": N, "likes": N, "dislikes": N, "like_rate": float, "recent_dislikes": [list of records]}
    """
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=days)
    total, likes, dislikes = 0, 0, 0
    recent_dislikes = []

    for f in sorted(_feedback_dir().glob("feedback_*.jsonl")):
        try:
            file_date_str = f.stem.split("_")[1]
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            if file_date < cutoff:
                continue
        except (IndexError, ValueError):
            continue

        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                total += 1
                if rec.get("rating") == "like":
                    likes += 1
                else:
                    dislikes += 1
                    recent_dislikes.append(rec)
            except json.JSONDecodeError:
                continue

    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "like_rate": likes / total * 100 if total > 0 else 0,
        "recent_dislikes": recent_dislikes[:20],
    }
