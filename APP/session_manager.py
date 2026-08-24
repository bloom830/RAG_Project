"""
会话持久化管理模块
职责：会话的增删查改、持久化到本地 JSON 文件
"""

import json
import uuid
from pathlib import Path
from datetime import datetime


def get_sessions_dir() -> Path:
    """获取会话目录，延迟创建"""
    d = Path(__file__).resolve().parent.parent / "sessions"
    d.mkdir(exist_ok=True)
    return d


# ============= 基础 CRUD =============

def list_sessions() -> list[dict]:
    """列出所有会话，按最近修改时间倒序"""
    sessions_dir = get_sessions_dir()
    sessions = []
    for f in sessions_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "id": f.stem,
                "title": data.get("title", "新对话"),
                "created_at": data.get("created_at", ""),
                "message_count": len(data.get("messages", [])),
                "updated_at": f.stat().st_mtime,
            })
        except Exception:
            continue
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def load_session(session_id: str) -> dict | None:
    """加载指定会话的完整数据"""
    path = get_sessions_dir() / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_session(session_id: str, title: str, messages: list, created_at: str):
    """保存会话到本地文件"""
    path = get_sessions_dir() / f"{session_id}.json"
    data = {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "messages": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_session(session_id: str):
    """删除会话文件"""
    path = get_sessions_dir() / f"{session_id}.json"
    if path.exists():
        path.unlink()


def generate_session_title(messages: list) -> str:
    """根据第一条用户消息生成会话标题"""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
            return text[:20] + ("..." if len(text) > 20 else "")
    return "新对话"


# ============= 会话切换辅助 =============

def switch_to_session(session_id: str) -> dict | None:
    """加载目标会话数据，返回 (messages, created_at) 或 None"""
    data = load_session(session_id)
    if data:
        return {
            "messages": data.get("messages", []),
            "created_at": data.get("created_at", datetime.now().isoformat()),
        }
    return None


def create_new_session() -> dict:
    """创建新会话，返回初始 session_state 信息"""
    return {
        "session_id": str(uuid.uuid4())[:8],
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
