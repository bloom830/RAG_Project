"""
聊天工具模块
职责：闲聊/问候检测、闲聊回复
"""

GREETINGS = {
    "你好", "您好", "hi", "hello", "hey", "哈喽", "嗨",
    "在吗", "在么", "在不在", "你是谁", "你叫什么", "你能做什么",
    "早上好", "下午好", "晚上好", "早安", "晚安",
    "thanks", "thank you", "谢谢",
}


def is_greeting(text: str) -> bool:
    """判断是否为闲聊/问候，无需走 RAG"""
    t = text.strip().lower().rstrip("!.?。！？~,， ")
    if t in GREETINGS:
        return True
    if len(t) <= 4 and any(g in t for g in GREETINGS):
        return True
    return False


def greeting_reply() -> str:
    """闲聊场景的固定回复"""
    return (
        "你好！我是**智策通**，专注于中国 AI 产业、政策、企业相关问题的政策智能问答助手。\n\n"
        "我可以帮你：\n"
        "- 查找政策原文与数据（如 OPC 政策、税收优惠）\n"
        "- 解读行业报告（AI 红包大战、Seedance 2.0、AI 学习机等）\n"
        "- 回答事实型 / 数据型 / 推理型问题\n"
        "- 支持多轮追问，上下文自动记忆\n\n"
        "你可以在下方点击快捷问题开始体验，或直接输入你的问题 🙂"
    )


def build_welcome_html() -> str:
    """开屏欢迎语 HTML"""
    return """
    <div class="welcome-box">
        <h3>👋 你好！我是智策通</h3>
        <p>专注于 <b>AI 产业 / 政策 / 企业</b> 领域的智能问答助手。</p>
        <p>📚 我能帮你：</p>
        <ul>
            <li>查询最新 AI 政策与行业动态</li>
            <li>解读具体政策的核心条款与数字</li>
            <li>对比不同时间段的市场趋势</li>
            <li>严格基于参考资料回答，<b>不编造</b></li>
            <li>💬 支持多轮追问，上下文自动记忆</li>
            <li>🌐 知识库未覆盖时自动联网搜索补充</li>
            <li>👍👎 对回答点赞/点踩帮助我改进</li>
        </ul>
        <p>👇 点下方示例问题开始，或直接输入你的问题</p>
    </div>
    """


def format_source_card(i: int, src: dict) -> str:
    """格式化单个来源卡片 HTML"""
    return f"""
    <div class="source-card">
        <b>来源 {i}</b>: {src['filename']}<br>
        📅 日期: {src['date']}<br>
        📝 摘要: {src['preview']}
    </div>
    """
