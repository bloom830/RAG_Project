"""
样式常量 — 供 app.py 注入
"""

CSS = """
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
    /* 会话列表样式 */
    .session-item {
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .session-item:hover {
        background-color: #f0f0f0;
    }
    .session-item.active {
        background-color: #e3f2fd;
        border-left: 3px solid #1f77b4;
    }
    .session-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #333;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .session-meta {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.15rem;
    }
    /* 用户消息气泡 */
    .user-bubble {
        background: #1f77b4;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.3rem 0 0.3rem auto;
        max-width: 80%;
        text-align: right;
        line-height: 1.5;
    }
    /* AI 消息气泡 */
    .ai-bubble {
        background: #f0f2f6;
        color: #333;
        padding: 0.6rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.3rem auto 0.3rem 0;
        max-width: 80%;
        line-height: 1.6;
    }
    /* 分隔线 */
    .chat-divider {
        text-align: center;
        color: #aaa;
        font-size: 0.75rem;
        margin: 0.5rem 0;
        padding: 0.2rem 0;
    }
    /* 浅蓝色按钮 - 覆盖"新建对话" secondary 样式 */
    button[kind="secondary"]:has(p:contains("新建对话")),
    button[data-testid="baseButton-secondary"] p:contains("新建对话") {
        color: #1f77b4 !important;
    }
    .stButton > button:has(> div > p:contains("新建对话")) {
        background: linear-gradient(135deg, #e3f2fd 0%, #e8f4f8 100%) !important;
        border: 1px solid #bbdefb !important;
        color: #1f77b4 !important;
        font-weight: 500 !important;
    }
    .stButton > button:has(> div > p:contains("新建对话")):hover {
        background: linear-gradient(135deg, #bbdefb 0%, #e3f2fd 100%) !important;
        border-color: #90caf9 !important;
    }
</style>
"""
