# 智策通 · 政策智能问答助手

基于 RAG（检索增强生成）的政策与行业分析智能问答系统，使用 Streamlit 构建 Web 界面。
![智策通系统主界面](images/screenshot_main.png)

## 功能特性

- **RAG 问答**：基于本地知识库（PDF / MD / TXT）检索增强生成
- **混合检索**：FAISS 语义检索 + BM25 关键词检索，RRF 融合 + Cross-Encoder 精排
- **多轮对话**：自动携带上下文，支持追问
- **会话管理**：新建、切换、删除，本地 JSON 持久化
- **反馈系统**：对每条回答点赞 / 点踩，数据存入 `feedback/`
- **转人工客服**：侧边栏提供二维码或联系方式入口
- **联网搜索兜底（可选）**：DuckDuckGo 或 SerpAPI
- **友好 UI**：Streamlit 构建，响应式布局

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| 大模型 | 智谱 GLM-4-Flash |
| 嵌入模型 | 智谱 embedding-2 |
| 向量库 | FAISS |
| 关键词检索 | BM25 |
| 精排模型 | BAAI/bge-reranker-base |
| 文档加载 | PyPDFLoader / TextLoader |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/bloom830/RAG_Project.git
cd RAG_Project
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install streamlit langchain langchain-community langchain-core langchain-text-splitters faiss-cpu zhipuai python-dotenv pypdf pypdfium2 pytesseract transformers torch
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key
# 可选
# SERPAPI_KEY=your_serpapi_key
# TRANSFER_CONTACT=微信: xxx
```

### 4. 准备知识库

修改 `APP/rag_engine.py` 中的 `DATA_SOURCES` 列表，指向你的文档目录（支持 PDF、MD、TXT）。

### 5. 启动应用

```bash
streamlit run APP/app.py
```

浏览器打开 `http://localhost:8502`。

## 项目结构

```
RAG_Project/
├── APP/
│   ├── app.py                 # 主入口
│   ├── chat_utils.py          # 闲聊检测、欢迎语
│   ├── feedback.py            # 点赞/点踩反馈
│   ├── rag_engine.py          # RAG 核心（加载、检索、Chain）
│   ├── rag_utils.py           # 嵌入、OCR 工具
│   ├── session_manager.py     # 会话 CRUD
│   ├── style.py               # CSS 样式
│   ├── transfer_service.py    # 转人工
│   └── web_search.py          # 联网搜索
├── sessions/                  # 会话数据（自动生成）
├── feedback/                  # 反馈数据（自动生成）
├── resources/                 # 客服二维码等资源
├── initial_test/              # 测试与调优脚本
└── .env                       # 环境变量（需自行创建）
```

## 测试数据集

本项目的检索方案评测基于 **CRUD-RAG**——由 IAAR-Shanghai 发布的、面向大模型检索增强生成（RAG）的中文综合评测基准。该基准提供了用于评估 RAG 系统检索与生成质量的配套数据集，以及在其上运行实验的教程。

- **来源**：[IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)
- **用途**：作为本项目的公开评测集，用于验证混合检索方案在中文政策问答场景下的关键词命中、数字准确率与问答通过率等指标。
- **使用方式**：按其官方说明准备数据集与嵌入模型（如 `sentence-transformers/bge-base-zh-v1.5`），并结合本仓库 `initial_test/` 下的评测脚本运行。

> 如使用该系统数据集进行研究或二次开发，请遵循其官方仓库中的许可证与引用要求。

## 运行测试

```bash
cd initial_test
python crud_eval.py
python tune_weights.py
```

## 许可证

MIT License
