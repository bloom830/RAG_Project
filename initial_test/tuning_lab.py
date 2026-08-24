"""
RAG Tuning Lab
Usage: python tuning_lab.py
Features: all key parameters are variables for easy A/B testing.
"""
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag_utils import ZhipuAIEmbeddings, load_all_documents, BASE_DIR

# === Data source configuration ===
# Point to a directory (recursively loads PDF/DOCX/MD) or a single file.
# Examples:
#   DATA_SOURCE = BASE_DIR / "Resources" / "openaxo-main"            # all markdown files
#   DATA_SOURCE = BASE_DIR / "Resources" / "openaxo-main" / "2026"   # only 2026
#   DATA_SOURCE = BASE_DIR / "openaxo-main" / "2026" / "03"  # March 2026 only
DATA_SOURCE = BASE_DIR / "openaxo-main" / "2026" / "03"  # March 2026 only
# ===========================

print(f"[INFO] API Key: {'OK' if os.getenv('ZHIPUAI_API_KEY') else 'MISSING'}")
print(f"[INFO] Data source: {DATA_SOURCE}")


# ==================== Tunable parameters ====================
# You only need to change this section!

CHUNK_SIZE = 200          # chunk size (try 200 / 500 / 2000)
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K = 5                 # number of retrieved chunks (try 1 / 3 / 10)

# Three prompt styles (uncomment one)
PROMPT_STYLE = "strict"   # concise / detailed / strict
# ============================================================


PROMPTS = {
    "concise": """You are an enterprise knowledge base assistant. Answer the user's question based on the provided reference materials.
If the answer is not in the reference materials, say you don't know instead of making things up.
Keep the answer concise and clear.

Reference materials:
{context}

User question: {question}""",

    "detailed": """You are a senior enterprise knowledge base analyst. Based on the reference materials below, provide a thorough, professional answer.
Requirements:
1. Fully integrate all relevant information from the reference materials
2. Organize the answer in a structured way (bullet points are fine)
3. If the reference materials contain specific data or cases, cite them
4. If the reference materials are insufficient, clearly state which aspects were not found

Reference materials:
{context}

User question: {question}""",

    "strict": """You are an enterprise knowledge base assistant. Answer STRICTLY according to these rules:
1. Only answer based on the reference materials; do not add information from outside
2. If the answer is in the reference materials, quote the relevant sentences verbatim
3. If the answer is not in the reference materials, you must answer "The reference materials do not contain relevant information"
4. Do not make any inferences or assumptions

Reference materials:
{context}

User question: {question}""",
}


def build_rag_chain(chunks):
    """Build the RAG chain."""
    print(f"\n[INFO] Building embeddings ({len(chunks)} chunks)...")
    embeddings = ZhipuAIEmbeddings(model="embedding-2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatZhipuAI(model="glm-4-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPTS[PROMPT_STYLE])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return retriever, prompt, llm, format_docs


def ask_question(rag_components, question, show_chunks=False):
    """Ask a question."""
    retriever, prompt, llm, format_docs = rag_components

    if show_chunks:
        print(f"\n{'='*60}")
        print(f"Retrieved {TOP_K} chunks (chunk_size={CHUNK_SIZE}):")
        print(f"{'='*60}")
        retrieved = retriever.invoke(question)
        for i, doc in enumerate(retrieved, 1):
            src = doc.metadata.get('source', '?').split('/')[-1].split('\\')[-1]
            page = doc.metadata.get('page', '?')
            print(f"\n[Chunk {i}] (source: {src}, page: {page})")
            print(f"{'-'*40}")
            print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
        print(f"{'='*60}")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print(f"\n[Answer]:")
    print(f"{'-'*60}")
    answer = rag_chain.invoke(question)
    print(answer)
    print(f"{'-'*60}")
    return answer


# ==================== Main ====================
if __name__ == "__main__":
    print(f"\nCurrent configuration:")
    print(f"   chunk_size = {CHUNK_SIZE}")
    print(f"   chunk_overlap = {CHUNK_OVERLAP}")
    print(f"   top_k = {TOP_K}")
    print(f"   prompt_style = {PROMPT_STYLE}")

    print(f"\nLoading documents...")
    docs = load_all_documents(DATA_SOURCE)
    if not docs:
        print(f"[ERROR] No documents loaded. Check the path: {DATA_SOURCE}")
        exit(1)
    print(f"   Total: {len(docs)} pages")

    print(f"\nSplitting (chunk_size={CHUNK_SIZE})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks")

    if not chunks:
        print(f"[ERROR] No chunks after splitting. Documents may be empty or unreadable.")
        exit(1)

    rag_components = build_rag_chain(chunks)

    print(f"\n{'='*60}")
    print(f"RAG Tuning Lab is ready!")
    print(f"{'='*60}")
    print(f"Commands:")
    print(f"   - Type a question directly -> get an answer")
    print(f"   - Type 'show <question>'  -> show retrieved chunks + answer")
    print(f"   - Type 'quit' / 'exit'    -> exit")
    print()

    while True:
        user_input = input("Your question (or 'show <question>' to inspect retrieval): ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "quit()", "退出"):
            print("Goodbye!")
            break

        show_chunks = False
        if user_input.startswith("show "):
            show_chunks = True
            user_input = user_input[5:].strip()

        ask_question(rag_components, user_input, show_chunks=show_chunks)
        print()
