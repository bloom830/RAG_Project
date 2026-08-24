"""Shared helpers for RAG scripts: native ZhipuAI embeddings and PDF loading with OCR fallback."""
import os
import hashlib
import pickle
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = BASE_DIR / "embedding_cache.pkl"


class ZhipuAIEmbeddings(Embeddings):
    """Native zhipuai SDK embeddings with batching and on-disk cache."""

    def __init__(self, model: str = "embedding-2", api_key: str = None,
                 cache_file=None, use_cache: bool = True):
        self.model = model
        self.client = ZhipuAI(api_key=api_key or os.getenv("ZHIPUAI_API_KEY"))
        self.use_cache = use_cache
        cache_path = cache_file or DEFAULT_CACHE_PATH
        self.cache_path = Path(cache_path) if not isinstance(cache_path, Path) else cache_path
        self._cache = {}
        self._dirty = False
        self._load_cache()

    def _load_cache(self):
        if not self.use_cache:
            return
        try:
            f = self.cache_path
            if not f.exists():
                return
            with open(f, "rb") as fp:
                payload = pickle.load(fp)
            if isinstance(payload, dict) and payload.get("model") == self.model:
                self._cache = payload.get("data", {})
        except Exception:
            pass

    def _save_cache(self):
        if not self.use_cache or not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"model": self.model, "data": self._cache}, f)
        except Exception:
            pass

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _batch_api_call(self, texts: List[str],
                        max_items: int = 64,
                        max_chars: int = 8000) -> List[List[float]]:
        """Call ZhipuAI embeddings API with batching to avoid 1214/1210 errors.

        Recursively splits a batch that exceeds either the item or char limit
        until every sub-batch is safe to send. Single texts longer than
        max_chars are truncated to prevent infinite recursion.
        """
        # 过滤空字符串并截断单条超长文本
        safe_texts = [t.strip()[:max_chars] for t in texts if t and t.strip()]
        if not safe_texts:
            return []

        # 长度对齐：如果原始输入有空字符串，返回占位符保持索引一致
        result_map = {}
        non_empty = []
        for i, t in enumerate(texts):
            if t and t.strip():
                result_map[i] = None
                non_empty.append((i, t.strip()[:max_chars]))

        def _send(batch: List[tuple[int, str]]) -> List[tuple[int, List[float]]]:
            """Recursively shrink batch until it satisfies both limits, then call API."""
            if not batch:
                return []
            total_chars = sum(len(t) for _, t in batch)
            if len(batch) > max_items or total_chars > max_chars:
                mid = max(len(batch) // 2, 1)
                return _send(batch[:mid]) + _send(batch[mid:])
            resp = self.client.embeddings.create(
                model=self.model,
                input=[t for _, t in batch],
            )
            embeddings = [item.embedding for item in resp.data]
            return [(idx, emb) for (idx, _), emb in zip(batch, embeddings)]

        indexed_results = _send(non_empty)
        for idx, emb in indexed_results:
            result_map[idx] = emb

        # 按原始顺序返回
        return [result_map.get(i, [0.0] * 1024) for i in range(len(texts))]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        for i, t in enumerate(texts):
            t = t.replace("\n", " ").strip()
            if not t:
                results.append([0.0] * 1024)
                continue
            h = self._text_hash(t)
            if self.use_cache and h in self._cache:
                results.append(self._cache[h])
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)
                results.append([0.0] * 1024)  # placeholder
        if uncached_texts:
            new_embeddings = self._batch_api_call(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                if self.use_cache:
                    self._cache[self._text_hash(texts[idx].replace("\n", " ").strip())] = emb
            self._dirty = True
            self._save_cache()
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _ensure_tesseract_path():
    """Ensure tesseract binary is on PATH for OCR."""
    import shutil
    import os
    path = shutil.which("tesseract")
    if path and os.path.isfile(path):
        return
    # Try common Windows install paths
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            tess_dir = os.path.dirname(c)
            os.environ["PATH"] = tess_dir + os.pathsep + os.environ.get("PATH", "")
            try:
                import pytesseract
                pytesseract.tesseract_cmd = c
            except ImportError:
                pass
            return


def _ocr_page_with_pypdfium2(file_path: str, page_index: int,
                             tesseract_lang: str = "chi_sim+eng") -> str:
    """Render a single page with pypdfium2 and OCR it with pytesseract."""
    import pypdfium2 as pdfium
    import pytesseract
    pdf = pdfium.PdfDocument(file_path)
    page = pdf[page_index]
    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    text = pytesseract.image_to_string(pil_image, lang=tesseract_lang)
    pdf.close()
    return text


def load_pdf_with_ocr(file_path: str, verbose: bool = False) -> List[Document]:
    """Load a PDF: first try direct text extraction, then OCR fallback for image-only pages."""
    _ensure_tesseract_path()
    docs: List[Document] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                docs.append(Document(page_content=text,
                                     metadata={"source": file_path, "page": i + 1}))
                continue
            # No text → OCR fallback
            try:
                import pytesseract
                import pypdfium2
                langs = pytesseract.get_languages(config="")
                ocr_lang = "chi_sim+eng" if "chi_sim" in langs else "eng"
                if verbose:
                    print(f"  [OCR] page {i+1} (lang={ocr_lang})")
                text = _ocr_page_with_pypdfium2(file_path, i, ocr_lang)
                docs.append(Document(page_content=text.strip(),
                                     metadata={"source": file_path, "page": i + 1}))
            except ImportError:
                # OCR libs missing
                if verbose:
                    print(f"  [WARN] page {i+1} has no text and OCR libs not installed")
            except Exception as e:
                if verbose:
                    print(f"  [WARN] OCR failed on page {i+1}: {e}")
    except Exception as e:
        if verbose:
            print(f"  [ERROR] PDF load failed: {e}")
    return docs


def load_md_file(file_path) -> Document:
    """Read a markdown / text file as a single Document."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            content = f.read().strip()
    except Exception:
        content = ""
    return Document(page_content=content, metadata={"source": str(file_path)})


def load_all_documents(source, verbose: bool = False) -> List[Document]:
    """Load documents from a directory (recursively) or a single file.

    Supports: PDF (with OCR fallback), MD/TXT, DOCX, DOC.
    """
    all_docs: List[Document] = []

    def _is_lock(p: Path) -> bool:
        return p.name.startswith("~$")

    src = Path(source)

    if src.is_file():
        suffix = src.suffix.lower()
        if _is_lock(src):
            return []
        if verbose:
            print(f"  File: {src.name}")
        if suffix == ".pdf":
            try:
                pages = load_pdf_with_ocr(str(src), verbose=verbose)
                for d in pages:
                    d.metadata["filename"] = src.name
                all_docs.extend(pages)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")
        elif suffix in (".docx", ".doc"):
            try:
                docs = UnstructuredWordDocumentLoader(str(src)).load()
                for d in docs:
                    d.metadata["filename"] = src.name
                all_docs.extend(docs)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")
        elif suffix in (".md", ".txt"):
            doc = load_md_file(src)
            doc.metadata["filename"] = src.name
            all_docs.append(doc)
        return all_docs

    if src.is_dir():
        for pdf_path in sorted(src.rglob("*.pdf")):
            if _is_lock(pdf_path):
                continue
            if verbose:
                print(f"  PDF: {pdf_path.name}")
            try:
                pages = load_pdf_with_ocr(str(pdf_path), verbose=verbose)
                for d in pages:
                    d.metadata["filename"] = pdf_path.name
                all_docs.extend(pages)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")
        for docx_path in sorted(src.rglob("*.docx")):
            if _is_lock(docx_path):
                continue
            if verbose:
                print(f"  DOCX: {docx_path.name}")
            try:
                docs = UnstructuredWordDocumentLoader(str(docx_path)).load()
                for d in docs:
                    d.metadata["filename"] = docx_path.name
                all_docs.extend(docs)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")
        for doc_path in sorted(src.rglob("*.doc")):
            if _is_lock(doc_path):
                continue
            if verbose:
                print(f"  DOC: {doc_path.name}")
            try:
                docs = UnstructuredWordDocumentLoader(str(doc_path)).load()
                for d in docs:
                    d.metadata["filename"] = doc_path.name
                all_docs.extend(docs)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")
        # MD/TXT
        md_count = 0
        for md_path in sorted(src.rglob("*.md")) + sorted(src.rglob("*.txt")):
            if _is_lock(md_path):
                continue
            if verbose and md_path.suffix.lower() == ".md":
                print(f"  MD: {md_path.name}")
                md_count += 1
            doc = load_md_file(md_path)
            doc.metadata["filename"] = md_path.name
            # Add relative path source for traceability
            try:
                doc.metadata["source"] = str(md_path.relative_to(src))
            except Exception:
                doc.metadata["source"] = str(md_path)
            all_docs.append(doc)
        if verbose and md_count == 0:
            # quick sanity check
            pass
    return all_docs
