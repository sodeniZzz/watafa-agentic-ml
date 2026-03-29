"""RAG: FAISS vector store over knowledge/ directory."""

import hashlib
import json
import logging
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.io_utils import ROOT_PATH
from src.utils.llm_utils import get_embedding_model_name, get_openrouter_api_key, get_url

logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)

KNOWLEDGE_DIR = ROOT_PATH / "knowledge"
CACHE_DIR = KNOWLEDGE_DIR / ".faiss_cache"
HASH_FILE = CACHE_DIR / "files_hash.txt"


def _knowledge_files():
    """List .md and .ipynb files in knowledge/."""
    return [p for p in sorted(KNOWLEDGE_DIR.iterdir())
            if p.is_file() and p.suffix in (".md", ".ipynb")]


def _compute_hash() -> str:
    """SHA256 over sorted (name, size, mtime) of knowledge files."""
    entries = []
    for p in _knowledge_files():
        s = p.stat()
        entries.append(f"{p.name}:{s.st_size}:{s.st_mtime_ns}")
    return hashlib.sha256("|".join(entries).encode()).hexdigest()


def _load_documents():
    """Read .md and .ipynb files from knowledge/ into text chunks."""
    texts = []
    for p in _knowledge_files():
        if p.suffix == ".md":
            texts.append(p.read_text(encoding="utf-8"))
        elif p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            for cell in nb.get("cells", []):
                if cell["cell_type"] in ("markdown", "code"):
                    texts.append("".join(cell["source"]))
    return texts


@lru_cache(maxsize=1)
def _get_store():
    """Build or load FAISS store. Returns None if knowledge/ is empty."""
    if not _knowledge_files():
        logger.info("No knowledge files found, RAG disabled")
        return None

    embeddings = OpenAIEmbeddings(
        model=get_embedding_model_name(),
        api_key=get_openrouter_api_key(),
        base_url=get_url(),
    )

    current_hash = _compute_hash()

    if CACHE_DIR.exists() and HASH_FILE.exists():
        if HASH_FILE.read_text().strip() == current_hash:
            logger.info("RAG store loaded from cache")
            return FAISS.load_local(
                str(CACHE_DIR), embeddings, allow_dangerous_deserialization=True
            )
        logger.info("Knowledge files changed, rebuilding RAG store")
    else:
        logger.info("No RAG cache found, building store")

    raw_texts = _load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.create_documents(raw_texts)

    store = FAISS.from_documents(chunks, embeddings)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(CACHE_DIR))
    HASH_FILE.write_text(current_hash)
    logger.info("RAG store built and cached (%d chunks)", len(chunks))
    return store


def init_store():
    """Pre-build the vector store (call before graph execution)."""
    _get_store()


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve top_k relevant chunks from the knowledge base."""
    store = _get_store()
    if store is None:
        return ""
    docs = store.similarity_search(query, k=top_k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
