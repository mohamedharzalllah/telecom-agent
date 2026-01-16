from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.setup import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PDF_DIR,
    VECTOR_DB_PATH,
    _ensure_api_key,
)


def _persisted_index_exists(path: Path) -> bool:
    """Return True if the FAISS index is already stored on disk."""
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def _build_and_save_vector_store(base_path: Path) -> FAISS:
    """Build the vector store from PDFs and persist it to disk."""
    api_key = _ensure_api_key()

    loader = PyPDFDirectoryLoader(PDF_DIR)
    docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    base_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(base_path))
    return vector_store


def _load_vector_store(base_path: Path) -> FAISS:
    """Load a persisted FAISS vector store from disk."""
    api_key = _ensure_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return FAISS.load_local(str(base_path), embeddings, allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)  # pyright: ignore[reportUndefinedVariable]
def get_vector_store() -> FAISS:
    """
    Load the FAISS vector store from disk if present, otherwise build and persist it.

    Cached in-process to avoid repeated loads within the same runtime.
    """
    base_path = VECTOR_DB_PATH
    if _persisted_index_exists(base_path):
        return _load_vector_store(base_path)
    return _build_and_save_vector_store(base_path)
