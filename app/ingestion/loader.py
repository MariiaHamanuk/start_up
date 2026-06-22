from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document


def load_pdf(path: str) -> List[Document]:
    return PyPDFLoader(path).load()


def load_text(path: str) -> List[Document]:
    return TextLoader(path, encoding="utf-8").load()


def load_url(url: str) -> List[Document]:
    return WebBaseLoader(url).load()


def load_document(source: str) -> List[Document]:
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    path = Path(source)
    if path.suffix.lower() == ".pdf":
        return load_pdf(source)
    return load_text(source)
