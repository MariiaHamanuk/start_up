import os
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: str) -> List[Document]:
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "page": i + 1},
            ))
    return docs


def load_text(path: str) -> List[Document]:
    text = Path(path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path})]


def load_url(url: str) -> List[Document]:
    headers = {"User-Agent": os.getenv("USER_AGENT", "startup-postmortem-rag/1.0")}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url})]


def load_document(source: str) -> List[Document]:
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    path = Path(source)
    if path.suffix.lower() == ".pdf":
        return load_pdf(source)
    return load_text(source)
