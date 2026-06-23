from typing import List

from langchain_core.documents import Document

_reranker = None


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


def get_reranker(model_name: str) -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name)
    return _reranker
