from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings

EMBEDDING_DIM = 768

PROMPT = ChatPromptTemplate.from_template(
    """You are an expert analyst of startup post-mortems and failure stories.
Answer the question using only the context below.
Always cite the specific startup or source you are drawing from.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.gemini_api_key,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    needs_recreate = False

    if settings.collection_name in existing and settings.use_hybrid_search:
        col_info = client.get_collection(settings.collection_name)
        vectors_cfg = col_info.config.params.vectors
        has_named_dense = isinstance(vectors_cfg, dict) and "dense" in vectors_cfg
        has_sparse = bool(getattr(col_info.config.params, "sparse_vectors_config", None))
        if not (has_named_dense and has_sparse):
            print(f"Recreating '{settings.collection_name}' to support hybrid search (re-ingest your documents).")
            client.delete_collection(settings.collection_name)
            needs_recreate = True

    if settings.collection_name not in existing or needs_recreate:
        if settings.use_hybrid_search:
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config={"dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
            )
        else:
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )


def get_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection(client)

    if settings.use_hybrid_search:
        from langchain_qdrant import FastEmbedSparse, RetrievalMode
        return QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=get_embeddings(),
            sparse_embedding=FastEmbedSparse(model_name=settings.sparse_model),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )


def build_qdrant_filter(
    industry: str | None = None,
    year: int | None = None,
    stage: str | None = None,
    startup_name: str | None = None,
) -> Filter | None:
    conditions = []
    if industry:
        conditions.append(FieldCondition(key="metadata.industry", match=MatchValue(value=industry.lower())))
    if year:
        conditions.append(FieldCondition(key="metadata.year", range=Range(gte=year, lte=year)))
    if stage:
        conditions.append(FieldCondition(key="metadata.stage", match=MatchValue(value=stage.lower())))
    if startup_name:
        conditions.append(FieldCondition(key="metadata.startup_name", match=MatchValue(value=startup_name.lower())))
    return Filter(must=conditions) if conditions else None


def get_retriever(top_k: int | None = None, qdrant_filter: Filter | None = None):
    search_kwargs: dict = {"k": top_k or settings.top_k}
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter
    return get_vector_store().as_retriever(search_kwargs=search_kwargs)


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')} | "
        f"Startup: {doc.metadata.get('startup_name', 'N/A')} | "
        f"Industry: {doc.metadata.get('industry', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
    )


def retrieve_and_rerank(
    question: str,
    top_k: int | None = None,
    qdrant_filter: Filter | None = None,
) -> List[Document]:
    retriever = get_retriever(top_k=top_k, qdrant_filter=qdrant_filter)
    docs = retriever.invoke(question)
    if settings.use_reranker and docs:
        from app.retrieval.reranker import get_reranker
        docs = get_reranker(settings.reranker_model).rerank(question, docs, top_k=settings.rerank_top_k)
    return docs


def build_chain(qdrant_filter: Filter | None = None):
    retriever = get_retriever(qdrant_filter=qdrant_filter)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain, retriever
