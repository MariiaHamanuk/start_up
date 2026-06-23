import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.retrieval.chain import (
    PROMPT,
    build_qdrant_filter,
    format_docs,
    get_llm,
    get_vector_store,
    retrieve_and_rerank,
)

router = APIRouter()

DATA_DIR = Path("data/postmortems")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class IngestMetadata(BaseModel):
    startup_name: str | None = None
    industry: str | None = None
    year: int | None = None
    stage: str | None = None


class URLIngestRequest(BaseModel):
    url: str
    metadata: IngestMetadata = IngestMetadata()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filter_industry: str | None = None
    filter_year: int | None = None
    filter_stage: str | None = None
    filter_startup: str | None = None
    use_reranker: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


def _apply_metadata(docs, meta: IngestMetadata):
    for doc in docs:
        if meta.startup_name:
            doc.metadata["startup_name"] = meta.startup_name.lower()
        if meta.industry:
            doc.metadata["industry"] = meta.industry.lower()
        if meta.year:
            doc.metadata["year"] = meta.year
        if meta.stage:
            doc.metadata["stage"] = meta.stage.lower()
    return docs


@router.post("/ingest/upload")
async def ingest_file(
    file: UploadFile = File(...),
    startup_name: str | None = None,
    industry: str | None = None,
    year: int | None = None,
    stage: str | None = None,
):
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = load_document(str(file_path))
        docs = _apply_metadata(docs, IngestMetadata(startup_name=startup_name, industry=industry, year=year, stage=stage))
        chunks = chunk_documents(docs)
        get_vector_store().add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Ingested {len(chunks)} chunks from {file.filename}"}


@router.post("/ingest/url")
async def ingest_url(request: URLIngestRequest):
    try:
        docs = load_document(request.url)
        docs = _apply_metadata(docs, request.metadata)
        chunks = chunk_documents(docs)
        get_vector_store().add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Ingested {len(chunks)} chunks from {request.url}"}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        qdrant_filter = build_qdrant_filter(
            industry=request.filter_industry,
            year=request.filter_year,
            stage=request.filter_stage,
            startup_name=request.filter_startup,
        )

        docs = retrieve_and_rerank(
            question=request.question,
            top_k=request.top_k,
            qdrant_filter=qdrant_filter,
        )

        answer = (PROMPT | get_llm() | StrOutputParser()).invoke({
            "context": format_docs(docs),
            "question": request.question,
        })

        sources = list({doc.metadata.get("source", "Unknown") for doc in docs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=sources, chunks_used=len(docs))
