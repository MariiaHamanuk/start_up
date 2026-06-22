import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.retrieval.chain import get_chain, get_vector_store

router = APIRouter()

DATA_DIR = Path("data/postmortems")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class URLRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/ingest/upload")
async def ingest_file(file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = load_document(str(file_path))
        chunks = chunk_documents(docs)
        get_vector_store().add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Ingested {len(chunks)} chunks from {file.filename}"}


@router.post("/ingest/url")
async def ingest_url(request: URLRequest):
    try:
        docs = load_document(request.url)
        chunks = chunk_documents(docs)
        get_vector_store().add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Ingested {len(chunks)} chunks from {request.url}"}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        chain, retriever = get_chain()
        docs = retriever.invoke(request.question)
        sources = list({doc.metadata.get("source", "Unknown") for doc in docs})
        answer = chain.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=sources)
