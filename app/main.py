from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Startup Post-Mortem RAG",
    description="Ask questions across startup failure stories with source citations.",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
