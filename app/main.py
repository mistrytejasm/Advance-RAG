from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.documents import router as documents_router
from app.api.embeddings import router as embeddings_router
from app.api.query import router as query_router
from app.api.monitoring_router import router as monitoring_router
from app.database.mongodb_client import mongo_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: logging and warm ready state
    yield
    # Graceful shutdown: close MongoDB connections and pools
    mongo_client.close()


app = FastAPI(
    title="Advance RAG System",
    version="5.0.0",
    lifespan=lifespan,
)

app.include_router(upload_router)
app.include_router(documents_router)
app.include_router(embeddings_router)
app.include_router(query_router)
app.include_router(monitoring_router)


@app.get("/")
def root():
    return {
        "message": "Advance RAG Ingestion Service Running"
    }