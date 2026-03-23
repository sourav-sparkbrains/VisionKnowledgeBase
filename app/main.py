"""
FastAPI application entrypoint.
Mounts all routers, runs startup initialization for DB collections,
and serves static files for image access.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import get_logger
from app.db.qdrant_client import qdrant_db
from app.db.mongo_client import mongo_db
from app.api.v1.routes.ingest import ingestion_router
from app.api.v1.routes.query import query_router
from app.api.v1.routes.images import images_router
from app.api.v1.routes.auth import auth_router

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    logger.info("Starting Visual Knowledge Base...")

    await qdrant_db.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vector_size=768
    )

    await mongo_db.create_collection(settings.MONGO_COLLECTION)
    await mongo_db.create_collection(settings.MONGO_USER_COLLECTION)

    logger.info("VKB startup complete.")
    yield
    logger.info("VKB shutting down.")


app = FastAPI(
    title="Visual Knowledge Base",
    description="AI-powered image memory and retrieval system.",
    version="0.1.0",
    lifespan=lifespan
)

app.mount(
    "/static",
    StaticFiles(directory=settings.STORAGE_BASE_PATH),
    name="static"
)

# mount routers
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(images_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1/auth")


@app.get("/", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "app": settings.APP_NAME}