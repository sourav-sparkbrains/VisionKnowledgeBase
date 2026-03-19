"""
Query routes — handles semantic search and RAG answer generation.
"""
from typing import Annotated
from fastapi import APIRouter, HTTPException, Depends, status

from app.api.v1.dependencies import get_current_user
from app.core.logging import get_logger
from app.models.query import QueryRequest, QueryResponse
from app.services.query_service import QueryService
from app.services.embedding_service import embedding_service
from app.db.mongo_client import mongo_db
from app.db.qdrant_client import qdrant_db

logger = get_logger()
query_router = APIRouter()


@query_router.post("/query", tags=["Query"])
async def query_images(payload: QueryRequest,
                       user_id: str = Annotated[str, Depends(get_current_user)]) -> QueryResponse:
    """
    Semantic search over image library with RAG answer generation.
    :param payload: QueryRequest with query text
    :user_id : user id
    :return: QueryResponse with answer and cited source images
    """
    try:
        namespace = user_id

        query_service = QueryService(
            embedding_service=embedding_service,
            mongo_db=mongo_db,
            qdrant_db=qdrant_db
        )

        return await query_service.query(
            query_text=payload.query,
            namespace=namespace
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_BAD_REQUEST,
            detail="Failed to generate response. Please try again."
        )