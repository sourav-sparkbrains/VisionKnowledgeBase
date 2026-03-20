"""
    Qdrant vector database client — singleton wrapper.
    Handles collection management, vector upsert, search and delete.
    All methods are async. Raises DatabaseError on any failure.
    Import qdrant_db singleton — never instantiate QdrantDB directly.
"""

from qdrant_client import AsyncQdrantClient, models
from typing import Self

from app.core.logging import get_logger
from app.core.config import settings
from app.core.exceptions import DatabaseError

logger = get_logger()

class QdrantDB:
    """
        Singleton Qdrant client.
        Connection is created once on first instantiation and reused across all requests.
    """

    _instance: Self | None = None

    def __new__(cls) -> Self:
        """Create or return existing singleton instance."""
        if cls._instance is None:
            logger.info("Connecting to Qdrant for the first time...")
            cls._instance = super().__new__(cls)

            host = settings.QDRANT_HOST
            port = settings.QDRANT_PORT
            api_key = settings.QDRANT_API_KEY or None

            cls._instance.client= AsyncQdrantClient(host=host, port=port,api_key=api_key)

        return cls._instance

    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
            Create a Qdrant collection if it does not already exist.
            Called once at app startup — not on every request.
            :param collection_name: name of the collection to create
            :param vector_size: dimensionality of vectors (512 for CLIP)
        """
        try:
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise DatabaseError(
                message=f"Qdrant create_collection failed: {str(e)}",
                user_message="Storage service unavailable.",
                error_code=503,
            )

    async def upsert_vector(self,vector_id: str, vector: list[float], payload: dict) -> None:
        """
            Insert or update a vector in Qdrant.
            If vector_id already exists, overwrites it.
            :param vector_id: unique identifier for this vector (matches MongoDB id)
            :param vector: CLIP embedding as list of floats
            :param payload: metadata stored alongside vector (namespace, image_id)
        """
        try:
            await self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.info(f"Upserted vector: {vector_id}")
        except Exception as e:
            logger.error(f"Failed to upsert vector: {str(e)}")
            raise DatabaseError(
                message=f"Qdrant upsert failed: {str(e)}",
                user_message="Failed to store image. Please try again.",
                error_code=503,
            )

    async def search_vector(self, query_vector: list[float], limit: int, namespace: str) -> list[models.ScoredPoint]:
        """
            Search for similar vectors filtered by namespace.
            Returns top `limit` results ordered by cosine similarity.
            :param query_vector: CLIP embedding of user query text
            :param limit: number of results to return
            :param namespace: user/team scope to search within
            :return: list of ScoredPoint objects with id, score, payload
        """
        try:
            result = await self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=limit,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="namespace",
                            match=models.MatchValue(value=namespace),
                        )
                    ]
                )
            )
            logger.info(f"Results found: {result}")
            return result.points
        except Exception as e:
            logger.error(f"Failed to search vector: {str(e)}")
            raise DatabaseError(
                message=f"Qdrant search failed: {str(e)}",
                user_message="Search unavailable. Please try again.",
                error_code=503,
            )

    async def delete_vector(self, vector_id: str) -> None:
        """
            Delete a vector by id.
            Called when an image is deleted — removes from Qdrant atomically.
            :param vector_id: id of vector to delete (matches MongoDB id)
        """
        try:
            await self.client.delete(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points_selector=models.PointIdsList(
                    points=[vector_id]
                )
            )
            logger.warning(f"Deleted vector: {vector_id}")
        except Exception as e:
            logger.error(f"Failed to delete vector: {str(e)}")
            raise DatabaseError(
                message=f"Qdrant delete failed: {str(e)}",
                user_message="Failed to delete image. Please try again.",
                error_code=503,
            )

qdrant_db = QdrantDB()
