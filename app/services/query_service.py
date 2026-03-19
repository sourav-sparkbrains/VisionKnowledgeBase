"""
Query service — handles semantic search and RAG answer generation.
Embeds text query → searches Qdrant → fetches MongoDB metadata → generates cited answer.
"""

import asyncio

from app.providers.local_llm_provider import local_llm
from app.core.logging import get_logger
from app.core.config import settings
from app.models.image import ImageResponse
from app.models.query import QueryResponse
from app.services.embedding_service import EmbeddingService
from app.db.mongo_client import MongoDB
from app.db.qdrant_client import QdrantDB
from app.utils.text_utils import RAG_PROMPT

logger = get_logger()

class QueryService:
    """
    Semantic search + RAG pipeline.
    Text query → CLIP embed → Qdrant search → MongoDB fetch → LLM answer.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        mongo_db: MongoDB,
        qdrant_db: QdrantDB
    ):
        self.embedding_service = embedding_service
        self.mongo_db = mongo_db
        self.qdrant_db = qdrant_db
        self.llm = local_llm

    async def query(
        self,
        query_text: str,
        namespace: str,
    ) -> QueryResponse:
        """
        Full RAG pipeline — embed query, search, fetch, generate answer.
        :param query_text: user's natural language question
        :param namespace: user/team scope to search within
        :return: QueryResponse with answer and cited source images
        """

        # step 1 — embed query text using CLIP
        query_vector = await asyncio.to_thread(
            self.embedding_service.model.encode,
            query_text,
            convert_to_numpy=True
        )
        query_vector = query_vector.tolist()
        logger.info(f"Query embedded: {len(query_vector)} dims")

        # step 2 — search Qdrant for similar vectors
        results = await self.qdrant_db.search_vector(
            query_vector=query_vector,
            limit=settings.TOP_K,
            namespace=namespace
        )
        logger.info(f"Qdrant returned {len(results)} results")

        if not results:
            return QueryResponse(
                answer="No relevant images found for your query.",
                source_images=[]
            )

        # step 3 — fetch image records from MongoDB
        records = []
        for hit in results:
            image_id = hit.payload.get("image_id")
            try:
                record = await self.mongo_db.get_image(image_id)
                records.append(record)
            except Exception as e:
                logger.warning(f"Could not fetch image {image_id}: {e}")
                continue

        # step 4 — build context for LLM
        context = "\n".join([
            f"Image {i+1}: {r['caption']} | tags: {', '.join(r['tags'])}"
            for i, r in enumerate(records)
        ])

        prompt = RAG_PROMPT.format(
            query=query_text,
            context=context
        )

        # step 5 — generate answer with Local LLM
        answer = await self.llm.generate(prompt)
        logger.info("RAG answer generated")

        # step 6 — build source images for response
        source_images = [
            ImageResponse(
                id=r["id"],
                caption=r["caption"],
                tags=r["tags"],
                image_type=r["image_type"],
                created_at=r["created_at"],
                image_url=f"{settings.BASE_URL}/static/{r['storage_path']}"
            )
            for r in records
        ]

        return QueryResponse(
            answer=answer,
            source_images=source_images
        )
