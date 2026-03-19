"""
Embedding service — generates CLIP vector embeddings from image bytes.
Uses local Sentence-Transformers for free, offline inference.
"""

import asyncio
import io
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger
from app.core.config import settings
from app.core.exceptions import EmbeddingFailed

logger = get_logger()

class EmbeddingService:
    """
    Generates image embeddings using CLIP via local Sentence-Transformers.
    Returns 512-dim float vector stored in Qdrant for semantic search.
    """

    def __init__(self):
        """Initialize local CLIP model."""
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

    async def get_embedding(self, image_bytes: bytes) -> list[float]:
        """
        Generate CLIP embedding from raw image bytes locally.
        :param image_bytes: raw image bytes
        :return: float vector (usually 512-dim for CLIP)
        :raises EmbeddingFailed: if processing fails
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            embedding = await asyncio.to_thread(
                self.model.encode,
                image,
                convert_to_numpy=True
            )

            result = embedding.tolist()

            logger.info(f"Generated local embedding of size: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise EmbeddingFailed(
                message=f"Local CLIP embedding failed: {str(e)}",
                user_message="Failed to process image locally.",
                error_code=500
            )

embedding_service = EmbeddingService()
