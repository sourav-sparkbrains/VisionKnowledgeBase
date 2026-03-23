"""
Embedding service — generates sigLIP vector embeddings from image bytes.
"""

import io
import asyncio
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from app.core.logging import get_logger
from app.core.exceptions import EmbeddingFailed

logger = get_logger()

class EmbeddingService:
    """
    Generates image embeddings using CLIP via local Sentence-Transformers.
    Returns 512-dim float vector stored in Qdrant for semantic search.
    """

    def __init__(self):
        """Initialize local CLIP model."""
        # self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        # self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224",
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    async def get_embedding(self, image_bytes: bytes) -> list[float]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            def _embed():
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                return features[0].tolist()

            result = await asyncio.to_thread(_embed)
            logger.info(f"Generated embedding of size: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise EmbeddingFailed(
                message=f"Local CLIP embedding failed: {str(e)}",
                user_message="Failed to process image locally.",
                error_code=500
            )

    async def get_text_embedding(self, text: str) -> list[float]:
        """Generate SigLIP embedding from text query."""
        try:
            def _embed():
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True
                )
                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                return features[0].tolist()

            result = await asyncio.to_thread(_embed)
            logger.info(f"Generated text embedding of size: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise EmbeddingFailed(
                message=f"SigLIP text embedding failed: {str(e)}",
                user_message="Failed to process query.",
                error_code=500
            )

embedding_service = EmbeddingService()
