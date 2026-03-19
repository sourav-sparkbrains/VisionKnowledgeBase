"""
Vision service — wraps vision provider to extract structured metadata from images.
Decouples ingestion logic from specific provider implementations.
"""

from app.core.logging import get_logger
from app.core.exceptions import VisionModelFailed
from app.models.image import VisionResult
from app.providers.base import VisionProvider

logger = get_logger()


class VisionService:
    """
    Thin wrapper around VisionProvider.
    Receives provider via DI — swap Groq for Ollama via config, zero code changes.
    """

    def __init__(self, vision_provider: VisionProvider):
        """
        :param vision_provider: any class implementing VisionProvider protocol
        """
        self.provider = vision_provider

    async def get_metadata(self, image_bytes: bytes) -> VisionResult:
        """
        Analyze image and return structured metadata.
        :param image_bytes: raw image bytes
        :return: VisionResult with caption, tags, image_type
        :raises VisionModelFailed: if provider call fails
        """
        try:
            result = await self.provider.analyze(image_bytes)
            logger.info(f"Vision metadata extracted: {result.image_type}")
            return result
        except VisionModelFailed:
            raise
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise VisionModelFailed(
                message=f"Vision service failed: {str(e)}",
                user_message="Failed to analyze image. Please try again.",
                error_code=503
            )