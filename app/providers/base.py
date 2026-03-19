"""
VisionProvider Protocol — defines the contract all vision providers must implement.
Any class with analyze(image_bytes: bytes) -> VisionResult satisfies this protocol.
"""

from typing import Protocol
from app.models.image import VisionResult


class VisionProvider(Protocol):
    """Contract for vision model providers — Groq, Ollama, etc."""

    async def analyze(self, image_bytes: bytes) -> VisionResult:
        """
        Analyze image bytes and return structured vision result.
        :param image_bytes: raw image bytes
        :return: VisionResult with caption, tags, image_type
        """
        ...