"""
Groq-based implementation of the VisionProvider.

Encodes images to base64, sends them to Groq's vision model,
and returns structured output (caption, tags, image type).
"""

import json
import base64
from groq import AsyncGroq
from groq.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)
from groq.types.chat.completion_create_params import (
    ResponseFormatResponseFormatJsonObject
)

from app.core.logging import get_logger
from app.core.config import settings
from app.models.image import VisionResult
from app.core.exceptions import VisionModelFailed
from app.utils.text_utils import VISION_PROMPT

logger = get_logger()

class GroqProvider:
    """
    Vision provider using Groq's API.
    Sends image as base64 to vision model, parses structured JSON response.
    """

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    @staticmethod
    def _build_message(image_b64: str) -> ChatCompletionUserMessageParam:
        """
        Build a properly typed vision message.
        """
        return {
            "role": "user",
            "content": [
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:image/jpeg;base64,{image_b64}"}
                ),
                ChatCompletionContentPartTextParam(
                    type="text",
                    text=VISION_PROMPT
                ),
            ],
        }

    async def analyze(self, image_bytes: bytes) -> VisionResult:
        """
        Analyze image using Groq vision model.
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            messages: list[ChatCompletionUserMessageParam] = [
                self._build_message(image_b64)
            ]
            response_format: ResponseFormatResponseFormatJsonObject = {
                "type": "json_object"
            }
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from vision model")

            parsed = json.loads(content)
            logger.info(f"Vision analysis complete: {parsed.get('image_type')}")
            return VisionResult(**parsed)

        except Exception as e:
            logger.error(f"Groq vision analysis failed: {e}")

            raise VisionModelFailed(
                message=f"Groq analyze failed: {str(e)}",
                user_message="Failed to analyze image. Please try again.",
                error_code=503,
            )