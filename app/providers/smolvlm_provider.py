"""
SmolVLM vision provider — implements VisionProvider protocol using local SmolVLM-256M model.
Runs fully offline on CPU, lightweight (500MB), no API key needed.
"""

import io
import asyncio
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

from app.core.logging import get_logger
from app.models.image import VisionResult, ImageType
from app.core.exceptions import VisionModelFailed
from app.core.config import settings

logger = get_logger()

MODEL_ID = settings.VISION_MODEL

type_map = {
    "screenshot": ImageType.SCREENSHOT,
    "whiteboard": ImageType.WHITEBOARD,
    "chart": ImageType.CHART,
    "diagram": ImageType.DIAGRAM,
    "photo": ImageType.PHOTO,
    "other": ImageType.OTHER
}


class SmolVLMProvider:
    """
    Local vision provider using SmolVLM-500M-Instruct.
    Fully private, CPU friendly, ~500MB RAM usage.
    Model loaded once at startup and reused across all requests.
    """

    def __init__(self):
        """Load SmolVLM processor and model once at startup."""
        logger.info("Loading SmolVLM-500M model locally...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        logger.info("SmolVLM-500M loaded successfully")

    def _generate(self, image: Image.Image, prompt_text: str, max_tokens: int = 128) -> str:
        """
        Run synchronous inference — called via asyncio.to_thread.
        :param image: PIL image
        :param prompt_text: instruction text
        :param max_tokens: max tokens to generate
        :return: generated text string
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )
        full_output = self.processor.decode(outputs[0], skip_special_tokens=True)

        if "Assistant:" in full_output:
            return full_output.split("Assistant:")[-1].strip()
        return full_output.strip()

    async def analyze(self, image_bytes: bytes) -> VisionResult:
        """
        Analyze image and return structured metadata using SmolVLM.
        Runs sync model calls in thread to avoid blocking async event loop.
        :param image_bytes: raw image bytes
        :return: VisionResult with caption, tags, image_type
        :raises VisionModelFailed: if model call or parsing fails
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            caption = await asyncio.to_thread(
                self._generate, image,
                "Describe in detail what is shown in this image including any text, UI elements, diagrams or content visible.",
                128
            )

            tags_raw = await asyncio.to_thread(
                self._generate, image,
                "What are 5 topics or objects in this image? Answer with exactly 5 single words separated by commas.",
                64
            )
            tags = [t.strip().strip('.') for t in tags_raw.split(',')][:5]

            type_raw = await asyncio.to_thread(
                self._generate, image,
                "Is this image a screenshot, whiteboard, chart, diagram or photo? Answer with one word only.",
                10
            )
            image_type = type_map.get(
                type_raw.lower().strip().strip('.'),
                ImageType.OTHER
            )

            logger.info(f"SmolVLM analysis complete: {image_type}")
            return VisionResult(
                caption=caption,
                tags=tags,
                image_type=image_type
            )

        except Exception as e:
            logger.error(f"SmolVLM analysis failed: {str(e)}")
            raise VisionModelFailed(
                message=f"SmolVLM analyze failed: {str(e)}",
                user_message="Failed to analyze image. Please try again.",
                error_code=503
            )