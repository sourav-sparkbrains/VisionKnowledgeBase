"""
Qwen2-VL vision provider — implements VisionProvider protocol.
Runs fully offline on CPU, better chart understanding than SmolVLM.
"""

import io
import asyncio
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

from app.core.logging import get_logger
from app.models.image import VisionResult, ImageType
from app.core.exceptions import VisionModelFailed
from app.core.config import settings

logger = get_logger()

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

type_map = {
    "screenshot": ImageType.SCREENSHOT,
    "whiteboard": ImageType.WHITEBOARD,
    "chart": ImageType.CHART,
    "diagram": ImageType.DIAGRAM,
    "photo": ImageType.PHOTO,
    "other": ImageType.OTHER
}


class QwenProvider:
    """
    Local vision provider using Qwen2-VL-2B-Instruct.
    Fully private, CPU, ~4GB RAM.
    Model loaded once at startup and reused across all requests.
    """

    def __init__(self):
        """Load Qwen2-VL model and processor once at startup."""
        logger.info(f"Loading {MODEL_ID}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model.eval()
        logger.info(f"{MODEL_ID} loaded successfully")

    def _generate(self, image: Image.Image, prompt_text: str, max_tokens: int = 256) -> str:
        """
        Run synchronous Qwen2-VL inference.
        :param image: PIL image
        :param prompt_text: instruction text
        :param max_tokens: max tokens to generate
        :return: generated text string
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )

        generated = outputs[:, inputs.input_ids.shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()

    async def analyze(self, image_bytes: bytes) -> VisionResult:
        """
        Analyze image and return structured metadata using Qwen2-VL.
        :param image_bytes: raw image bytes
        :return: VisionResult with caption, tags, image_type
        :raises VisionModelFailed: if model call fails
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            caption = await asyncio.to_thread(
                self._generate, image,
                "Describe in detail what is shown in this image including any text, data, charts, UI elements or diagrams visible. Be specific about numbers and labels if present.",
                256
            )

            tags_raw = await asyncio.to_thread(
                self._generate, image,
                "List exactly 5 keywords describing this image content, separated by commas. Single words only.",
                64
            )
            tags = [t.strip().strip('.') for t in tags_raw.split(',')][:5]

            type_raw = await asyncio.to_thread(
                self._generate, image,
                "Classify this image in one word only: screenshot, whiteboard, chart, diagram, photo, or other.",
                10
            )
            image_type = type_map.get(
                type_raw.lower().strip().strip('.'),
                ImageType.OTHER
            )

            logger.info(f"Qwen analysis complete: {image_type}")
            return VisionResult(
                caption=caption,
                tags=tags,
                image_type=image_type
            )

        except Exception as e:
            logger.error(f"Qwen analysis failed: {str(e)}")
            raise VisionModelFailed(
                message=f"Qwen analyze failed: {str(e)}",
                user_message="Failed to analyze image. Please try again.",
                error_code=503
            )