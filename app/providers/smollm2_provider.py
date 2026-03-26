"""

"""

import torch
import asyncio
from transformers import pipeline, BitsAndBytesConfig

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger()


class SmolLM2Provider:
    """

    """

    def __init__(self):
        """
        """
        logger.info("Loading smollm2...")

        self.pipe = pipeline(
            "text-generation",
            model= "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            torch_dtype=torch.float32,
            device="cpu"
        )

        logger.info("Local smollm2 loaded successfully")


    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt}
        ]

        result = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,
        )

        output = result[0]["generated_text"]
        if isinstance(output, list):
            return output[-1]["content"].strip()

        return output.strip()

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        """
        try:
            return await asyncio.to_thread(
                self._generate,
                prompt,
                max_tokens
            )
        except Exception as e:
            logger.error(f"Local smollm2 generation failed: {str(e)}")
            raise

smollm2 = SmolLM2Provider()
