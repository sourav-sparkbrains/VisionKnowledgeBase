"""
Local LLM provider module.

Implements a lightweight, fully offline text generation service using
TinyLlama-1.1B-Chat-v1.0 via Hugging Face transformers pipeline.

The model is loaded once at initialization and reused for all requests.
Async compatibility is achieved by offloading blocking inference to a thread.
"""

import torch
import asyncio
from transformers import pipeline

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger()


class LocalLLMProvider:
    """
    Local language model provider for text generation.
    Uses TinyLlama-1.1B-Chat-v1.0 running on CPU via Hugging Face pipeline.
    Designed for use in RAG pipelines where privacy and offline capability
    are required.
    """

    def __init__(self):
        """
        Initialize the local LLM provider.
        Loads the pretrained TinyLlama-1.1B-Chat-v1.0 model into memory.
        This operation is performed once at startup to avoid repeated loading.
        """
        logger.info("Loading local LLM ...")

        self.pipe = pipeline(
            "text-generation",
            model= settings.LOCAL_LLM,
            torch_dtype=torch.float32,
            device="cpu"
        )

        logger.info("Local LLM loaded successfully")

    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt}
        ]

        result = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        output = result[0]["generated_text"]
        if isinstance(output, list):
            return output[-1]["content"].strip()

        return output.strip()

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Asynchronous text generation method.
        Wraps the blocking `_generate` method using asyncio.to_thread
        to prevent blocking the event loop.
        :param prompt: Input prompt for the model.
        :param max_tokens: Maximum number of tokens to generate.
        """
        try:
            return await asyncio.to_thread(
                self._generate,
                prompt,
                max_tokens
            )
        except Exception as e:
            logger.error(f"Local LLM generation failed: {str(e)}")
            raise

local_llm = LocalLLMProvider()
