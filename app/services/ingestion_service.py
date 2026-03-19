"""
Ingestion service — orchestrates the full image upload pipeline.
Validates, deduplicates, processes and stores images across all three stores.
"""

import uuid
import hashlib
from app.core.logging import get_logger
from app.core.config import settings
from app.core.exceptions import (
    FileTooLarge, InvalidFileType, DuplicateImage
)
from app.models.image import ImageRecord
from app.services.vision_service import VisionService
from app.services.embedding_service import EmbeddingService
from app.services.storage_service import StorageService
from app.db.mongo_client import MongoDB
from app.db.qdrant_client import QdrantDB

logger = get_logger()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}


class IngestionService:
    """
    Orchestrates full image ingestion pipeline.
    Validates → deduplicates → stores → analyzes → embeds → persists.
    All dependencies injected — swap providers without changing this class.
    """

    def __init__(
        self,
        vision_service: VisionService,
        embedding_service: EmbeddingService,
        storage_service: StorageService,
        mongo_db: MongoDB,
        qdrant_db: QdrantDB
    ):
        self.vision_service = vision_service
        self.embedding_service = embedding_service
        self.storage_service = storage_service
        self.mongo_db = mongo_db
        self.qdrant_db = qdrant_db

    async def ingest(
        self,
        file_bytes: bytes,
        file_name: str,
        namespace: str = "default"
    ) -> ImageRecord:
        """
        Full ingestion pipeline — validate, dedup, store, analyze, embed, persist.
        :param file_bytes: raw image bytes
        :param file_name: original filename
        :param namespace: user/team scope
        :return: ImageRecord stored in MongoDB
        :raises FileTooLarge: if file exceeds MAX_IMAGE_SIZE_MB
        :raises InvalidFileType: if file extension not allowed
        :raises DuplicateImage: if image already exists
        """

        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise FileTooLarge(
                message=f"{file_name} exceeds {settings.MAX_IMAGE_SIZE_MB}MB",
                user_message=f"File too large. Max size is {settings.MAX_IMAGE_SIZE_MB}MB.",
                error_code=413
            )

        ext = file_name.split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileType(
                message=f"{file_name} has unsupported extension: {ext}",
                user_message=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                error_code=415
            )

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = await self.mongo_db.get_images(namespace)
        if any(img.get("file_hash") == file_hash for img in existing):
            raise DuplicateImage(
                message=f"Image with hash {file_hash} already exists",
                user_message="This image has already been uploaded.",
                error_code=409
            )

        unique_name = f"{uuid.uuid4()}.{ext}"
        storage_path = await self.storage_service.save_image(file_bytes, unique_name)
        logger.info(f"Saved to filesystem: {storage_path}")

        vision_result = await self.vision_service.get_metadata(file_bytes)
        logger.info(f"Vision complete: {vision_result.image_type}")

        vector = await self.embedding_service.get_embedding(file_bytes)
        logger.info(f"Embedding generated: {len(vector)} dims")

        image_id = str(uuid.uuid4())
        vector_id = str(uuid.uuid4())

        record = ImageRecord(
            id=image_id,
            file_hash=file_hash,
            storage_path=storage_path,
            caption=vision_result.caption,
            tags=vision_result.tags,
            image_type=vision_result.image_type,
            vector_id=vector_id,
            namespace=namespace
        )

        await self.mongo_db.insert_image(record.model_dump())
        logger.info(f"Saved to MongoDB: {image_id}")

        await self.qdrant_db.upsert_vector(
            vector_id=vector_id,
            vector=vector,
            payload={
                "image_id": image_id,
                "namespace": namespace
            }
        )
        logger.info(f"Saved to Qdrant: {vector_id}")

        return record