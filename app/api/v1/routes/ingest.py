"""
Ingestion routes — handles image upload and processing pipeline.
"""
from typing import Annotated
from fastapi import APIRouter, HTTPException, File, UploadFile, status, Depends

from app.core.logging import get_logger
from app.core.exceptions import DuplicateImage, FileTooLarge, InvalidFileType
from app.services.embedding_service import embedding_service
from app.services.vision_service import VisionService
from app.services.storage_service import StorageService
from app.services.ingestion_service import IngestionService
from app.providers.smolvlm_provider import SmolVLMProvider
from app.db.mongo_client import mongo_db
from app.db.qdrant_client import qdrant_db
from app.models.image import ImageRecord
from app.api.v1.dependencies import get_current_user


logger = get_logger()
ingestion_router = APIRouter()

_vision_service = VisionService(SmolVLMProvider())
_storage_service = StorageService()


@ingestion_router.post("/ingest", tags=["Ingest"])
async def ingest(
        user_id:Annotated[str, Depends(get_current_user)],
        file: UploadFile = File(...)
                 ) -> ImageRecord:
    """
    Upload and process an image through full ingestion pipeline.
    :param file: image file to upload
    :user_id : user id
    :return: ImageRecord with extracted metadata
    """
    try:
        file_bytes = await file.read()

        ingest_service = IngestionService(
            vision_service=_vision_service,
            embedding_service=embedding_service,
            storage_service=_storage_service,
            mongo_db=mongo_db,
            qdrant_db=qdrant_db
        )

        return await ingest_service.ingest(
            file_bytes=file_bytes,
            file_name=file.filename,
            namespace=user_id.strip().lower()
        )

    except DuplicateImage as e:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT, detail=e.user_message)
    except FileTooLarge as e:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=e.user_message)
    except InvalidFileType as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=e.user_message)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingestion failed. Please try again."
        )