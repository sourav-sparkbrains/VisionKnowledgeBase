"""
Image routes — list, fetch and delete images from the knowledge base.
"""

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.db.mongo_client import mongo_db
from app.db.qdrant_client import qdrant_db
from app.db.storage_client import storage_client
from app.core.exceptions import DatabaseError, ImageNotFound
from app.models.image import ImageRecord, ImageResponse
from app.core.config import settings

logger = get_logger()
images_router = APIRouter()


@images_router.get("/images", tags=["Images"])
async def get_images(namespace: str = "default") -> list[ImageResponse]:
    """
    Fetch all images for a given namespace.
    :param namespace: namespace to filter images
    :return: list of image responses
    """
    try:
        results = await mongo_db.get_images(namespace)
        return [
            ImageResponse(
                id=r["id"],
                caption=r["caption"],
                tags=r["tags"],
                image_type=r["image_type"],
                created_at=r["created_at"],
                image_url=f"{settings.BASE_URL}/static/{r['storage_path']}"
            )
            for r in results
        ]
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=e.user_message)
    except Exception as e:
        logger.error(f"Get images failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch images.")


@images_router.get("/images/{image_id}", tags=["Images"])
async def get_image(image_id: str) -> ImageResponse:
    """
    Fetch a single image by ID.
    :param image_id: unique image id
    :return: image response
    """
    try:
        r = await mongo_db.get_image(image_id)
        return ImageResponse(
            id=r["id"],
            caption=r["caption"],
            tags=r["tags"],
            image_type=r["image_type"],
            created_at=r["created_at"],
            image_url=f"{settings.BASE_URL}/static/{r['storage_path']}"
        )
    except ImageNotFound as e:
        raise HTTPException(status_code=404, detail=e.user_message)
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=e.user_message)
    except Exception as e:
        logger.error(f"Get image failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch image.")


@images_router.delete("/images/{image_id}", tags=["Images"])
async def delete_image(image_id: str) -> dict:
    """
    Delete image from all three stores — MongoDB, Qdrant, filesystem.
    Fetches record first to get vector_id and storage_path.
    :param image_id: unique image id
    :return: confirmation message
    """
    try:
        record = await mongo_db.get_image(image_id)

        await mongo_db.delete_image(image_id)
        await qdrant_db.delete_vector(record["vector_id"])
        await storage_client.delete_file(record["storage_path"])

        logger.info(f"Deleted image {image_id} from all stores")
        return {"message": f"Image {image_id} deleted successfully"}

    except ImageNotFound as e:
        raise HTTPException(status_code=404, detail=e.user_message)
    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=e.user_message)
    except Exception as e:
        logger.error(f"Delete image failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete image.")