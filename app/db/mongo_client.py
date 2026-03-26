"""
MongoDB async client — singleton wrapper using Motor.
Handles all image record CRUD operations.
All methods are async. Raises DatabaseError on failures, ImageNotFound when record missing.
Import mongo_db singleton — never instantiate MongoDB directly.
"""

import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi

from app.core.logging import get_logger
from app.core.config import settings
from app.core.exceptions import DatabaseError, ImageNotFound

logger = get_logger()


class MongoDB:
    """
    Singleton MongoDB client using Motor for async operations.
    Connection is created once on first instantiation and reused across all requests.
    """
    _instance: None = None

    def __new__(cls):
        """Create or return existing singleton instance."""
        if cls._instance is None:
            logger.info("Connecting to MongoDB...")
            cls._instance = super().__new__(cls)
            cls._instance.client = AsyncIOMotorClient(
                settings.MONGO_URI,
                server_api=ServerApi('1'),
                tls=True,
                tlsCAFile=certifi.where()
            )
            cls._instance.db = cls._instance.client[settings.MONGO_DB_NAME]
        return cls._instance

    async def create_collection(self, collection_name: str) -> None:
        """
        Create a MongoDB collection if it does not already exist.
        Called once at app startup — not on every request.
        :param collection_name: name of the collection to create
        """
        try:
            collection_list = await self.db.list_collection_names()
            if collection_name not in collection_list:
                await self.db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise DatabaseError(
                message=f"MongoDB create_collection failed: {str(e)}",
                user_message="Storage service unavailable.",
                error_code=503
            )

    def get_collection(self, collection_name: str):
        """
        Return a MongoDB collection object by name.
        :param collection_name: name of the collection
        :return: Motor async collection
        """
        try:
            return self.db[collection_name]
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise DatabaseError(
                message=f"Failed to get collection: {str(e)}",
                user_message="Storage service unavailable.",
                error_code=503
            )

    async def insert_image(self, image_record: dict) -> None:
        """
        Insert an image record into MongoDB.
        Expects a pre-validated dict — validation is the service layer's responsibility.
        :param image_record: ImageRecord as dict, ready to store
        """
        try:
            await self.db[settings.MONGO_COLLECTION].insert_one(image_record)
            logger.info(f"Inserted image record: {image_record.get('id')}")
        except Exception as e:
            logger.error(f"Failed to insert image record: {e}")
            raise DatabaseError(
                message=f"MongoDB insert failed: {str(e)}",
                user_message="Failed to store image. Please try again.",
                error_code=503
            )

    async def get_image(self, image_record_id: str) -> dict:
        """
        Fetch a single image record by id.
        :param image_record_id: unique image id
        :return: image record as dict
        :raises ImageNotFound: if no record exists with given id
        """
        try:
            result = await self.db[settings.MONGO_COLLECTION].find_one(
                {'id': image_record_id}
            )
            if result is None:
                raise ImageNotFound(
                    message=f"Image {image_record_id} not found in MongoDB",
                    user_message="Image not found.",
                    error_code=404
                )
            logger.info(f"Fetched image record: {image_record_id}")
            return result
        except ImageNotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch image {image_record_id}: {e}")
            raise DatabaseError(
                message=f"MongoDB get_image failed: {str(e)}",
                user_message="Failed to fetch image. Please try again.",
                error_code=503
            )

    async def get_images(self, namespace: str) -> list[dict]:
        """
        Fetch all image records for a given namespace.
        :param namespace: user or team scope to filter by
        :return: list of image records as dicts
        """
        try:
            cursor = self.db[settings.MONGO_COLLECTION].find(
                {'namespace': namespace}
            )
            images = []
            async for document in cursor:
                images.append(document)
            logger.info(f"Fetched {len(images)} images for namespace: {namespace}")
            return images
        except Exception as e:
            logger.error(f"Failed to fetch images for namespace {namespace}: {e}")
            raise DatabaseError(
                message=f"MongoDB get_images failed: {str(e)}",
                user_message="Failed to fetch images. Please try again.",
                error_code=503
            )

    async def delete_image(self, image_record_id: str) -> None:
        """
        Delete a single image record by id.
        :param image_record_id: unique image id to delete
        """
        try:
            await self.db[settings.MONGO_COLLECTION].delete_one(
                {'id': image_record_id}
            )
            logger.info(f"Deleted image record: {image_record_id}")
        except Exception as e:
            logger.error(f"Failed to delete image {image_record_id}: {e}")
            raise DatabaseError(
                message=f"MongoDB delete failed: {str(e)}",
                user_message="Failed to delete image. Please try again.",
                error_code=503
            )


mongo_db = MongoDB()