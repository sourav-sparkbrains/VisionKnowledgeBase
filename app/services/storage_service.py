"""
Storage service — thin wrapper around StorageClient.
Handles raw image file operations: save, read, delete, exists.
"""

from app.core.logging import get_logger
from app.core.exceptions import DuplicateImage, ImageNotFound
from app.db.storage_client import storage_client

logger = get_logger()


class StorageService:
    """
    Delegates file operations to StorageClient.
    Exists as a service layer to keep ingestion logic clean.
    """

    def __init__(self):
        self.client = storage_client

    async def save_image(self, file_bytes: bytes, file_name: str) -> str:
        """
        Save image to filesystem.
        :param file_bytes: raw image bytes
        :param file_name: filename to save as
        :return: relative file path
        :raises DuplicateImage: if file already exists
        :raises DatabaseError: if write fails
        """
        try:
            path = await self.client.save_file(file_bytes, file_name)
            logger.info(f"Image saved: {file_name}")
            return path
        except DuplicateImage:
            raise
        except Exception as e:
            logger.error(f"Failed to save image {file_name}: {e}")
            raise

    async def get_image(self, file_name: str) -> bytes:
        """
        Read image bytes from filesystem.
        :param file_name: filename to read
        :return: raw image bytes
        :raises ImageNotFound: if file does not exist
        """
        try:
            data = await self.client.get_file(file_name)
            logger.info(f"Image retrieved: {file_name}")
            return data
        except ImageNotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to get image {file_name}: {e}")
            raise

    async def delete_image(self, file_name: str) -> None:
        """
        Delete image from filesystem.
        :param file_name: filename to delete
        :raises ImageNotFound: if file does not exist
        """
        try:
            await self.client.delete_file(file_name)
            logger.info(f"Image deleted: {file_name}")
        except ImageNotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to delete image {file_name}: {e}")
            raise

    def image_exists(self, file_name: str) -> bool:
        """
        Check if image exists on filesystem.
        :param file_name: filename to check
        :return: True if exists, False otherwise
        """
        return self.client.is_exists(file_name)