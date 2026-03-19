"""
Local filesystem storage client.
Handles raw image file operations — save, read, delete, exists.
Uses aiofiles for non-blocking async file I/O.
Not a singleton — stateless, no shared resources.
"""

import aiofiles
from pathlib import Path

from app.core.logging import get_logger
from app.core.config import settings
from app.core.exceptions import DatabaseError, DuplicateImage, ImageNotFound

logger = get_logger()


class StorageClient:
    """
    Handles raw image file operations on local filesystem.
    Base path is configured via STORAGE_BASE_PATH in settings.
    """

    def __init__(self):
        self.base_path = Path(settings.STORAGE_BASE_PATH)

    async def save_file(self, file_bytes: bytes, file_name: str) -> str:
        """
        Save image bytes to disk.
        Creates base directory if it doesn't exist.
        :param file_bytes: raw image bytes
        :param file_name: filename to save as
        :return: relative file path as string
        :raises DuplicateImage: if file already exists
        :raises DatabaseError: if write fails
        """
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            file_path = self.base_path / file_name

            if file_path.exists():
                logger.error(f"File already exists: {file_name}")
                raise DuplicateImage(
                    message=f"File {file_name} already exists on disk",
                    user_message="This image has already been uploaded.",
                    error_code=409
                )

            async with aiofiles.open(file_path, mode="wb") as f:
                await f.write(file_bytes)

            logger.info(f"Saved file: {file_name}")
            return str(file_path)

        except DuplicateImage:
            raise
        except Exception as e:
            logger.error(f"Failed to save file {file_name}: {e}")
            raise DatabaseError(
                message=f"File save failed: {str(e)}",
                user_message="Failed to save image. Please try again.",
                error_code=503
            )

    async def get_file(self, file_name: str) -> bytes:
        """
        Read image bytes from disk.
        :param file_name: filename to read
        :return: raw image bytes
        :raises ImageNotFound: if file does not exist
        :raises DatabaseError: if read fails
        """
        try:
            file_path = self.base_path / file_name

            if not file_path.is_file():
                logger.error(f"File not found: {file_name}")
                raise ImageNotFound(
                    message=f"File {file_name} not found on filesystem",
                    user_message="Image not found.",
                    error_code=404
                )

            async with aiofiles.open(file_path, mode="rb") as f:
                return await f.read()

        except ImageNotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to read file {file_name}: {e}")
            raise DatabaseError(
                message=f"File read failed: {str(e)}",
                user_message="Failed to retrieve image. Please try again.",
                error_code=503
            )

    async def delete_file(self, file_name: str) -> None:
        """
        Delete image file from disk.
        :param file_name: filename to delete
        :raises ImageNotFound: if file does not exist
        :raises DatabaseError: if deletion fails
        """
        try:
            file_path = self.base_path / file_name

            if not file_path.is_file():
                logger.error(f"File not found: {file_name}")
                raise ImageNotFound(
                    message=f"File {file_name} not found on filesystem",
                    user_message="Image not found.",
                    error_code=404
                )

            file_path.unlink()
            logger.info(f"Deleted file: {file_name}")

        except ImageNotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file {file_name}: {e}")
            raise DatabaseError(
                message=f"File deletion failed: {str(e)}",
                user_message="Failed to delete image. Please try again.",
                error_code=503
            )

    def is_exists(self, file_name: str) -> bool:
        """
        Check if file exists on disk.
        Synchronous — pathlib.is_file() is instant, no I/O wait.
        :param file_name: filename to check
        :return: True if exists, False otherwise
        """
        return (self.base_path / file_name).is_file()


storage_client = StorageClient()