"""
Custom exceptions for VKB.
All exceptions inherit from VKBException.
- message      → full detail, goes to logs only
- user_message → clean string, returned to API caller
- error_code   → maps to HTTP status code
"""

class VKBException(Exception):
    def __init__(self, message: str, user_message: str, error_code: int) -> None:
        """Base exception for all VKB errors."""
        super().__init__(message)
        self.message = message # for developers (logs)
        self.user_message = user_message # for user (frontend)
        self.error_code = error_code

class DuplicateImage(VKBException):
    """Raised when an image with the same hash already exists."""
    pass

class ImageNotFound(VKBException):
    """Raised when a requested image does not exist."""
    pass

class FileTooLarge(VKBException):
    """Raised when uploaded file exceeds MAX_IMAGE_SIZE_MB."""
    pass

class InvalidFileType(VKBException):
    """Raised when uploaded file is not a supported image type."""
    pass

class VisionModelFailed(VKBException):
    """Raised when LLaVA/Groq vision call fails."""
    pass

class EmbeddingFailed(VKBException):
    """Raised when CLIP embedding generation fails."""
    pass

class DatabaseError(VKBException):
    """Raised when any DB operation fails (Qdrant or MongoDB)."""
    pass

class AuthenticationFailed(VKBException):
    """Raised when authentication fails."""
    pass

class MissingDataError(VKBException):
    """Raised when a required data was not found."""
    pass