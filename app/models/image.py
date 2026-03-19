"""
    Pydantic models for image data.
    - ImageType: enum restricting valid image classifications
    - VisionResult: output from LLaVA vision analysis
    - ImageRecord: full image document stored in MongoDB
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone

class ImageType(str, Enum):
    SCREENSHOT = "screenshot"
    WHITEBOARD = "whiteboard"
    CHART = "chart"
    DIAGRAM = "diagram"
    PHOTO = "photo"
    OTHER = "other"

class ImageRecord(BaseModel):
    """Full image document as stored in MongoDB."""
    id: str
    file_hash: str
    storage_path: str
    caption: str
    tags: list[str]
    image_type: ImageType
    vector_id: str
    namespace: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class VisionResult(BaseModel):
    """Direct output from LLaVA — caption, tags, image type only."""
    caption: str
    tags: list[str]
    image_type: ImageType

class ImageResponse(BaseModel):
    """Public facing image model — returned to API consumers."""
    id:         str
    caption:    str
    tags:       list[str]
    image_type: ImageType
    created_at: datetime
    image_url:  str