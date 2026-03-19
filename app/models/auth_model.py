import uuid
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timezone

class UserRegister(BaseModel):
    fullname: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    token: str

class DBUser(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fullname: str
    email: EmailStr
    password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True