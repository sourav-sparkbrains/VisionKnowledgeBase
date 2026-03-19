"""
Auth routes — register and login. Returns JWT on success.
"""

from fastapi import APIRouter, HTTPException
from app.models.auth_model import UserRegister, UserLogin, DBUser, UserResponse
from app.db.mongo_client import mongo_db
from app.core.config import settings
from app.utils.auth_utils import get_hashed_password, verify_password, create_access_token
from app.core.exceptions import AuthenticationFailed
from app.core.logging import get_logger

logger = get_logger()
auth_router = APIRouter()

_collection = mongo_db.get_collection(settings.MONGO_USER_COLLECTION)


@auth_router.post("/register", tags=["Auth"])
async def register_user(input: UserRegister) -> dict:
    """Register new user — checks duplicate email, hashes password, stores in MongoDB."""
    try:
        existing = await _collection.find_one({"email": input.email}, {"_id": 0})
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered.")

        user = DBUser(
            fullname=input.fullname,
            email=input.email,
            password=get_hashed_password(input.password)
        )
        await _collection.insert_one(user.model_dump())
        logger.info(f"Registered: {input.email}")
        return {"message": "User registered successfully."}

    except HTTPException:
        raise
    except AuthenticationFailed as e:
        raise HTTPException(status_code=500, detail=e.user_message)
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to register.")


@auth_router.post("/login", tags=["Auth"])
async def login_user(input: UserLogin) -> UserResponse:
    """Login — verifies credentials, returns JWT containing user_id."""
    try:
        user = await _collection.find_one(
            {"email": input.email},
            {"_id": 0, "password": 1, "id": 1}
        )
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials.")

        if not verify_password(input.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials.")

        token = create_access_token({
            "user_id": user["id"],
            "email": input.email
        })
        logger.info(f"Login: {input.email}")
        return UserResponse(token=token)

    except HTTPException:
        raise
    except AuthenticationFailed as e:
        raise HTTPException(status_code=500, detail=e.user_message)
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to login.")