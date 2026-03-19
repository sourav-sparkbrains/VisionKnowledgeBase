"""
Auth routes — register and login. Returns JWT on success.
"""

import jwt
from typing import Annotated
from fastapi import APIRouter, HTTPException, Depends, status
from app.models.auth_model import UserRegister, UserLogin, DBUser, UserResponse, RefreshTokenRequest
from app.db.mongo_client import mongo_db
from app.core.config import settings
from app.api.v1.dependencies import get_current_user
from app.utils.auth_utils import get_hashed_password, verify_password, create_access_token, create_refresh_token
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
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered.")

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.user_message)
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register.")


@auth_router.post("/login", tags=["Auth"])
async def login_user(input: UserLogin) -> UserResponse:
    """Login — verifies credentials, returns JWT containing user_id."""
    try:
        user = await _collection.find_one(
            {"email": input.email},
            {"_id": 0, "password": 1, "id": 1}
        )
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

        if not verify_password(input.password, user["password"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

        access_token = create_access_token({
            "user_id": user["id"],
            "email": input.email
        })
        refresh_token = create_refresh_token(
            {
                "user_id": user["id"],
                "email": input.email
            }
        )
        logger.info(f"Login: {input.email}")

        await _collection.update_one(
            {"id": user["id"]},
            {"$set": {"refresh_token": refresh_token}}
        )

        return UserResponse(access_token=access_token, refresh_token=refresh_token)

    except HTTPException:
        raise
    except AuthenticationFailed as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.user_message)
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to login.")

@auth_router.post("/refresh", tags=["Auth"])
async def get_refresh_token(input: RefreshTokenRequest) -> dict:
    """
    Issue new access token using valid refresh token.
    Decodes refresh token, verifies against DB, returns new access token.
    :param input: RefreshTokenRequest with refresh_token
    :return: new access_token
    """
    try:
        payload = jwt.decode(
            input.refresh_token,
            settings.REFRESH_SECRET_KEY,
            algorithms=["HS256"]
        )
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token — user_id missing.")

        stored_data = await _collection.find_one(
            {"id": user_id},
            {"_id": 0, "refresh_token": 1, "email": 1}
        )
        if not stored_data:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

        if stored_data["refresh_token"] != input.refresh_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token mismatch.")

        new_access_token = create_access_token({
            "user_id": user_id,
            "email": stored_data["email"]
        })

        logger.info(f"Access token refreshed for user: {user_id}")
        return {"access_token": new_access_token}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired. Login again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to refresh token.")


@auth_router.post("/logout", tags=["Auth"])
async def logout(user_id: str = Annotated[str,Depends(get_current_user)]) -> dict:
    """
    Logout user by clearing refresh token from DB.
    Even if attacker has the token — it's now invalid.
    :param user_id: extracted from JWT via Depends
    :return: success message
    """
    try:
        await _collection.update_one(
            {"id": user_id},
            {"$set": {"refresh_token": None}}
        )
        logger.info(f"User logged out: {user_id}")
        return {"message": "Logged out successfully."}

    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to logout.")