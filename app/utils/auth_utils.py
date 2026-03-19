"""
Authentication utility functions.
Handles password hashing, verification and JWT token creation/decoding.
"""

import bcrypt
import jwt
from datetime import datetime, timedelta, timezone

from app.core.logging import get_logger
from app.core.exceptions import AuthenticationFailed, MissingDataError
from app.core.config import settings

logger = get_logger()


def get_hashed_password(password: str) -> str:
    """Hash plain text password using bcrypt."""
    try:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        raise AuthenticationFailed(
            message=f"Password hashing failed: {e}",
            user_message="Failed to process password.",
            error_code=500
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain text password against bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        raise AuthenticationFailed(
            message=f"Password verification failed: {e}",
            user_message="Failed to verify password.",
            error_code=500
        )


def create_access_token(data: dict) -> str:
    """Create signed JWT with expiry. data must contain user_id."""
    try:
        if not data:
            raise MissingDataError(
                message="Token data empty",
                user_message="Token data empty.",
                error_code=400
            )
        data["exp"] = datetime.now(timezone.utc) + timedelta(
            hours=settings.EXPIRY_TIME
        )
        return jwt.encode(data, settings.SECRET_KEY, algorithm="HS256")
    except MissingDataError:
        raise
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise AuthenticationFailed(
            message=f"Token creation failed: {e}",
            user_message="Failed to create token.",
            error_code=500
        )


def decode_access_token(token: str) -> dict:
    """Decode and verify JWT. Raises ExpiredSignatureError or InvalidTokenError."""
    try:
        return jwt.decode(
            token, settings.SECRET_KEY, algorithms=["HS256"]
        )
    except jwt.ExpiredSignatureError:
        raise
    except jwt.InvalidTokenError:
        raise
    except Exception as e:
        logger.error(f"Token decoding failed: {e}")
        raise AuthenticationFailed(
            message=f"Token decoding failed: {e}",
            user_message="Failed to decode token.",
            error_code=500
        )