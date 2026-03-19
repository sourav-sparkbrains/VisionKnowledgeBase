"""
FastAPI dependencies — extracts user_id from JWT for protected routes.
"""

import jwt
from typing import Annotated
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from app.core.logging import get_logger
from app.utils.auth_utils import decode_access_token

logger = get_logger()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> str:
    """
    Extract user_id from JWT — returns it as namespace for routes.
    :param token: JWT bearer token
    :return: user_id string
    """
    try:
        payload = decode_access_token(token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_current_user failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed.")