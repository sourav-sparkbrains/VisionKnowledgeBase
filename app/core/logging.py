"""
    Configures get_logger() for use across all modules.
    Returns a named logger per calling module.
    Logs: timestamp | app_name | module.path | level | message
"""

import os
import inspect
import logging

from app.core.config import settings

APP_NAME = settings.APP_NAME

def _configure_logging() -> None:
    """Set up root logger once at startup."""
    logging.basicConfig(
        level=logging.INFO if settings.APP_ENV == "development" else logging.WARNING,
        format=f"%(asctime)s - {settings.APP_NAME} - %(name)s - %(levelname)s - %(message)s",
    )

def get_logger() -> logging.Logger:
    """
        Configured a logger.
        :return: timestamp | application_name | module.path | level | message
    """
    frame = inspect.stack()[1]
    file_path = str(os.path.splitext(os.path.relpath(frame[1]))[0])
    name = file_path.replace(os.sep,'.')
    logger = logging.getLogger(name)

    return logger

_configure_logging()