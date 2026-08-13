"""FastAPI dependency injection."""
import os
from typing import Optional
from fastapi import Depends, Request

from ..engine import ReviewEngine
from ..storage import StorageBackend, MemoryStorage, JsonFileStorage


def get_engine(request: Request) -> ReviewEngine:
    """Retrieve the ReviewEngine instance attached to the app state."""
    return request.app.state.engine


def get_ws_manager(request: Request):
    """Retrieve the WSManager instance attached to the app state."""
    return request.app.state.ws_manager


def get_configs_dir(request: Request) -> str:
    """Retrieve the configs_dir path attached to the app state."""
    return request.app.state.configs_dir
