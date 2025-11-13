"""Stub for omnibase_core.models.core"""

from typing import Any
from uuid import uuid4


class ModelContainer:
    """Stub for ONEX container model."""

    def __init__(self, value: Any = None, container_type: str = "config", **kwargs):
        self.value = value
        self.container_type = container_type
        self.container_id = uuid4()


__all__ = ["ModelContainer"]
