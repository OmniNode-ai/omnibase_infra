"""Stub for omnibase_core.nodes.node_effect"""

from typing import Any
from uuid import uuid4


class NodeEffect:
    """Base class for ONEX Effect nodes - stub implementation."""

    def __init__(self, container: Any):
        """Initialize node with container."""
        self.container = container
        self.node_id = uuid4()

    async def execute_effect(self, contract: Any) -> Any:
        """Execute effect - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute_effect")

    async def initialize(self) -> None:
        """Initialize node resources - optional."""
        pass

    async def cleanup(self) -> None:
        """Cleanup node resources - optional."""
        pass

    def get_metadata_loader(self) -> Any:
        """Get metadata loader - optional, returns None by default."""
        return None


__all__ = ["NodeEffect"]
