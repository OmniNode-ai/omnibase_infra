"""Decorator for automatic agent registration."""

import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from .models import AgentMetadata, AgentType

logger = logging.getLogger(__name__)

# Type variable for decorator
T = TypeVar("T")

# Global registry for decorator-based registration
# This is populated at module import time
_DECORATOR_REGISTRY: dict[str, dict[str, Any]] = {}


def register_agent(
    agent_id: str,
    capabilities: list[str],
    agent_type: AgentType,
    version: str,
    description: str,
    **metadata_kwargs: Any,
) -> Callable[[T], T]:
    """
    Decorator for agent self-registration.

    This decorator stores agent metadata in a global registry that can be used
    by the AgentRegistry to automatically discover and register agents.

    Performance: <1ms per registration

    Args:
        agent_id: Unique agent identifier
        capabilities: List of capability tags
        agent_type: Agent type enum
        version: Semantic version (e.g., "1.0.0")
        description: Human-readable description
        **metadata_kwargs: Additional metadata fields (priority, max_concurrent_tasks, etc.)

    Returns:
        Decorator function

    Raises:
        ValueError: If agent_id is already registered
        TypeError: If decorated class/function is invalid

    Example:
        ```python
        @register_agent(
            agent_id="contract_inferencer_v1",
            capabilities=["contract_inference", "yaml_parsing"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Infers contracts from YAML specifications",
            priority=90,
            max_concurrent_tasks=10
        )
        class ContractInferencerAgent:
            async def execute(self, task: Task) -> Result:
                # Agent implementation
                pass
        ```
    """

    def decorator(cls_or_func: T) -> T:
        # Validate not already registered
        if agent_id in _DECORATOR_REGISTRY:
            raise ValueError(f"Agent '{agent_id}' is already registered")

        # Validate class/function
        if inspect.isclass(cls_or_func):
            if not (hasattr(cls_or_func, "execute") or hasattr(cls_or_func, "agent")):
                raise TypeError(
                    f"Agent class '{agent_id}' must have 'execute()' or 'agent' attribute"
                )
        elif not callable(cls_or_func):
            raise TypeError(f"Agent '{agent_id}' must be a class or callable")

        # Create metadata
        metadata = AgentMetadata(
            agent_type=agent_type,
            version=version,
            description=description,
            **metadata_kwargs,
        )

        # Store agent metadata in global registry
        _DECORATOR_REGISTRY[agent_id] = {
            "agent_id": agent_id,
            "agent_type": agent_type.value,
            "capabilities": capabilities,
            "description": description,
            "class_or_function": cls_or_func,
            "is_class": inspect.isclass(cls_or_func),
            "module_name": cls_or_func.__module__,
            "metadata": metadata,
        }

        # Attach registration info to class/function
        cls_or_func._agent_id = agent_id  # type: ignore
        cls_or_func._capabilities = capabilities  # type: ignore
        cls_or_func._metadata = metadata  # type: ignore

        logger.info(
            f"[AgentDecorator] Registered agent: '{agent_id}' "
            f"({len(_DECORATOR_REGISTRY)} total)"
        )

        return cls_or_func

    return decorator


def get_registered_agents() -> dict[str, dict[str, Any]]:
    """
    Get all registered agents and their metadata.

    This function returns all agents registered via the @register_agent decorator.

    Returns:
        Dictionary mapping agent_id to metadata

    Performance: <5ms for 100+ agents
    """
    return _DECORATOR_REGISTRY.copy()


def agent_is_registered(agent_id: str) -> bool:
    """
    Check if an agent is registered.

    Args:
        agent_id: Agent ID to check

    Returns:
        True if agent is registered, False otherwise

    Performance: <1ms
    """
    return agent_id in _DECORATOR_REGISTRY


def get_agent_metadata(agent_id: str) -> dict[str, Any] | None:
    """
    Get metadata for a registered agent.

    Args:
        agent_id: Agent identifier

    Returns:
        Agent metadata dictionary or None if not found

    Performance: <1ms
    """
    return _DECORATOR_REGISTRY.get(agent_id)


def find_agents_by_capability(capability: str) -> list[dict[str, Any]]:
    """
    Find all agents with a specific capability.

    Args:
        capability: Capability to search for

    Returns:
        List of agent metadata dictionaries

    Performance: <5ms for 100+ agents
    """
    return [
        metadata
        for metadata in _DECORATOR_REGISTRY.values()
        if capability in metadata["capabilities"]
    ]


def get_agent_instance(agent_id: str, **init_kwargs: Any) -> Any:
    """
    Get instance of a registered agent.

    Args:
        agent_id: Agent identifier
        **init_kwargs: Initialization keyword arguments

    Returns:
        Agent instance (for classes) or function (for functions)

    Raises:
        ValueError: If agent not registered
        TypeError: If agent is a function and init_kwargs provided
    """
    if agent_id not in _DECORATOR_REGISTRY:
        raise ValueError(f"Agent '{agent_id}' not registered")

    metadata = _DECORATOR_REGISTRY[agent_id]
    cls_or_func = metadata["class_or_function"]

    if metadata["is_class"]:
        return cls_or_func(**init_kwargs)
    else:
        if init_kwargs:
            raise TypeError(
                f"Agent '{agent_id}' is a function, cannot pass init_kwargs"
            )
        return cls_or_func


def clear_registry() -> None:
    """
    Clear decorator registry.

    Warning: This is primarily for testing. Use with caution in production.
    """
    _DECORATOR_REGISTRY.clear()
