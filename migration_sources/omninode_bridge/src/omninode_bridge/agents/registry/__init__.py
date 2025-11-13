"""
Agent Registration & Discovery System.

This module provides production-ready agent registry with:
- Dynamic agent registration with capability tracking
- Fast agent discovery with caching (<5ms cache hit)
- Capability matching with confidence scoring (0.0-1.0)
- ThreadSafeState integration for centralized storage
- Heartbeat monitoring for agent health

Performance Targets:
- Registration: <50ms per agent
- Discovery (cache hit): <5ms
- Discovery (cache miss): <100ms
- Cache hit rate: 85-95%
"""

from .cache import CacheEntry, CacheManager, CacheStats
from .decorators import (
    agent_is_registered,
    find_agents_by_capability,
    get_agent_instance,
    get_agent_metadata,
    get_registered_agents,
    register_agent,
)
from .exceptions import (
    AgentNotFoundError,
    AgentRegistryError,
    DuplicateAgentError,
    NoAgentFoundError,
)
from .matcher import CapabilityMatchEngine
from .models import (
    AgentInfo,
    AgentMetadata,
    AgentStatus,
    AgentType,
    ConfidenceScore,
    RegistrationResult,
    Task,
)
from .registry import AgentRegistry

__all__ = [
    # Core Registry
    "AgentRegistry",
    # Data Models
    "AgentInfo",
    "AgentMetadata",
    "AgentStatus",
    "AgentType",
    "Task",
    "RegistrationResult",
    "ConfidenceScore",
    # Capability Matching
    "CapabilityMatchEngine",
    # Caching
    "CacheManager",
    "CacheEntry",
    "CacheStats",
    # Decorators
    "register_agent",
    "get_registered_agents",
    "agent_is_registered",
    "get_agent_metadata",
    "find_agents_by_capability",
    "get_agent_instance",
    # Exceptions
    "AgentRegistryError",
    "AgentNotFoundError",
    "NoAgentFoundError",
    "DuplicateAgentError",
]
