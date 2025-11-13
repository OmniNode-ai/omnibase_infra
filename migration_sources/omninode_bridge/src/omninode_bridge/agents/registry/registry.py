"""Agent registration and discovery system."""

import logging
import time
from datetime import datetime
from typing import Any, Optional

from ..coordination.thread_safe_state import ThreadSafeState
from .cache import CacheManager
from .exceptions import AgentNotFoundError, DuplicateAgentError, NoAgentFoundError
from .matcher import CapabilityMatchEngine
from .models import AgentInfo, AgentMetadata, AgentStatus, RegistrationResult, Task

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Agent registration and discovery system.

    Features:
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

    Example:
        ```python
        from omninode_bridge.agents.coordination import ThreadSafeState
        from omninode_bridge.agents.registry import AgentRegistry

        state = ThreadSafeState()
        registry = AgentRegistry(state=state, enable_cache=True)

        # Register agent
        result = registry.register_agent(
            agent_id="contract_inferencer",
            capabilities=["contract_inference", "yaml_parsing"],
            metadata=AgentMetadata(
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description="Infers contracts from YAML"
            )
        )

        # Match task to agent
        task = Task(
            task_type="contract_inference",
            required_capabilities=["contract_inference"]
        )
        agent, confidence = registry.match_agent(task)
        print(f"Matched: {agent.agent_id} with confidence {confidence:.2f}")
        ```
    """

    def __init__(
        self,
        state: ThreadSafeState,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 300,
        cache_max_size: int = 1000,
    ) -> None:
        """
        Initialize agent registry.

        Args:
            state: ThreadSafeState instance for centralized storage
            enable_cache: Enable caching for discovery (default: True)
            cache_ttl_seconds: Cache TTL in seconds (default: 300s = 5min)
            cache_max_size: Maximum cache entries (default: 1000)
        """
        self.state = state
        self.capability_matcher = CapabilityMatchEngine()

        # Initialize cache if enabled
        self.cache: Optional[CacheManager] = None
        if enable_cache:
            self.cache = CacheManager(
                max_size=cache_max_size, ttl_seconds=cache_ttl_seconds
            )

        # Initialize registry in ThreadSafeState
        self._initialize_state()

    def register_agent(
        self, agent_id: str, capabilities: list[str], metadata: AgentMetadata
    ) -> RegistrationResult:
        """
        Register an agent with capabilities.

        Performance Target: <50ms per registration

        Args:
            agent_id: Unique agent identifier
            capabilities: List of capability tags (e.g., ["contract_inference", "llm"])
            metadata: Agent metadata (version, description, etc.)

        Returns:
            RegistrationResult with success status and details

        Raises:
            DuplicateAgentError: If agent_id is already registered
            ValueError: If inputs are invalid
        """
        start_time = time.time()

        # 1. Validate inputs
        if not agent_id:
            raise ValueError("agent_id cannot be empty")

        if not capabilities:
            raise ValueError("capabilities cannot be empty")

        # 2. Check for duplicates
        if self._agent_exists(agent_id):
            raise DuplicateAgentError(agent_id)

        # 3. Create agent info
        agent_info = AgentInfo(
            agent_id=agent_id,
            capabilities=capabilities,
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        # 4. Store in ThreadSafeState
        agents = self.state.get("agents", {})
        agents[agent_id] = agent_info.model_dump()

        # Update capability index
        capability_index = self.state.get("capability_index", {})
        for capability in capabilities:
            if capability not in capability_index:
                capability_index[capability] = set()
            capability_index[capability].add(agent_id)

        # Atomic update
        self.state.set("agents", agents, changed_by="registry")
        self.state.set("capability_index", capability_index, changed_by="registry")

        # 5. Invalidate cache (since new agent available)
        if self.cache:
            self.cache.invalidate_all()

        # 6. Update metrics
        registration_time_ms = (time.time() - start_time) * 1000
        self._update_metrics("registration_time_ms", registration_time_ms)

        logger.info(
            f"Registered agent '{agent_id}' with capabilities {capabilities} "
            f"in {registration_time_ms:.2f}ms"
        )

        return RegistrationResult(
            success=True,
            agent_id=agent_id,
            message=f"Agent '{agent_id}' registered successfully",
            registration_time_ms=registration_time_ms,
        )

    def discover_agents(
        self, capability: str, status_filter: Optional[AgentStatus] = AgentStatus.ACTIVE
    ) -> list[AgentInfo]:
        """
        Discover agents by capability.

        Performance: O(1) lookup via capability index

        Args:
            capability: Capability tag to search for
            status_filter: Filter by agent status (default: ACTIVE only)

        Returns:
            List of matching agents (sorted by registration time)
        """
        capability_index = self.state.get("capability_index", {})
        agent_ids = capability_index.get(capability, set())

        agents_data = self.state.get("agents", {})
        matching_agents = []

        for agent_id in agent_ids:
            if agent_id in agents_data:
                agent_data = agents_data[agent_id]
                agent_info = AgentInfo(**agent_data)

                # Apply status filter
                if status_filter is None or agent_info.status == status_filter:
                    matching_agents.append(agent_info)

        # Sort by registration time (newest first)
        matching_agents.sort(key=lambda a: a.registered_at, reverse=True)

        return matching_agents

    def match_agent(self, task: Task) -> tuple[AgentInfo, float]:
        """
        Match task to best agent with confidence scoring.

        Performance:
        - Cache hit: <5ms (85-95% of requests)
        - Cache miss: <100ms (capability matching + scoring)

        Args:
            task: Task to match (contains requirements, capabilities, etc.)

        Returns:
            Tuple of (best_agent, confidence_score)
            confidence_score is 0.0-1.0 (1.0 = perfect match)

        Raises:
            NoAgentFoundError: If no suitable agent found
        """
        start_time = time.time()

        # 1. Check cache
        if self.cache:
            cache_key = self._generate_cache_key(task)
            cached = self.cache.get(cache_key)

            if cached is not None:
                # Cache hit!
                elapsed_ms = (time.time() - start_time) * 1000
                self._update_metrics("routing_time_ms", elapsed_ms)
                self._update_metrics("cache_hits", 1)

                agent_info = AgentInfo(**cached["agent_info"])
                return agent_info, cached["confidence"]

        # 2. Cache miss - perform capability matching
        agents_data = self.state.get("agents", {})
        active_agents = []

        for agent_data in agents_data.values():
            agent_info = AgentInfo(**agent_data)
            if agent_info.status == AgentStatus.ACTIVE:
                active_agents.append(agent_info)

        if not active_agents:
            raise NoAgentFoundError(
                "No active agents available",
                required_capabilities=task.required_capabilities,
            )

        # 3. Score all agents
        scored_agents = []
        for agent in active_agents:
            score = self.capability_matcher.score_agent(agent=agent, task=task)
            scored_agents.append((agent, score.total))

        # 4. Sort by confidence (highest first)
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # 5. Get best match
        best_agent, best_confidence = scored_agents[0]

        # 6. Check minimum confidence threshold
        if best_confidence < 0.3:
            raise NoAgentFoundError(
                f"No suitable agent found (best confidence: {best_confidence:.2f})",
                required_capabilities=task.required_capabilities,
            )

        # 7. Cache result
        if self.cache:
            self.cache.set(
                cache_key,
                {"agent_info": best_agent.model_dump(), "confidence": best_confidence},
            )

        # 8. Update metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics("routing_time_ms", elapsed_ms)
        self._update_metrics("cache_misses", 1)

        logger.debug(
            f"Matched task {task.task_id} to agent '{best_agent.agent_id}' "
            f"with confidence {best_confidence:.2f} in {elapsed_ms:.2f}ms"
        )

        return best_agent, best_confidence

    def get_agent(self, agent_id: str) -> AgentInfo:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentInfo

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        agents = self.state.get("agents", {})

        if agent_id not in agents:
            raise AgentNotFoundError(agent_id)

        return AgentInfo(**agents[agent_id])

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        agents = self.state.get("agents", {})

        if agent_id not in agents:
            raise AgentNotFoundError(agent_id)

        agent_data = agents[agent_id]
        agent_info = AgentInfo(**agent_data)

        # Remove from agents
        del agents[agent_id]

        # Remove from capability index
        capability_index = self.state.get("capability_index", {})
        for capability in agent_info.capabilities:
            if capability in capability_index:
                capability_index[capability].discard(agent_id)
                if not capability_index[capability]:
                    del capability_index[capability]

        # Atomic update
        self.state.set("agents", agents, changed_by="registry")
        self.state.set("capability_index", capability_index, changed_by="registry")

        # Invalidate cache
        if self.cache:
            self.cache.invalidate_all()

        logger.info(f"Unregistered agent '{agent_id}'")

    def list_agents(self) -> list[AgentInfo]:
        """
        List all registered agents.

        Returns:
            List of all agents (sorted by registration time, newest first)
        """
        agents_data = self.state.get("agents", {})
        agents = [AgentInfo(**data) for data in agents_data.values()]

        # Sort by registration time (newest first)
        agents.sort(key=lambda a: a.registered_at, reverse=True)

        return agents

    def heartbeat(self, agent_id: str) -> None:
        """
        Update agent heartbeat timestamp.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        agents = self.state.get("agents", {})

        if agent_id not in agents:
            raise AgentNotFoundError(agent_id)

        agent_data = agents[agent_id]
        agent_info = AgentInfo(**agent_data)

        # Update heartbeat
        agent_info.last_heartbeat = datetime.utcnow()
        agent_info.status = AgentStatus.ACTIVE

        agents[agent_id] = agent_info.model_dump()
        self.state.set("agents", agents, changed_by="heartbeat")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics or empty dict if caching disabled
        """
        if self.cache:
            stats = self.cache.get_stats()
            return stats.model_dump()
        return {}

    # Private methods

    def _agent_exists(self, agent_id: str) -> bool:
        """Check if agent exists."""
        agents = self.state.get("agents", {})
        return agent_id in agents

    def _generate_cache_key(self, task: Task) -> str:
        """Generate cache key from task."""
        requirements = sorted(task.requirements)
        capabilities = sorted(task.required_capabilities)

        key_parts = [task.task_type, "|".join(requirements), "|".join(capabilities)]

        return ":".join(key_parts)

    def _update_metrics(self, metric_name: str, value: float) -> None:
        """Update registry metrics."""
        metrics = self.state.get("registry_metrics", {})

        if metric_name not in metrics:
            metrics[metric_name] = {
                "count": 0,
                "sum": 0.0,
                "min": float("inf"),
                "max": 0.0,
            }

        metrics[metric_name]["count"] += 1
        metrics[metric_name]["sum"] += value
        metrics[metric_name]["min"] = min(metrics[metric_name]["min"], value)
        metrics[metric_name]["max"] = max(metrics[metric_name]["max"], value)

        self.state.set("registry_metrics", metrics, changed_by="registry")

    def _initialize_state(self) -> None:
        """Initialize registry state in ThreadSafeState."""
        if not self.state.has("agents"):
            self.state.set("agents", {}, changed_by="registry")

        if not self.state.has("capability_index"):
            self.state.set("capability_index", {}, changed_by="registry")

        if not self.state.has("registry_metrics"):
            self.state.set("registry_metrics", {}, changed_by="registry")
