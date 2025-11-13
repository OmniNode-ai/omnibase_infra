# Agent Registration & Discovery Architecture

**Document Version**: 1.0
**Date**: 2025-11-06
**Status**: Design Phase
**Target Implementation**: Wave 3 (Phase 4 - Agent Framework Integration)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [ThreadSafeState Lock API](#2-threadsafestate-lock-api)
3. [Class Design](#3-class-design)
4. [Registration Protocol](#4-registration-protocol)
5. [Discovery Mechanism](#5-discovery-mechanism)
6. [Capability Matching](#6-capability-matching)
7. [Caching Design](#7-caching-design)
8. [Integration Design](#8-integration-design)
9. [Error Handling](#9-error-handling)
10. [Testing Strategy](#10-testing-strategy)
11. [Implementation Plan](#11-implementation-plan)

---

## 1. Architecture Overview

### 1.1 Purpose

The Agent Registry provides a **production-ready agent registration and discovery system** with:
- **Dynamic Registration**: Agents register at startup with capabilities
- **Fast Discovery**: <5ms agent lookup with caching (85-95% cache hit rate)
- **Capability Matching**: Intelligent task-to-agent routing with confidence scoring (0.0-1.0)
- **ThreadSafeState Integration**: Centralized state management
- **Type Safety**: Strong Pydantic models for all data structures

### 1.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Registry                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AgentRegistry (Core)                         │   │
│  │  - register_agent()                                       │   │
│  │  - discover_agents()                                      │   │
│  │  - match_agent()                                          │   │
│  │  - get_agent()                                            │   │
│  │  - unregister_agent()                                     │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                              │
│       ┌───────────┼───────────┐                                 │
│       │           │           │                                 │
│       ↓           ↓           ↓                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│  │CapMatch │ │  Cache  │ │ State   │                           │
│  │ Engine  │ │ Manager │ │ Manager │                           │
│  └─────────┘ └─────────┘ └─────────┘                           │
│       ↓           ↓           ↓                                 │
│  ┌─────────────────────────────────┐                           │
│  │    ThreadSafeState Storage      │                           │
│  │  - agents: Dict[str, AgentInfo] │                           │
│  │  - cache: Dict[str, CacheEntry] │                           │
│  │  - metrics: RegistryMetrics     │                           │
│  └─────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ↓
        ┌───────────────┴────────────────┐
        │                                │
        ↓                                ↓
┌──────────────┐              ┌──────────────────┐
│ Kafka Events │              │ Performance      │
│ - agent.reg  │              │ Metrics          │
│ - agent.dereg│              │ - routing_time   │
│ - agent.match│              │ - cache_hit_rate │
└──────────────┘              └──────────────────┘
```

### 1.3 Registration Flow

```
Agent Startup
    │
    ↓
┌─────────────────────────────┐
│ @register_agent Decorator   │
│ - agent_id: str             │
│ - capabilities: List[str]   │
│ - metadata: AgentMetadata   │
└──────────┬──────────────────┘
           │
           ↓
┌─────────────────────────────┐
│ AgentRegistry.register()    │
│ 1. Validate metadata        │
│ 2. Check duplicates         │
│ 3. Store in ThreadSafeState │
│ 4. Publish Kafka event      │
│ 5. Start heartbeat          │
└──────────┬──────────────────┘
           │
           ↓
┌─────────────────────────────┐
│ ThreadSafeState             │
│ - agents[agent_id] = info   │
│ - Indexed by capabilities   │
└─────────────────────────────┘
```

### 1.4 Discovery Flow

```
Task Arrives
    │
    ↓
┌─────────────────────────────┐
│ AgentRegistry.match_agent() │
│ 1. Check cache (key=task)   │
│ 2. If hit: return <5ms      │
│ 3. If miss: capability match│
└──────────┬──────────────────┘
           │
           ↓ (cache miss)
┌─────────────────────────────┐
│ CapabilityMatchEngine       │
│ 1. Extract task requirements│
│ 2. Score all agents (0-1.0) │
│ 3. Rank by confidence       │
│ 4. Return top match         │
└──────────┬──────────────────┘
           │
           ↓
┌─────────────────────────────┐
│ CacheManager.set()          │
│ - Cache result (TTL=300s)   │
│ - LRU eviction (max=1000)   │
└──────────┬──────────────────┘
           │
           ↓
Return (agent_info, confidence)
```

---

## 2. ThreadSafeState Lock API

### 2.1 Lock Usage Pattern

**IMPORTANT**: All lock operations in this architecture use **asynchronous context managers**.

The `ThreadSafeState` class provides two lock types:
- `read_lock()` - For read-only operations (multiple readers allowed)
- `write_lock()` - For write operations (exclusive access)

**Correct Usage** (Async Context Manager):
```python
# Read operations
async with self.state.read_lock():
    agents = self.state.get("agents", {})
    # Perform read operations

# Write operations
async with self.state.write_lock():
    agents = self.state.get("agents", {})
    agents[agent_id] = agent_info
    self.state.set("agents", agents)
```

**Incorrect Usage** ❌:
```python
# DO NOT use synchronous context manager
with self.state.write_lock():  # ❌ WRONG
    # This will cause issues in async context
```

### 2.2 Lock API Rationale

- **Async-First Design**: All registry operations are asynchronous
- **Non-Blocking**: Async locks don't block the event loop
- **Thread Safety**: Protects shared state across concurrent operations
- **Consistent Pattern**: All 13 lock usage sites in this document use `async with`

### 2.3 Lock Best Practices

1. **Minimize Lock Duration**: Keep critical sections short
2. **Read Lock for Reads**: Use `read_lock()` for read-only operations (allows concurrent reads)
3. **Write Lock for Mutations**: Use `write_lock()` for any state modifications
4. **No Nested Locks**: Avoid nested lock acquisitions to prevent deadlocks
5. **Early Release**: Release locks before I/O operations (Kafka, external APIs)

---

## 3. Class Design

### 3.1 Core Classes

#### 3.1.1 AgentRegistry

```python
from typing import Dict, List, Tuple, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
import asyncio

class AgentRegistry:
    """
    Agent registration and discovery system.

    Features:
    - Dynamic agent registration with capability tracking
    - Fast agent discovery with caching (<5ms cache hit)
    - Capability matching with confidence scoring (0.0-1.0)
    - ThreadSafeState integration for centralized storage
    - Kafka event publishing for observability

    Performance Targets:
    - Registration: <50ms per agent
    - Discovery (cache hit): <5ms
    - Discovery (cache miss): <100ms
    - Cache hit rate: 85-95%
    """

    def __init__(
        self,
        state: ThreadSafeState,
        kafka_producer: Optional[KafkaProducer] = None,
        enable_cache: bool = True,
        enable_heartbeat: bool = True,
        cache_ttl_seconds: int = 300,
        cache_max_size: int = 1000
    ) -> None:
        """
        Initialize agent registry.

        Args:
            state: ThreadSafeState instance for centralized storage
            kafka_producer: Optional Kafka producer for event publishing
            enable_cache: Enable caching for discovery (default: True)
            enable_heartbeat: Enable agent heartbeat monitoring (default: True)
            cache_ttl_seconds: Cache TTL in seconds (default: 300s = 5min)
            cache_max_size: Maximum cache entries (default: 1000)
        """
        self.state = state
        self.kafka_producer = kafka_producer

        # Initialize components
        self.capability_matcher = CapabilityMatchEngine()

        if enable_cache:
            self.cache = CacheManager(
                max_size=cache_max_size,
                ttl_seconds=cache_ttl_seconds
            )
        else:
            self.cache = None

        # Initialize registry in ThreadSafeState (async initialization)
        self._init_task = asyncio.create_task(self._initialize_state())

        # Start heartbeat monitor
        if enable_heartbeat:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        metadata: AgentMetadata
    ) -> RegistrationResult:
        """
        Register an agent with capabilities.

        Args:
            agent_id: Unique agent identifier
            capabilities: List of capability tags (e.g., ["contract_inference", "llm"])
            metadata: Agent metadata (version, description, etc.)

        Returns:
            RegistrationResult with success status and details

        Raises:
            ValueError: If agent_id is duplicate or invalid
            ValidationError: If metadata is invalid

        Performance: <50ms per registration
        """
        start_time = time.time()

        # 1. Validate inputs
        if not agent_id:
            raise ValueError("agent_id cannot be empty")

        if not capabilities:
            raise ValueError("capabilities cannot be empty")

        # 2. Check for duplicates
        if await self._agent_exists(agent_id):
            raise ValueError(f"Agent '{agent_id}' already registered")

        # 3. Validate metadata
        try:
            validated_metadata = AgentMetadata.model_validate(metadata)
        except Exception as e:
            raise ValidationError(f"Invalid metadata: {e}")

        # 4. Create agent info
        agent_info = AgentInfo(
            agent_id=agent_id,
            capabilities=capabilities,
            metadata=validated_metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE
        )

        # 5. Store in ThreadSafeState
        async with self.state.write_lock():
            agents = self.state.get("agents", {})
            agents[agent_id] = agent_info

            # Update capability index
            capability_index = self.state.get("capability_index", {})
            for capability in capabilities:
                if capability not in capability_index:
                    capability_index[capability] = set()
                capability_index[capability].add(agent_id)

            self.state.set("agents", agents)
            self.state.set("capability_index", capability_index)

        # 6. Invalidate cache (since new agent available)
        if self.cache:
            self.cache.invalidate_all()

        # 7. Publish Kafka event
        if self.kafka_producer:
            await self._publish_registration_event(agent_info)

        # 8. Update metrics
        registration_time_ms = (time.time() - start_time) * 1000
        await self._update_metrics("registration_time_ms", registration_time_ms)

        return RegistrationResult(
            success=True,
            agent_id=agent_id,
            message=f"Agent '{agent_id}' registered successfully",
            registration_time_ms=registration_time_ms
        )

    async def discover_agents(
        self,
        capability: str,
        status_filter: Optional[AgentStatus] = AgentStatus.ACTIVE
    ) -> List[AgentInfo]:
        """
        Discover agents by capability.

        Args:
            capability: Capability tag to search for
            status_filter: Filter by agent status (default: ACTIVE only)

        Returns:
            List of matching agents (sorted by registration time)

        Performance: O(1) lookup via capability index
        """
        async with self.state.read_lock():
            capability_index = self.state.get("capability_index", {})
            agent_ids = capability_index.get(capability, set())

            agents = self.state.get("agents", {})
            matching_agents = [
                agents[agent_id]
                for agent_id in agent_ids
                if agent_id in agents
                and (status_filter is None or agents[agent_id].status == status_filter)
            ]

        # Sort by registration time (newest first)
        matching_agents.sort(key=lambda a: a.registered_at, reverse=True)

        return matching_agents

    async def match_agent(
        self,
        task: Task
    ) -> Tuple[AgentInfo, float]:
        """
        Match task to best agent with confidence scoring.

        Args:
            task: Task to match (contains requirements, capabilities, etc.)

        Returns:
            Tuple of (best_agent, confidence_score)
            confidence_score is 0.0-1.0 (1.0 = perfect match)

        Raises:
            NoAgentFoundError: If no suitable agent found

        Performance:
        - Cache hit: <5ms (85-95% of requests)
        - Cache miss: <100ms (capability matching + scoring)
        """
        start_time = time.time()

        # 1. Check cache
        if self.cache:
            cache_key = self._generate_cache_key(task)
            cached = self.cache.get(cache_key)

            if cached is not None:
                # Cache hit!
                elapsed_ms = (time.time() - start_time) * 1000
                await self._update_metrics("routing_time_ms", elapsed_ms)
                await self._update_metrics("cache_hits", 1)

                return cached["agent_info"], cached["confidence"]

        # 2. Cache miss - perform capability matching
        async with self.state.read_lock():
            agents = self.state.get("agents", {})
            active_agents = [
                agent for agent in agents.values()
                if agent.status == AgentStatus.ACTIVE
            ]

        if not active_agents:
            raise NoAgentFoundError("No active agents available")

        # 3. Score all agents
        scored_agents = []
        for agent in active_agents:
            confidence = await self.capability_matcher.score_agent(
                agent=agent,
                task=task
            )
            scored_agents.append((agent, confidence))

        # 4. Sort by confidence (highest first)
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # 5. Get best match
        best_agent, best_confidence = scored_agents[0]

        # 6. Check minimum confidence threshold
        if best_confidence < 0.3:
            raise NoAgentFoundError(
                f"No suitable agent found (best confidence: {best_confidence:.2f})"
            )

        # 7. Cache result
        if self.cache:
            self.cache.set(
                cache_key,
                {"agent_info": best_agent, "confidence": best_confidence}
            )

        # 8. Update metrics
        elapsed_ms = (time.time() - start_time) * 1000
        await self._update_metrics("routing_time_ms", elapsed_ms)
        await self._update_metrics("cache_misses", 1)

        # 9. Publish match event
        if self.kafka_producer:
            await self._publish_match_event(task, best_agent, best_confidence)

        return best_agent, best_confidence

    async def get_agent(self, agent_id: str) -> AgentInfo:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentInfo

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        async with self.state.read_lock():
            agents = self.state.get("agents", {})

            if agent_id not in agents:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found")

            return agents[agent_id]

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        async with self.state.write_lock():
            agents = self.state.get("agents", {})

            if agent_id not in agents:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found")

            agent_info = agents[agent_id]

            # Remove from agents
            del agents[agent_id]

            # Remove from capability index
            capability_index = self.state.get("capability_index", {})
            for capability in agent_info.capabilities:
                if capability in capability_index:
                    capability_index[capability].discard(agent_id)
                    if not capability_index[capability]:
                        del capability_index[capability]

            self.state.set("agents", agents)
            self.state.set("capability_index", capability_index)

        # Invalidate cache
        if self.cache:
            self.cache.invalidate_all()

        # Publish deregistration event
        if self.kafka_producer:
            await self._publish_deregistration_event(agent_id)

    # Private methods

    async def _agent_exists(self, agent_id: str) -> bool:
        """Check if agent exists"""
        async with self.state.read_lock():
            agents = self.state.get("agents", {})
            return agent_id in agents

    def _generate_cache_key(self, task: Task) -> str:
        """Generate cache key from task"""
        # Hash based on task requirements and capabilities
        requirements = sorted(task.requirements)
        capabilities = sorted(task.required_capabilities)

        key_parts = [
            task.task_type,
            "|".join(requirements),
            "|".join(capabilities)
        ]

        return ":".join(key_parts)

    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats and mark inactive agents"""
        while True:
            await asyncio.sleep(30)  # Check every 30s

            async with self.state.write_lock():
                agents = self.state.get("agents", {})
                now = datetime.utcnow()

                for agent_id, agent_info in agents.items():
                    # Mark inactive if no heartbeat for 5min
                    if (now - agent_info.last_heartbeat).total_seconds() > 300:
                        agent_info.status = AgentStatus.INACTIVE

                self.state.set("agents", agents)

    async def _publish_registration_event(self, agent_info: AgentInfo) -> None:
        """Publish agent registration event to Kafka"""
        event = {
            "event_type": "agent.registered.v1",
            "agent_id": agent_info.agent_id,
            "capabilities": agent_info.capabilities,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": agent_info.metadata.model_dump()
        }
        await self.kafka_producer.send("agent-registration-events", event)

    async def _publish_deregistration_event(self, agent_id: str) -> None:
        """Publish agent deregistration event to Kafka"""
        event = {
            "event_type": "agent.deregistered.v1",
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.kafka_producer.send("agent-registration-events", event)

    async def _publish_match_event(
        self,
        task: Task,
        agent: AgentInfo,
        confidence: float
    ) -> None:
        """Publish agent match event to Kafka"""
        event = {
            "event_type": "agent.matched.v1",
            "task_id": str(task.task_id),
            "agent_id": agent.agent_id,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.kafka_producer.send("agent-routing-events", event)

    async def _update_metrics(self, metric_name: str, value: float) -> None:
        """Update registry metrics"""
        async with self.state.write_lock():
            metrics = self.state.get("registry_metrics", {})

            if metric_name not in metrics:
                metrics[metric_name] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": 0.0
                }

            metrics[metric_name]["count"] += 1
            metrics[metric_name]["sum"] += value
            metrics[metric_name]["min"] = min(metrics[metric_name]["min"], value)
            metrics[metric_name]["max"] = max(metrics[metric_name]["max"], value)

            self.state.set("registry_metrics", metrics)

    async def _initialize_state(self) -> None:
        """Initialize registry state in ThreadSafeState"""
        async with self.state.write_lock():
            if not self.state.get("agents"):
                self.state.set("agents", {})

            if not self.state.get("capability_index"):
                self.state.set("capability_index", {})

            if not self.state.get("registry_metrics"):
                self.state.set("registry_metrics", {})
```

#### 3.1.2 Data Models

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
from uuid import UUID

class AgentStatus(str, Enum):
    """Agent status enum"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class AgentType(str, Enum):
    """Agent type enum"""
    CONTRACT_INFERENCER = "contract_inferencer"
    TEMPLATE_SELECTOR = "template_selector"
    BUSINESS_LOGIC_GENERATOR = "business_logic_generator"
    VALIDATOR = "validator"
    DEBUG = "debug"
    ANALYZER = "analyzer"
    RESEARCHER = "researcher"

class AgentMetadata(BaseModel):
    """Agent metadata"""
    agent_type: AgentType
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: str
    author: Optional[str] = None
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    priority: int = Field(default=50, ge=0, le=100)

    # Performance characteristics
    avg_execution_time_ms: Optional[float] = None
    success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Additional metadata
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)

class AgentInfo(BaseModel):
    """Complete agent information"""
    agent_id: str
    capabilities: List[str]
    metadata: AgentMetadata
    registered_at: datetime
    last_heartbeat: datetime
    status: AgentStatus = AgentStatus.ACTIVE

    # Runtime metrics
    active_tasks: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0

class Task(BaseModel):
    """Task to be matched to an agent"""
    task_id: UUID
    task_type: str
    requirements: List[str] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=100)
    timeout_seconds: int = Field(default=300)

    # Context for matching
    complexity: Optional[str] = None  # "simple", "medium", "complex"
    domain: Optional[str] = None
    node_type: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

class RegistrationResult(BaseModel):
    """Result of agent registration"""
    success: bool
    agent_id: str
    message: str
    registration_time_ms: float
    errors: List[str] = Field(default_factory=list)

class ConfidenceScore(BaseModel):
    """Confidence score breakdown"""
    total: float = Field(..., ge=0.0, le=1.0)
    capability_score: float = Field(..., ge=0.0, le=1.0)
    load_score: float = Field(..., ge=0.0, le=1.0)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    success_rate_score: float = Field(..., ge=0.0, le=1.0)

    explanation: str
```

#### 3.1.3 CapabilityMatchEngine

```python
class CapabilityMatchEngine:
    """
    Capability matching engine with multi-criteria scoring.

    Scoring Algorithm:
    - Capability Match: 40% weight (Jaccard similarity)
    - Load Balance: 20% weight (1 - active_tasks / max_tasks)
    - Priority: 20% weight (agent priority / 100)
    - Success Rate: 20% weight (success_rate)

    Total Score: weighted sum (0.0-1.0)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize capability matcher.

        Args:
            weights: Custom weights for scoring criteria
                     Default: {"capability": 0.4, "load": 0.2, "priority": 0.2, "success_rate": 0.2}
        """
        self.weights = weights or {
            "capability": 0.4,
            "load": 0.2,
            "priority": 0.2,
            "success_rate": 0.2
        }

    async def score_agent(
        self,
        agent: AgentInfo,
        task: Task
    ) -> float:
        """
        Score agent suitability for task.

        Args:
            agent: Agent to score
            task: Task to match

        Returns:
            Confidence score (0.0-1.0)
        """
        # 1. Capability matching (Jaccard similarity)
        agent_caps = set(agent.capabilities)
        task_caps = set(task.required_capabilities)

        if not task_caps:
            capability_score = 1.0  # No specific requirements
        else:
            intersection = len(agent_caps & task_caps)
            union = len(agent_caps | task_caps)
            capability_score = intersection / union if union > 0 else 0.0

        # 2. Load balancing
        max_tasks = agent.metadata.max_concurrent_tasks
        active_tasks = agent.active_tasks

        if max_tasks > 0:
            load_score = 1.0 - (active_tasks / max_tasks)
        else:
            load_score = 0.0

        # 3. Priority match
        priority_score = agent.metadata.priority / 100.0

        # 4. Success rate
        success_rate_score = agent.metadata.success_rate or 0.5

        # 5. Calculate weighted total
        total_score = (
            self.weights["capability"] * capability_score +
            self.weights["load"] * load_score +
            self.weights["priority"] * priority_score +
            self.weights["success_rate"] * success_rate_score
        )

        return total_score
```

#### 3.1.4 CacheManager

```python
from collections import OrderedDict
from typing import Optional, Any
import time

class CacheManager:
    """
    LRU cache with TTL for agent discovery results.

    Performance Targets:
    - Get: <5ms (cache hit)
    - Set: <10ms
    - Cache hit rate: 85-95%

    Eviction Policy:
    - LRU (Least Recently Used)
    - TTL-based expiration
    - Max size limit
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300
    ) -> None:
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cache entries (default: 1000)
            ttl_seconds: Time-to-live in seconds (default: 300s = 5min)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired

        Performance: <5ms
        """
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            # Expired - remove
            del self.cache[key]
            self.misses += 1
            return None

        # Cache hit - move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1

        return entry.value

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache

        Performance: <10ms
        """
        # Check if key exists
        if key in self.cache:
            # Update existing
            self.cache[key] = CacheEntry(value=value, timestamp=time.time())
            self.cache.move_to_end(key)
        else:
            # New entry
            if len(self.cache) >= self.max_size:
                # Evict LRU (first item)
                self.cache.popitem(last=False)
                self.evictions += 1

            self.cache[key] = CacheEntry(value=value, timestamp=time.time())

    def invalidate(self, key: str) -> None:
        """Invalidate specific cache entry"""
        if key in self.cache:
            del self.cache[key]

    def invalidate_all(self) -> None:
        """Invalidate entire cache"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }

class CacheEntry(BaseModel):
    """Cache entry with timestamp"""
    value: Any
    timestamp: float
```

---

## 4. Registration Protocol

### 4.1 Decorator-Based Registration

```python
# NOTE: This is design/example code showing the proposed registration pattern.
# The get_global_registry() function is not yet implemented in the codebase.

from typing import Callable, TypeVar, List
from functools import wraps
import asyncio
from omninode_bridge.agents.registry import AgentRegistry, AgentMetadata, AgentType

T = TypeVar('T')

def register_agent(
    agent_id: str,
    capabilities: List[str],
    agent_type: AgentType,
    version: str,
    description: str,
    **metadata_kwargs
) -> Callable[[T], T]:
    """
    Decorator for agent self-registration.

    Usage:
        @register_agent(
            agent_id="contract_inferencer_v1",
            capabilities=["contract_inference", "yaml_parsing"],
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Infers contracts from YAML specifications"
        )
        class ContractInferencer:
            async def execute(self, task: Task) -> Result:
                # Agent implementation
                pass

    Args:
        agent_id: Unique agent identifier
        capabilities: List of capability tags
        agent_type: Agent type enum
        version: Semantic version (e.g., "1.0.0")
        description: Human-readable description
        **metadata_kwargs: Additional metadata fields

    Returns:
        Decorator function
    """
    def decorator(cls_or_func: T) -> T:
        # Create metadata
        metadata = AgentMetadata(
            agent_type=agent_type,
            version=version,
            description=description,
            **metadata_kwargs
        )

        # Get global registry instance
        # NOTE: get_global_registry() would return a singleton AgentRegistry instance
        registry = get_global_registry()

        # Register agent (sync version - called at import time)
        asyncio.run(
            registry.register_agent(
                agent_id=agent_id,
                capabilities=capabilities,
                metadata=metadata
            )
        )

        # Attach registration info to class/function
        cls_or_func._agent_id = agent_id
        cls_or_func._capabilities = capabilities
        cls_or_func._metadata = metadata

        return cls_or_func

    return decorator
```

### 4.2 Manual Registration

```python
# NOTE: This is design/example code showing the proposed registration pattern.
# The get_global_registry() function is not yet implemented in the codebase.

from typing import Any, List
from omninode_bridge.agents.registry import AgentRegistry, AgentMetadata, RegistrationResult

# For agents that need programmatic registration
async def register_agent_programmatic(
    agent_id: str,
    agent_instance: Any,
    capabilities: List[str],
    metadata: AgentMetadata
) -> RegistrationResult:
    """
    Programmatically register an agent instance.

    Args:
        agent_id: Unique agent identifier
        agent_instance: Agent instance (must have execute() method)
        capabilities: List of capability tags
        metadata: Agent metadata

    Returns:
        RegistrationResult
    """
    # NOTE: get_global_registry() would return a singleton AgentRegistry instance
    registry = get_global_registry()

    # Validate agent has execute method
    if not hasattr(agent_instance, "execute"):
        raise ValueError("Agent must have execute() method")

    # Register
    result = await registry.register_agent(
        agent_id=agent_id,
        capabilities=capabilities,
        metadata=metadata
    )

    # Store instance reference
    registry._agent_instances[agent_id] = agent_instance

    return result
```

### 4.3 Heartbeat Protocol

```python
class HeartbeatManager:
    """
    Manages agent heartbeat lifecycle.

    Heartbeat Interval: 60s (configurable)
    Timeout Threshold: 5min (3 missed heartbeats)
    """

    def __init__(
        self,
        agent_id: str,
        registry: AgentRegistry,
        interval_seconds: int = 60
    ) -> None:
        self.agent_id = agent_id
        self.registry = registry
        self.interval_seconds = interval_seconds
        self.heartbeat_task = None

    async def start(self) -> None:
        """Start heartbeat loop"""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop heartbeat loop"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop"""
        while True:
            await asyncio.sleep(self.interval_seconds)

            try:
                await self.registry.heartbeat(self.agent_id)
            except Exception as e:
                logger.error(f"Heartbeat failed for {self.agent_id}: {e}")

    async def heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat timestamp"""
        async with self.state.write_lock():
            agents = self.state.get("agents", {})

            if agent_id in agents:
                agents[agent_id].last_heartbeat = datetime.utcnow()
                agents[agent_id].status = AgentStatus.ACTIVE
                self.state.set("agents", agents)
```

---

## 5. Discovery Mechanism

### 5.1 Capability-Based Lookup

```python
async def discover_by_capability(
    registry: AgentRegistry,
    capability: str,
    min_confidence: float = 0.5
) -> List[Tuple[AgentInfo, float]]:
    """
    Discover agents by capability with confidence scoring.

    Args:
        registry: AgentRegistry instance
        capability: Capability tag
        min_confidence: Minimum confidence threshold (default: 0.5)

    Returns:
        List of (agent, confidence) tuples sorted by confidence
    """
    # Get all agents with capability
    agents = await registry.discover_agents(capability)

    # Score each agent
    scored_agents = []
    for agent in agents:
        confidence = _calculate_capability_confidence(agent, capability)

        if confidence >= min_confidence:
            scored_agents.append((agent, confidence))

    # Sort by confidence (highest first)
    scored_agents.sort(key=lambda x: x[1], reverse=True)

    return scored_agents

def _calculate_capability_confidence(
    agent: AgentInfo,
    capability: str
) -> float:
    """
    Calculate confidence for capability match.

    Factors:
    - Exact match: 1.0
    - Success rate: 0-0.2 boost
    - Load: 0-0.2 penalty
    """
    # Base confidence (exact match)
    confidence = 1.0 if capability in agent.capabilities else 0.0

    # Success rate boost (up to +0.2)
    if agent.metadata.success_rate:
        confidence += agent.metadata.success_rate * 0.2

    # Load penalty (up to -0.2)
    load_ratio = agent.active_tasks / agent.metadata.max_concurrent_tasks
    confidence -= load_ratio * 0.2

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, confidence))
```

### 5.2 Tag-Based Filtering

```python
async def discover_by_tags(
    registry: AgentRegistry,
    tags: List[str],
    match_all: bool = False
) -> List[AgentInfo]:
    """
    Discover agents by metadata tags.

    Args:
        registry: AgentRegistry instance
        tags: List of tags to match
        match_all: If True, agent must have all tags (AND logic)
                   If False, agent must have at least one tag (OR logic)

    Returns:
        List of matching agents
    """
    async with registry.state.read_lock():
        agents = registry.state.get("agents", {})

        matching_agents = []
        for agent in agents.values():
            agent_tags = set(agent.metadata.tags)
            required_tags = set(tags)

            if match_all:
                # AND logic - must have all tags
                if required_tags.issubset(agent_tags):
                    matching_agents.append(agent)
            else:
                # OR logic - must have at least one tag
                if agent_tags & required_tags:
                    matching_agents.append(agent)

    return matching_agents
```

### 5.3 Fuzzy Matching

```python
from difflib import SequenceMatcher

async def discover_fuzzy(
    registry: AgentRegistry,
    query: str,
    min_similarity: float = 0.7
) -> List[Tuple[AgentInfo, float]]:
    """
    Discover agents using fuzzy string matching on descriptions.

    Args:
        registry: AgentRegistry instance
        query: Search query
        min_similarity: Minimum similarity threshold (default: 0.7)

    Returns:
        List of (agent, similarity) tuples sorted by similarity
    """
    async with registry.state.read_lock():
        agents = registry.state.get("agents", {})

        scored_agents = []
        for agent in agents.values():
            # Calculate similarity with description
            similarity = SequenceMatcher(
                None,
                query.lower(),
                agent.metadata.description.lower()
            ).ratio()

            if similarity >= min_similarity:
                scored_agents.append((agent, similarity))

    # Sort by similarity (highest first)
    scored_agents.sort(key=lambda x: x[1], reverse=True)

    return scored_agents
```

---

## 6. Capability Matching

### 6.1 Scoring Algorithm

```python
class ConfidenceScorer:
    """
    Multi-criteria confidence scoring for agent-task matching.

    Scoring Criteria:
    1. Capability Match (40% weight)
       - Jaccard similarity between agent capabilities and task requirements
       - Perfect match: 1.0, No overlap: 0.0

    2. Load Balance (20% weight)
       - 1.0 - (active_tasks / max_tasks)
       - Prefer agents with lower load

    3. Priority (20% weight)
       - agent.priority / 100
       - Higher priority agents scored higher

    4. Success Rate (20% weight)
       - Historical success rate (0.0-1.0)
       - Prefer agents with proven track record

    Total Score: weighted sum (0.0-1.0)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "capability": 0.4,
            "load": 0.2,
            "priority": 0.2,
            "success_rate": 0.2
        }

    def score(
        self,
        agent: AgentInfo,
        task: Task,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """Calculate confidence score with detailed breakdown"""

        # 1. Capability matching
        capability_score = self._score_capabilities(agent, task)

        # 2. Load balancing
        load_score = self._score_load(agent)

        # 3. Priority
        priority_score = agent.metadata.priority / 100.0

        # 4. Success rate
        success_rate_score = agent.metadata.success_rate or 0.5

        # Calculate weighted total
        total = (
            self.weights["capability"] * capability_score +
            self.weights["load"] * load_score +
            self.weights["priority"] * priority_score +
            self.weights["success_rate"] * success_rate_score
        )

        # Generate explanation
        explanation = self._generate_explanation(
            agent, task, capability_score, load_score,
            priority_score, success_rate_score
        )

        return ConfidenceScore(
            total=total,
            capability_score=capability_score,
            load_score=load_score,
            priority_score=priority_score,
            success_rate_score=success_rate_score,
            explanation=explanation
        )

    def _score_capabilities(
        self,
        agent: AgentInfo,
        task: Task
    ) -> float:
        """Score capability match using Jaccard similarity"""
        agent_caps = set(agent.capabilities)
        task_caps = set(task.required_capabilities)

        if not task_caps:
            return 1.0  # No specific requirements

        intersection = len(agent_caps & task_caps)
        union = len(agent_caps | task_caps)

        return intersection / union if union > 0 else 0.0

    def _score_load(self, agent: AgentInfo) -> float:
        """Score load balance (prefer agents with lower load)"""
        max_tasks = agent.metadata.max_concurrent_tasks
        active_tasks = agent.active_tasks

        if max_tasks <= 0:
            return 0.0

        return 1.0 - (active_tasks / max_tasks)

    def _generate_explanation(
        self,
        agent: AgentInfo,
        task: Task,
        cap_score: float,
        load_score: float,
        pri_score: float,
        sr_score: float
    ) -> str:
        """Generate human-readable explanation"""
        parts = [
            f"Agent '{agent.agent_id}' scored {cap_score:.2f} for capability match",
            f"Load: {load_score:.2f} ({agent.active_tasks}/{agent.metadata.max_concurrent_tasks} tasks)",
            f"Priority: {pri_score:.2f}",
            f"Success Rate: {sr_score:.2f}"
        ]

        return " | ".join(parts)
```

### 6.2 Routing Decisions

```python
async def route_task_to_agent(
    registry: AgentRegistry,
    task: Task,
    min_confidence: float = 0.5,
    fallback_strategy: str = "best_effort"
) -> Tuple[AgentInfo, ConfidenceScore]:
    """
    Route task to best agent with confidence scoring.

    Args:
        registry: AgentRegistry instance
        task: Task to route
        min_confidence: Minimum confidence threshold (default: 0.5)
        fallback_strategy: Strategy when no agent meets threshold
            - "best_effort": Return best agent even if below threshold
            - "fail": Raise error if no agent meets threshold

    Returns:
        Tuple of (selected_agent, confidence_score)

    Raises:
        NoAgentFoundError: If no suitable agent found (fail strategy)
    """
    # Get best match
    best_agent, confidence = await registry.match_agent(task)

    if confidence < min_confidence:
        if fallback_strategy == "fail":
            raise NoAgentFoundError(
                f"No agent meets minimum confidence {min_confidence} "
                f"(best: {confidence:.2f})"
            )
        else:
            # Best effort - return best agent despite low confidence
            logger.warning(
                f"Agent confidence {confidence:.2f} below threshold {min_confidence}, "
                f"using best effort strategy"
            )

    return best_agent, confidence
```

---

## 7. Caching Design

### 7.1 Cache Strategy

**Cache Key Design**:
```python
def generate_cache_key(task: Task) -> str:
    """
    Generate deterministic cache key from task.

    Key Components:
    - task_type (e.g., "contract_inference")
    - required_capabilities (sorted list)
    - complexity (if specified)
    - domain (if specified)

    Format: "task_type|cap1,cap2|complexity|domain"

    Examples:
    - "contract_inference|yaml_parsing,llm|medium|database"
    - "template_selection|pattern_matching|simple|"
    """
    parts = [
        task.task_type,
        ",".join(sorted(task.required_capabilities)),
        task.complexity or "",
        task.domain or ""
    ]

    return "|".join(parts)
```

**Cache Invalidation**:
```python
class CacheInvalidationStrategy:
    """
    Cache invalidation triggers.

    Invalidation Events:
    1. Agent Registration - Invalidate all (new agent available)
    2. Agent Deregistration - Invalidate all (agent removed)
    3. Agent Status Change - Invalidate entries for that agent
    4. TTL Expiration - Automatic expiration after 5min
    5. Manual Invalidation - On-demand cache clearing
    """

    def __init__(self, cache: CacheManager, registry: AgentRegistry):
        self.cache = cache
        self.registry = registry

    def on_agent_registered(self, agent_id: str) -> None:
        """Invalidate all cache entries (new agent available)"""
        self.cache.invalidate_all()

    def on_agent_deregistered(self, agent_id: str) -> None:
        """Invalidate all cache entries (agent removed)"""
        self.cache.invalidate_all()

    def on_agent_status_changed(self, agent_id: str, new_status: AgentStatus) -> None:
        """Invalidate entries for specific agent"""
        # Find cache keys that resolved to this agent
        # (requires tracking cache key -> agent_id mapping)
        self._invalidate_agent_entries(agent_id)

    def _invalidate_agent_entries(self, agent_id: str) -> None:
        """Invalidate cache entries for specific agent"""
        # TODO: Implement cache invalidation logic
        # Implementation requires agent_id tracking in cache entries
        # Suggested approach:
        # 1. Maintain reverse mapping: agent_id -> List[cache_keys]
        # 2. On agent status change, look up affected cache keys
        # 3. Invalidate those specific entries
        pass
```

### 7.2 Cache Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cache Hit Rate** | 85-95% | `cache_hits / (cache_hits + cache_misses)` |
| **Cache Hit Latency** | <5ms | Time from `cache.get()` to return (hit) |
| **Cache Miss Latency** | <100ms | Time from `cache.get()` to match_agent() return |
| **Cache Size** | 1000 entries | Max entries before LRU eviction |
| **Cache TTL** | 300s (5min) | Entry expiration time |

### 7.3 Cache Warming

```python
async def warm_cache(
    registry: AgentRegistry,
    common_tasks: List[Task]
) -> Dict[str, Any]:
    """
    Pre-warm cache with common task patterns.

    Args:
        registry: AgentRegistry instance
        common_tasks: List of common task patterns

    Returns:
        Warming statistics
    """
    start_time = time.time()
    warmed = 0
    errors = 0

    for task in common_tasks:
        try:
            # Perform match (will cache result)
            await registry.match_agent(task)
            warmed += 1
        except Exception as e:
            logger.error(f"Cache warming failed for task {task.task_id}: {e}")
            errors += 1

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "warmed": warmed,
        "errors": errors,
        "total_time_ms": elapsed_ms,
        "avg_time_ms": elapsed_ms / len(common_tasks) if common_tasks else 0
    }
```

---

## 8. Integration Design

### 8.1 ThreadSafeState Integration

```python
# ThreadSafeState Schema for Agent Registry

state_schema = {
    "agents": {
        # Key: agent_id
        # Value: AgentInfo (serialized dict)
        "agent_1": {
            "agent_id": "contract_inferencer_v1",
            "capabilities": ["contract_inference", "yaml_parsing"],
            "metadata": {...},
            "registered_at": "2025-11-06T10:00:00Z",
            "last_heartbeat": "2025-11-06T10:05:00Z",
            "status": "active",
            "active_tasks": 3,
            "total_tasks_completed": 150,
            "total_tasks_failed": 5
        }
    },

    "capability_index": {
        # Key: capability tag
        # Value: Set of agent_ids
        "contract_inference": {"agent_1", "agent_2"},
        "template_selection": {"agent_3"},
        "llm": {"agent_1", "agent_4"}
    },

    "registry_metrics": {
        "registration_time_ms": {
            "count": 10,
            "sum": 450.0,
            "min": 30.0,
            "max": 60.0
        },
        "routing_time_ms": {
            "count": 1000,
            "sum": 8500.0,
            "min": 2.0,
            "max": 150.0
        },
        "cache_hits": {"count": 850},
        "cache_misses": {"count": 150}
    }
}
```

### 8.2 Kafka Event Publishing

```python
# Kafka Topics for Agent Registry

topics = [
    "agent-registration-events",    # Agent registration/deregistration
    "agent-routing-events",         # Task routing decisions
    "agent-heartbeat-events"        # Heartbeat status updates (optional)
]

# Event Schemas

# 1. Agent Registration Event
registration_event = {
    "event_type": "agent.registered.v1",
    "agent_id": "contract_inferencer_v1",
    "capabilities": ["contract_inference", "yaml_parsing"],
    "metadata": {
        "agent_type": "contract_inferencer",
        "version": "1.0.0",
        "description": "Infers contracts from YAML specifications"
    },
    "timestamp": "2025-11-06T10:00:00Z"
}

# 2. Agent Deregistration Event
deregistration_event = {
    "event_type": "agent.deregistered.v1",
    "agent_id": "contract_inferencer_v1",
    "reason": "shutdown",
    "timestamp": "2025-11-06T12:00:00Z"
}

# 3. Agent Match Event
match_event = {
    "event_type": "agent.matched.v1",
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_id": "contract_inferencer_v1",
    "confidence": 0.85,
    "routing_time_ms": 45.2,
    "cache_hit": false,
    "timestamp": "2025-11-06T10:05:00Z"
}
```

### 8.3 Performance Metrics Integration

```python
class RegistryMetricsCollector:
    """Collect and publish registry performance metrics"""

    def __init__(
        self,
        registry: AgentRegistry,
        prometheus_exporter: Optional[PrometheusExporter] = None
    ):
        self.registry = registry
        self.prometheus_exporter = prometheus_exporter

    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect all registry metrics"""
        async with self.registry.state.read_lock():
            metrics = self.registry.state.get("registry_metrics", {})

            # Calculate averages
            routing_avg = (
                metrics.get("routing_time_ms", {}).get("sum", 0) /
                max(metrics.get("routing_time_ms", {}).get("count", 1), 1)
            )

            registration_avg = (
                metrics.get("registration_time_ms", {}).get("sum", 0) /
                max(metrics.get("registration_time_ms", {}).get("count", 1), 1)
            )

            # Calculate cache hit rate
            cache_hits = metrics.get("cache_hits", {}).get("count", 0)
            cache_misses = metrics.get("cache_misses", {}).get("count", 0)
            total_requests = cache_hits + cache_misses
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

            return {
                "routing_time_avg_ms": routing_avg,
                "routing_time_p99_ms": metrics.get("routing_time_ms", {}).get("max", 0),
                "registration_time_avg_ms": registration_avg,
                "cache_hit_rate": cache_hit_rate,
                "total_agents": len(self.registry.state.get("agents", {})),
                "active_agents": self._count_active_agents()
            }

    async def publish_to_prometheus(self) -> None:
        """Publish metrics to Prometheus"""
        if not self.prometheus_exporter:
            return

        metrics = await self.collect_metrics()

        # Publish to Prometheus
        self.prometheus_exporter.gauge(
            "agent_registry_routing_time_avg_ms",
            metrics["routing_time_avg_ms"]
        )

        self.prometheus_exporter.gauge(
            "agent_registry_cache_hit_rate",
            metrics["cache_hit_rate"]
        )

        self.prometheus_exporter.gauge(
            "agent_registry_total_agents",
            metrics["total_agents"]
        )

        self.prometheus_exporter.gauge(
            "agent_registry_active_agents",
            metrics["active_agents"]
        )
```

---

## 9. Error Handling

### 9.1 Error Types

```python
class AgentRegistryError(Exception):
    """Base exception for agent registry errors"""
    pass

class AgentNotFoundError(AgentRegistryError):
    """Agent not found in registry"""
    pass

class NoAgentFoundError(AgentRegistryError):
    """No suitable agent found for task"""
    pass

class DuplicateAgentError(AgentRegistryError):
    """Agent already registered with same ID"""
    pass

class AgentUnavailableError(AgentRegistryError):
    """Agent exists but is unavailable (inactive, at capacity, etc.)"""
    pass

class ValidationError(AgentRegistryError):
    """Validation error for agent metadata or task"""
    pass
```

### 9.2 Error Handling Strategies

```python
async def match_agent_with_fallback(
    registry: AgentRegistry,
    task: Task,
    fallback_agents: Optional[List[str]] = None
) -> Tuple[AgentInfo, float]:
    """
    Match agent with fallback strategy.

    Strategy:
    1. Try primary matching (capability-based)
    2. If no agent found or confidence < 0.3, try fallback agents
    3. If all fail, raise NoAgentFoundError

    Args:
        registry: AgentRegistry instance
        task: Task to match
        fallback_agents: List of fallback agent IDs (optional)

    Returns:
        Tuple of (agent, confidence)

    Raises:
        NoAgentFoundError: If no suitable agent found
    """
    try:
        # Try primary matching
        agent, confidence = await registry.match_agent(task)

        if confidence >= 0.3:
            return agent, confidence

        logger.warning(
            f"Primary match confidence {confidence:.2f} below threshold, "
            f"trying fallback agents"
        )

    except NoAgentFoundError:
        logger.warning("No agent found via primary matching, trying fallback agents")

    # Try fallback agents
    if fallback_agents:
        for agent_id in fallback_agents:
            try:
                agent = await registry.get_agent(agent_id)

                # Check if agent is available
                if agent.status == AgentStatus.ACTIVE:
                    # Return with low confidence (fallback)
                    return agent, 0.2

            except AgentNotFoundError:
                continue

    # All fallback strategies failed
    raise NoAgentFoundError(
        f"No suitable agent found for task {task.task_id} "
        f"(tried primary matching + {len(fallback_agents or [])} fallback agents)"
    )
```

### 9.3 Retry Logic

```python
async def register_agent_with_retry(
    registry: AgentRegistry,
    agent_id: str,
    capabilities: List[str],
    metadata: AgentMetadata,
    max_retries: int = 3,
    backoff_seconds: float = 1.0
) -> RegistrationResult:
    """
    Register agent with retry logic.

    Args:
        registry: AgentRegistry instance
        agent_id: Agent identifier
        capabilities: List of capabilities
        metadata: Agent metadata
        max_retries: Maximum retry attempts (default: 3)
        backoff_seconds: Initial backoff in seconds (default: 1.0)

    Returns:
        RegistrationResult

    Raises:
        AgentRegistryError: If all retries fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = await registry.register_agent(
                agent_id=agent_id,
                capabilities=capabilities,
                metadata=metadata
            )

            return result

        except (ConnectionError, asyncio.TimeoutError) as e:
            last_error = e

            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = backoff_seconds * (2 ** attempt)
                logger.warning(
                    f"Registration failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time:.1f}s: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                # Final attempt failed
                raise AgentRegistryError(
                    f"Registration failed after {max_retries} attempts: {e}"
                ) from last_error

        except Exception as e:
            # Non-retryable error
            raise AgentRegistryError(f"Registration failed: {e}") from e

    # Should never reach here
    raise AgentRegistryError("Unexpected registration failure")
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/unit/test_agent_registry.py

import pytest
from datetime import datetime
from omninode_bridge.agents.registry import AgentRegistry
from omninode_bridge.agents.models import (
    AgentInfo, AgentMetadata, AgentType, AgentStatus, Task
)

@pytest.fixture
async def registry(thread_safe_state):
    """Create test registry"""
    registry = AgentRegistry(state=thread_safe_state, enable_cache=True)
    yield registry
    # Cleanup
    await registry.shutdown()

@pytest.mark.asyncio
async def test_register_agent(registry):
    """Test agent registration"""
    # Arrange
    agent_id = "test_agent_1"
    capabilities = ["test_capability"]
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent"
    )

    # Act
    result = await registry.register_agent(
        agent_id=agent_id,
        capabilities=capabilities,
        metadata=metadata
    )

    # Assert
    assert result.success is True
    assert result.agent_id == agent_id
    assert result.registration_time_ms < 50  # <50ms target

    # Verify agent stored
    agent = await registry.get_agent(agent_id)
    assert agent.agent_id == agent_id
    assert agent.capabilities == capabilities
    assert agent.status == AgentStatus.ACTIVE

@pytest.mark.asyncio
async def test_register_duplicate_agent(registry):
    """Test duplicate agent registration fails"""
    # Arrange
    agent_id = "test_agent_1"
    capabilities = ["test_capability"]
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent"
    )

    # Register first time
    await registry.register_agent(agent_id, capabilities, metadata)

    # Act & Assert
    with pytest.raises(ValueError, match="already registered"):
        await registry.register_agent(agent_id, capabilities, metadata)

@pytest.mark.asyncio
async def test_discover_agents_by_capability(registry):
    """Test capability-based discovery"""
    # Arrange - register 3 agents
    for i in range(3):
        await registry.register_agent(
            agent_id=f"agent_{i}",
            capabilities=["test_cap_1"] if i < 2 else ["test_cap_2"],
            metadata=AgentMetadata(
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description=f"Agent {i}"
            )
        )

    # Act
    agents_cap1 = await registry.discover_agents("test_cap_1")
    agents_cap2 = await registry.discover_agents("test_cap_2")

    # Assert
    assert len(agents_cap1) == 2
    assert len(agents_cap2) == 1
    assert all(agent.status == AgentStatus.ACTIVE for agent in agents_cap1)

@pytest.mark.asyncio
async def test_match_agent_cache_hit(registry):
    """Test agent matching with cache hit"""
    # Arrange - register agent
    agent_id = "test_agent"
    capabilities = ["contract_inference"]
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent"
    )
    await registry.register_agent(agent_id, capabilities, metadata)

    task = Task(
        task_id=uuid4(),
        task_type="contract_inference",
        required_capabilities=["contract_inference"]
    )

    # Act - first match (cache miss)
    start_time = time.time()
    agent1, conf1 = await registry.match_agent(task)
    first_time_ms = (time.time() - start_time) * 1000

    # Act - second match (cache hit)
    start_time = time.time()
    agent2, conf2 = await registry.match_agent(task)
    second_time_ms = (time.time() - start_time) * 1000

    # Assert
    assert agent1.agent_id == agent2.agent_id
    assert conf1 == conf2
    assert second_time_ms < 5  # <5ms cache hit target
    assert second_time_ms < first_time_ms  # Cache hit faster

@pytest.mark.asyncio
async def test_match_agent_confidence_scoring(registry):
    """Test confidence scoring for agent matching"""
    # Arrange - register agents with different characteristics
    await registry.register_agent(
        agent_id="high_priority",
        capabilities=["contract_inference", "yaml_parsing"],
        metadata=AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="High priority agent",
            priority=90,
            success_rate=0.95
        )
    )

    await registry.register_agent(
        agent_id="low_priority",
        capabilities=["contract_inference"],
        metadata=AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Low priority agent",
            priority=30,
            success_rate=0.70
        )
    )

    task = Task(
        task_id=uuid4(),
        task_type="contract_inference",
        required_capabilities=["contract_inference", "yaml_parsing"]
    )

    # Act
    agent, confidence = await registry.match_agent(task)

    # Assert
    assert agent.agent_id == "high_priority"  # Should select high priority agent
    assert confidence > 0.7  # Should have high confidence

@pytest.mark.asyncio
async def test_unregister_agent(registry):
    """Test agent unregistration"""
    # Arrange
    agent_id = "test_agent"
    capabilities = ["test_capability"]
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent"
    )
    await registry.register_agent(agent_id, capabilities, metadata)

    # Act
    await registry.unregister_agent(agent_id)

    # Assert
    with pytest.raises(AgentNotFoundError):
        await registry.get_agent(agent_id)

    # Capability index should be updated
    agents = await registry.discover_agents("test_capability")
    assert len(agents) == 0
```

### 10.2 Integration Tests

```python
# tests/integration/test_agent_registry_integration.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_registration_discovery_flow(
    thread_safe_state,
    kafka_producer
):
    """Test complete registration and discovery flow"""
    # Arrange
    registry = AgentRegistry(
        state=thread_safe_state,
        kafka_producer=kafka_producer,
        enable_cache=True
    )

    # Act - Register multiple agents
    agents = [
        ("contract_inferencer", ["contract_inference", "yaml_parsing"]),
        ("template_selector", ["template_selection", "pattern_matching"]),
        ("business_logic_gen", ["llm", "code_generation"])
    ]

    for agent_id, capabilities in agents:
        await registry.register_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            metadata=AgentMetadata(
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description=f"{agent_id} agent"
            )
        )

    # Assert - Discovery works
    contract_agents = await registry.discover_agents("contract_inference")
    assert len(contract_agents) == 1
    assert contract_agents[0].agent_id == "contract_inferencer"

    # Assert - Matching works
    task = Task(
        task_id=uuid4(),
        task_type="contract_inference",
        required_capabilities=["contract_inference", "yaml_parsing"]
    )

    agent, confidence = await registry.match_agent(task)
    assert agent.agent_id == "contract_inferencer"
    assert confidence > 0.5

    # Assert - Kafka events published
    # (Verify Kafka consumer received registration events)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_performance(registry):
    """Test cache hit rate and performance"""
    # Arrange - register agent
    await registry.register_agent(
        agent_id="test_agent",
        capabilities=["test_cap"],
        metadata=AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent"
        )
    )

    task = Task(
        task_id=uuid4(),
        task_type="test",
        required_capabilities=["test_cap"]
    )

    # Act - perform 100 matches
    cache_hit_times = []

    for i in range(100):
        start_time = time.time()
        await registry.match_agent(task)
        elapsed_ms = (time.time() - start_time) * 1000

        if i > 0:  # Skip first (cache miss)
            cache_hit_times.append(elapsed_ms)

    # Assert - cache hit rate and performance
    cache_stats = registry.cache.get_stats()
    assert cache_stats["hit_rate"] > 0.85  # >85% hit rate

    avg_cache_hit_time = sum(cache_hit_times) / len(cache_hit_times)
    assert avg_cache_hit_time < 5  # <5ms average cache hit time
```

### 10.3 Performance Tests

```python
# tests/performance/test_agent_registry_performance.py

@pytest.mark.performance
@pytest.mark.asyncio
async def test_registration_performance(registry):
    """Test registration performance target (<50ms)"""
    # Arrange
    num_agents = 100
    registration_times = []

    # Act
    for i in range(num_agents):
        start_time = time.time()

        await registry.register_agent(
            agent_id=f"perf_agent_{i}",
            capabilities=[f"cap_{i % 10}"],
            metadata=AgentMetadata(
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description=f"Performance test agent {i}"
            )
        )

        elapsed_ms = (time.time() - start_time) * 1000
        registration_times.append(elapsed_ms)

    # Assert
    avg_time = sum(registration_times) / len(registration_times)
    p99_time = sorted(registration_times)[int(len(registration_times) * 0.99)]

    assert avg_time < 50  # <50ms average
    assert p99_time < 100  # <100ms p99

@pytest.mark.performance
@pytest.mark.asyncio
async def test_discovery_performance(registry):
    """Test discovery performance (<100ms for cache miss)"""
    # Arrange - register 100 agents
    for i in range(100):
        await registry.register_agent(
            agent_id=f"agent_{i}",
            capabilities=[f"cap_{i % 10}"],
            metadata=AgentMetadata(
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description=f"Agent {i}"
            )
        )

    # Act - perform discovery
    start_time = time.time()
    agents = await registry.discover_agents("cap_5")
    elapsed_ms = (time.time() - start_time) * 1000

    # Assert
    assert len(agents) == 10  # Should find 10 agents
    assert elapsed_ms < 100  # <100ms for discovery

@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_matching(registry):
    """Test concurrent agent matching performance"""
    # Arrange - register agents
    for i in range(10):
        await registry.register_agent(
            agent_id=f"agent_{i}",
            capabilities=[f"cap_{i % 3}"],
            metadata=AgentMetadata(
                agent_type=AgentType.CONTRACT_INFERENCER,
                version="1.0.0",
                description=f"Agent {i}"
            )
        )

    # Create 1000 tasks
    tasks = [
        Task(
            task_id=uuid4(),
            task_type="test",
            required_capabilities=[f"cap_{i % 3}"]
        )
        for i in range(1000)
    ]

    # Act - match all tasks concurrently
    start_time = time.time()

    match_coros = [registry.match_agent(task) for task in tasks]
    results = await asyncio.gather(*match_coros)

    elapsed_ms = (time.time() - start_time) * 1000

    # Assert
    assert len(results) == 1000
    assert all(r[1] > 0.3 for r in results)  # All matches > 0.3 confidence
    assert elapsed_ms < 10000  # <10s for 1000 concurrent matches

    # Check cache hit rate
    cache_stats = registry.cache.get_stats()
    assert cache_stats["hit_rate"] > 0.85  # >85% hit rate
```

---

## 11. Implementation Plan

### 11.1 File Structure

```
src/omninode_bridge/agents/
├── __init__.py
├── registry/
│   ├── __init__.py
│   ├── registry.py              # AgentRegistry class
│   ├── models.py                # Data models (AgentInfo, Task, etc.)
│   ├── capability_matcher.py   # CapabilityMatchEngine
│   ├── cache.py                 # CacheManager
│   ├── decorators.py            # @register_agent decorator
│   └── exceptions.py            # Custom exceptions
└── examples/
    ├── register_agent_example.py
    └── match_agent_example.py

tests/
├── unit/
│   └── agents/
│       └── test_agent_registry.py
├── integration/
│   └── agents/
│       └── test_agent_registry_integration.py
└── performance/
    └── agents/
        └── test_agent_registry_performance.py
```

### 11.2 Implementation Order

**Phase 1: Core Registry (Week 1)**
1. ✅ Data models (`models.py`)
2. ✅ AgentRegistry basic implementation (`registry.py`)
3. ✅ ThreadSafeState integration
4. ✅ Unit tests for registration/discovery

**Phase 2: Capability Matching (Week 2)**
1. ✅ CapabilityMatchEngine (`capability_matcher.py`)
2. ✅ Confidence scoring algorithm
3. ✅ Unit tests for matching

**Phase 3: Caching (Week 3)**
1. ✅ CacheManager implementation (`cache.py`)
2. ✅ LRU eviction + TTL expiration
3. ✅ Cache integration with AgentRegistry
4. ✅ Performance tests for cache hit rate

**Phase 4: Decorator & Heartbeat (Week 4)**
1. ✅ `@register_agent` decorator (`decorators.py`)
2. ✅ Heartbeat monitoring
3. ✅ Agent status management
4. ✅ Integration tests

**Phase 5: Integration & Polish (Week 5)**
1. ✅ Kafka event publishing
2. ✅ Prometheus metrics
3. ✅ Error handling
4. ✅ Documentation
5. ✅ End-to-end tests

### 11.3 Dependencies

**External Dependencies**:
- `pydantic>=2.0`: Data validation and modeling
- `asyncio`: Async/await support
- `aiokafka`: Kafka event publishing
- `prometheus-client`: Metrics export

**Internal Dependencies**:
- `ThreadSafeState`: Centralized state management (Wave 3 dependency)
- `KafkaProducer`: Event publishing infrastructure
- `PrometheusExporter`: Metrics collection

### 11.4 Success Criteria

**Functionality**:
- ✅ Agents can register with capabilities
- ✅ Discovery works via capability/tag/fuzzy matching
- ✅ Matching produces confidence scores (0.0-1.0)
- ✅ Cache hit rate: 85-95%
- ✅ ThreadSafeState integration complete

**Performance**:
- ✅ Registration: <50ms per agent
- ✅ Discovery (cache hit): <5ms
- ✅ Discovery (cache miss): <100ms
- ✅ Cache hit rate: 85-95%

**Quality**:
- ✅ Unit test coverage: >90%
- ✅ Integration tests: All critical paths
- ✅ Performance tests: All targets validated
- ✅ Documentation: Complete API docs + examples

---

## Summary

This architecture provides a **production-ready agent registration and discovery system** with:

✅ **Dynamic Registration**: Decorator-based + programmatic registration
✅ **Fast Discovery**: <5ms cache hit, <100ms cache miss
✅ **Intelligent Matching**: Multi-criteria confidence scoring (0.0-1.0)
✅ **Caching**: 85-95% hit rate with LRU + TTL eviction
✅ **ThreadSafeState Integration**: Centralized state management
✅ **Type Safety**: Strong Pydantic models throughout
✅ **Observability**: Kafka events + Prometheus metrics
✅ **Error Handling**: Comprehensive error types + retry logic

**Ready for implementation in Wave 3** (Phase 4 - Agent Framework Integration).

---

**Document Status**: ✅ Complete
**Next Steps**: Begin Phase 1 implementation (Core Registry)
**Dependencies**: ThreadSafeState (Wave 3)
