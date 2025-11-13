"""
Coordination signal system for agent-to-agent communication.

This module provides a high-performance signal coordinator for multi-agent
code generation workflows with <100ms signal propagation target.

Performance Targets:
- Signal propagation: <100ms
- Thread-safe signal storage via ThreadSafeState
- Async signal delivery (non-blocking)
- Metrics collection for every signal

Example:
    ```python
    from omninode_bridge.agents.coordination import ThreadSafeState, SignalCoordinator
    from omninode_bridge.agents.metrics import MetricsCollector

    state = ThreadSafeState()
    metrics = MetricsCollector()
    coordinator = SignalCoordinator(state=state, metrics_collector=metrics)

    # Send signal
    success = await coordinator.signal_coordination_event(
        coordination_id="codegen-session-1",
        event_type="agent_initialized",
        event_data={
            "agent_id": "model-generator",
            "capabilities": ["pydantic_models", "type_hints"],
            "ready": True
        }
    )

    # Subscribe to signals
    async for signal in coordinator.subscribe_to_signals(
        coordination_id="codegen-session-1",
        agent_id="validator-generator",
        signal_types=["agent_completed", "dependency_resolved"]
    ):
        print(f"Received signal: {signal.signal_type}")
    ```
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Optional

from ..metrics.collector import MetricsCollector
from .signal_models import (
    CoordinationSignal,
    SignalMetrics,
    SignalSubscription,
    SignalType,
)
from .thread_safe_state import ThreadSafeState

logger = logging.getLogger(__name__)


class SignalCoordinator:
    """
    High-performance signal coordinator for agent-to-agent communication.

    Features:
    - Thread-safe signal storage using ThreadSafeState
    - Async signal propagation with <100ms target
    - Signal history tracking for debugging
    - Metrics collection for all signals
    - Signal subscription with filtering

    Performance:
    - Signal propagation: <100ms (target)
    - Storage operations: <2ms (ThreadSafeState)
    - Metrics overhead: <1ms per signal
    - Subscription filtering: O(n) where n = signals since last read

    Example:
        ```python
        coordinator = SignalCoordinator(state=state, metrics_collector=metrics)

        # Send completion signal
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={
                "agent_id": "model-gen",
                "result_summary": "Generated 5 models",
                "quality_score": 0.95,
                "execution_time_ms": 1234.5
            }
        )
        ```
    """

    def __init__(
        self,
        state: ThreadSafeState,
        metrics_collector: Optional[MetricsCollector] = None,
        max_history_size: int = 10000,
    ) -> None:
        """
        Initialize signal coordinator.

        Args:
            state: ThreadSafeState instance for centralized storage
            metrics_collector: Optional metrics collector for signal tracking
            max_history_size: Maximum signal history per coordination session
        """
        self.state = state
        self.metrics = metrics_collector
        self._max_history_size = max_history_size

        # Initialize signal channels in state
        self._initialize_state()

        # Signal delivery queues (in-memory for async delivery)
        self._delivery_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

        # Subscriptions tracking
        self._subscriptions: dict[str, list[SignalSubscription]] = defaultdict(list)

        logger.info(
            f"SignalCoordinator initialized: max_history={max_history_size}, "
            f"metrics_enabled={metrics_collector is not None}"
        )

    async def signal_coordination_event(
        self,
        coordination_id: str,
        event_type: str,
        event_data: dict,
        sender_agent_id: Optional[str] = None,
        recipient_agents: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Send coordination signal to participating agents.

        Performance Target: <100ms total (storage + delivery + metrics)

        Args:
            coordination_id: Coordination session identifier
            event_type: Signal type (agent_initialized, agent_completed, etc.)
            event_data: Signal-specific event data
            sender_agent_id: Agent that sent signal (default: "system")
            recipient_agents: List of recipient agent IDs (empty = broadcast)
            metadata: Additional metadata (correlation_id, priority, etc.)

        Returns:
            True if signal sent successfully, False otherwise

        Example:
            ```python
            # Send agent completed signal
            success = await coordinator.signal_coordination_event(
                coordination_id="session-1",
                event_type="agent_completed",
                event_data={
                    "agent_id": "model-gen",
                    "quality_score": 0.95
                },
                sender_agent_id="model-gen"
            )
            ```
        """
        start_time = time.time()

        try:
            # 1. Validate event type
            try:
                signal_type = SignalType(event_type)
            except ValueError:
                logger.error(f"Invalid signal type: {event_type}")
                return False

            # 2. Create coordination signal
            signal = CoordinationSignal(
                signal_type=signal_type,
                sender_agent_id=sender_agent_id or "system",
                recipient_agents=recipient_agents or [],
                event_data=event_data,
                metadata=metadata or {},
                coordination_id=coordination_id,
            )

            # 3. Store signal in ThreadSafeState
            await self._store_signal(coordination_id, signal)

            # 4. Deliver to subscribed agents (async, non-blocking)
            asyncio.create_task(self._deliver_signal(coordination_id, signal))

            # 5. Update metrics
            propagation_time_ms = (time.time() - start_time) * 1000

            if self.metrics:
                await self.metrics.record_timing(
                    metric_name="signal_propagation_time_ms",
                    duration_ms=propagation_time_ms,
                    tags={
                        "signal_type": signal_type.value,
                        "coordination_id": coordination_id,
                    },
                    correlation_id=coordination_id,
                )

                await self.metrics.record_counter(
                    metric_name="signals_sent",
                    count=1,
                    tags={
                        "signal_type": signal_type.value,
                        "coordination_id": coordination_id,
                    },
                )

            # 6. Update signal metrics
            await self._update_signal_metrics(
                coordination_id, signal_type.value, propagation_time_ms
            )

            logger.debug(
                f"Signal sent: {signal_type.value} for {coordination_id} "
                f"in {propagation_time_ms:.2f}ms"
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to send signal {event_type} for {coordination_id}: {e}",
                exc_info=True,
            )
            return False

    async def subscribe_to_signals(
        self,
        coordination_id: str,
        agent_id: str,
        signal_types: Optional[list[str]] = None,
    ) -> AsyncIterator[CoordinationSignal]:
        """
        Subscribe to coordination signals with filtering.

        Yields signals as they arrive (async iterator pattern).

        Args:
            coordination_id: Coordination session to subscribe to
            agent_id: Subscribing agent ID
            signal_types: List of signal types to receive (None = all types)

        Yields:
            CoordinationSignal objects matching subscription

        Example:
            ```python
            # Subscribe to completion and dependency signals
            async for signal in coordinator.subscribe_to_signals(
                coordination_id="session-1",
                agent_id="validator-gen",
                signal_types=["agent_completed", "dependency_resolved"]
            ):
                print(f"Received: {signal.signal_type}")
                if signal.signal_type == "dependency_resolved":
                    # Process dependency
                    pass
            ```
        """
        # 1. Create subscription
        subscription = SignalSubscription(
            coordination_id=coordination_id,
            agent_id=agent_id,
            signal_types=[SignalType(st) for st in signal_types]
            if signal_types
            else [],
        )

        # 2. Register subscription
        self._subscriptions[coordination_id].append(subscription)

        logger.info(
            f"Agent '{agent_id}' subscribed to signals for {coordination_id} "
            f"(types: {signal_types or 'all'})"
        )

        # 3. Create delivery queue for this agent
        queue_key = f"{coordination_id}:{agent_id}"
        if queue_key not in self._delivery_queues:
            self._delivery_queues[queue_key] = asyncio.Queue()

        queue = self._delivery_queues[queue_key]

        try:
            # 4. Yield signals from queue
            while True:
                signal = await queue.get()

                # Check if signal matches subscription filters
                if self._matches_subscription(signal, subscription):
                    # Update metrics
                    if self.metrics:
                        await self.metrics.record_counter(
                            metric_name="signals_received",
                            count=1,
                            tags={
                                "signal_type": signal.signal_type.value,
                                "agent_id": agent_id,
                            },
                        )

                    yield signal

        except asyncio.CancelledError:
            # Cleanup subscription on cancellation
            self._subscriptions[coordination_id].remove(subscription)
            logger.info(
                f"Agent '{agent_id}' unsubscribed from {coordination_id} signals"
            )
            raise

    def get_signal_history(
        self,
        coordination_id: str,
        filters: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> list[CoordinationSignal]:
        """
        Retrieve signal history with optional filtering.

        Args:
            coordination_id: Coordination session identifier
            filters: Optional filters (signal_type, sender_agent_id, etc.)
            limit: Maximum number of signals to return (None = all)

        Returns:
            List of coordination signals (most recent first)

        Example:
            ```python
            # Get last 10 completion signals
            history = coordinator.get_signal_history(
                coordination_id="session-1",
                filters={"signal_type": "agent_completed"},
                limit=10
            )
            ```
        """
        # Get signal history from ThreadSafeState
        history_key = f"signal_history:{coordination_id}"
        history_data = self.state.get(history_key, [])

        # Reconstruct signals
        signals = [CoordinationSignal(**data) for data in history_data]

        # Apply filters
        if filters:
            signals = [s for s in signals if self._matches_filters(s, filters)]

        # Sort by timestamp (most recent first)
        signals.sort(key=lambda s: s.timestamp, reverse=True)

        # Apply limit
        if limit is not None:
            signals = signals[:limit]

        return signals

    def get_signal_metrics(self, coordination_id: str) -> SignalMetrics:
        """
        Get signal metrics for coordination session.

        Args:
            coordination_id: Coordination session identifier

        Returns:
            SignalMetrics with aggregated statistics

        Example:
            ```python
            metrics = coordinator.get_signal_metrics("session-1")
            print(f"Total signals: {metrics.total_signals_sent}")
            print(f"Avg propagation: {metrics.average_propagation_ms:.2f}ms")
            ```
        """
        metrics_key = f"signal_metrics:{coordination_id}"
        metrics_data = self.state.get(metrics_key, {})

        if not metrics_data:
            return SignalMetrics()

        return SignalMetrics(**metrics_data)

    # Private methods

    async def _store_signal(
        self, coordination_id: str, signal: CoordinationSignal
    ) -> None:
        """
        Store signal in ThreadSafeState with history management.

        Performance: <2ms (ThreadSafeState target)
        """
        history_key = f"signal_history:{coordination_id}"

        # Get existing history
        history = self.state.get(history_key, [])

        # Add new signal
        history.append(signal.model_dump())

        # Limit history size (keep most recent)
        if len(history) > self._max_history_size:
            history = history[-self._max_history_size :]

        # Store updated history
        self.state.set(history_key, history, changed_by="signal_coordinator")

    async def _deliver_signal(
        self, coordination_id: str, signal: CoordinationSignal
    ) -> None:
        """
        Deliver signal to subscribed agents (async, non-blocking).

        Uses in-memory queues for fast delivery to active subscribers.
        """
        # Get all subscriptions for this coordination session
        subscriptions = self._subscriptions.get(coordination_id, [])

        for subscription in subscriptions:
            # Check if signal should be delivered to this agent
            if self._matches_subscription(signal, subscription):
                queue_key = f"{coordination_id}:{subscription.agent_id}"
                queue = self._delivery_queues.get(queue_key)

                if queue:
                    # Non-blocking put
                    try:
                        await asyncio.wait_for(queue.put(signal), timeout=0.1)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Signal delivery timeout for agent '{subscription.agent_id}'"
                        )

    def _matches_subscription(
        self, signal: CoordinationSignal, subscription: SignalSubscription
    ) -> bool:
        """Check if signal matches subscription filters."""
        # Check signal types filter
        if subscription.signal_types and signal.signal_type not in subscription.signal_types:
            return False

        # Check sender filter
        if (
            subscription.sender_filter
            and signal.sender_agent_id != subscription.sender_filter
        ):
            return False

        # Check recipient filter (if broadcast or explicitly addressed)
        if signal.recipient_agents and subscription.agent_id not in signal.recipient_agents:
            return False

        return True

    def _matches_filters(self, signal: CoordinationSignal, filters: dict) -> bool:
        """Check if signal matches provided filters."""
        for key, value in filters.items():
            if key == "signal_type":
                if signal.signal_type.value != value:
                    return False
            elif key == "sender_agent_id":
                if signal.sender_agent_id != value:
                    return False
            elif key in signal.event_data:
                if signal.event_data[key] != value:
                    return False

        return True

    async def _update_signal_metrics(
        self, coordination_id: str, signal_type: str, propagation_time_ms: float
    ) -> None:
        """Update aggregated signal metrics."""
        metrics_key = f"signal_metrics:{coordination_id}"

        # Get existing metrics
        metrics_data = self.state.get(metrics_key, {})
        metrics = SignalMetrics(**metrics_data) if metrics_data else SignalMetrics()

        # Update metrics
        metrics.total_signals_sent += 1

        # Update average propagation time (running average)
        total_time = metrics.average_propagation_ms * (metrics.total_signals_sent - 1)
        total_time += propagation_time_ms
        metrics.average_propagation_ms = total_time / metrics.total_signals_sent

        # Update max propagation time
        if propagation_time_ms > metrics.max_propagation_ms:
            metrics.max_propagation_ms = propagation_time_ms

        # Update signals by type
        metrics.signals_by_type[signal_type] = (
            metrics.signals_by_type.get(signal_type, 0) + 1
        )

        # Store updated metrics
        self.state.set(metrics_key, metrics.model_dump(), changed_by="signal_coordinator")

    def _initialize_state(self) -> None:
        """Initialize signal state in ThreadSafeState."""
        # Signal history will be created on-demand per coordination session
        # No global initialization needed
        pass

    # ISignalCoordinator interface adapter methods
    # These methods provide compatibility with the ISignalCoordinator interface
    # used by DependencyResolver

    async def signal_event(
        self,
        coordination_id: str,
        event_type: str,
        event_data: dict[str, Any],
        priority: int = 5,
    ) -> None:
        """
        Signal a coordination event (adapter for ISignalCoordinator).

        This is an adapter method that delegates to signal_coordination_event()
        to provide compatibility with ISignalCoordinator interface.

        Args:
            coordination_id: Coordination session ID
            event_type: Type of event to signal (e.g., "dependency_resolved")
            event_data: Event payload data
            priority: Signal priority (1-10, higher = more urgent)
        """
        # Map priority to metadata
        metadata = {"priority": priority}

        # Delegate to main signal method
        await self.signal_coordination_event(
            coordination_id=coordination_id,
            event_type=event_type,
            event_data=event_data,
            sender_agent_id=event_data.get("agent_id", "system"),
            recipient_agents=None,  # Broadcast
            metadata=metadata,
        )

    async def check_agent_completion(
        self,
        coordination_id: str,
        agent_id: str,
        completion_event: str = "completion",
    ) -> bool:
        """
        Check if specific agent has completed (adapter for ISignalCoordinator).

        Args:
            coordination_id: Coordination session ID
            agent_id: Agent identifier to check
            completion_event: Completion event name to look for

        Returns:
            True if agent has signaled completion, False otherwise
        """
        # Get signal history for this coordination session
        signals = self.get_signal_history(
            coordination_id=coordination_id,
            filters={"signal_type": "agent_completed"},
        )

        # Check if any completion signal is from the target agent
        for signal in signals:
            if signal.sender_agent_id == agent_id:
                return True

        return False

    async def get_coordination_signals(
        self,
        coordination_id: str,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get coordination signals for session (adapter for ISignalCoordinator).

        Args:
            coordination_id: Coordination session ID
            event_type: Filter by event type (None = all events)

        Returns:
            List of signal dictionaries matching criteria
        """
        # Build filters
        filters = {}
        if event_type:
            filters["signal_type"] = event_type

        # Get signal history
        signals = self.get_signal_history(
            coordination_id=coordination_id,
            filters=filters if filters else None,
        )

        # Convert to dictionaries for compatibility
        return [signal.model_dump() for signal in signals]
