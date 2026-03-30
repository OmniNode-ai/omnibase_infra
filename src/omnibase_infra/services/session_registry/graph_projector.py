# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka-to-Memgraph graph projector for session coordination.

Consumes omniclaude hook events from Kafka and projects them into
Memgraph as a session coordination graph. Events without a task_id
in their payload are skipped.

All mutations use MERGE for idempotency (Doctrine D3: replay-safe).
Conflict emission is advisory only (Doctrine D6: projection/control
separation).

Consumer group: omnibase_infra.session_registry.graph_project.v1
Subscription: regex pattern ``onex\\.evt\\.omniclaude\\..*``

Part of the Multi-Session Coordination Layer (OMN-6850, Task 9).
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer  # type: ignore[import-untyped]

from omnibase_infra.services.session_registry.enum_node_label import (
    EnumNodeLabel,
)
from omnibase_infra.services.session_registry.enum_relationship_type import (
    EnumRelationshipType,
)
from omnibase_infra.services.session_registry.model_config_graph_projector import (
    ModelConfigGraphProjector,
)
from omnibase_infra.services.session_registry.model_graph_mutation import (
    ModelGraphMutation,
)

__all__ = [
    "ModelConfigGraphProjector",
    "ModelGraphMutation",
    "build_graph_mutations",
    "SessionGraphProjector",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repo extraction helper
# ---------------------------------------------------------------------------

_REPO_PATTERN = re.compile(r"(?:omni_home|omni_worktrees/[^/]+)/([^/]+)")


def _extract_repo(file_path: str) -> str | None:
    """Extract repository name from a file path.

    Looks for patterns like ``omni_home/<repo>/`` or
    ``omni_worktrees/<ticket>/<repo>/``.

    Args:
        file_path: Absolute or relative file path.

    Returns:
        Repository name or None if not extractable.
    """
    match = _REPO_PATTERN.search(file_path)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Mutation builder (pure function, no I/O)
# ---------------------------------------------------------------------------


def build_graph_mutations(
    event: dict[str, object],
) -> list[ModelGraphMutation]:
    """Build Cypher MERGE mutations from a hook event payload.

    Skips events that lack a ``task_id`` field. Handles:

    - **All events with task_id + session_id**: MERGE Session, Task,
      Session-[:WORKS_ON]->Task.
    - **tool.executed with file_path**: MERGE File, Task-[:TOUCHES]->File,
      extract repo and create File-[:BELONGS_TO]->Repository.
    - **coordination.signal with pr_merged**: MERGE PullRequest,
      Task-[:PRODUCED]->PullRequest, PR-[:BELONGS_TO]->Repository.
    - **coordination.signal with rebase_needed / related_task_id**:
      MERGE Task-[:DEPENDS_ON]->other Task.

    Args:
        event: Deserialized JSON event payload (envelope or inner payload).

    Returns:
        List of ModelGraphMutation instances. Empty if event has no task_id.
    """
    # Navigate into payload if this is an envelope
    payload: dict[str, object] = event.get("payload", event)  # type: ignore[assignment]

    task_id = payload.get("task_id")
    if not task_id:
        return []

    session_id = payload.get("session_id")
    if not session_id:
        return []

    mutations: list[ModelGraphMutation] = []

    # --- Base: MERGE Session + Task + WORKS_ON ---
    mutations.append(
        ModelGraphMutation(
            cypher=(
                f"MERGE (s:{EnumNodeLabel.SESSION} {{session_id: $session_id}}) "
                f"MERGE (t:{EnumNodeLabel.TASK} {{task_id: $task_id}}) "
                f"MERGE (s)-[:{EnumRelationshipType.WORKS_ON}]->(t)"
            ),
            params={"session_id": str(session_id), "task_id": str(task_id)},
        )
    )

    event_type = str(payload.get("event_type", ""))

    # --- tool.executed with file_path ---
    if event_type == "tool.executed":
        file_path = payload.get("file_path")
        if file_path:
            file_path_str = str(file_path)
            mutations.append(
                ModelGraphMutation(
                    cypher=(
                        f"MERGE (t:{EnumNodeLabel.TASK} {{task_id: $task_id}}) "
                        f"MERGE (f:{EnumNodeLabel.FILE} {{path: $path}}) "
                        f"MERGE (t)-[:{EnumRelationshipType.TOUCHES}]->(f)"
                    ),
                    params={"task_id": str(task_id), "path": file_path_str},
                )
            )
            repo = _extract_repo(file_path_str)
            if repo:
                mutations.append(
                    ModelGraphMutation(
                        cypher=(
                            f"MERGE (f:{EnumNodeLabel.FILE} {{path: $path}}) "
                            f"MERGE (r:{EnumNodeLabel.REPOSITORY} {{name: $repo}}) "
                            f"MERGE (f)-[:{EnumRelationshipType.BELONGS_TO}]->(r)"
                        ),
                        params={"path": file_path_str, "repo": repo},
                    )
                )

    # --- coordination.signal ---
    elif event_type == "coordination.signal":
        signal_type = str(payload.get("signal_type", ""))

        # PR merged
        if signal_type == "pr_merged":
            pr_number = payload.get("pr_number")
            repo_raw = payload.get("repo")
            repo_name: str | None = str(repo_raw) if repo_raw else None
            if pr_number is not None:
                pr_id = f"{repo_name}#{pr_number}" if repo_name else str(pr_number)
                mutations.append(
                    ModelGraphMutation(
                        cypher=(
                            f"MERGE (t:{EnumNodeLabel.TASK} {{task_id: $task_id}}) "
                            f"MERGE (p:{EnumNodeLabel.PULL_REQUEST} {{pr_id: $pr_id}}) "
                            f"MERGE (t)-[:{EnumRelationshipType.PRODUCED}]->(p)"
                        ),
                        params={"task_id": str(task_id), "pr_id": pr_id},
                    )
                )
                if repo_name:
                    repo_str = repo_name
                    mutations.append(
                        ModelGraphMutation(
                            cypher=(
                                f"MERGE (p:{EnumNodeLabel.PULL_REQUEST} {{pr_id: $pr_id}}) "
                                f"MERGE (r:{EnumNodeLabel.REPOSITORY} {{name: $repo}}) "
                                f"MERGE (p)-[:{EnumRelationshipType.BELONGS_TO}]->(r)"
                            ),
                            params={"pr_id": pr_id, "repo": repo_str},
                        )
                    )

        # Rebase needed / dependency
        elif signal_type == "rebase_needed":
            related_task_id = payload.get("related_task_id")
            if related_task_id:
                mutations.append(
                    ModelGraphMutation(
                        cypher=(
                            f"MERGE (t:{EnumNodeLabel.TASK} {{task_id: $task_id}}) "
                            f"MERGE (other:{EnumNodeLabel.TASK} {{task_id: $related_task_id}}) "
                            f"MERGE (t)-[:{EnumRelationshipType.DEPENDS_ON}]->(other)"
                        ),
                        params={
                            "task_id": str(task_id),
                            "related_task_id": str(related_task_id),
                        },
                    )
                )

    return mutations


# ---------------------------------------------------------------------------
# Projector (consumer lifecycle)
# ---------------------------------------------------------------------------


class SessionGraphProjector:
    """Kafka consumer that projects session events into Memgraph.

    Uses regex topic subscription to receive all omniclaude hook events.
    For each event with a task_id, builds MERGE mutations and executes
    them against Memgraph.

    Lifecycle:
        1. ``start()`` -- connect to Kafka and Memgraph
        2. ``run()`` -- consume loop (blocking)
        3. ``stop()`` -- graceful shutdown

    Example::

        config = ModelConfigGraphProjector()
        projector = SessionGraphProjector(config)
        await projector.start()
        try:
            await projector.run()
        finally:
            await projector.stop()
    """

    def __init__(self, config: ModelConfigGraphProjector) -> None:
        self._config = config
        self._running = False
        self._consumer: AIOKafkaConsumer | None = None
        self._mgclient: object | None = None  # mgclient.Connection (no stub)

    async def start(self) -> None:
        """Connect to Kafka and Memgraph."""
        import mgclient  # type: ignore[import-untyped]
        from aiokafka import AIOKafkaConsumer  # type: ignore[import-untyped]

        self._mgclient = mgclient.connect(
            host=self._config.memgraph_host,
            port=self._config.memgraph_port,
        )

        consumer = AIOKafkaConsumer(
            bootstrap_servers=self._config.bootstrap_servers,
            group_id=self._config.consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        consumer.subscribe(pattern=self._config.topic_pattern)
        await consumer.start()
        self._consumer = consumer
        self._running = True

        logger.info(
            "SessionGraphProjector started",
            extra={
                "consumer_group": self._config.consumer_group,
                "topic_pattern": self._config.topic_pattern,
                "memgraph_uri": self._config.bolt_uri,
            },
        )

    async def stop(self) -> None:
        """Graceful shutdown: stop Kafka consumer and close Memgraph."""
        self._running = False

        consumer = self._consumer
        if consumer is not None:
            try:
                await consumer.stop()
            except Exception:  # noqa: BLE001 -- boundary
                logger.warning("Error stopping Kafka consumer", exc_info=True)
            finally:
                self._consumer = None

        mgconn = self._mgclient
        if mgconn is not None:
            try:
                mgconn.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 -- boundary
                logger.warning("Error closing Memgraph connection", exc_info=True)
            finally:
                self._mgclient = None

        logger.info("SessionGraphProjector stopped")

    async def run(self) -> None:
        """Main consume loop. Blocks until ``stop()`` is called."""
        consumer = self._consumer
        if not self._running or consumer is None:
            raise RuntimeError("Projector not started. Call start() first.")

        async for message in consumer:
            if not self._running:
                break

            try:
                value = message.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                event: dict[str, object] = json.loads(value)
                mutations = build_graph_mutations(event)

                if mutations:
                    self._execute_mutations(mutations)

                await consumer.commit()

            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSON message",
                    extra={"topic": message.topic, "offset": message.offset},
                )
                await consumer.commit()

            except Exception:
                logger.exception(
                    "Error processing message",
                    extra={"topic": message.topic, "offset": message.offset},
                )

    def _execute_mutations(self, mutations: list[ModelGraphMutation]) -> None:
        """Execute a batch of Cypher mutations against Memgraph."""
        mgconn = self._mgclient
        if mgconn is None:
            logger.error("Memgraph connection not available")
            return

        cursor = mgconn.cursor()  # type: ignore[attr-defined]
        for mutation in mutations:
            try:
                cursor.execute(mutation.cypher, mutation.params)
            except Exception:
                logger.exception(
                    "Failed to execute Cypher mutation",
                    extra={"cypher": mutation.cypher, "params": mutation.params},
                )
        try:
            mgconn.commit()  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to commit Memgraph transaction")
