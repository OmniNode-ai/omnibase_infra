# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read a gateway forwarder's state through the Docker CLI.

Runs on the lab host that runs the forwarder, or with ``DOCKER_HOST`` pointing
at that host's Docker daemon (for example ``ssh://<user>@<lab host>``). Three
bounded read-only calls: ``docker inspect`` for the process state and start
time, ``docker logs --since`` over one inbound-topic retry interval plus a
margin for the transport's ``kafka_transport_topic_refused`` and
``kafka_transport_topic_admitted`` lines, and ``docker exec ... cat`` of the
mounted broker-ref map for the cloud leg's bootstrap. It mutates nothing.

Ticket: OMN-19930
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_board_probe_effect.models import (
    ModelForwarderRefusedTopicRequest,
    ModelForwarderStateObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.protocols import (
    ProtocolForwarderStateReader,
)


def parse_refused_topics(log_text: str) -> tuple[str, ...]:
    """Replay exact event tokens and topic fields in log order."""
    refused: set[str] = set()
    for line in log_text.splitlines():
        fields = line.split()
        topic = next(
            (
                field.removeprefix("topic=")
                for field in fields
                if field.startswith("topic=")
            ),
            "",
        )
        if not topic:
            continue
        if "kafka_transport_topic_refused" in fields:
            refused.add(topic)
        if "kafka_transport_topic_admitted" in fields:
            refused.discard(topic)
    return tuple(sorted(refused))


class HandlerDockerForwarderStateReader(ProtocolForwarderStateReader):
    """Observe a forwarder through bounded Docker CLI calls off the event loop."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        docker: str = "docker",
        timeout_seconds: float = 30.0,
        window_margin_seconds: int = 60,
    ) -> None:
        self._runner = runner
        self._now = now
        self._docker = docker
        self._timeout_seconds = timeout_seconds
        self._window_margin_seconds = window_margin_seconds

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler (host I/O)."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: effect, reads a container on the host."""
        return EnumHandlerTypeCategory.EFFECT

    async def observe(
        self, request: ModelForwarderRefusedTopicRequest
    ) -> ModelForwarderStateObservation:
        """Read process state, recent topic events, and the mounted broker map."""
        return await asyncio.to_thread(self._observe, request)

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return self._runner(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=self._timeout_seconds,
        )

    def _failed_read(
        self, argv: list[str], reason: str
    ) -> ModelForwarderStateObservation:
        return ModelForwarderStateObservation(
            read_ok=False,
            read_error=f"{' '.join(argv)}: {reason.strip()[:500]}",
            observed_at=self._now(),
        )

    def _observe(
        self, request: ModelForwarderRefusedTopicRequest
    ) -> ModelForwarderStateObservation:
        argv = [
            self._docker,
            "inspect",
            "--format",
            "{{json .State}}",
            request.forwarder_container,
        ]
        try:
            state_result = self._run(argv)
            state_output = f"{state_result.stderr}\n{state_result.stdout}".strip()
            if state_result.returncode != 0:
                return self._failed_read(
                    argv, f"exit {state_result.returncode}: {state_output}"
                )
            try:
                state = json.loads(state_result.stdout)
                if not isinstance(state, dict) or not isinstance(
                    state.get("Running"), bool
                ):
                    raise ValueError("expected a state mapping with boolean Running")
                running = state["Running"]
                started_at_text = state.get("StartedAt")
                if not isinstance(started_at_text, str):
                    raise ValueError("expected an RFC3339 StartedAt string")
                started_at = datetime.fromisoformat(started_at_text).replace(
                    microsecond=0
                )
            except ValueError as exc:
                return self._failed_read(argv, f"{exc}: {state_output}")

            window_seconds = (
                request.retry_interval_seconds + self._window_margin_seconds
            )
            argv = [
                self._docker,
                "logs",
                "--since",
                f"{window_seconds}s",
                request.forwarder_container,
            ]
            logs_result = self._run(argv)
            log_text = f"{logs_result.stdout}\n{logs_result.stderr}"
            if logs_result.returncode != 0 and running:
                return self._failed_read(
                    argv, f"exit {logs_result.returncode}: {log_text}"
                )

            argv = [
                self._docker,
                "exec",
                request.forwarder_container,
                "cat",
                request.broker_ref_map_path,
            ]
            broker_result = self._run(argv)
            observed_cloud_broker = ""
            if broker_result.returncode == 0:
                broker_map = yaml.safe_load(broker_result.stdout)
                if isinstance(broker_map, Mapping):
                    broker = broker_map.get(request.cloud_broker_ref)
                    if isinstance(broker, str):
                        observed_cloud_broker = broker.strip()

            return ModelForwarderStateObservation(
                read_ok=True,
                running=running,
                started_at=started_at,
                observed_at=self._now(),
                window_seconds=window_seconds,
                refused_topics=parse_refused_topics(log_text),
                observed_cloud_broker=observed_cloud_broker,
            )
        except (subprocess.TimeoutExpired, OSError, yaml.YAMLError) as exc:
            return self._failed_read(argv, str(exc))


__all__ = ["HandlerDockerForwarderStateReader", "parse_refused_topics"]
