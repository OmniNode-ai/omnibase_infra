# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observe C28 through bounded Docker, HTTP and checkout pytest calls."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import time
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
    observe_lane,
    run_negative,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_boot_changed import (
    ConsumerFlowBootChangedError,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
    ConsumerFlowInputError,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_observation import (
    ModelConsumerFlowObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)
from omnibase_infra.nodes.node_board_probe_effect.models.typed_dict_consumer_flow import (
    TypedDictConsumerFlowCollected,
)
from omnibase_infra.nodes.node_board_probe_effect.protocols.protocol_consumer_flow_target import (
    ProtocolConsumerFlowTarget,
)


class HandlerDockerConsumerFlowTarget(ProtocolConsumerFlowTarget):
    """Create a transport per request and collect off the asyncio event loop."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        urlopen: Callable[
            ..., AbstractContextManager[IO[bytes]]
        ] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        repo_root: Path | None = None,
    ) -> None:
        self._runner = runner
        self._urlopen = urlopen
        self._sleep = sleep
        self._monotonic = monotonic
        self._repo_root = repo_root if repo_root is not None else Path.cwd()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Effect: Docker, HTTP and local negative controls."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelConsumerFlowRequest
    ) -> ModelConsumerFlowObservation:
        """Contract entrypoint for board_probe.observe_consumer_flow."""
        return await self.observe(request)

    async def observe(
        self, request: ModelConsumerFlowRequest
    ) -> ModelConsumerFlowObservation:
        """Return an unreadable observation for every input/transport failure."""
        return await asyncio.to_thread(self._observe, request)

    def _observe(
        self, request: ModelConsumerFlowRequest
    ) -> ModelConsumerFlowObservation:
        lane = ConsumerFlowLane(
            docker=request.docker_bin,
            base_url=request.base_url,
            runner=self._runner,
            urlopen=self._urlopen,
            sleep=self._sleep,
            monotonic=self._monotonic,
        )
        try:
            obs: TypedDictConsumerFlowCollected = {}
            for attempt in range(request.attempts):
                try:
                    obs = observe_lane(
                        lane,
                        samples=request.samples,
                        interval=request.sample_interval,
                        settle_seconds=request.settle_seconds,
                        injection_wait=request.injection_wait,
                    )
                    break
                except ConsumerFlowBootChangedError:
                    if attempt + 1 == request.attempts:
                        raise
            request.scratch.mkdir(parents=True, exist_ok=True)
            obs["negative"] = run_negative(
                self._repo_root,
                shlex.split(request.pytest_cmd),
                request.scratch.resolve(),
                runner=self._runner,
            )
            return ModelConsumerFlowObservation(read_ok=True, **obs)
        except (
            ConsumerFlowInputError,
            OSError,
            subprocess.SubprocessError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
        ) as exc:
            return ModelConsumerFlowObservation(
                read_ok=False, read_error=f"{type(exc).__name__}: {exc}"
            )
