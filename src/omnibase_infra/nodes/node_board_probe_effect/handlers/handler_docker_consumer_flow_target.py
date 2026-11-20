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
    ConsumerFlowLaneUnsettledError,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_lane import (
    ModelConsumerFlowLane,
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


def read_folded_profiles(repo_root: Path, compose_project: str) -> frozenset[str]:
    """The roles the shipped policy folds into main on the lane of ``compose_project``.

    A project the policy does not name, and a checkout that ships no policy
    file, fold nothing, so every runtime container is expected there (the
    strict reading). A policy file that exists and cannot be read raises
    ``OSError``, which the probe grades INDETERMINATE. The shipped policy file
    pairs ``<PREFIX>_COMPOSE_PROJECT`` with ``<PREFIX>_RUNTIME_MAIN_FOLDED_PROFILES``.
    """
    values: dict[str, str] = {}
    try:
        policy = (repo_root / "docker" / "runtime-policy.env").read_text(
            encoding="utf-8"
        )
    except FileNotFoundError:
        return frozenset()
    for line in policy.splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            values[key.strip()] = value.strip()
    prefixes = [
        key.removesuffix("_COMPOSE_PROJECT")
        for key, value in values.items()
        if key.endswith("_COMPOSE_PROJECT") and value == compose_project
    ]
    if len(prefixes) != 1:
        return frozenset()
    folded = values.get(f"{prefixes[0]}_RUNTIME_MAIN_FOLDED_PROFILES", "")
    return frozenset(role.strip() for role in folded.split(",") if role.strip())


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
        try:
            containers = ModelConsumerFlowLane.from_compose_project(
                request.compose_project,
                folded_profiles=read_folded_profiles(
                    self._repo_root, request.compose_project
                ),
            )
        except (OSError, ValueError) as exc:
            return ModelConsumerFlowObservation(
                read_ok=False, read_error=f"{type(exc).__name__}: {exc}"
            )
        lane = ConsumerFlowLane(
            docker=request.docker_bin,
            base_url=request.base_url,
            containers=containers,
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
                        boot_grace_seconds=request.boot_grace_seconds,
                        injection_wait=request.injection_wait,
                    )
                    break
                except ConsumerFlowBootChangedError:
                    if attempt + 1 == request.attempts:
                        raise
                except ConsumerFlowLaneUnsettledError as exc:
                    # OMN-20410: the .201 dev runtime flaps unhealthy for minutes
                    # at a time (a redeploy, or its own degraded health monitor),
                    # so one settle window that ran out is not yet a verdict. The
                    # next attempt waits a whole fresh window; a lane that never
                    # converges still ends unreadable, which grades INDETERMINATE.
                    if attempt + 1 == request.attempts:
                        raise ConsumerFlowLaneUnsettledError(
                            f"{exc} (after {request.attempts} settle window(s))"
                        ) from exc
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
