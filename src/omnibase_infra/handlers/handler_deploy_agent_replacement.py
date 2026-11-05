# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-local replacement effect for bus-commanded deploy-agent operations.

The executor supplies its host I/O boundary, as it does for the dev boot guard.
The handler owns the hot-patch admission rule and the existing settle/retry
behaviour, so both the first attempt and recovery stay behind the same gate.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import ModuleType

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_core.protocols.event_bus.protocol_dispatch_bus_client_transport import (
    ProtocolDispatchBusClientTransport,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_lifecycle import (
    ProtocolEventBusLifecycle,
)
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_agent_replacement_config import (
    ModelDeployAgentReplacementConfig,
)
from omnibase_infra.models.model_deploy_agent_replacement_request import (
    ModelDeployAgentReplacementRequest,
)
from omnibase_infra.models.model_deploy_agent_replacement_result import (
    ModelDeployAgentReplacementResult,
)

logger = logging.getLogger(__name__)
REMOVAL_IN_PROGRESS = "is already in progress"


class HotpatchLedgerPreflightError(RuntimeError):
    """A container replacement was refused by the hot-patch ledger preflight."""


class HandlerDeployAgentReplacement:
    """Gate a host-local replacement and supervise a recreate's cleanup/retry."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._config = container.get_service(ModelDeployAgentReplacementConfig)
        self.execution = self._config.execution
        self.removed = self._config.removed

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDeployAgentReplacementRequest
    ) -> ModelDeployAgentReplacementResult:
        if request.operation == "preflight":
            self._guard(request)
            return ModelDeployAgentReplacementResult(returncode=0)
        return self._recreate(request)

    @staticmethod
    def _result(
        returncode: int,
        stdout: str | None,
        stderr: str | None,
    ) -> ModelDeployAgentReplacementResult:
        return ModelDeployAgentReplacementResult(
            returncode=returncode,
            stdout=stdout or "",
            stderr=stderr or "",
        )

    def _guard(self, request: ModelDeployAgentReplacementRequest) -> None:
        """Refuse a container replacement over a live recorded hot-patch.

        Mirrors ``guard_hotpatch_ledger`` in handler_runtime_deploy.sh: no ledger on
        the host means nothing was recorded and the gate is skipped with a warning;
        a ledger with no gate script, no ``OMNI_HOME`` or no readable build ref is a
        refusal, and so is any non-zero exit of the preflight itself.
        """
        ledger = self._config.ledger_path
        if not ledger.is_file():
            logger.warning(
                "No hot-patch ledger at %s; nothing recorded on this host, gate skipped",
                ledger,
            )
            return
        gate = (
            Path(self.execution.REPO_DIR) / "scripts" / "preflight_hotpatch_ledger.py"
        )
        if not gate.is_file():
            raise HotpatchLedgerPreflightError(
                f"hot-patch ledger exists at {ledger} but the gate script is missing: "
                f"{gate}; refusing to replace containers over recorded hot-patches"
            )
        clones_root = self._config.clones_root.strip()
        if not clones_root:
            raise HotpatchLedgerPreflightError(
                f"hot-patch ledger exists at {ledger} but OMNI_HOME is unset; cannot "
                "resolve build-input clones for the hot-patch preflight"
            )
        head = self.execution._run(
            ["git", "-C", self.execution.REPO_DIR, "rev-parse", "HEAD"],
            timeout=self._config.ref_timeout,
            env=self._config.git_environment,
        )
        if head.returncode != 0 or not head.stdout.strip():
            raise HotpatchLedgerPreflightError(
                f"hot-patch ledger exists at {ledger} but the build ref of {self.execution.REPO_DIR} "
                f"could not be read: {head.stderr.strip()[:400]}"
            )
        cmd = [
            self._config.python_executable,
            str(gate),
            "--lane",
            request.compose_project.removeprefix("omnibase-infra").removeprefix("-")
            or "dev",
            "--ledger",
            str(ledger),
            "--clones-root",
            clones_root,
            "--build-ref",
            f"omnibase_infra={head.stdout.strip()}",
        ]
        result = self.execution._run(
            cmd, timeout=self._config.gate_timeout, env=self._config.git_environment
        )
        if result.stdout.strip():
            logger.info("%s", result.stdout.strip())
        if result.returncode != 0:
            raise HotpatchLedgerPreflightError(
                f"hot-patch ledger preflight refused the {request.lane} replacement "
                f"(exit={result.returncode}): "
                f"{(result.stderr or result.stdout).strip()[:1500]}"
            )

    def _recreate(
        self, request: ModelDeployAgentReplacementRequest
    ) -> ModelDeployAgentReplacementResult:
        removed = self.removed
        argv = list(request.argv)
        timeout = request.timeout
        if timeout is None:
            raise ValueError("recreate requires a timeout")
        env = request.env
        services = request.services
        lane = self.execution.EnumRuntimeLane(request.lane)
        self._guard(request)

        def settle() -> None:
            removed.extend(
                self.execution.settle_service_containers(lane, services).removed
            )

        try:
            first = self.execution._run(argv, timeout=timeout, env=env)
        except self.execution.subprocess.TimeoutExpired:
            removed.extend(self.execution.remove_recreate_leftovers(lane, services))
            settle()
            logger.warning(
                "compose recreate of %s timed out at %ss; leftovers removed: %s; "
                "retrying once (OMN-20864)",
                ", ".join(services),
                timeout,
                removed or "none",
            )
        else:
            if first.returncode == 0:
                return self._result(first.returncode, first.stdout, first.stderr)
            settle()
            if REMOVAL_IN_PROGRESS not in f"{first.stderr or ''}{first.stdout or ''}":
                return self._result(first.returncode, first.stdout, first.stderr)
            logger.warning(
                "compose recreate of %s met a removal still in progress; settled "
                "(removed: %s) and retrying once (OMN-18475)",
                ", ".join(services),
                removed or "none",
            )
        try:
            second = self.execution._run(argv, timeout=timeout, env=env)
        except self.execution.subprocess.TimeoutExpired:
            removed.extend(self.execution.remove_recreate_leftovers(lane, services))
            settle()
            logger.exception(
                "compose recreate of %s timed out twice at %ss; leftovers removed: %s",
                ", ".join(services),
                timeout,
                removed or "none",
            )
            raise
        if second.returncode != 0:
            settle()
        return self._result(second.returncode, second.stdout, second.stderr)


def run_replacement(
    execution: ModuleType,
    *,
    lane: StrEnum,
    environment: Mapping[str, str],
    contract: dict[str, object],
    bus: ProtocolDispatchBusClientTransport,
    lifecycle: ProtocolEventBusLifecycle,
    argv: list[str] | None = None,
    timeout: int | None = None,
    services: Sequence[str] = (),
    env: dict[str, str] | None = None,
    removed: list[str] | None = None,
) -> ModelDeployAgentReplacementResult:
    """Inject host I/O and dispatch the contract over the supplied bus.

    The agent already consumes a signed deploy command. Its admission effect
    stays on this host even when the runtime it is about to replace is down.
    """
    policy = contract["replacement_policy"]
    if not isinstance(policy, dict):
        raise ValueError("replacement_policy must be a mapping")
    config = ModelDeployAgentReplacementConfig(
        execution=execution,
        ledger_path=Path(
            environment.get("HOTPATCH_LEDGER_PATH") or policy["ledger_path"]
        ),
        clones_root=environment.get("OMNI_HOME", ""),
        git_environment=scrub_git_location_env(environment),
        python_executable=execution.sys.executable,
        gate_timeout=policy["gate_timeout"],
        ref_timeout=policy["ref_timeout"],
    )
    request = ModelDeployAgentReplacementRequest(
        operation="preflight" if argv is None else "recreate",
        lane=lane.value,
        compose_project=execution.lane_config_for(lane).compose_project,
        argv=tuple(argv or ()),
        timeout=timeout,
        services=tuple(services),
        env=env,
    )
    try:
        result = run_coro_sync(_dispatch(config, request, contract, bus, lifecycle))
    finally:
        if removed is not None:
            removed.extend(config.removed)
    return result


async def _dispatch(
    config: ModelDeployAgentReplacementConfig,
    request: ModelDeployAgentReplacementRequest,
    contract: dict[str, object],
    bus: ProtocolDispatchBusClientTransport,
    lifecycle: ProtocolEventBusLifecycle,
) -> ModelDeployAgentReplacementResult:
    """Wire one contract-declared command and terminal event to its handler."""
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        ModelDeployAgentReplacementConfig, config
    )
    await container.service_registry.register_instance(
        HandlerDeployAgentReplacement, HandlerDeployAgentReplacement(container)
    )
    handler = container.get_service(HandlerDeployAgentReplacement)
    await lifecycle.start()
    declaration = contract["host_dispatch"]
    if not isinstance(declaration, dict):
        raise ValueError("replacement contract host_dispatch must be a mapping")
    command_topic = str(declaration["command_topic"])
    terminal_topic = str(declaration["terminal_topic"])
    answers: list[ModelDeployAgentReplacementResult] = []
    errors: list[Exception] = []

    async def on_terminal(message: ModelEventMessage) -> None:
        answers.append(
            ModelDeployAgentReplacementResult.model_validate_json(message.value)
        )

    async def on_command(message: ModelEventMessage) -> None:
        try:
            result = handler.handle(
                ModelDeployAgentReplacementRequest.model_validate_json(message.value)
            )
        except (
            HotpatchLedgerPreflightError,
            config.execution.subprocess.SubprocessError,
            OSError,
            ValueError,
        ) as exc:
            errors.append(exc)
            result = ModelDeployAgentReplacementResult(
                returncode=1, error_message=str(exc)
            )
        await bus.publish(terminal_topic, None, result.model_dump_json().encode())

    try:
        await bus.subscribe(
            terminal_topic, group_id=terminal_topic, on_message=on_terminal
        )
        await bus.subscribe(
            command_topic, group_id=command_topic, on_message=on_command
        )
        await bus.publish(command_topic, None, request.model_dump_json().encode())
        if errors:
            raise errors[0]
        if len(answers) != 1:
            raise RuntimeError("replacement effect produced no unique terminal result")
        return answers[0]
    finally:
        await lifecycle.shutdown()
