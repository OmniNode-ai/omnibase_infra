# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that runs `docker compose up -d` to provision local services.

Invariants:
  I3 — Monkeypatch discipline: `import asyncio` at module top; patch via
       `handler_mod.asyncio.create_subprocess_exec` and
       `handler_mod.asyncio.open_connection`.
  I4 — Port semantics: uses asyncio.open_connection (OPEN check).
       Never uses connect_ex here.
  I7 — Compose file path: handler receives `compose_file_path` as an already-
       resolved string. Validates that it exists; does NOT call
       resolve_compose_file().

Ticket: OMN-3493
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
    ModelLocalProvisionEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_LOCALHOST = "localhost"


async def _poll_port_open(host: str, port: int, max_wait: float) -> bool:
    """Poll until the TCP port is open or max_wait seconds elapse.

    Uses asyncio.open_connection (OPEN check — Invariant I4).
    Exponential backoff starting at 0.5 s, capped by remaining time.

    Args:
        host: Hostname or IP to connect to.
        port: TCP port number to probe.
        max_wait: Maximum total seconds to wait.

    Returns:
        True if the port opened within max_wait, False if timeout was reached.
    """
    deadline = time.monotonic() + max_wait
    delay = 0.5
    while time.monotonic() < deadline:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=min(1.0, remaining),
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (TimeoutError, OSError):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, remaining)
    return False


class HandlerLocalProvision:
    """Runs `docker compose up -d` and polls ports until healthy or timeout.

    Handler for the ``provision_local`` operation of
    ``NodeLocalProvisionEffect``.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the handler with a DI container."""
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the handler type enum value."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the handler category enum value."""
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler (no-op for this handler)."""
        self._initialized = True
        logger.info("HandlerLocalProvision initialized")

    async def shutdown(self) -> None:
        """Shut down the handler (no-op for this handler)."""
        self._initialized = False
        logger.info("HandlerLocalProvision shutdown")

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """Provision local services via docker compose.

        Envelope keys:
            topology: ModelDeploymentTopology — defines services and profiles.
            compose_file_path: str — already-resolved path to the compose file.
            correlation_id: UUID — for tracing.
            max_wait_seconds: int — max seconds to wait per service (default 120).

        Returns:
            ModelHandlerOutput wrapping ModelLocalProvisionEffectOutput.
        """
        from omnibase_core.models.core.model_deployment_topology import (
            ModelDeploymentTopology,
        )

        correlation_id_raw = envelope.get("correlation_id")
        _context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="provision_local",
            target_name="docker_compose",
        )
        corr_id: UUID = (
            correlation_id_raw if isinstance(correlation_id_raw, UUID) else uuid4()
        )

        # --- I7: validate compose_file_path boundary ---
        compose_file_path_raw = envelope.get("compose_file_path")
        if not isinstance(compose_file_path_raw, str):
            return _make_error_output(
                corr_id,
                "compose_file_path must be a string",
            )
        compose_path = Path(compose_file_path_raw)
        if not compose_path.exists():
            raise RuntimeError(
                f"Compose file not found: {compose_path}. "
                "Set ONEX_COMPOSE_FILE or use --compose-file."
            )

        topology_raw = envelope.get("topology")
        if not isinstance(topology_raw, ModelDeploymentTopology):
            return _make_error_output(
                corr_id, "topology must be a ModelDeploymentTopology"
            )

        max_wait_seconds_raw = envelope.get("max_wait_seconds", 120)
        max_wait_seconds: int = (
            int(max_wait_seconds_raw)
            if isinstance(max_wait_seconds_raw, (int, float))
            else 120
        )

        topology: ModelDeploymentTopology = topology_raw

        # --- Collect unique profiles (deduplication) ---
        profiles: set[str] = set()
        local_service_names: list[str] = []
        local_services = topology.local_services()
        for svc_name, svc in local_services.items():
            if svc.local is not None:
                if svc.local.compose_profile:
                    profiles.add(svc.local.compose_profile)
                local_service_names.append(svc.local.compose_service)

        if not local_service_names:
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-local-provision",
                result=ModelLocalProvisionEffectOutput(
                    success=True,
                    correlation_id=corr_id,
                    services_started=(),
                ),
            )

        # --- Build docker compose command (sorted profile flags for determinism) ---
        profile_flags: list[str] = [
            item for p in sorted(profiles) for item in ["--profile", p]
        ]
        cmd: list[str] = (
            ["docker", "compose", "-f", str(compose_path)]
            + profile_flags
            + ["up", "-d"]
            + local_service_names
        )

        logger.info("Running: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout_bytes, stderr_bytes = await proc.communicate()
        returncode = proc.returncode

        if returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            logger.error(
                "docker compose up failed (rc=%d): %s", returncode, stderr_text
            )
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-local-provision",
                result=ModelLocalProvisionEffectOutput(
                    success=False,
                    correlation_id=corr_id,
                    services_started=(),
                    error=stderr_text
                    or f"docker compose exited with code {returncode}",
                ),
            )

        # --- Poll ports until healthy or timeout ---
        services_started: list[str] = []
        services_failed: list[str] = []

        for svc_name, svc in local_services.items():
            if svc.local is None:
                continue
            host = _LOCALHOST
            port = svc.local.host_port
            opened = await _poll_port_open(host, port, float(max_wait_seconds))
            if opened:
                services_started.append(svc.local.compose_service)
                logger.info(
                    "Service %s healthy on %s:%d",
                    svc_name,
                    host,
                    port,
                )
            else:
                services_failed.append(svc.local.compose_service)
                logger.warning(
                    "Service %s did not open port %d within %ds",
                    svc_name,
                    port,
                    max_wait_seconds,
                )

        all_ok = len(services_failed) == 0
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-local-provision",
            result=ModelLocalProvisionEffectOutput(
                success=all_ok,
                correlation_id=corr_id,
                services_started=tuple(services_started),
                error=(
                    f"Timed out waiting for services: {services_failed}"
                    if services_failed
                    else None
                ),
            ),
        )


def _make_error_output(
    corr_id: UUID,
    error_message: str,
) -> ModelHandlerOutput:
    """Build an error ModelHandlerOutput wrapping ModelLocalProvisionEffectOutput."""
    return ModelHandlerOutput.for_compute(
        input_envelope_id=uuid4(),
        correlation_id=corr_id,
        handler_id="handler-local-provision",
        result=ModelLocalProvisionEffectOutput(
            success=False,
            correlation_id=corr_id,
            error=error_message,
        ),
    )


__all__: list[str] = ["HandlerLocalProvision"]
