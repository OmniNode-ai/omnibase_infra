# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that runs `docker compose down` to tear down local services.

Ticket: OMN-3493
"""

from __future__ import annotations

import asyncio
import logging
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


class HandlerLocalTeardown:
    """Stops and removes local services via `docker compose down`.

    Handler for the ``teardown_local`` operation of
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
        logger.info("HandlerLocalTeardown initialized")

    async def shutdown(self) -> None:
        """Shut down the handler (no-op for this handler)."""
        self._initialized = False
        logger.info("HandlerLocalTeardown shutdown")

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """Tear down local services via docker compose down.

        Envelope keys:
            topology: ModelDeploymentTopology — defines services and profiles.
            compose_file_path: str — already-resolved path to the compose file.
            correlation_id: UUID — for tracing.

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
            operation="teardown_local",
            target_name="docker_compose",
        )
        corr_id: UUID = (
            correlation_id_raw if isinstance(correlation_id_raw, UUID) else uuid4()
        )

        compose_file_path_raw = envelope.get("compose_file_path")
        if not isinstance(compose_file_path_raw, str):
            return _make_error_output(corr_id, "compose_file_path must be a string")
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
        topology: ModelDeploymentTopology = topology_raw

        profiles: set[str] = set()
        local_service_names: list[str] = []
        local_services = topology.local_services()
        for _svc_name, svc in local_services.items():
            if svc.local is not None:
                if svc.local.compose_profile:
                    profiles.add(svc.local.compose_profile)
                local_service_names.append(svc.local.compose_service)

        if not local_service_names:
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-local-teardown",
                result=ModelLocalProvisionEffectOutput(
                    success=True,
                    correlation_id=corr_id,
                    services_stopped=(),
                ),
            )

        profile_flags: list[str] = [
            item for p in sorted(profiles) for item in ["--profile", p]
        ]
        cmd: list[str] = (
            ["docker", "compose", "-f", str(compose_path)]
            + profile_flags
            + ["down"]
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
                "docker compose down failed (rc=%d): %s", returncode, stderr_text
            )
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-local-teardown",
                result=ModelLocalProvisionEffectOutput(
                    success=False,
                    correlation_id=corr_id,
                    services_stopped=(),
                    error=stderr_text
                    or f"docker compose exited with code {returncode}",
                ),
            )

        services_stopped = tuple(
            svc.local.compose_service
            for svc in local_services.values()
            if svc.local is not None
        )
        logger.info("Services stopped: %s", services_stopped)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-local-teardown",
            result=ModelLocalProvisionEffectOutput(
                success=True,
                correlation_id=corr_id,
                services_stopped=services_stopped,
            ),
        )


def _make_error_output(corr_id: UUID, error_message: str) -> ModelHandlerOutput:
    """Build an error ModelHandlerOutput wrapping ModelLocalProvisionEffectOutput."""
    return ModelHandlerOutput.for_compute(
        input_envelope_id=uuid4(),
        correlation_id=corr_id,
        handler_id="handler-local-teardown",
        result=ModelLocalProvisionEffectOutput(
            success=False,
            correlation_id=corr_id,
            error=error_message,
        ),
    )


__all__: list[str] = ["HandlerLocalTeardown"]
