# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that coordinates all 4 setup effect nodes sequentially.

Sequential workflow:
    I8 (Cloud gate) → Preflight → Local provision → Infisical → Post-provision validate

Invariants:
    I5 — Orchestrator output has no ``result`` field: ModelSetupOrchestratorOutput
         intentionally omits ``result``. Enforced by model definition and strict test.
    I6 — Event types are constants: All emitted event types must be in
         SETUP_EVENT_TYPES frozenset.
    I8 — Cloud gate is a hard stop: If any service has mode=CLOUD, emit
         setup.cloud.unavailable with gated_services list and return immediately.
         Do NOT proceed to preflight. Do NOT convert cloud to disabled at runtime.

Ticket: OMN-3495
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums import EnumDeploymentMode
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_setup_orchestrator.constants.setup_event_types import (
    SETUP_EVENT_TYPES,
)
from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_event import (
    ModelSetupEvent,
)
from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_orchestrator_output import (
    ModelSetupOrchestratorOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.nodes.node_setup_orchestrator.protocols.protocol_infisical_effect import (
        ProtocolInfisicalEffect,
    )
    from omnibase_infra.nodes.node_setup_orchestrator.protocols.protocol_preflight_effect import (
        ProtocolPreflightEffect,
    )
    from omnibase_infra.nodes.node_setup_orchestrator.protocols.protocol_provision_effect import (
        ProtocolProvisionEffect,
    )
    from omnibase_infra.nodes.node_setup_orchestrator.protocols.protocol_validate_effect import (
        ProtocolValidateEffect,
    )

logger = logging.getLogger(__name__)


class HandlerSetupOrchestrator:
    """Coordinates all 4 setup effect nodes sequentially.

    Workflow:
        1. Cloud gate check (I8): hard stop if any service mode == CLOUD.
        2. Preflight: validate all prerequisites.
        3. Local provision: start Docker Compose services.
        4. Infisical setup: bootstrap Infisical (skipped if not in topology).
        5. Post-provision validate: health-check all deployed services.

    All effect nodes are injected via protocol interfaces, enabling full
    test isolation without subprocess or Docker dependencies.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        preflight: ProtocolPreflightEffect,
        provision: ProtocolProvisionEffect,
        infisical: ProtocolInfisicalEffect,
        validate: ProtocolValidateEffect,
    ) -> None:
        """Initialize with container and effect-node protocol dependencies.

        Args:
            container: ONEX dependency injection container.
            preflight: Protocol implementation for preflight checks.
            provision: Protocol implementation for local Docker Compose provisioning.
            infisical: Protocol implementation for Infisical setup.
            validate: Protocol implementation for post-provision health validation.
        """
        self._container = container
        self._preflight = preflight
        self._provision = provision
        self._infisical = infisical
        self._validate = validate
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler."""
        self._initialized = True
        logger.info("HandlerSetupOrchestrator initialized")

    async def shutdown(self) -> None:
        """Shut down the handler."""
        self._initialized = False
        logger.info("HandlerSetupOrchestrator shutdown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _event(
        self,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> ModelSetupEvent:
        """Construct a ModelSetupEvent.

        Validates that event_type is in SETUP_EVENT_TYPES (I6).

        Raises:
            ValueError: If event_type is not in SETUP_EVENT_TYPES.
        """
        if event_type not in SETUP_EVENT_TYPES:
            raise ValueError(
                f"I6 violated: event_type '{event_type}' is not in SETUP_EVENT_TYPES"
            )
        return ModelSetupEvent(
            event_type=event_type,
            payload=payload or {},
        )

    def _emit(
        self,
        events: list[ModelSetupEvent],
        correlation_id: UUID,
    ) -> ModelHandlerOutput[ModelSetupOrchestratorOutput]:
        """Wrap a list of events in a ModelHandlerOutput."""
        output = ModelSetupOrchestratorOutput(
            correlation_id=correlation_id,
            events=tuple(events),
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id="handler-setup-orchestrator",
            result=output,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelSetupOrchestratorOutput]:
        """Coordinate all 4 setup effect nodes in sequence.

        Envelope keys:
            topology: ModelDeploymentTopology — deployment topology.
            correlation_id: UUID — correlation ID for distributed tracing.
            compose_file_path: str — path to Docker Compose file for provision step.
            dry_run: bool — if True, pass dry_run hint to sub-effects (default False).
        """
        topology_raw = envelope.get("topology")
        correlation_id_raw = envelope.get("correlation_id")
        compose_file_path_raw = envelope.get("compose_file_path", "")
        compose_file_path = str(compose_file_path_raw)

        # Resolve or generate correlation_id
        if isinstance(correlation_id_raw, UUID):
            corr_id = correlation_id_raw
        else:
            corr_id = uuid4()

        # We need the topology as a ModelDeploymentTopology.
        # Import here to avoid circular imports at module level.
        from omnibase_core.models.core.model_deployment_topology import (
            ModelDeploymentTopology,
        )

        if not isinstance(topology_raw, ModelDeploymentTopology):
            raise TypeError(
                f"envelope['topology'] must be ModelDeploymentTopology, "
                f"got {type(topology_raw)}"
            )
        topology: ModelDeploymentTopology = topology_raw

        return await self.handle(topology, corr_id, compose_file_path)

    async def handle(
        self,
        topology: ModelDeploymentTopology,
        correlation_id: UUID,
        compose_file_path: str,
    ) -> ModelHandlerOutput[ModelSetupOrchestratorOutput]:
        """Coordinate the full sequential setup workflow.

        Invariant I8: Cloud gate runs FIRST. If any service has mode=CLOUD,
        emit setup.cloud.unavailable with gated_services and return immediately.
        Do NOT proceed to preflight.

        Args:
            topology: Deployment topology for the setup workflow.
            correlation_id: UUID for distributed tracing.
            compose_file_path: Path to Docker Compose file for provision step.

        Returns:
            ModelHandlerOutput containing ModelSetupOrchestratorOutput with all
            emitted lifecycle events.
        """
        # ---------------------------------------------------------------
        # I8: Cloud gate — hard stop before preflight
        # ---------------------------------------------------------------
        cloud_services = [
            name
            for name, svc in topology.services.items()
            if svc.mode == EnumDeploymentMode.CLOUD
        ]
        if cloud_services:
            logger.warning(
                "Cloud gate triggered: services=%s. Halting setup.", cloud_services
            )
            return self._emit(
                [
                    self._event(
                        "setup.cloud.unavailable",
                        {"gated_services": cloud_services},
                    )
                ],
                correlation_id,
            )

        events: list[ModelSetupEvent] = []

        # ---------------------------------------------------------------
        # Step 1: Preflight
        # ---------------------------------------------------------------
        events.append(self._event("setup.preflight.started"))
        preflight_result = await self._preflight.run_preflight(correlation_id)
        if not preflight_result.passed:
            failed_checks = [
                c.check_key for c in preflight_result.checks if not c.passed
            ]
            events.append(
                self._event(
                    "setup.preflight.failed",
                    {"failed_checks": failed_checks},
                )
            )
            logger.error("Preflight failed: %s", failed_checks)
            return self._emit(events, correlation_id)
        events.append(self._event("setup.preflight.completed"))

        # ---------------------------------------------------------------
        # Step 2: Local provision
        # ---------------------------------------------------------------
        events.append(self._event("setup.provision.started"))
        provision_result = await self._provision.provision_local(
            topology,
            compose_file_path,
            correlation_id,
        )
        if not provision_result.success:
            events.append(
                self._event(
                    "setup.provision.failed",
                    {"error": provision_result.error or "unknown provision error"},
                )
            )
            logger.error("Provision failed: %s", provision_result.error)
            return self._emit(events, correlation_id)
        events.append(self._event("setup.provision.completed"))

        # ---------------------------------------------------------------
        # Step 3: Infisical (only if infisical is in topology and enabled)
        # ---------------------------------------------------------------
        if topology.is_service_enabled("infisical"):
            events.append(self._event("setup.infisical.started"))
            infisical_result = await self._infisical.setup_infisical(
                topology,
                correlation_id,
            )
            if infisical_result.status == "failed":
                events.append(
                    self._event(
                        "setup.infisical.failed",
                        {"error": infisical_result.error or "unknown infisical error"},
                    )
                )
                logger.error("Infisical setup failed: %s", infisical_result.error)
                return self._emit(events, correlation_id)
            if infisical_result.status == "skipped":
                events.append(self._event("setup.infisical.skipped"))
            else:
                events.append(self._event("setup.infisical.completed"))
        else:
            events.append(self._event("setup.infisical.skipped"))

        # ---------------------------------------------------------------
        # Step 4: Post-provision validate
        # ---------------------------------------------------------------
        events.append(self._event("setup.validate.started"))
        validate_result = await self._validate.validate_services(
            topology,
            correlation_id,
        )
        if not validate_result.all_healthy:
            events.append(
                self._event(
                    "setup.validate.failed",
                    {
                        "error": validate_result.error
                        or "one or more services unhealthy"
                    },
                )
            )
            logger.error("Post-provision validation failed: %s", validate_result.error)
            return self._emit(events, correlation_id)
        events.append(self._event("setup.validate.completed"))

        # ---------------------------------------------------------------
        # All steps completed successfully
        # ---------------------------------------------------------------
        events.append(self._event("setup.completed"))
        logger.info("Setup orchestration completed successfully")
        return self._emit(events, correlation_id)


__all__: list[str] = ["HandlerSetupOrchestrator"]
