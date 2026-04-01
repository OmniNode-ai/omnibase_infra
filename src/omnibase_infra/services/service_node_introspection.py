# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Standalone node introspection service for kernel-level introspection.

Provides a ProtocolNodeIntrospection implementation that the ServiceKernel
can inject into RuntimeHostProcess to publish introspection events with
full metadata, capabilities, and event_bus fields from the node's contract.

Without this service, RuntimeHostProcess receives introspection_service=None
and silently skips all introspection publishing, leaving the omnidash Node
Registry and Topic Registry empty.

Related:
    - OMN-5609: Wire contract data into introspection emission path
    - OMN-1930: Phase 1 - Fix Auto-Introspection

.. versionadded:: 0.24.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID, uuid4

import yaml

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.contracts.model_contract_base import ModelContractBase
from omnibase_core.models.contracts.subcontracts.model_event_bus_subcontract import (
    ModelEventBusSubcontract,
)
from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.models.discovery.model_introspection_config import (
    ModelIntrospectionConfig,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)

logger = logging.getLogger(__name__)


def _attach_event_bus(
    contract: ModelContractBase,
    event_bus_sub: ModelEventBusSubcontract,
) -> ModelContractBase:
    """Attach event_bus subcontract to a contract instance.

    MixinNodeIntrospection._extract_event_bus_config() reads event_bus
    topics via ``getattr(contract, 'event_bus', None)``.  Because
    ModelContractBase uses ``extra='forbid'``, the ``event_bus`` YAML
    section is NOT parsed into the Pydantic model.  This function
    sets the attribute directly on the instance's ``__dict__`` to
    bypass Pydantic's field validation.

    Args:
        contract: Parsed contract (ModelContractBase subclass).
        event_bus_sub: Parsed event_bus subcontract from the same YAML.

    Returns:
        The same contract instance with ``event_bus`` attribute attached.
    """
    # Pydantic frozen models block __setattr__, so write to __dict__ directly.
    contract.__dict__["event_bus"] = event_bus_sub
    return contract


class ServiceNodeIntrospection(MixinNodeIntrospection):
    """Standalone introspection service for kernel-level node announcement.

    Wraps MixinNodeIntrospection to provide ProtocolNodeIntrospection
    conformance. The kernel creates one instance per runtime, passing
    the node's contract so that introspection events include:

    - metadata.description (from contract.description)
    - event_bus (from contract event_bus subcontract topics)
    - contract_capabilities (from ContractCapabilityExtractor)
    - declared_capabilities (from contract node_capabilities block)

    Thread Safety:
        Inherits thread safety from MixinNodeIntrospection.
        All async methods are safe for concurrent calls.

    Example::

        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=Path("contracts/runtime"),
            event_bus=kafka_bus,
            node_name="my-service",
            environment="local",
        )
        await service.publish_introspection(EnumIntrospectionReason.STARTUP)
    """

    # Instance attribute for description override when contract parsing fails
    _description_override: str | None = None

    async def get_introspection_data(self) -> ModelNodeIntrospectionEvent:
        """Override to inject description from raw YAML when contract parsing fails.

        OMN-6405: ModelContractBase uses extra='forbid' and rejects rich contract
        YAML fields. When the contract can't be parsed, the mixin produces
        metadata.description=None. This override injects the description extracted
        directly from the YAML file.
        """
        event = await super().get_introspection_data()
        if (
            self._description_override
            and event.metadata is not None
            and event.metadata.description is None
        ):
            # ModelNodeMetadata is not frozen — direct assignment is safe
            event.metadata.description = self._description_override
        return event

    @classmethod
    def from_kernel_config(
        cls,
        *,
        event_bus: object | None,
        node_name: str,
        node_id: UUID | None = None,
        node_type: EnumNodeKind = EnumNodeKind.ORCHESTRATOR,
        version: str = "1.0.0",
        environment: str = "local",
        description: str | None = None,
        contract: ModelContractBase | None = None,
        event_bus_subcontract: ModelEventBusSubcontract | None = None,
    ) -> ServiceNodeIntrospection:
        """Create an introspection service from kernel configuration.

        Args:
            event_bus: Event bus for publishing introspection events.
            node_name: Service/node name (from runtime_config.yaml).
            node_id: Unique node ID. Generated if not provided.
            node_type: ONEX node type classification.
            version: Node version string.
            environment: Deployment environment (local, staging, production).
            description: Human-readable description. Falls back to contract
                description if not provided.
            contract: Parsed ONEX contract for capability extraction.
            event_bus_subcontract: Parsed event_bus section from the contract
                YAML. When provided, the introspection event will include
                subscribe/publish topics.

        Returns:
            Initialized ServiceNodeIntrospection ready for publishing.
        """
        instance = cls()

        effective_description = description
        if effective_description is None and contract is not None:
            effective_description = getattr(contract, "description", None)

        # Attach event_bus subcontract to the contract so the mixin can read
        # topics via getattr(contract, 'event_bus', None).
        if contract is not None and event_bus_subcontract is not None:
            _attach_event_bus(contract, event_bus_subcontract)

        config = ModelIntrospectionConfig(
            node_id=node_id or uuid4(),
            node_type=node_type,
            node_name=node_name,
            event_bus=event_bus,
            version=version,
            env=environment,
            service=node_name,
            contract=contract,
        )
        instance.initialize_introspection(config)

        # OMN-6405: When full contract parsing fails (extra="forbid" rejects
        # rich YAML fields), the mixin's _introspection_contract is None and
        # get_introspection_data() produces metadata.description=None. Store
        # the description extracted from raw YAML so our get_introspection_data
        # override can inject it into the event metadata.
        instance._description_override = effective_description

        logger.info(
            "ServiceNodeIntrospection created for %s (description=%s, "
            "has_contract=%s, has_event_bus_sub=%s)",
            node_name,
            effective_description or "(none)",
            contract is not None,
            event_bus_subcontract is not None,
        )

        return instance

    @classmethod
    def from_contract_dir(
        cls,
        *,
        contracts_dir: Path,
        event_bus: object | None,
        node_name: str,
        node_id: UUID | None = None,
        environment: str = "local",
    ) -> ServiceNodeIntrospection:
        """Create an introspection service by loading contract from directory.

        Scans contracts_dir for contract.yaml and parses both the contract
        model and the event_bus subcontract (which is a separate YAML section
        not included in the Pydantic contract model).

        Args:
            contracts_dir: Directory containing contract YAML files.
            event_bus: Event bus for publishing introspection events.
            node_name: Service/node name.
            node_id: Unique node ID. Generated if not provided.
            environment: Deployment environment.

        Returns:
            Initialized ServiceNodeIntrospection.
        """
        contract, event_bus_sub = _try_load_contract(contracts_dir)

        # Extract metadata from contract Pydantic model if available
        # OMN-6405: Also extract name from contract so introspection events
        # carry the contract-defined node name instead of the runtime config
        # name (e.g., "node_registration_orchestrator" not "runtime_config").
        description = None
        effective_node_name = node_name
        version = "1.0.0"
        node_type = EnumNodeKind.ORCHESTRATOR

        if contract is not None:
            description = getattr(contract, "description", None)
            contract_name = getattr(contract, "name", None)
            if isinstance(contract_name, str) and contract_name:
                effective_node_name = contract_name
            contract_version = getattr(contract, "contract_version", None)
            if contract_version is not None:
                version = (
                    f"{contract_version.major}.{contract_version.minor}"
                    f".{contract_version.patch}"
                )
            raw_node_type = getattr(contract, "node_type", None)
            if raw_node_type is not None:
                node_type = _map_node_type(raw_node_type)
        else:
            # OMN-6405: Contract Pydantic parsing may fail when the YAML has
            # fields not declared in the model (extra="forbid"). Fall back to
            # reading description, node_type, and version directly from the
            # raw YAML so introspection events still carry useful metadata.
            raw_yaml = _try_read_raw_yaml(contracts_dir)
            if raw_yaml is not None:
                description = raw_yaml.get("description")
                if isinstance(description, str) and description:
                    pass  # Keep it
                else:
                    description = None
                raw_name = raw_yaml.get("name")
                if isinstance(raw_name, str) and raw_name:
                    effective_node_name = raw_name
                raw_nt = raw_yaml.get("node_type")
                if raw_nt is not None:
                    node_type = _map_node_type(raw_nt)
                cv = raw_yaml.get("contract_version")
                if isinstance(cv, dict):
                    version = f"{cv.get('major', 0)}.{cv.get('minor', 0)}.{cv.get('patch', 0)}"

        return cls.from_kernel_config(
            event_bus=event_bus,
            node_name=effective_node_name,
            node_id=node_id,
            node_type=node_type,
            version=version,
            environment=environment,
            description=description,
            contract=contract,
            event_bus_subcontract=event_bus_sub,
        )


def _try_read_raw_yaml(contracts_dir: Path) -> dict[str, object] | None:
    """Read the first contract.yaml as a raw dict without Pydantic validation.

    Used as a fallback when full contract parsing fails (OMN-6405) to extract
    description, node_type, and version directly from YAML.
    """
    candidates = [contracts_dir / "contract.yaml"]
    if contracts_dir.is_dir():
        for child in sorted(contracts_dir.iterdir()):
            if child.is_dir():
                candidate = child / "contract.yaml"
                if candidate.exists():
                    candidates.append(candidate)

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            continue
    return None


def _try_load_contract(
    contracts_dir: Path,
) -> tuple[ModelContractBase | None, ModelEventBusSubcontract | None]:
    """Attempt to load a contract and its event_bus subcontract from a directory.

    Tries contract.yaml first, then falls back to subdirectory contracts.
    Both elements are None on failure.

    Args:
        contracts_dir: Directory to search for contract files.

    Returns:
        Tuple of (contract, event_bus_subcontract).
    """
    candidates = [
        contracts_dir / "contract.yaml",
    ]
    if contracts_dir.is_dir():
        for child in sorted(contracts_dir.iterdir()):
            if child.is_dir():
                candidate = child / "contract.yaml"
                if candidate.exists():
                    candidates.append(candidate)

    for contract_path in candidates:
        if not contract_path.exists():
            continue
        try:
            with contract_path.open(encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if not isinstance(raw, dict):
                continue

            contract = _parse_contract_dict(raw)
            if contract is None:
                continue

            # Parse event_bus subcontract separately (not part of contract model)
            event_bus_sub = _parse_event_bus_subcontract(raw)

            logger.debug(
                "Loaded contract from %s (name=%s, description=%s, has_event_bus=%s)",
                contract_path,
                contract.name,
                contract.description[:50] if contract.description else "(none)",
                event_bus_sub is not None,
            )
            return contract, event_bus_sub
        except Exception as e:  # noqa: BLE001 — boundary: graceful degradation for contract loading
            logger.debug(
                "Could not parse contract from %s: %s",
                contract_path,
                e,
            )
            continue

    logger.debug("No valid contract found in %s", contracts_dir)
    return None, None


def _parse_contract_dict(raw: dict[str, object]) -> ModelContractBase | None:
    """Parse a raw YAML dict into the correct ModelContract subclass.

    Strips the ``event_bus`` key before parsing because ModelContractBase
    uses ``extra='forbid'`` and does not declare ``event_bus`` as a field.
    The event_bus section is parsed separately by ``_parse_event_bus_subcontract``.

    Args:
        raw: Parsed YAML dictionary.

    Returns:
        Concrete ModelContractBase subclass or None on failure.
    """
    from omnibase_core.models.contracts.model_contract_compute import (
        ModelContractCompute,
    )
    from omnibase_core.models.contracts.model_contract_effect import (
        ModelContractEffect,
    )
    from omnibase_core.models.contracts.model_contract_orchestrator import (
        ModelContractOrchestrator,
    )
    from omnibase_core.models.contracts.model_contract_reducer import (
        ModelContractReducer,
    )

    node_type = str(raw.get("node_type", "")).upper()

    contract_class: type[ModelContractBase]
    if "ORCHESTRATOR" in node_type:
        contract_class = ModelContractOrchestrator
    elif "REDUCER" in node_type:
        contract_class = ModelContractReducer
    elif "COMPUTE" in node_type:
        contract_class = ModelContractCompute
    else:
        contract_class = ModelContractEffect

    # Strip keys that are not part of the contract model (extra="forbid").
    # Contract YAMLs may contain extra sections (event_bus, mcp, time_injection,
    # workflow_coordination, etc.) that are NOT declared as Pydantic fields.
    # Only pass keys that the model actually declares. (OMN-6405)
    model_fields = set(contract_class.model_fields.keys())
    filtered = {k: v for k, v in raw.items() if k in model_fields}

    try:
        return contract_class(**filtered)
    except Exception as e:  # noqa: BLE001 — boundary: graceful degradation for contract parsing
        logger.debug("Failed to parse contract as %s: %s", contract_class.__name__, e)
        return None


def _parse_event_bus_subcontract(
    raw: dict[str, object],
) -> ModelEventBusSubcontract | None:
    """Parse the event_bus section from raw contract YAML.

    Args:
        raw: Full parsed YAML dictionary.

    Returns:
        Parsed ModelEventBusSubcontract or None if section is absent/invalid.
    """
    event_bus_data = raw.get("event_bus")
    if not isinstance(event_bus_data, dict):
        return None

    try:
        return ModelEventBusSubcontract(**event_bus_data)
    except Exception as e:  # noqa: BLE001 — boundary: graceful degradation for subcontract parsing
        logger.debug("Failed to parse event_bus subcontract: %s", e)
        return None


def _map_node_type(raw_node_type: object) -> EnumNodeKind:
    """Map a contract node_type value to EnumNodeKind.

    Args:
        raw_node_type: Node type from contract (EnumNodeType or string).

    Returns:
        Corresponding EnumNodeKind value.
    """
    type_str = str(raw_node_type).upper()

    if "EFFECT" in type_str:
        return EnumNodeKind.EFFECT
    elif "COMPUTE" in type_str:
        return EnumNodeKind.COMPUTE
    elif "REDUCER" in type_str:
        return EnumNodeKind.REDUCER
    elif "ORCHESTRATOR" in type_str:
        return EnumNodeKind.ORCHESTRATOR
    else:
        return EnumNodeKind.EFFECT


__all__ = ["ServiceNodeIntrospection"]
