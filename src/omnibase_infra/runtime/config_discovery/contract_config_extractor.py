# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract configuration extractor for ONEX Infrastructure.

Scans ONEX contract YAML files and extracts configuration requirements by
inspecting only Pydantic-backed fields:

    - ``metadata.transport_type`` -- the transport type declared in metadata
    - ``handler_routing.handlers[].handler_type`` -- handler-level transport types
    - ``dependencies[].type == "environment"`` -- explicit env var dependencies

This extractor does NOT parse untyped YAML sections. It only inspects
fields that are backed by Pydantic models in omnibase_core/omnibase_infra.

Thread Safety:
    This class is stateless and safe for concurrent use.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.models.model_config_requirement import (
    ModelConfigRequirement,
)
from omnibase_infra.runtime.config_discovery.models.model_config_requirements import (
    ModelConfigRequirements,
)
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)

logger = logging.getLogger(__name__)

# Mapping from contract transport_type string values to enum members.
# Uses EnumInfraTransportType values (e.g., "db" -> DATABASE).
_TRANSPORT_STR_MAP: dict[str, EnumInfraTransportType] = {
    t.value: t for t in EnumInfraTransportType
}

# Additional aliases used in contracts (e.g., "database" -> DATABASE).
_TRANSPORT_ALIASES: dict[str, EnumInfraTransportType] = {
    # Database aliases
    "database": EnumInfraTransportType.DATABASE,
    "postgres": EnumInfraTransportType.DATABASE,
    "postgresql": EnumInfraTransportType.DATABASE,
    # Message broker aliases
    "redis": EnumInfraTransportType.VALKEY,
    "redpanda": EnumInfraTransportType.KAFKA,
    # Secret management aliases
    "secret": EnumInfraTransportType.INFISICAL,
    "secrets": EnumInfraTransportType.INFISICAL,
    # Graph database aliases (Memgraph/Neo4j)
    # "memgraph" handler_type -> GRAPH transport (GRAPH_HOST, GRAPH_PORT, GRAPH_PROTOCOL)
    "memgraph": EnumInfraTransportType.GRAPH,
    # "intent" handler_type -> wraps HandlerGraph for Memgraph intent storage
    "intent": EnumInfraTransportType.GRAPH,
    # Filesystem aliases — handlers that read/write local filesystem artifacts
    # "repo_state" -> reads git repository state from local filesystem
    "repo_state": EnumInfraTransportType.FILESYSTEM,
    # "rrh_storage" -> writes RRH result JSON artifacts to local filesystem
    "rrh_storage": EnumInfraTransportType.FILESYSTEM,
    # Internal/runtime aliases — pure-compute or environment-inspection handlers
    # with no external service credentials.  Mapping to RUNTIME or INMEMORY
    # causes the extractor to record zero config keys (both have empty key
    # tuples in _TRANSPORT_KEYS), which is the correct behaviour.
    #
    # "architecture_validation" -> pure COMPUTE, validates ONEX architecture rules
    "architecture_validation": EnumInfraTransportType.RUNTIME,
    # "auth_gate" -> pure COMPUTE, evaluates work-authorization cascade
    "auth_gate": EnumInfraTransportType.RUNTIME,
    # "ledger_projection" -> pure COMPUTE, projects events into audit ledger
    "ledger_projection": EnumInfraTransportType.RUNTIME,
    # "validation_ledger_projection" -> pure COMPUTE, projects validation events
    "validation_ledger_projection": EnumInfraTransportType.RUNTIME,
    # "rrh_validate" -> pure COMPUTE, evaluates RRH validation rules
    "rrh_validate": EnumInfraTransportType.RUNTIME,
    # "runtime_target" -> collects deployment runtime target context from env
    "runtime_target": EnumInfraTransportType.RUNTIME,
    # "toolchain" -> collects build-tool versions from the local environment
    "toolchain": EnumInfraTransportType.RUNTIME,
    # "mock" -> in-memory test/mock handler, no external credentials
    "mock": EnumInfraTransportType.INMEMORY,
}


def _resolve_transport(value: str) -> EnumInfraTransportType | None:
    """Resolve a string to an EnumInfraTransportType.

    Tries the enum value map first, then aliases. Returns None if
    the string does not match any known transport type.
    """
    normalized = value.strip().lower()
    result = _TRANSPORT_STR_MAP.get(normalized)
    if result is not None:
        return result
    return _TRANSPORT_ALIASES.get(normalized)


class ContractConfigExtractor:
    """Extracts configuration requirements from ONEX contract YAML files.

    Usage::

        extractor = ContractConfigExtractor()
        reqs = extractor.extract_from_paths([
            Path("src/omnibase_infra/nodes/handlers/db/contract.yaml"),
            Path("src/omnibase_infra/nodes/handlers/consul/contract.yaml"),
        ])
        for req in reqs.requirements:
            print(f"{req.key} from {req.transport_type.value}")
    """

    def __init__(self) -> None:
        self._transport_map = TransportConfigMap()

    def extract_from_yaml(self, contract_path: Path) -> ModelConfigRequirements:
        """Extract config requirements from a single contract YAML file.

        Scans three Pydantic-backed sections:
            1. ``metadata.transport_type``
            2. ``handler_routing.handlers[].handler_type``
            3. ``dependencies[].type == "environment"``

        Args:
            contract_path: Path to the contract YAML file.

        Returns:
            ``ModelConfigRequirements`` with discovered requirements.
        """
        requirements: list[ModelConfigRequirement] = []
        transport_types: list[EnumInfraTransportType] = []
        errors: list[str] = []

        try:
            raw = contract_path.read_text(encoding="utf-8")
            # ONEX_EXCLUDE: any_type - yaml.safe_load returns untyped dict from contract YAML
            raw_data: object = yaml.safe_load(raw)
        except Exception as exc:
            return ModelConfigRequirements(
                contract_paths=(contract_path,),
                errors=(f"Failed to parse {contract_path}: {exc}",),
            )

        if raw_data is None:
            # Empty file -- valid but nothing to extract.
            return ModelConfigRequirements(contract_paths=(contract_path,))

        if not isinstance(raw_data, dict):
            return ModelConfigRequirements(
                contract_paths=(contract_path,),
                errors=(
                    f"Contract {contract_path} has a non-mapping YAML root "
                    f"(got {type(raw_data).__name__})",
                ),
            )

        data: dict[str, object] = raw_data

        # 1. metadata.transport_type
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            transport_str = metadata.get("transport_type", "")
            if transport_str:
                transport = _resolve_transport(str(transport_str))
                if transport is not None:
                    if transport not in transport_types:
                        transport_types.append(transport)
                    for key in self._transport_map.keys_for_transport(transport):
                        requirements.append(
                            ModelConfigRequirement(
                                key=key,
                                transport_type=transport,
                                source_contract=contract_path,
                                source_field="metadata.transport_type",
                            )
                        )
                else:
                    errors.append(
                        f"Unknown transport_type '{transport_str}' in "
                        f"{contract_path}::metadata.transport_type"
                    )

        # 2. handler_routing.handlers[].handler_type
        handler_routing = data.get("handler_routing", {})
        if isinstance(handler_routing, dict):
            handlers = handler_routing.get("handlers", [])
            if isinstance(handlers, list):
                for idx, handler_entry in enumerate(handlers):
                    if not isinstance(handler_entry, dict):
                        continue
                    handler_type_str = handler_entry.get("handler_type", "")
                    if not handler_type_str:
                        continue
                    transport = _resolve_transport(str(handler_type_str))
                    if transport is not None:
                        if transport not in transport_types:
                            transport_types.append(transport)
                        for key in self._transport_map.keys_for_transport(transport):
                            requirements.append(
                                ModelConfigRequirement(
                                    key=key,
                                    transport_type=transport,
                                    source_contract=contract_path,
                                    source_field=f"handler_routing.handlers[{idx}].handler_type",
                                )
                            )
                    else:
                        errors.append(
                            f"Unknown handler_type '{handler_type_str}' in "
                            f"{contract_path}::handler_routing.handlers[{idx}]"
                        )

        # 3. dependencies[].type == "environment"
        dependencies = data.get("dependencies", [])
        if isinstance(dependencies, list):
            for idx, dep in enumerate(dependencies):
                if not isinstance(dep, dict):
                    continue
                if dep.get("type") != "environment":
                    continue
                env_var = dep.get("env_var", "")
                if not env_var:
                    continue
                dep_required = dep.get("required", True)
                requirements.append(
                    ModelConfigRequirement(
                        key=str(env_var),
                        transport_type=EnumInfraTransportType.RUNTIME,
                        source_contract=contract_path,
                        source_field=f"dependencies[{idx}]",
                        required=bool(dep_required),
                    )
                )

        return ModelConfigRequirements(
            requirements=tuple(requirements),
            transport_types=tuple(transport_types),
            contract_paths=(contract_path,),
            errors=tuple(errors),
        )

    def extract_from_paths(self, contract_paths: list[Path]) -> ModelConfigRequirements:
        """Extract config requirements from multiple contract files.

        Scans each path. If a path is a directory, recursively finds
        all contract YAML files within it. Discovered patterns:

        - ``contract.yaml`` -- canonical node contract (original convention)
        - ``contract_*.yaml`` -- named contract variant (e.g. ``contract_omniclaude_runtime.yaml``)

        Both patterns are supported so that repos adopting the named-contract
        convention (introduced in OMN-2990) are visible to the seeder without
        requiring files to be renamed or moved.

        Args:
            contract_paths: List of file or directory paths to scan.

        Returns:
            Merged ``ModelConfigRequirements`` from all contracts.
        """
        result = ModelConfigRequirements()

        files_to_scan: list[Path] = []
        for path in contract_paths:
            if path.is_dir():
                # Collect both naming conventions and deduplicate via a set
                # before sorting so the final scan order is deterministic.
                matched: set[Path] = set(path.rglob("contract.yaml"))
                matched.update(path.rglob("contract_*.yaml"))
                files_to_scan.extend(sorted(matched))
            elif path.is_file():
                files_to_scan.append(path)
            else:
                result = result.merge(
                    ModelConfigRequirements(
                        errors=(f"Path does not exist: {path}",),
                    )
                )

        for contract_file in files_to_scan:
            extracted = self.extract_from_yaml(contract_file)
            result = result.merge(extracted)

        logger.info(
            "Config extraction complete: %d requirements from %d contracts "
            "(%d transport types, %d errors)",
            len(result.requirements),
            len(result.contract_paths),
            len(result.transport_types),
            len(result.errors),
        )

        return result
