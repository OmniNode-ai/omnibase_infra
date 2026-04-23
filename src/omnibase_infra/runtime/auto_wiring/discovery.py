# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract auto-discovery engine for ONEX auto-wiring.

Scans ``onex.nodes`` entry points, locates sibling ``contract.yaml`` files,
parses the contract subset needed for wiring, and builds a
:class:`ModelAutoWiringManifest`.

This module is **pure** — no handler imports, no Kafka connections, no I/O
beyond reading YAML files from disk.

Part of OMN-7653: Contract auto-discovery from onex.nodes entry points.
"""

from __future__ import annotations

import inspect
import logging
from importlib.metadata import entry_points
from pathlib import Path

import yaml

from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelDiscoveryError,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    ModelRuntimeProfilePolicy,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "onex.nodes"


def discover_contracts() -> ModelAutoWiringManifest:
    """Scan all ``onex.nodes`` entry points and build an auto-wiring manifest.

    For each entry point, the engine:

    1. Loads the entry point to obtain the node class (but does NOT instantiate it).
    2. Resolves the ``contract.yaml`` file adjacent to the module that defines the class.
    3. Parses the YAML and extracts the fields needed for wiring.

    Errors on individual entry points are captured — they never abort the full scan.

    Returns:
        A :class:`ModelAutoWiringManifest` with all discovered contracts and errors.
    """
    contracts: list[ModelDiscoveredContract] = []
    errors: list[ModelDiscoveryError] = []

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "unknown"
        dist_version = dist.version if dist is not None else "0.0.0"

        try:
            node_cls = ep.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load entry point '%s' from '%s': %s",
                ep.name,
                dist_name,
                exc,
            )
            errors.append(
                ModelDiscoveryError(
                    entry_point_name=ep.name,
                    package_name=dist_name,
                    error=f"Failed to load entry point: {exc}",
                )
            )
            continue

        try:
            contract_path = _resolve_contract_path(node_cls)
        except (FileNotFoundError, TypeError) as exc:
            logger.warning(
                "No contract.yaml for entry point '%s' from '%s': %s",
                ep.name,
                dist_name,
                exc,
            )
            errors.append(
                ModelDiscoveryError(
                    entry_point_name=ep.name,
                    package_name=dist_name,
                    error=str(exc),
                )
            )
            continue

        try:
            contract = _parse_contract(
                contract_path=contract_path,
                entry_point_name=ep.name,
                package_name=dist_name,
                package_version=dist_version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse contract for '%s' from '%s': %s",
                ep.name,
                dist_name,
                exc,
            )
            errors.append(
                ModelDiscoveryError(
                    entry_point_name=ep.name,
                    package_name=dist_name,
                    error=f"Failed to parse contract: {exc}",
                )
            )
            continue

        contracts.append(contract)
        logger.info(
            "Discovered contract: %s (%s) from %s %s",
            contract.name,
            contract.node_type,
            dist_name,
            dist_version,
        )

    return ModelAutoWiringManifest(
        contracts=tuple(contracts),
        errors=tuple(errors),
    )


def discover_contracts_from_paths(
    contract_paths: list[Path],
) -> ModelAutoWiringManifest:
    """Build a manifest from explicit contract.yaml file paths.

    Useful for testing and for environments where entry points are not
    available (e.g. running directly from source).

    Args:
        contract_paths: List of paths to contract.yaml files.

    Returns:
        A :class:`ModelAutoWiringManifest` with all discovered contracts and errors.
    """
    contracts: list[ModelDiscoveredContract] = []
    errors: list[ModelDiscoveryError] = []

    for path in contract_paths:
        name = path.parent.name
        try:
            contract = _parse_contract(
                contract_path=path,
                entry_point_name=name,
                package_name="local",
                package_version="0.0.0",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse contract at %s: %s", path, exc)
            errors.append(
                ModelDiscoveryError(
                    entry_point_name=name,
                    package_name="local",
                    error=f"Failed to parse contract: {exc}",
                )
            )
            continue
        contracts.append(contract)

    return ModelAutoWiringManifest(
        contracts=tuple(contracts),
        errors=tuple(errors),
    )


def _resolve_contract_path(node_cls: type) -> Path:
    """Resolve the contract.yaml path for a node class or module.

    Strategy:
    1. If the object has a ``contract_path`` attribute, use it directly.
    2. For namespace packages (no ``__file__``), search each path in ``__path__``.
    3. Otherwise, look for ``contract.yaml`` in the same directory as the
       module that defines the class.

    Raises:
        FileNotFoundError: If no contract.yaml can be located.
        TypeError: If ``inspect.getfile`` cannot locate the source (re-raised
            from caller's ``except (FileNotFoundError, TypeError)`` guard).
    """
    # Strategy 1: explicit contract_path attribute
    explicit = getattr(node_cls, "contract_path", None)
    if explicit is not None:
        p = Path(str(explicit))
        if p.is_file():
            return p

    # Strategy 2: namespace package — has __path__ but no __file__
    # Entry points may resolve to namespace packages (directories without
    # __init__.py).  inspect.getfile raises TypeError for these; check
    # __path__ entries directly instead.
    pkg_paths = getattr(node_cls, "__path__", None)
    if pkg_paths is not None:
        for pkg_dir in pkg_paths:
            candidate = Path(pkg_dir) / "contract.yaml"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No contract.yaml found in namespace package paths: {list(pkg_paths)}"
        )

    # Strategy 3: sibling contract.yaml (class or regular module)
    source_file = inspect.getfile(node_cls)
    module_dir = Path(source_file).parent
    candidate = module_dir / "contract.yaml"
    if candidate.is_file():
        return candidate

    # Strategy 4: parent directory (for cases where node.py is in a subdir)
    parent_candidate = module_dir.parent / "contract.yaml"
    if parent_candidate.is_file():
        return parent_candidate

    name = getattr(node_cls, "__name__", repr(node_cls))
    raise FileNotFoundError(
        f"No contract.yaml found for {name} "
        f"(searched {module_dir} and {module_dir.parent})"
    )


def _parse_contract(
    *,
    contract_path: Path,
    entry_point_name: str,
    package_name: str,
    package_version: str,
) -> ModelDiscoveredContract:
    """Parse a contract.yaml file into a ModelDiscoveredContract.

    Only reads the fields needed for auto-wiring. Unknown fields are ignored.
    """
    with open(contract_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw).__name__}")

    # Extract contract version
    cv_raw = raw.get("contract_version", {})
    if isinstance(cv_raw, dict):
        contract_version = ModelContractVersion(
            major=cv_raw.get("major", 0),
            minor=cv_raw.get("minor", 0),
            patch=cv_raw.get("patch", 0),
        )
    else:
        contract_version = ModelContractVersion(major=0, minor=0, patch=0)

    # Extract event bus wiring
    event_bus: ModelEventBusWiring | None = None
    eb_raw = raw.get("event_bus")
    if isinstance(eb_raw, dict):
        event_bus = ModelEventBusWiring(
            subscribe_topics=tuple(eb_raw.get("subscribe_topics", [])),
            publish_topics=tuple(eb_raw.get("publish_topics", [])),
        )

    # Extract handler routing — new format (handler_routing:) or legacy (handler:)
    handler_routing: ModelHandlerRouting | None = None
    hr_raw = raw.get("handler_routing")
    if isinstance(hr_raw, dict):
        handler_routing = _parse_handler_routing(hr_raw)
    else:
        h_raw = raw.get("handler")
        if isinstance(h_raw, dict):
            handler_routing = _parse_legacy_handler(h_raw)

    runtime_profiles: dict[str, ModelRuntimeProfilePolicy] = {}
    runtime_profiles_raw = raw.get("runtime_profiles")
    if isinstance(runtime_profiles_raw, dict):
        for profile_name, policy_raw in runtime_profiles_raw.items():
            if not isinstance(profile_name, str) or not isinstance(policy_raw, dict):
                continue
            runtime_profiles[profile_name] = ModelRuntimeProfilePolicy(
                eligible=policy_raw.get("eligible"),
                optional=bool(policy_raw.get("optional", False)),
            )

    return ModelDiscoveredContract(
        name=raw.get("name", entry_point_name),
        node_type=raw.get("node_type", "UNKNOWN"),
        description=raw.get("description", ""),
        contract_version=contract_version,
        node_version=str(raw.get("node_version", "1.0.0")),
        contract_path=contract_path,
        entry_point_name=entry_point_name,
        package_name=package_name,
        package_version=package_version,
        event_bus=event_bus,
        handler_routing=handler_routing,
        runtime_profiles=runtime_profiles,
    )


def _parse_handler_routing(hr_raw: dict) -> ModelHandlerRouting:
    """Parse the handler_routing section from a contract YAML dict."""
    entries: list[ModelHandlerRoutingEntry] = []
    for h in hr_raw.get("handlers", []):
        if not isinstance(h, dict):
            continue
        handler_ref_raw = h.get("handler")
        if not isinstance(handler_ref_raw, dict):
            continue
        handler_ref = ModelHandlerRef(
            name=handler_ref_raw.get("name", ""),
            module=handler_ref_raw.get("module", ""),
        )
        event_model: ModelHandlerRef | None = None
        em_raw = h.get("event_model")
        if isinstance(em_raw, dict):
            event_model = ModelHandlerRef(
                name=em_raw.get("name", ""),
                module=em_raw.get("module", ""),
            )
        entries.append(
            ModelHandlerRoutingEntry(
                handler=handler_ref,
                event_model=event_model,
                operation=h.get("operation"),
                event_type=h.get("event_type"),
            )
        )
    return ModelHandlerRouting(
        routing_strategy=hr_raw.get("routing_strategy", "unknown"),
        handlers=tuple(entries),
    )


def _parse_legacy_handler(h_raw: dict) -> ModelHandlerRouting | None:
    """Synthesize a ModelHandlerRouting from the legacy handler: key.

    Legacy format:
        handler:
          module: some.module.path
          class: HandlerClassName
          input_model: some.module.path.ModelClassName   # optional

    Maps to payload_type_match routing with a single handler entry.
    Returns None if the required module/class fields are missing.
    """
    module = h_raw.get("module", "")
    class_name = h_raw.get("class", "")
    if not module or not class_name:
        return None

    handler_ref = ModelHandlerRef(name=class_name, module=module)

    event_model: ModelHandlerRef | None = None
    input_model_str = h_raw.get("input_model")
    if isinstance(input_model_str, str) and "." in input_model_str:
        last_dot = input_model_str.rfind(".")
        event_model = ModelHandlerRef(
            name=input_model_str[last_dot + 1 :],
            module=input_model_str[:last_dot],
        )

    entry = ModelHandlerRoutingEntry(
        handler=handler_ref,
        event_model=event_model,
        operation=None,
    )
    return ModelHandlerRouting(
        routing_strategy="payload_type_match",
        handlers=(entry,),
    )
