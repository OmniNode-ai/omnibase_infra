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
from collections.abc import Mapping
from importlib.metadata import entry_points
from pathlib import Path

import yaml

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelDiscoveryError,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.util_contract_content_hash import (
    contract_content_hash,
)
from omnibase_infra.utils.util_runtime_packages import (
    get_active_runtime_packages,
    is_autowiring_discovery_cache_enabled,
    is_gateway_cloud_mirroring_enabled,
    is_runtime_package_active,
    is_runtime_topic_active,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "onex.nodes"


def _contract_targets_active_runtime_packages(
    contract: ModelDiscoveredContract,
    active_packages: frozenset[str] | None,
) -> bool:
    """Return True when a contract's publish topics stay within the active runtime surface.

    Only publish_topics are checked because a contract that publishes into a
    package domain IS part of that package's producer surface and should only
    be loaded when the package is active.  subscribe_topics are read-only
    consumption — a projection or reducer can subscribe to topics from an
    inactive producer package without that package needing to be active.
    """
    if contract.event_bus is None:
        return True

    compatibility_publish_topics = set(contract.compatibility_publish_topics)
    return all(
        is_runtime_topic_active(topic, active_packages)
        or topic in compatibility_publish_topics
        for topic in contract.event_bus.publish_topics
    )


def _contract_requires_cloud_gateway(raw: dict) -> bool:
    """Return True when a contract declares a cloud gateway forwarding leg.

    Detected structurally via ``config.gateway_forwarder.cloud_leg`` rather than
    by node name, so any node that mirrors between the local bus and a hosted
    cloud Kafka edge is treated uniformly. Such a node must only be wired on
    lanes where the cloud leg is provisioned (OMN-13809).
    """
    config = raw.get("config")
    if not isinstance(config, dict):
        return False
    forwarder = config.get("gateway_forwarder")
    if not isinstance(forwarder, dict):
        return False
    return isinstance(forwarder.get("cloud_leg"), dict)


def _skip_dormant_cloud_gateway(contract: ModelDiscoveredContract) -> bool:
    """Return True when a cloud-gateway contract must be skipped on this lane.

    A contract that declares a cloud gateway leg is dormant unless cloud
    mirroring is explicitly enabled. Skipping it prevents the runtime from
    subscribing its ``ModelGatewayEnvelope`` handlers to bare domain topics
    whose real payloads are domain models — the ``ValidationError``-on-every-
    delegation failure fixed in OMN-13809.
    """
    if not contract.requires_cloud_gateway:
        return False
    if is_gateway_cloud_mirroring_enabled():
        return False
    logger.info(
        "Skipping contract '%s' because it declares a cloud gateway leg but "
        "cloud mirroring is not enabled on this lane (set %s to enable)",
        contract.name,
        "ONEX_GATEWAY_CLOUD_MIRRORING_ENABLED",
    )
    return True


class _DiscoveryMemo:
    """One-slot holder for the last completed scan.

    A holder rather than three module globals so the memo can be replaced by
    mutation instead of rebinding, which keeps the hot path free of ``global``.
    Module state is deliberate: the sweep this exists to make cheap calls
    ``discover_contracts()`` from a fresh task every ~316s within one process.
    """

    __slots__ = ("inputs", "manifest", "stats")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.inputs: tuple[object, ...] | None = None
        self.stats: Mapping[str, tuple[int, int]] | None = None
        self.manifest: ModelAutoWiringManifest | None = None

    def store(
        self,
        inputs: tuple[object, ...],
        stats: Mapping[str, tuple[int, int]],
        manifest: ModelAutoWiringManifest,
    ) -> None:
        self.inputs = inputs
        self.stats = stats
        self.manifest = manifest


_DISCOVERY_MEMO = _DiscoveryMemo()


def discover_contracts_cache_clear() -> None:
    """Drop the memo. For tests, and for any caller that installs a package."""
    _DISCOVERY_MEMO.clear()


def _discovery_inputs(
    active_packages: frozenset[str] | None,
) -> tuple[object, ...]:
    """Everything outside the contract files that can change the manifest.

    The entry-point set (name, distribution, version) plus the two environment
    switches the scan's skip predicates read:
    ``_contract_targets_active_runtime_packages`` via ``active_packages`` and
    ``_skip_dormant_cloud_gateway`` via cloud mirroring. A contract file's own
    bytes are covered separately by the stat map, so they are not repeated here.
    """
    eps = tuple(
        sorted(
            (
                ep.name,
                ep.dist.name if ep.dist is not None else "unknown",
                ep.dist.version if ep.dist is not None else "0.0.0",
            )
            for ep in entry_points(group=ENTRY_POINT_GROUP)
        )
    )
    packages = tuple(sorted(active_packages)) if active_packages is not None else None
    return (eps, packages, is_gateway_cloud_mirroring_enabled())


def _stat_map(paths: tuple[Path, ...]) -> dict[str, tuple[int, int]] | None:
    """Mtime+size per contract file, or None if any of them cannot be stat'd.

    Returning None on a missing file deliberately forces the full scan rather
    than treating the absence as "unchanged" — the scan is what turns a vanished
    contract into a recorded :class:`ModelDiscoveryError`.
    """
    out: dict[str, tuple[int, int]] = {}
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            return None
        out[str(path)] = (st.st_mtime_ns, st.st_size)
    return out


def _resolve_active_contract_paths(
    active_packages: frozenset[str] | None,
) -> tuple[tuple[Path, ...], bool]:
    """Resolve each active entry point's contract path without parsing it.

    This is the cheap half of the scan — measured in the deployed pod on
    2026-09-26 at 0.052s for 544 entry points against 24.099s for the parsing
    half — and it is what lets the memo be validated rather than trusted.

    Returns ``(paths, complete)``. ``complete`` is False when any entry point
    failed to load or resolve, which forces the full scan so the error is
    recorded instead of being served from the memo.
    """
    paths: list[Path] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "unknown"
        if not is_runtime_package_active(dist_name, active_packages):
            continue
        try:
            node_cls = ep.load()
            paths.append(_resolve_contract_path(node_cls))
        except Exception:  # noqa: BLE001 - any failure means "rescan properly"
            return tuple(paths), False
    return tuple(paths), True


def discover_contracts() -> ModelAutoWiringManifest:
    """Scan all ``onex.nodes`` entry points and build an auto-wiring manifest.

    **Memoized since OMN-19373.** A repeat call whose entry points, runtime-lane
    environment and contract-file mtimes/sizes are all unchanged returns the
    previous manifest instead of re-parsing. Measured in the deployed
    ``omninode-runtime-effects`` pod on onex-dev, 2026-09-26: this function took
    24.036s and an immediate second call took 23.922s, of which
    ``_parse_contract`` over 544 entry points was 24.099s — the sweep was
    re-reading files that cannot change inside a container image.

    That mattered beyond CPU. ``omninode-runtime-effects`` re-runs this sweep
    every ~316s, and a gateway heartbeat issued during one was answered only as
    the sweep ended — beat 6 of the 2026-09-26T15:46Z readback returned 200
    after 15.333s, a 25.103s server-recorded gap against OMN-15957's 20s
    acceptance tolerance. Making the repeat scan cheap is what removes that
    stall; widening timeouts only moved where it showed up.

    The memo is validated, not trusted: every call re-reads the entry-point set
    and stats every contract file, and any difference — including a file that
    has vanished — triggers the full scan. Set
    ``ONEX_AUTOWIRING_DISCOVERY_CACHE=0`` to disable it.

    For each entry point, the engine:

    1. Loads the entry point to obtain the node class (but does NOT instantiate it).
    2. Resolves the ``contract.yaml`` file adjacent to the module that defines the class.
    3. Parses the YAML and extracts the fields needed for wiring.

    Errors on individual entry points are captured — they never abort the full scan.

    Duplicate contract names (same ``name`` field from two different packages) are
    detected and surfaced as :class:`ModelDiscoveryError` entries.  The first
    occurrence wins; subsequent duplicates are dropped to prevent
    ``ONEX_CORE_064_DUPLICATE_REGISTRATION`` crashes at dispatcher registration
    time (OMN-11958).

    Returns:
        A :class:`ModelAutoWiringManifest` with all discovered contracts and errors.
    """
    active_packages = get_active_runtime_packages()

    if not is_autowiring_discovery_cache_enabled():
        manifest, _ = _scan_contracts(active_packages)
        return manifest

    inputs = _discovery_inputs(active_packages)
    cached = _DISCOVERY_MEMO.manifest
    if cached is not None and _DISCOVERY_MEMO.inputs == inputs:
        paths, complete = _resolve_active_contract_paths(active_packages)
        if complete:
            current = _stat_map(paths)
            if current is not None and current == _DISCOVERY_MEMO.stats:
                logger.debug(
                    "Auto-wiring discovery served from memo: "
                    "%d contracts, %d contract files unchanged",
                    cached.total_discovered,
                    len(current),
                )
                return cached

    manifest, scanned_paths = _scan_contracts(active_packages)
    stats = _stat_map(scanned_paths)
    if stats is not None:
        _DISCOVERY_MEMO.store(inputs, stats, manifest)
    return manifest


def _scan_contracts(
    active_packages: frozenset[str] | None,
) -> tuple[ModelAutoWiringManifest, tuple[Path, ...]]:
    """Run the full scan, parsing every active entry point's contract.

    Returns the manifest and every contract path that was resolved. The paths
    come back so the caller can fingerprint them, and they include files whose
    contract was then skipped (inactive package domain, dormant cloud gateway,
    duplicate name) because editing any of those changes the outcome.
    """
    contracts: list[ModelDiscoveredContract] = []
    errors: list[ModelDiscoveryError] = []
    resolved_paths: list[Path] = []
    # Tracks first-seen package for each contract name — used to detect
    # cross-package duplicates before they reach the dispatch engine.
    seen_contract_names: dict[str, str] = {}

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "unknown"
        dist_version = dist.version if dist is not None else "0.0.0"
        if not is_runtime_package_active(dist_name, active_packages):
            logger.debug(
                "Skipping contract discovery for inactive runtime package '%s'",
                dist_name,
            )
            continue

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

        resolved_paths.append(contract_path)

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

        if not _contract_targets_active_runtime_packages(contract, active_packages):
            logger.info(
                "Skipping contract '%s' from '%s' because it targets an inactive runtime package domain",
                contract.name,
                dist_name,
            )
            continue

        if _skip_dormant_cloud_gateway(contract):
            continue

        # Duplicate contract name guard (OMN-11958): two packages shipping a
        # contract with the same ``name`` field would produce identical dispatcher
        # IDs and crash with ONEX_CORE_064_DUPLICATE_REGISTRATION.  Surface the
        # collision as a discovery error and skip the duplicate so the runtime
        # boots cleanly.  The first occurrence (by entry_point iteration order)
        # wins; the owning package should remove the stale copy.
        if contract.name in seen_contract_names:
            first_owner = seen_contract_names[contract.name]
            error_msg = (
                f"Duplicate contract name '{contract.name}' already registered "
                f"by package '{first_owner}'. Skipping registration from "
                f"'{dist_name}' to prevent DUPLICATE_REGISTRATION crash. "
                f"Remove the stale entry point from one of these packages."
            )
            logger.error(
                "DUPLICATE_REGISTRATION prevented: contract='%s' "
                "first_owner='%s' duplicate_package='%s' "
                "entry_point='%s'",
                contract.name,
                first_owner,
                dist_name,
                ep.name,
            )
            errors.append(
                ModelDiscoveryError(
                    entry_point_name=ep.name,
                    package_name=dist_name,
                    error=error_msg,
                )
            )
            continue

        seen_contract_names[contract.name] = dist_name
        contracts.append(contract)
        logger.info(
            "Discovered contract: %s (%s) from %s %s",
            contract.name,
            contract.node_type,
            dist_name,
            dist_version,
        )

    return (
        ModelAutoWiringManifest(
            contracts=tuple(contracts),
            errors=tuple(errors),
        ),
        tuple(resolved_paths),
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
    active_packages = get_active_runtime_packages()

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
        if not _contract_targets_active_runtime_packages(contract, active_packages):
            logger.info(
                "Skipping contract '%s' from explicit path %s because it targets an inactive runtime package domain",
                contract.name,
                path,
            )
            continue
        if _skip_dormant_cloud_gateway(contract):
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


def _parse_bool_field(raw_dict: dict, field_name: str, default: bool = False) -> bool:
    value = raw_dict.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(
            f"event_bus.{field_name} must be a boolean when provided, "
            f"got {type(value).__name__}"
        )
    return value


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
            dlq_topics=tuple(eb_raw.get("dlq_topics", [])),
            consumer_group=eb_raw.get("consumer_group"),
            consumer_purpose=eb_raw.get("consumer_purpose"),
            plugin_managed=_parse_bool_field(eb_raw, "plugin_managed", False),
            tenant_scoped_ingress=_parse_bool_field(
                eb_raw, "tenant_scoped_ingress", False
            ),
            terminal_event=eb_raw.get("terminal_event"),
        )

    # Extract handler routing — new format (handler_routing:) or legacy (handler:)
    handler_routing: ModelHandlerRouting | None = None
    hr_raw = raw.get("handler_routing")
    h_raw = raw.get("handler")
    if isinstance(hr_raw, dict):
        handler_routing = _parse_handler_routing(
            hr_raw,
            contract_path=contract_path,
            parse_default_handler=not isinstance(h_raw, dict),
        )
        if not handler_routing.handlers and isinstance(h_raw, dict):
            handler_routing = _parse_legacy_handler(h_raw)
    elif isinstance(h_raw, dict):
        handler_routing = _parse_legacy_handler(h_raw)

    runtime_profiles = _extract_runtime_profiles(raw)
    runtime_lanes = _extract_runtime_lanes(raw)
    db_io_raw = raw.get("db_io")
    db_io = (
        ModelDbOwnershipSubcontract.model_validate(db_io_raw)
        if db_io_raw is not None
        else None
    )

    return ModelDiscoveredContract(
        name=raw.get("name", entry_point_name),
        node_type=raw.get("node_type", "UNKNOWN"),
        description=raw.get("description", ""),
        contract_version=contract_version,
        node_version=_render_node_version(raw.get("node_version")),
        contract_path=contract_path,
        # OMN-18708: hash the file this contract was parsed from, on the
        # discovery pass that parsed it, so the introspection manifest served
        # by a running lane carries the same triple the image label was
        # stamped with. Computed here rather than lazily by a reader: the file
        # is open on this code path and may not exist on any other.
        contract_content_hash=contract_content_hash(contract_path),
        entry_point_name=entry_point_name,
        package_name=package_name,
        package_version=package_version,
        runtime_profiles=runtime_profiles,
        runtime_lanes=runtime_lanes,
        compatibility_publish_topics=raw.get("compatibility_publish_topics"),
        terminal_event=(
            raw.get("terminal_event")
            if raw.get("terminal_event") is not None
            else event_bus.terminal_event
            if event_bus is not None
            else None
        ),
        requires_cloud_gateway=_contract_requires_cloud_gateway(raw),
        event_bus=event_bus,
        handler_routing=handler_routing,
        db_io=db_io,
    )


def _render_node_version(raw_value: object) -> str:
    """Render a contract's declared ``node_version`` as a version STRING.

    OMN-18708. This field was ``str(raw.get("node_version", "1.0.0"))``, which
    is correct for the 77 contracts that declare a string and produces a Python
    dict repr for the 67 that declare the ``{major, minor, patch}`` mapping --
    ``"{'major': 1, 'minor': 0, 'patch': 0}"``. That value was already being
    served on ``/v1/introspection/manifest``; it becomes load-bearing here,
    because it is one third of the triple an image label is stamped with and a
    promotion gate compares.

    A mapping renders as ``major.minor.patch``, using 0 for an absent or
    non-integer component rather than guessing at a different shape. Anything
    else renders with ``str`` exactly as before, so no contract that already
    declared a string moves.
    """
    if isinstance(raw_value, dict):

        def _component(key: str) -> int:
            value = raw_value.get(key, 0)
            return (
                value if isinstance(value, int) and not isinstance(value, bool) else 0
            )

        return f"{_component('major')}.{_component('minor')}.{_component('patch')}"
    if raw_value is None:
        return "1.0.0"
    return str(raw_value)


def _extract_runtime_profiles(raw: dict) -> tuple[str, ...]:
    """Extract contract-declared runtime profile ownership.

    Preferred location is top-level ``runtime_profiles``. ``descriptor`` is also
    supported so node contracts can keep operational ownership metadata next to
    existing archetype/purity declarations.
    """
    profiles_raw = raw.get("runtime_profiles")
    descriptor_raw = raw.get("descriptor")
    if profiles_raw is None and isinstance(descriptor_raw, dict):
        profiles_raw = descriptor_raw.get("runtime_profiles")

    if profiles_raw is None:
        return ()
    if isinstance(profiles_raw, str):
        raw_values = (profiles_raw,)
    elif isinstance(profiles_raw, list | tuple):
        raw_values = tuple(profiles_raw)
    else:
        raise ValueError("runtime_profiles must be a string or list of strings")

    profiles: list[str] = []
    for raw_profile in raw_values:
        if not isinstance(raw_profile, str):
            raise ValueError("runtime_profiles entries must be strings")
        profile = raw_profile.strip().lower()
        if not profile:
            raise ValueError("runtime_profiles entries cannot be blank")
        profiles.append(profile)
    return tuple(dict.fromkeys(profiles))


def _extract_runtime_lanes(raw: Mapping[str, object]) -> ModelRuntimeLaneScope | None:
    """Extract the contract-declared runtime lane scope (OMN-19408).

    Read from the top-level ``runtime_lanes`` key only. Absent means unscoped.
    A present value is validated by the core model: an unregistered lane, a
    blank entry or an empty list raises, which the caller records as a
    discovery error for this contract -- a malformed scope must never degrade
    to "unscoped", because that attaches the node on exactly the lanes the
    author tried to keep it off.
    """
    lanes_raw = raw.get("runtime_lanes")
    if lanes_raw is None:
        return None
    return ModelRuntimeLaneScope.model_validate({"lanes": lanes_raw})


def _parse_handler_routing(
    hr_raw: dict,
    *,
    contract_path: Path | None = None,
    parse_default_handler: bool = True,
) -> ModelHandlerRouting:
    """Parse the handler_routing section from a contract YAML dict.

    Fail-closed (OMN-14141): a ``handlers[]`` entry that cannot be parsed into a
    dispatcher raises ``ValueError`` instead of being silently skipped. The
    historical FLAT ``handler_class:`` / ``handler_module:`` string schema — which
    ``ModelHandlerRoutingEntry`` does not understand — previously fell through the
    ``continue`` here and produced ``handlers=()`` with NO error. Auto-wiring
    then reported ``EnumWiringOutcome.WIRED`` with zero dispatchers while still
    subscribing to and committing Kafka offsets on the topic: silent
    phantom-wiring (the WI-14 root cause, OMN-14139/OMN-14135). Every routed
    handler MUST carry a nested ``handler: {name, module}`` mapping.

    The legacy top-level ``handler: {module, class}`` fallback in
    ``_parse_contract`` is unaffected: those contracts declare an EMPTY / ABSENT
    ``handlers`` list, so this loop never runs and never raises — the zero-length
    parse still triggers the fallback.
    """
    entries: list[ModelHandlerRoutingEntry] = []
    handlers_raw = hr_raw.get("handlers")
    if parse_default_handler and not isinstance(handlers_raw, list):
        default_handler = hr_raw.get("default_handler")
        if (
            isinstance(default_handler, str)
            and default_handler
            and ":" in default_handler
        ):
            module_ref, class_name = default_handler.rsplit(":", 1)
            entries.append(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=class_name.strip(),
                        module=_resolve_default_handler_module(
                            module_ref.strip(), contract_path
                        ),
                    ),
                )
            )
    handlers_iter = handlers_raw if isinstance(handlers_raw, list) else []
    for index, h in enumerate(handlers_iter):
        if not isinstance(h, dict):
            raise ValueError(
                f"handler_routing.handlers[{index}] must be a mapping, got "
                f"{type(h).__name__} — cannot be parsed into a dispatcher "
                "(OMN-14141)."
            )
        handler_ref_raw = h.get("handler")
        if not isinstance(handler_ref_raw, dict):
            raise ValueError(
                f"handler_routing.handlers[{index}] is missing a nested "
                "'handler: {name, module}' mapping (found keys: "
                f"{sorted(h.keys())}). The flat 'handler_class'/'handler_module' "
                "schema is not parseable and silently produces zero dispatchers, "
                "which then phantom-wires the subscribed topic with no live "
                "handler (OMN-14141). Use the nested shape:\n"
                "    handler:\n"
                "      name: <HandlerClassName>\n"
                "      module: <module.path>"
            )
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
        elif isinstance(em_raw, str):
            module_name, separator, model_name = em_raw.rpartition(".")
            if not separator or not module_name or not model_name:
                raise ValueError(
                    f"handler_routing.handlers[{index}].event_model must be a "
                    "fully qualified 'module.Model' string or a {name, module} "
                    "mapping"
                )
            event_model = ModelHandlerRef(name=model_name, module=module_name)
        entries.append(
            ModelHandlerRoutingEntry(
                handler=handler_ref,
                event_model=event_model,
                operation=h.get("operation"),
                event_type=h.get("event_type"),
                message_category=h.get("message_category"),
                topic=h.get("topic"),
            )
        )
    return ModelHandlerRouting(
        routing_strategy=hr_raw.get("routing_strategy", "unknown"),
        handlers=tuple(entries),
    )


def _resolve_default_handler_module(module_ref: str, contract_path: Path | None) -> str:
    """Resolve ``default_handler`` module shorthand for package-discovered contracts."""
    if not module_ref or "." in module_ref or contract_path is None:
        return module_ref

    contract_dir = contract_path.parent
    if not (contract_dir / f"{module_ref}.py").exists():
        return module_ref

    package_parts: list[str] = []
    current = contract_dir
    while (current / "__init__.py").exists():
        package_parts.append(current.name)
        current = current.parent
    package_parts.reverse()
    if not package_parts:
        return module_ref
    return ".".join((*package_parts, module_ref))


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
