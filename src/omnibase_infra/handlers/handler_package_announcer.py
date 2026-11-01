# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot step 0: announce the installed packages' contracts and build the node catalog (OMN-21590).

The metadata reader in this module is the one place that names the ``onex.nodes``
entry-point group. It reads each installed package's own metadata, resolves the
contract file, hashes its bytes in the canonical form and returns records; nothing
here imports a handler or constructs a node. The announcer turns the records into
``ModelNodeAnnouncement`` values and publishes them on a bus the lane overlay binds
(core's in-memory bus before the container exists, a durable bus on a lane that has
one); a catalog service subscribes to the same topic and folds each announcement into
core's pure ``ModelNodeCatalog``. The runtime then reads that catalog instead of
scanning entry points and pushing contracts to the registry.

The fold is one function whichever bus carried the announcement, so the catalog is the
same on the in-memory bus, on a recording bus and on a real broker (plan 3.11.3).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import UUID

import yaml
from pydantic import ValidationError

from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_announcement_operation import EnumAnnouncementOperation
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events import ModelContractRegisteredEvent
from omnibase_core.models.events.contract_registration.model_announced_package import (
    ModelAnnouncedPackage,
)
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.nodes.package_announce.model_catalog_outcome import (
    ModelCatalogOutcome,
)
from omnibase_core.models.nodes.package_announce.model_node_announce_policy import (
    ModelNodeAnnouncePolicy,
)
from omnibase_core.models.nodes.package_announce.model_node_catalog import (
    ModelNodeCatalog,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.protocols.services.protocol_announce_bus import ProtocolAnnounceBus
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_contract_content_hash import (
    ContractContentHashError,
)
from omnibase_infra.models.node_announce.model_node_announcement_topic import (
    ModelNodeAnnouncementTopic,
)
from omnibase_infra.models.node_announce.model_package_announce_request import (
    ModelPackageAnnounceRequest,
)
from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.models.node_announce.model_package_contract_record import (
    ModelPackageContractRecord,
)
from omnibase_infra.models.node_announce.model_package_inventory import (
    ModelPackageInventory,
)
from omnibase_infra.runtime.util_contract_content_hash import contract_content_hash
from omnibase_infra.utils.util_runtime_packages import is_runtime_package_active

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP: Final = "onex.nodes"

_SETTLE_TIMEOUT_SECONDS: Final = 60.0

# What importing an arbitrary installed node module can raise; the reader records it and
# goes on, so one broken package never stops boot.
_LOAD_FAILURES: Final = (
    ImportError,
    AttributeError,
    SyntaxError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    LookupError,
)

# What a malformed announcement or an unreadable installed file can raise in the fold.
_RECORD_FAILURES: Final = (
    ValidationError,
    ModelOnexError,
    ValueError,
    OSError,
    yaml.YAMLError,
)

_INSTALLED_FILE = Callable[[ModelNodeAnnouncement], tuple[bytes | None, Path | None]]


def resolve_contract_path(node_cls: type) -> Path:
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


def read_contract_yaml(contract_path: Path) -> object:
    """Read contract YAML safely for discovery and pre-subscription wiring.

    Cold wiring re-reads contract extensions after discovery. Keep those reads
    on the same fast safe parser so they do not restore the pure-Python parse
    delay before the first consumer bind (OMN-18843). I/O and parse errors
    propagate to the caller, which owns the field's refusal semantics.
    """
    with open(contract_path) as f:
        # OMN-18843: cold discovery gates the first consumer bind. Use the
        # safe LibYAML parser when available; the memo only helps later scans.
        # Both loaders reject Python object tags, with a portable fallback.
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)(f)
        try:
            raw = loader.get_single_data()
        finally:
            loader.dispose()

    return raw


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _declared_topics(document: dict[object, object]) -> tuple[str, ...]:
    event_bus = document.get("event_bus")
    if not isinstance(event_bus, dict):
        return ()
    topics: dict[str, None] = {}
    for key in ("subscribe_topics", "publish_topics", "dlq_topics"):
        for topic in _string_tuple(event_bus.get(key)):
            topics[topic] = None
    return tuple(topics)


def _read_contract(
    *, entry_point_name: str, distribution: str, version: str, contract_path: Path
) -> ModelPackageContractRecord:
    try:
        raw_bytes = contract_path.read_bytes()
        document = read_contract_yaml(contract_path)
        if not isinstance(document, dict):
            raise ValueError(f"Expected YAML dict, got {type(document).__name__}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return ModelPackageContractRecord(
            entry_point=entry_point_name,
            distribution=distribution,
            version=version,
            contract_path=contract_path,
            error_stage="parse",
            error=str(exc),
        )
    name = document.get("name", entry_point_name)
    return ModelPackageContractRecord(
        entry_point=entry_point_name,
        distribution=distribution,
        version=version,
        contract_path=contract_path,
        contract_name=name if isinstance(name, str) else str(name),
        content_hash=ModelNodeAnnouncement.canonical_content_hash(raw_bytes),
        declared_topics=_declared_topics(document),
        runtime_profiles=_string_tuple(document.get("runtime_profiles")),
    )


def read_package_inventory(
    active_packages: frozenset[str] | None = None,
) -> ModelPackageInventory:
    """Read the installed package set's own metadata, in the order it lists the entries.

    Entries of an inactive runtime package are left out of ``records`` and kept in the
    ``fingerprint``. A failure to load or resolve one entry is a record with the stage
    that failed; it never stops the read of the others.
    """
    records: list[ModelPackageContractRecord] = []
    fingerprint: list[tuple[str, str, str]] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "unknown"
        dist_version = str(dist.version) if dist is not None else "0.0.0"
        fingerprint.append((ep.name, dist_name, dist_version))
        if not is_runtime_package_active(dist_name, active_packages):
            logger.debug(
                "Skipping contract discovery for inactive runtime package '%s'",
                dist_name,
            )
            continue
        try:
            node_cls = ep.load()
        except _LOAD_FAILURES as exc:
            logger.warning(
                "Failed to load entry point '%s' from '%s': %s",
                ep.name,
                dist_name,
                exc,
            )
            records.append(
                ModelPackageContractRecord(
                    entry_point=ep.name,
                    distribution=dist_name,
                    version=dist_version,
                    error_stage="load",
                    error=str(exc),
                )
            )
            continue
        try:
            contract_path = resolve_contract_path(node_cls)
        except (FileNotFoundError, TypeError) as exc:
            logger.warning(
                "No contract.yaml for entry point '%s' from '%s': %s",
                ep.name,
                dist_name,
                exc,
            )
            records.append(
                ModelPackageContractRecord(
                    entry_point=ep.name,
                    distribution=dist_name,
                    version=dist_version,
                    error_stage="resolve",
                    error=str(exc),
                )
            )
            continue
        records.append(
            _read_contract(
                entry_point_name=ep.name,
                distribution=dist_name,
                version=dist_version,
                contract_path=contract_path,
            )
        )
    return ModelPackageInventory(
        records=tuple(records), fingerprint=tuple(sorted(fingerprint))
    )


def announcer_identity(namespace: str) -> UUID:
    """A stable announcer id for a named announcer, so a restart keeps its identity."""
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return UUID(bytes=digest[:16], version=5)


def build_announcements(
    inventory: ModelPackageInventory, request: ModelPackageAnnounceRequest
) -> tuple[ModelNodeAnnouncement, ...]:
    """One ``announce`` per readable contract, in the inventory's order."""
    return tuple(
        ModelNodeAnnouncement(
            entity_id=record.contract_name,
            operation=EnumAnnouncementOperation.ANNOUNCE,
            content_hash=record.content_hash,
            package=ModelAnnouncedPackage(
                distribution=record.distribution, version=record.version
            ),
            node_kind=EnumAnnouncedNodeKind.NODE,
            declared_topics=record.declared_topics,
            announcer_id=request.announcer_id,
            generation=request.generation,
            announced_at=request.announced_at,
        )
        for record in inventory.records
        if record.error_stage is None
        and record.contract_name is not None
        and record.content_hash is not None
    )


def tier0_policy(
    inventory: ModelPackageInventory, announcer_id: UUID
) -> ModelNodeAnnouncePolicy:
    """The policy of a lane that binds none: admit what is installed.

    Handler prefixes are open and the active profiles are the ones the installed
    contracts name, so the catalog refuses nothing discovery used to accept; the
    runtime profile filter downstream still decides which node a runtime owns.
    """
    profiles = sorted({p for r in inventory.records for p in r.runtime_profiles})
    return ModelNodeAnnouncePolicy(
        trusted_handler_prefixes=("",),
        active_profiles=tuple(profiles),
        authoritative_announcers={
            r.distribution: announcer_id for r in inventory.records
        },
    )


def installed_file_reader(inventory: ModelPackageInventory) -> _INSTALLED_FILE:
    """Read the installed contract bytes for an announcement, as admission needs them."""
    paths = {
        (r.contract_name, r.distribution): r.contract_path
        for r in reversed(inventory.records)
        if r.contract_path is not None and r.contract_name is not None
    }

    def read(announcement: ModelNodeAnnouncement) -> tuple[bytes | None, Path | None]:
        path = paths.get((announcement.entity_id, announcement.package.distribution))
        if path is None:
            return None, None
        try:
            return path.read_bytes(), path
        except OSError:
            return None, path

    return read


def apply_announcement(
    catalog: ModelNodeCatalog,
    announcement: ModelNodeAnnouncement,
    read_installed: _INSTALLED_FILE,
) -> tuple[ModelNodeCatalog, ModelCatalogOutcome]:
    """Fold one announcement into the catalog; the same function on every bus."""
    installed_bytes, installed_path = read_installed(announcement)
    next_catalog, outcome, _changes = catalog.apply(
        announcement, installed_bytes=installed_bytes, installed_path=installed_path
    )
    return next_catalog, outcome


def local_announce_request(
    active_packages: frozenset[str] | None, *, workload: str = "main"
) -> ModelPackageAnnounceRequest:
    """The request of a caller that binds no overlay: Tier 0, this process, generation 1."""
    return ModelPackageAnnounceRequest(
        topic=ModelNodeAnnouncementTopic.suffix_for(workload),
        announcer_id=announcer_identity(f"omnibase_infra.package_announcer.{workload}"),
        generation=1,
        announced_at=datetime.now(UTC),
        active_packages=active_packages,
    )


def build_catalog_direct(
    inventory: ModelPackageInventory, request: ModelPackageAnnounceRequest
) -> ModelPackageAnnounceResult:
    """Fold the inventory's announcements with no bus, for a caller with no event loop.

    Produces the catalog ``HandlerPackageAnnouncer.handle`` produces for the same
    request, because both use ``build_announcements`` and ``apply_announcement``.
    """
    announcements = build_announcements(inventory, request)
    policy = request.policy or tier0_policy(inventory, request.announcer_id)
    catalog = ModelNodeCatalog(policy=policy)
    read_installed = installed_file_reader(inventory)
    outcomes: list[ModelCatalogOutcome] = []
    for announcement in announcements:
        catalog, outcome = apply_announcement(catalog, announcement, read_installed)
        outcomes.append(outcome)
    return ModelPackageAnnounceResult(
        inventory=inventory,
        announcements=announcements,
        catalog=catalog,
        outcomes=tuple(outcomes),
    )


class HandlerPackageAnnouncer:
    """Announce every installed contract on the bound bus and build the catalog from them.

    The bus is a dependency the lane overlay binds: core's in-memory bus for Tier 0,
    a durable bus where the lane has one. The handler holds no bus class name.
    """

    def __init__(self, bus: ProtocolAnnounceBus[object]) -> None:
        self._bus = bus

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelPackageAnnounceRequest
    ) -> ModelPackageAnnounceResult:
        """Read the metadata, publish the announcements, return the catalog."""
        inventory = read_package_inventory(request.active_packages)
        announcements = build_announcements(inventory, request)
        policy = request.policy or tier0_policy(inventory, request.announcer_id)
        read_installed = installed_file_reader(inventory)

        catalog = ModelNodeCatalog(policy=policy)
        seen = 0
        outcomes: list[ModelCatalogOutcome] = []
        settled = asyncio.Event()
        expected = len(announcements)
        if expected == 0:
            settled.set()

        async def on_message(message: object) -> None:
            nonlocal catalog, seen
            value = getattr(message, "value", None)
            try:
                announcement = ModelNodeAnnouncement.model_validate_json(value or b"")
                catalog, outcome = apply_announcement(
                    catalog, announcement, read_installed
                )
                outcomes.append(outcome)
            except _RECORD_FAILURES:
                logger.warning("Unreadable announcement skipped", exc_info=True)
            finally:
                seen += 1
                if seen >= expected:
                    settled.set()

        await self._bus.start()
        unsubscribe = await self._bus.subscribe(
            request.topic, on_message=on_message, group_id=request.catalog_group
        )
        try:
            for announcement in announcements:
                await self._bus.publish(
                    request.topic,
                    announcement.entity_id.encode("utf-8"),
                    announcement.model_dump_json().encode("utf-8"),
                )
            await asyncio.wait_for(settled.wait(), timeout=_SETTLE_TIMEOUT_SECONDS)
        finally:
            await unsubscribe()
        return ModelPackageAnnounceResult(
            inventory=inventory,
            announcements=announcements,
            catalog=catalog,
            outcomes=tuple(outcomes),
        )


def build_registered_events(
    contracts: Iterable[ModelDiscoveredContract], correlation_id: UUID
) -> tuple[ModelContractRegisteredEvent, ...]:
    """One registration event per readable contract; an unreadable one is logged and skipped."""
    events: list[ModelContractRegisteredEvent] = []
    for contract in contracts:
        try:
            contract_yaml = contract.contract_path.read_text(encoding="utf-8")
            content_hash = contract.contract_content_hash or contract_content_hash(
                contract.contract_path, correlation_id=correlation_id
            )
        except (OSError, UnicodeDecodeError, ContractContentHashError) as exc:
            logger.warning(
                "Contract registration skipped for %s: %s (correlation_id=%s)",
                contract.name,
                type(exc).__name__,
                correlation_id,
            )
            continue
        version = contract.contract_version
        events.append(
            ModelContractRegisteredEvent(
                node_name=contract.name,
                node_version=ModelSemVer(
                    major=version.major, minor=version.minor, patch=version.patch
                ),
                contract_hash=content_hash,
                contract_yaml=contract_yaml,
                correlation_id=correlation_id,
            )
        )
    return tuple(events)


async def run_boot_step0(
    *,
    workload: str,
    active_packages: frozenset[str] | None = None,
    environment: str = "boot",
) -> ModelPackageAnnounceResult:
    """Boot step 0: announce and catalog on a fresh in-memory bus, before any container.

    The in-memory bus is the Tier 0 binding of the announcer's bus. A lane whose
    overlay binds a durable bus constructs ``HandlerPackageAnnouncer`` with that bus
    and the same request instead; the catalog it returns is the same.
    """
    from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.utils.util_runtime_packages import (
        get_active_runtime_packages,
    )

    bus = EventBusInmemory(environment=environment, group="node-announce")
    packages = (
        active_packages
        if active_packages is not None
        else get_active_runtime_packages()
    )
    return await HandlerPackageAnnouncer(bus).handle(
        local_announce_request(packages, workload=workload)
    )


__all__ = [
    "ENTRY_POINT_GROUP",
    "HandlerPackageAnnouncer",
    "announcer_identity",
    "apply_announcement",
    "build_announcements",
    "build_catalog_direct",
    "build_registered_events",
    "installed_file_reader",
    "local_announce_request",
    "read_contract_yaml",
    "read_package_inventory",
    "run_boot_step0",
    "resolve_contract_path",
    "tier0_policy",
]
