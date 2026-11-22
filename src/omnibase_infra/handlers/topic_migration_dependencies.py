# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Constructor collaborators of ``HandlerTopicMigrationExecutor`` (OMN-26338).

Auto-wiring builds the handler's required ``provisioner`` and ``drain_proof_gate``
services before it constructs the handler (the builder stays in
``runtime/auto_wiring/handler_wiring.py``, where its raw Kafka clients are not a
freestanding module), so the handler keeps a strict constructor contract and
generic resolver Step 2 stays deterministic.

The provisioner scans every installed contract when it is constructed. That scan
held the event loop for about 12 s at one CPU, so :func:`warm_topic_migration_scan`
reads the same request on a worker thread first; the synchronous construction that
follows finds the stored scan and builds the same topic set.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.models.event_bus.model_contract_topic_scan import (
    ModelContractTopicScan,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.models import (
        ModelDiscoveredContract,
        ModelHandlerRoutingEntry,
    )

TOPIC_MIGRATION_EXECUTOR_DEPS = frozenset({"provisioner", "drain_proof_gate"})


def provisioner_dependencies(
    provisioner: object | None,
) -> dict[str, dict[str, object]]:
    """Explicit constructor dependencies that hand ``provisioner`` to the executor."""
    if provisioner is None:
        return {}
    return {"HandlerTopicMigrationExecutor": {"provisioner": provisioner}}


def required_handler_init_params(handler_cls: type) -> frozenset[str]:
    """Return required constructor parameter names for a handler class."""
    sig = inspect.signature(handler_cls)
    return frozenset(
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and param.default is inspect.Parameter.empty
    )


def _requires_migration_collaborators(entry: ModelHandlerRoutingEntry) -> bool:
    """True when the entry's handler requires exactly the migration collaborators."""
    with contextlib.suppress(Exception):  # detection only; preparation reports it
        module = importlib.import_module(entry.handler.module)
        params = required_handler_init_params(getattr(module, entry.handler.name))
        return bool(params) and params <= TOPIC_MIGRATION_EXECUTOR_DEPS
    return False


async def warm_topic_migration_scan(
    contract: ModelDiscoveredContract,
    root: Path,
    explicit_dependencies: dict[str, dict[str, object]] | None = None,
) -> None:
    """Read the provisioner's contract-topic scan of ``root`` off the loop.

    Does nothing unless a handler of the contract requires exactly the migration
    executor's collaborators. The warm read only decides where the scan runs, never
    whether wiring fails: a handler that cannot be imported, a missing contracts
    root and a failed scan are all left for the synchronous construction that
    follows, which raises the same error inside the same wiring boundary as before.
    """
    if contract.handler_routing is None or not root.is_dir():
        return
    held = explicit_dependencies or {}
    entries = [
        entry
        for entry in contract.handler_routing.handlers
        if "provisioner" not in held.get(entry.handler.name, {})
    ]
    if any(map(_requires_migration_collaborators, entries)):
        with contextlib.suppress(Exception):  # the synchronous construction re-raises
            await ModelContractTopicScan.read_off_loop(contracts_root=root)
