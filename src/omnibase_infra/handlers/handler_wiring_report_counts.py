# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the wiring-report counts statically from the boot catalog (OMN-26335).

The catalog is the manifest the kernel reads at boot. The code graph is read
without importing: a handler module is located with ``importlib.util.find_spec``
and its source parsed, so a handler that does not import cleanly at runtime is
still counted from its declaration.
"""

from __future__ import annotations

import ast
import importlib.util
from collections import defaultdict
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.wiring_ratchet import ModelWiringCounts
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)

_DUPLICATE_NAME_REASON = "duplicate_contract_name"


def _module_source(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _class_bases(source: Path) -> dict[str, tuple[str, ...]]:
    """Top-level class name to its base-name tails; imported names map to ()."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = tuple(
                base.attr if isinstance(base, ast.Attribute) else ast.unparse(base)
                for base in node.bases
            )
        elif isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                found.setdefault((alias.asname or alias.name).split(".")[0], ())
    return found


class HandlerWiringReportCounts:
    """Count failed, duplicate, quarantined wiring and unproduced subscriptions."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelAutoWiringManifest) -> ModelWiringCounts:
        """Count from the manifest and the handler sources it names."""
        failed = sum(1 for e in request.errors if e.reason != _DUPLICATE_NAME_REASON)
        duplicate_names = sum(
            1 for e in request.errors if e.reason == _DUPLICATE_NAME_REASON
        )
        bases_by_module: dict[str, dict[str, tuple[str, ...]] | None] = {}
        quarantined = 0
        for contract in request.contracts:
            routing = contract.handler_routing
            for entry in routing.handlers if routing is not None else ():
                module, name = entry.handler.module, entry.handler.name
                if module not in bases_by_module:
                    source = _module_source(module)
                    bases_by_module[module] = (
                        _class_bases(source) if source is not None else None
                    )
                bases = bases_by_module[module]
                if bases is None or name not in bases:
                    failed += 1
                elif "Protocol" in bases[name]:
                    quarantined += 1

        subscribed: dict[str, int] = defaultdict(int)
        published: set[str] = set()
        for contract in request.contracts:
            if contract.event_bus is None:
                continue
            published.update(contract.event_bus.publish_topics)
            for topic in contract.event_bus.subscribe_topics:
                subscribed[topic] += 1
        return ModelWiringCounts(
            failed=failed,
            duplicates=sum(1 for owners in subscribed.values() if owners > 1)
            + duplicate_names,
            quarantined=quarantined,
            unproduced_subscriptions=len(set(subscribed) - published),
        )


__all__ = ["HandlerWiringReportCounts"]
