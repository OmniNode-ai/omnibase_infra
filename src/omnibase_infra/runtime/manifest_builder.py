# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build a ModelRuntimeManifest from auto-wiring results (OMN-11196).

Called once at the end of the bootstrap sequence, after all startup phases:
contract discovery, ownership validation, handler registration, and topic
ownership. Produces a deterministic, hash-stable snapshot of the runtime
topology for observability and drift detection.
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone

from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)


def build_runtime_manifest(
    report: ModelAutoWiringReport,
    manifest: ModelAutoWiringManifest,
    runtime_profile: str,
    image_digest: str | None = None,
) -> object:
    """Build a ModelRuntimeManifest from auto-wiring results.

    Extracts wired/skipped/failed contracts, topics, and handlers from the
    wiring report and discovered manifest, then returns a frozen
    ModelRuntimeManifest ready for publication on the event bus.

    The import of ModelRuntimeManifest is deferred so that this module can be
    imported before omnibase_core PR #1098 lands (the model is gated behind a
    try/import in service_kernel.py as well).

    Args:
        report: The wiring report produced by wire_from_manifest().
        manifest: The filtered auto-wiring manifest (post-quarantine).
        runtime_profile: The RUNTIME_PROFILE value (e.g. "main").
        image_digest: Optional OCI image digest for the running container.

    Returns:
        A ModelRuntimeManifest instance (typed as object to allow graceful
        fallback when the model is not yet available in omnibase_core).

    Raises:
        ImportError: If omnibase_core.models.runtime_manifest is not installed.
    """
    from omnibase_core.models.runtime_manifest.model_manifest_contract import (
        ModelManifestContract,
    )
    from omnibase_core.models.runtime_manifest.model_manifest_handler import (
        ModelManifestHandler,
    )
    from omnibase_core.models.runtime_manifest.model_runtime_manifest import (
        ModelRuntimeManifest,
    )

    results_by_outcome: dict[str, list[ModelContractWiringResult]] = {
        EnumWiringOutcome.WIRED: [],
        EnumWiringOutcome.SKIPPED: [],
        EnumWiringOutcome.FAILED: [],
    }
    for result in report.results:
        results_by_outcome[result.outcome].append(result)

    # Build a lookup from contract name → discovered contract for metadata
    contract_by_name = {c.name: c for c in manifest.contracts}

    def _to_manifest_contract(
        result: ModelContractWiringResult,
    ) -> ModelManifestContract:
        discovered = contract_by_name.get(result.contract_name)
        version = str(discovered.contract_version) if discovered else "unknown"
        node_type = discovered.node_type if discovered else "unknown"
        # Stable hash: SHA-256 of "{name}:{version}"
        import hashlib

        contract_hash = hashlib.sha256(
            f"{result.contract_name}:{version}".encode()
        ).hexdigest()
        return ModelManifestContract(
            name=result.contract_name,
            version=version,
            node_type=node_type,
            contract_hash=contract_hash,
        )

    wired_contracts = tuple(
        _to_manifest_contract(r) for r in results_by_outcome[EnumWiringOutcome.WIRED]
    )
    skipped_contracts = tuple(
        _to_manifest_contract(r) for r in results_by_outcome[EnumWiringOutcome.SKIPPED]
    )
    failed_contracts = tuple(
        _to_manifest_contract(r) for r in results_by_outcome[EnumWiringOutcome.FAILED]
    )

    # Collect all publish topics from wired results
    owned_command_topics: set[str] = set()
    for result in results_by_outcome[EnumWiringOutcome.WIRED]:
        discovered = contract_by_name.get(result.contract_name)
        if discovered and discovered.event_bus:
            owned_command_topics.update(discovered.event_bus.publish_topics)

    # Collect all subscribe topics from wired results
    subscribed_event_topics: set[str] = set()
    for result in results_by_outcome[EnumWiringOutcome.WIRED]:
        subscribed_event_topics.update(result.topics_subscribed)

    # Collect all wired handlers from wired results
    handlers: list[ModelManifestHandler] = []
    for result in results_by_outcome[EnumWiringOutcome.WIRED]:
        discovered = contract_by_name.get(result.contract_name)
        if not discovered or not discovered.handler_routing:
            continue
        routing_strategy = discovered.handler_routing.routing_strategy or "unknown"
        for wiring_outcome in result.wirings:
            handlers.append(
                ModelManifestHandler(
                    name=wiring_outcome.handler_name,
                    module_path=result.contract_name,
                    routing_strategy=routing_strategy,
                )
            )

    return ModelRuntimeManifest(
        runtime_profile=runtime_profile,
        contracts=wired_contracts,
        owned_command_topics=frozenset(owned_command_topics),
        subscribed_event_topics=frozenset(subscribed_event_topics),
        handlers=tuple(handlers),
        skipped_contracts=skipped_contracts,
        failed_contracts=failed_contracts,
        ownership_violations=(),
        image_digest=image_digest,
        started_at=datetime.now(tz=UTC),
    )
