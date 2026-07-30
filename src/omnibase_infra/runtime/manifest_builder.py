# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build and publish the runtime manifest snapshot (OMN-11196 / OMN-15512).

Called once at the end of the bootstrap sequence, after all startup phases:
contract discovery, ownership validation, handler registration, and topic
ownership. Produces a deterministic, hash-stable snapshot of the runtime
topology for observability and drift detection.

OMN-15512: the snapshot also carries the boot attach-readiness aggregate, so
the NOT-READY blocker set (contract name + the topics whose readiness confirm
failed) reaches the ``runtime_manifests`` projection instead of dying in the
log stream. :func:`publish_runtime_manifest` is the seam a test can drive with
a recording bus — the kernel calls exactly this function, so a test that
asserts on the captured envelope is asserting on the artifact that runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.model_runtime_attach_readiness import (
        ModelRuntimeAttachReadiness,
    )
    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.runtime.models.model_runtime_manifest_published import (
        ModelRuntimeManifestPublished,
    )


def build_runtime_manifest(
    report: ModelAutoWiringReport,
    manifest: ModelAutoWiringManifest,
    runtime_profile: str,
    image_digest: str | None = None,
    attach_readiness: ModelRuntimeAttachReadiness | None = None,
) -> object:
    """Build the published runtime manifest from auto-wiring results.

    Extracts wired/skipped/failed contracts, topics, and handlers from the
    wiring report and discovered manifest, then returns a frozen
    ModelRuntimeManifestPublished ready for publication on the event bus.

    The import of the core manifest models is deferred so that this module can
    be imported before omnibase_core PR #1098 lands (the model is gated behind
    a try/import in service_kernel.py as well).

    Args:
        report: The wiring report produced by wire_from_manifest().
        manifest: The filtered auto-wiring manifest (post-quarantine).
        runtime_profile: The RUNTIME_PROFILE value (e.g. "main").
        image_digest: Optional OCI image digest for the running container.
        attach_readiness: Boot attach-readiness aggregate (OMN-15512). Narrowed
            to the blocker set before publication — see
            ``ModelRuntimeAttachReadiness.blockers_only``. ``None`` when the
            per-contract interleave did not run at all.

    Returns:
        A ModelRuntimeManifestPublished instance (typed as object to allow
        graceful fallback when the base model is not available in
        omnibase_core).

    Raises:
        ImportError: If omnibase_core.models.runtime_manifest is not installed.
    """
    from omnibase_core.models.runtime_manifest.model_manifest_contract import (
        ModelManifestContract,
    )
    from omnibase_core.models.runtime_manifest.model_manifest_handler import (
        ModelManifestHandler,
    )
    from omnibase_infra.runtime.models.model_runtime_manifest_published import (
        ModelRuntimeManifestPublished,
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

    wired_results = sorted(
        results_by_outcome[EnumWiringOutcome.WIRED],
        key=lambda r: (r.contract_name, r.package_name),
    )
    skipped_results = sorted(
        results_by_outcome[EnumWiringOutcome.SKIPPED],
        key=lambda r: (r.contract_name, r.package_name),
    )
    failed_results = sorted(
        results_by_outcome[EnumWiringOutcome.FAILED],
        key=lambda r: (r.contract_name, r.package_name),
    )

    wired_contracts = tuple(_to_manifest_contract(r) for r in wired_results)
    skipped_contracts = tuple(_to_manifest_contract(r) for r in skipped_results)
    failed_contracts = tuple(_to_manifest_contract(r) for r in failed_results)

    owned_command_topics: set[str] = set()
    for result in wired_results:
        discovered = contract_by_name.get(result.contract_name)
        if discovered and discovered.event_bus:
            owned_command_topics.update(discovered.event_bus.publish_topics)

    subscribed_event_topics: set[str] = set()
    for result in wired_results:
        subscribed_event_topics.update(result.topics_subscribed)

    handlers: list[ModelManifestHandler] = []
    for result in wired_results:
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

    return ModelRuntimeManifestPublished(
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
        attach_readiness=(
            attach_readiness.blockers_only() if attach_readiness is not None else None
        ),
    )


async def publish_runtime_manifest(
    *,
    event_bus: ProtocolEventBusLike,
    report: ModelAutoWiringReport,
    manifest: ModelAutoWiringManifest,
    runtime_profile: str,
    topic: str,
    correlation_id: UUID,
    image_digest: str | None = None,
    attach_readiness: ModelRuntimeAttachReadiness | None = None,
) -> ModelRuntimeManifestPublished:
    """Build the boot snapshot and publish it on the runtime-manifest topic.

    Extracted from ``service_kernel`` step 9.8 (OMN-15512) so the publish seam
    is drivable by a test with a recording bus. The kernel calls this exact
    function, so a test that asserts on the captured envelope payload asserts
    on the artifact that runs — not on a surrogate.

    Args:
        event_bus: The runtime event bus (``publish_envelope``).
        report: The wiring report produced by wire_from_manifest().
        manifest: The filtered auto-wiring manifest (post-quarantine).
        runtime_profile: The RUNTIME_PROFILE value (e.g. "main").
        topic: Resolved topic for SUFFIX_RUNTIME_MANIFEST_PUBLISHED.
        correlation_id: The boot correlation id, propagated onto the envelope.
        image_digest: Optional OCI image digest for the running container.
        attach_readiness: Boot attach-readiness aggregate (OMN-15512).

    Returns:
        The published payload, so callers and tests can assert on exactly what
        went onto the bus.

    Raises:
        ImportError: If omnibase_core.models.runtime_manifest is not installed.
    """
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.runtime.models.model_runtime_manifest_published import (
        ModelRuntimeManifestPublished,
    )

    payload = build_runtime_manifest(
        report=report,
        manifest=manifest,
        runtime_profile=runtime_profile,
        image_digest=image_digest,
        attach_readiness=attach_readiness,
    )
    if not isinstance(payload, ModelRuntimeManifestPublished):  # pragma: no cover
        raise TypeError(
            "build_runtime_manifest must return ModelRuntimeManifestPublished, "
            f"got {type(payload).__name__}"
        )

    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=payload,
        correlation_id=correlation_id,
        event_type="runtime-manifest-published",
        source_tool="service_kernel",
    )
    await event_bus.publish_envelope(envelope=envelope, topic=topic)
    return payload
