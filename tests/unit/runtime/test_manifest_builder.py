# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for manifest_builder.py (OMN-11196).

Verifies that build_runtime_manifest() produces a valid ModelRuntimeManifest
from mock auto-wiring results without importing any handler or node classes.

These tests are skipped when omnibase_core.models.runtime_manifest is not yet
available (PR #1098 not merged).
"""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
    ModelWiringOutcome,
)


def _runtime_manifest_importable() -> bool:
    try:
        import omnibase_core.models.runtime_manifest.model_runtime_manifest

        return True
    except ImportError:
        return False


runtime_manifest_available = pytest.mark.skipif(
    not _runtime_manifest_importable(),
    reason="omnibase_core.models.runtime_manifest not yet available (PR #1098)",
)


def _make_discovered_contract(
    name: str = "test-node",
    publish_topics: tuple[str, ...] = (),
    subscribe_topics: tuple[str, ...] = (),
    routing_strategy: str = "payload_type_match",
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=pathlib.Path("/fake/contract.yaml"),
        entry_point_name=f"onex.nodes.{name}",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            publish_topics=list(publish_topics),
            subscribe_topics=list(subscribe_topics),
        )
        if publish_topics or subscribe_topics
        else None,
        handler_routing=ModelHandlerRouting(
            routing_strategy=routing_strategy,
            handlers=[],
        )
        if routing_strategy
        else None,
    )


def _make_wired_result(
    contract_name: str = "test-node",
    package_name: str = "test-package",
    topics_subscribed: tuple[str, ...] = (),
    handlers: tuple[str, ...] = (),
) -> ModelContractWiringResult:
    wirings = tuple(
        ModelWiringOutcome(
            handler_name=h,
            resolution_outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
        )
        for h in handlers
    )
    return ModelContractWiringResult(
        contract_name=contract_name,
        package_name=package_name,
        outcome=EnumWiringOutcome.WIRED,
        topics_subscribed=topics_subscribed,
        wirings=wirings,
    )


def _make_skipped_result(
    contract_name: str = "skipped-node",
    package_name: str = "test-package",
    reason: str = "profile mismatch",
) -> ModelContractWiringResult:
    return ModelContractWiringResult(
        contract_name=contract_name,
        package_name=package_name,
        outcome=EnumWiringOutcome.SKIPPED,
        reason=reason,
    )


def _make_failed_result(
    contract_name: str = "failed-node",
    package_name: str = "test-package",
    reason: str = "handler not found",
) -> ModelContractWiringResult:
    return ModelContractWiringResult(
        contract_name=contract_name,
        package_name=package_name,
        outcome=EnumWiringOutcome.FAILED,
        reason=reason,
    )


@runtime_manifest_available
def test_build_manifest_empty_report() -> None:
    """build_runtime_manifest returns a valid manifest for an empty wiring report."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    report = ModelAutoWiringReport()
    manifest = ModelAutoWiringManifest()
    result = build_runtime_manifest(
        report=report,
        manifest=manifest,
        runtime_profile="main",
    )
    assert result.runtime_profile == "main"
    assert result.contracts == ()
    assert result.skipped_contracts == ()
    assert result.failed_contracts == ()
    assert result.owned_command_topics == frozenset()
    assert result.subscribed_event_topics == frozenset()
    assert result.handlers == ()
    assert result.ownership_violations == ()
    assert result.image_digest is None


@runtime_manifest_available
def test_build_manifest_wired_contracts_populate_correctly() -> None:
    """Wired contracts are projected into contracts tuple with correct metadata."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    discovered = _make_discovered_contract(
        name="node-alpha",
        publish_topics=("onex.cmd.svc.action.v1",),
        subscribe_topics=("onex.evt.svc.event.v1",),
    )
    wired = _make_wired_result(
        contract_name="node-alpha",
        topics_subscribed=("onex.evt.svc.event.v1",),
        handlers=("HandlerAlpha",),
    )
    report = ModelAutoWiringReport(results=(wired,))
    manifest = ModelAutoWiringManifest(contracts=(discovered,))

    result = build_runtime_manifest(
        report=report, manifest=manifest, runtime_profile="main"
    )

    assert len(result.contracts) == 1
    assert result.contracts[0].name == "node-alpha"
    assert result.contracts[0].node_type == "EFFECT_GENERIC"
    assert result.contracts[0].contract_hash != ""
    assert "onex.cmd.svc.action.v1" in result.owned_command_topics
    assert "onex.evt.svc.event.v1" in result.subscribed_event_topics
    assert len(result.handlers) == 1
    assert result.handlers[0].name == "HandlerAlpha"


@runtime_manifest_available
def test_build_manifest_skipped_and_failed_segregated() -> None:
    """Skipped/failed contracts go to their respective tuples, not contracts."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    wired = _make_wired_result(contract_name="good-node")
    skipped = _make_skipped_result(contract_name="skipped-node")
    failed = _make_failed_result(contract_name="failed-node")

    report = ModelAutoWiringReport(results=(wired, skipped, failed))
    manifest = ModelAutoWiringManifest(
        contracts=(
            _make_discovered_contract("good-node"),
            _make_discovered_contract("skipped-node"),
            _make_discovered_contract("failed-node"),
        )
    )

    result = build_runtime_manifest(
        report=report, manifest=manifest, runtime_profile="test"
    )

    assert len(result.contracts) == 1
    assert result.contracts[0].name == "good-node"
    assert len(result.skipped_contracts) == 1
    assert result.skipped_contracts[0].name == "skipped-node"
    assert len(result.failed_contracts) == 1
    assert result.failed_contracts[0].name == "failed-node"


@runtime_manifest_available
def test_build_manifest_image_digest_forwarded() -> None:
    """image_digest from caller is passed through to the manifest."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    report = ModelAutoWiringReport()
    manifest = ModelAutoWiringManifest()
    result = build_runtime_manifest(
        report=report,
        manifest=manifest,
        runtime_profile="production",
        image_digest="sha256:abc123",
    )
    assert result.image_digest == "sha256:abc123"


@runtime_manifest_available
def test_build_manifest_contract_hash_deterministic() -> None:
    """Calling build_runtime_manifest twice with identical input yields same contract_hash."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    discovered = _make_discovered_contract("stable-node")
    wired = _make_wired_result("stable-node")
    report = ModelAutoWiringReport(results=(wired,))
    manifest = ModelAutoWiringManifest(contracts=(discovered,))

    result_a = build_runtime_manifest(
        report=report, manifest=manifest, runtime_profile="main"
    )
    result_b = build_runtime_manifest(
        report=report, manifest=manifest, runtime_profile="main"
    )

    assert result_a.contract_hash == result_b.contract_hash
    assert result_a.topology_hash == result_b.topology_hash


def test_build_manifest_builder_importable() -> None:
    """manifest_builder module is importable regardless of omnibase_core PR status."""
    import importlib

    mod = importlib.import_module("omnibase_infra.runtime.manifest_builder")
    assert hasattr(mod, "build_runtime_manifest")


def test_build_manifest_raises_on_missing_core_model() -> None:
    """build_runtime_manifest raises ImportError when omnibase_core model is absent."""
    import sys

    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    patched = {
        "omnibase_core.models.runtime_manifest": None,
        "omnibase_core.models.runtime_manifest.model_runtime_manifest": None,
        "omnibase_core.models.runtime_manifest.model_manifest_contract": None,
        "omnibase_core.models.runtime_manifest.model_manifest_handler": None,
    }
    with patch.dict(sys.modules, patched):
        report = ModelAutoWiringReport()
        manifest = ModelAutoWiringManifest()
        with pytest.raises((ImportError, AttributeError)):
            build_runtime_manifest(
                report=report, manifest=manifest, runtime_profile="main"
            )


def test_suffix_runtime_manifest_published_importable() -> None:
    """SUFFIX_RUNTIME_MANIFEST_PUBLISHED is exported from topics.__init__."""
    from omnibase_infra.topics import SUFFIX_RUNTIME_MANIFEST_PUBLISHED

    assert SUFFIX_RUNTIME_MANIFEST_PUBLISHED.startswith("onex.evt.")
    assert "runtime-manifest-published" in SUFFIX_RUNTIME_MANIFEST_PUBLISHED
