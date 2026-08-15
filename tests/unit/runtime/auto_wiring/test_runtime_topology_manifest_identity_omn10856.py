# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED/GREEN: runtime topology manifest identity binding (OMN-10856).

The auto-wiring manifest served over ``/v1/introspection/manifest`` carried
only ``{contracts, errors}`` (OMN-7653) — it could not identify which
runtime profile produced it or which build it came from. The Linear ticket
DoD requires: "runtime_profile identity, contracts loaded, command topics
owned, event topics subscribed, handler registrations, image SHA +
deployment SHA".

This suite covers the identity-binding half of that requirement:
``ModelAutoWiringManifest.runtime_profile`` / ``.image_sha`` /
``.deployment_sha``, and the fail-fast, never-silently-defaulted semantics
of ``ModelRuntimeBuildSha`` (an unset source must surface as an explicit
``value=None, absent_reason=<why>`` marker, never a fabricated placeholder
value). Env-var resolution itself is NOT exercised here — that happens in
``service_kernel.py`` (the approved env-read boundary per
``scripts/check-env-reads.sh``); this suite drives
``ModelRuntimeBuildSha.from_raw`` / ``bind_introspection_manifest_identity``
with already-fetched values, matching how the kernel actually calls them.

Topology (command topics owned / event topics subscribed / handler
registrations) is already carried per-contract on
``ModelDiscoveredContract.event_bus`` / ``.handler_routing`` and is not
re-derived here — see ``ModelAutoWiringManifest.get_all_publish_topics`` /
``.get_all_subscribe_topics``, which already aggregate it from the single
source (the ``contracts`` tuple).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.auto_wiring.introspection_manifest_identity import (
    bind_introspection_manifest_identity,
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
from omnibase_infra.runtime.auto_wiring.models.model_runtime_build_sha import (
    ModelRuntimeBuildSha,
)

pytestmark = pytest.mark.unit


def _contract(name: str = "example_contract") -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="fake_pkg",
    )


class TestModelRuntimeBuildShaPresenceSemantics:
    """A build-identity SHA is present-with-value XOR absent-with-reason."""

    def test_present_value_forbids_absent_reason(self) -> None:
        sha = ModelRuntimeBuildSha(value="abc1234", absent_reason=None)
        assert sha.value == "abc1234"
        assert sha.absent_reason is None
        assert sha.is_present is True

    def test_absent_requires_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="absent_reason"):
            ModelRuntimeBuildSha(value=None, absent_reason=None)

    def test_present_value_with_reason_is_rejected(self) -> None:
        """A value cannot simultaneously carry an absence reason — that would
        let a caller silently confuse a real SHA with a placeholder."""
        with pytest.raises(ValidationError, match="absent_reason"):
            ModelRuntimeBuildSha(value="abc1234", absent_reason="ignored")

    def test_present_classmethod(self) -> None:
        sha = ModelRuntimeBuildSha.present("deadbeef")
        assert sha.value == "deadbeef"
        assert sha.absent_reason is None
        assert sha.is_present is True

    def test_present_classmethod_rejects_blank(self) -> None:
        with pytest.raises(ValueError, match="non-blank"):
            ModelRuntimeBuildSha.present("   ")

    def test_absent_classmethod(self) -> None:
        sha = ModelRuntimeBuildSha.absent("no source configured")
        assert sha.value is None
        assert sha.absent_reason == "no source configured"
        assert sha.is_present is False

    def test_from_raw_present(self) -> None:
        sha = ModelRuntimeBuildSha.from_raw("deadbeef", source_name="ONEX_TEST_SHA_VAR")
        assert sha.value == "deadbeef"
        assert sha.absent_reason is None
        assert sha.is_present is True

    def test_from_raw_none_is_explicit_absent_never_a_silent_default(self) -> None:
        """Fail-fast contract: a missing source must surface as a typed,
        reasoned absence — never fabricate a value, never raise, never
        silently pick a default string like 'unknown'."""
        sha = ModelRuntimeBuildSha.from_raw(
            None, source_name="ONEX_DEFINITELY_UNSET_VAR"
        )
        assert sha.value is None
        assert sha.is_present is False
        assert sha.absent_reason is not None
        assert "ONEX_DEFINITELY_UNSET_VAR" in sha.absent_reason

    def test_from_raw_blank_string_treated_as_unset(self) -> None:
        sha = ModelRuntimeBuildSha.from_raw("   ", source_name="ONEX_TEST_SHA_VAR")
        assert sha.value is None
        assert sha.absent_reason is not None


class TestModelAutoWiringManifestIdentityFields:
    """ModelAutoWiringManifest carries runtime_profile/image_sha/deployment_sha."""

    def test_default_construction_is_unenriched_but_typed(self) -> None:
        """Backward-compatible default: legacy construction sites (discovery,
        ~50 existing tests) do not need to supply identity — but the default
        must still be a typed absent marker, never a bare None/empty sentinel
        masquerading as a real value."""
        manifest = ModelAutoWiringManifest(contracts=(_contract(),), errors=())
        assert manifest.runtime_profile == ""
        assert isinstance(manifest.image_sha, ModelRuntimeBuildSha)
        assert manifest.image_sha.is_present is False
        assert isinstance(manifest.deployment_sha, ModelRuntimeBuildSha)
        assert manifest.deployment_sha.is_present is False

    def test_explicit_identity_construction(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(_contract(),),
            errors=(),
            runtime_profile="workers",
            image_sha=ModelRuntimeBuildSha.present("40a744d"),
            deployment_sha=ModelRuntimeBuildSha.absent(
                "ONEX_DEPLOYMENT_SHA is not set"
            ),
        )
        assert manifest.runtime_profile == "workers"
        assert manifest.image_sha.value == "40a744d"
        assert manifest.deployment_sha.is_present is False

    def test_manifest_is_frozen_and_extra_forbid_unchanged(self) -> None:
        """Guard against silently relaxing the model's existing strictness
        while adding the new fields."""
        manifest = ModelAutoWiringManifest(contracts=(), errors=())
        with pytest.raises(ValidationError):
            ModelAutoWiringManifest(contracts=(), errors=(), unexpected_field=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            manifest.runtime_profile = "mutated"  # type: ignore[misc]

    def test_serialized_manifest_identifies_profile_and_build(self) -> None:
        """DoD binding: the manifest content itself — not an out-of-band
        claim — must identify the runtime profile and the build (image SHA
        + deployment SHA, present-or-reasoned-absent)."""
        manifest = ModelAutoWiringManifest(
            contracts=(_contract(),),
            errors=(),
            runtime_profile="effects",
            image_sha=ModelRuntimeBuildSha.present("cafefeed"),
            deployment_sha=ModelRuntimeBuildSha.absent(
                "ONEX_DEPLOYMENT_SHA is not set"
            ),
        )
        dumped = manifest.model_dump()
        assert dumped["runtime_profile"] == "effects"
        assert dumped["image_sha"]["value"] == "cafefeed"
        assert dumped["deployment_sha"]["value"] is None
        assert dumped["deployment_sha"]["absent_reason"]


class TestBindIntrospectionManifestIdentity:
    """The single-source binder called from service_kernel.py's attach point.

    Reuses the already-discovered ``contracts``/``errors`` tuple verbatim —
    only the three identity fields are added — so topology is never
    re-derived in a second place. Env resolution itself is the kernel's job
    (approved env-read boundary); this binder only assembles
    already-classified ``ModelRuntimeBuildSha`` values.
    """

    def test_binds_identity_fields(self) -> None:
        base = ModelAutoWiringManifest(
            contracts=(_contract(), _contract("b")), errors=()
        )
        enriched = bind_introspection_manifest_identity(
            base,
            runtime_profile="workers",
            image_sha=ModelRuntimeBuildSha.present("sha256:abc123"),
            deployment_sha=ModelRuntimeBuildSha.present("d42cbe70"),
        )

        assert enriched.runtime_profile == "workers"
        assert enriched.image_sha.value == "sha256:abc123"
        assert enriched.deployment_sha.value == "d42cbe70"
        # Topology is reused verbatim, not re-derived.
        assert enriched.contracts == base.contracts
        assert enriched.errors == base.errors

    def test_binds_absent_shas_when_supplied_absent(self) -> None:
        base = ModelAutoWiringManifest(contracts=(_contract(),), errors=())
        enriched = bind_introspection_manifest_identity(
            base,
            runtime_profile="main",
            image_sha=ModelRuntimeBuildSha.absent("ONEX_IMAGE_DIGEST is not set"),
            deployment_sha=ModelRuntimeBuildSha.absent(
                "ONEX_DEPLOYMENT_SHA is not set"
            ),
        )

        assert enriched.runtime_profile == "main"
        assert enriched.image_sha.is_present is False
        assert enriched.deployment_sha.is_present is False
        assert "ONEX_IMAGE_DIGEST" in (enriched.image_sha.absent_reason or "")
        assert "ONEX_DEPLOYMENT_SHA" in (enriched.deployment_sha.absent_reason or "")
