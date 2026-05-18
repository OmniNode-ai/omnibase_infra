# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Proof-of-life integration test for platform hardening data flows (OMN-11210).

Exercises four data flows introduced by the OMN-11188 platform hardening epic:

1. Runtime manifest flow — ModelRuntimeManifest hash determinism
2. Evidence bundle flow — ModelStandardEvidenceBundle completeness + bundle_hash stability
3. Projection contract freshness — ModelProjectionContract required-field enforcement
4. Data provenance validation — EnumDataProvenance normalisation round-trip

Models from OMN-11190/11191/11192 are not yet in the installed omnibase_core
(v0.41.0); they are defined inline here until those PRs merge and the pin advances.
EnumDataProvenance and EnumDegradedBehavior ship in omnibase_core v0.41.0 (OMN-11189).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from omnibase_core.enums.enum_data_provenance import EnumDataProvenance
from omnibase_core.enums.enum_degraded_behavior import EnumDegradedBehavior

# ---------------------------------------------------------------------------
# Inline model definitions (mirror OMN-11190 / 11191 / 11192 PRs)
# Remove once omnibase_core >= v0.42.0 is pinned and the classes are importable.
# ---------------------------------------------------------------------------


class _ModelManifestContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    node_type: str = Field(..., min_length=1)
    contract_hash: str = Field(..., min_length=1)


class _ModelManifestHandler(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., min_length=1)
    module_path: str = Field(..., min_length=1)
    routing_strategy: str = Field(..., min_length=1)


class _ModelRuntimeManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    runtime_profile: str = Field(..., min_length=1)
    contracts: tuple[_ModelManifestContract, ...] = Field(default=())
    owned_command_topics: frozenset[str] = Field(default_factory=frozenset)
    subscribed_event_topics: frozenset[str] = Field(default_factory=frozenset)
    handlers: tuple[_ModelManifestHandler, ...] = Field(default=())
    skipped_contracts: tuple[_ModelManifestContract, ...] = Field(default=())
    failed_contracts: tuple[_ModelManifestContract, ...] = Field(default=())
    image_digest: str | None = Field(default=None)
    started_at: datetime = Field(...)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def contract_hash(self) -> str:
        sorted_hashes = sorted(c.contract_hash for c in self.contracts)
        payload = json.dumps(sorted_hashes, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def topology_hash(self) -> str:
        payload = json.dumps(
            {
                "runtime_profile": self.runtime_profile,
                "contract_hash": self.contract_hash,
                "owned_command_topics": sorted(self.owned_command_topics),
                "subscribed_event_topics": sorted(self.subscribed_event_topics),
                "handlers": sorted(h.name for h in self.handlers),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class _ModelArtifactEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    filename: str
    sha256: str
    write_order: int


class _ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: str
    artifacts: tuple[_ModelArtifactEntry, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bundle_hash(self) -> str:
        sorted_hashes = sorted(a.sha256 for a in self.artifacts)
        combined = "".join(sorted_hashes).encode()
        return hashlib.sha256(combined).hexdigest()


class _ModelStandardRunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: str
    runner: str
    started_at: datetime
    completed_at: datetime | None = None
    expected_artifacts: tuple[str, ...] = ()
    ticket_id: str | None = None


_ARTIFACT_TO_FIELD: dict[str, str] = {
    "run_manifest.json": "run_manifest",
    "contract_snapshot.json": "contract_snapshot",
    "input.json": "input_data",
    "output.json": "output_data",
    "verifier_result.json": "verifier_result",
    "artifact_manifest.json": "artifact_manifest",
}


class _ModelStandardEvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: str
    run_manifest: _ModelStandardRunManifest
    artifact_manifest: _ModelArtifactManifest | None = None
    contract_snapshot: dict[str, object] | None = None
    verifier_result: dict[str, object] | None = None
    input_data: dict[str, object] | None = None
    output_data: dict[str, object] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        for artifact in self.run_manifest.expected_artifacts:
            field_name = _ARTIFACT_TO_FIELD.get(artifact)
            if field_name is not None and getattr(self, field_name) is None:
                return False
        return True


class _ModelCursorContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    cursor_type: str = Field(..., min_length=1)
    supports_replay: bool = Field(...)


class _ModelProjectionContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    projection_name: str = Field(..., min_length=1)
    source_topics: tuple[str, ...] = Field(...)
    schema_model: str = Field(..., min_length=1)
    freshness_sla_seconds: int = Field(..., gt=0)
    freshness_field: str = Field(..., min_length=1)
    freshness_source_table: str = Field(..., min_length=1)
    degraded_semantics: EnumDegradedBehavior = Field(...)
    cursor: _ModelCursorContract = Field(...)
    ordering_contract_ref: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)

_CONTRACT_A = _ModelManifestContract(
    name="node_auth_effect",
    version="1",
    node_type="effect",
    contract_hash="aaa111",
)
_CONTRACT_B = _ModelManifestContract(
    name="node_payment_compute",
    version="2",
    node_type="compute",
    contract_hash="bbb222",
)
_HANDLER_A = _ModelManifestHandler(
    name="HandlerAuthEffect",
    module_path="omnibase_infra.nodes.node_auth_effect.handler",
    routing_strategy="topic",
)


# ---------------------------------------------------------------------------
# Test 1: Runtime manifest flow
# ---------------------------------------------------------------------------


class TestRuntimeManifestFlow:
    def test_contract_hash_deterministic(self) -> None:
        manifest1 = _ModelRuntimeManifest(
            runtime_profile="production",
            contracts=(_CONTRACT_A, _CONTRACT_B),
            started_at=_NOW,
        )
        manifest2 = _ModelRuntimeManifest(
            runtime_profile="production",
            contracts=(_CONTRACT_B, _CONTRACT_A),
            started_at=_NOW,
        )
        assert manifest1.contract_hash == manifest2.contract_hash

    def test_contract_hash_value(self) -> None:
        manifest = _ModelRuntimeManifest(
            runtime_profile="staging",
            contracts=(_CONTRACT_A,),
            started_at=_NOW,
        )
        expected = hashlib.sha256(
            json.dumps(sorted([_CONTRACT_A.contract_hash]), sort_keys=True).encode()
        ).hexdigest()
        assert manifest.contract_hash == expected

    def test_topology_hash_changes_on_profile(self) -> None:
        base = _ModelRuntimeManifest(
            runtime_profile="prod",
            contracts=(_CONTRACT_A,),
            started_at=_NOW,
        )
        other = _ModelRuntimeManifest(
            runtime_profile="staging",
            contracts=(_CONTRACT_A,),
            started_at=_NOW,
        )
        assert base.topology_hash != other.topology_hash

    def test_field_values_preserved(self) -> None:
        manifest = _ModelRuntimeManifest(
            runtime_profile="prod",
            contracts=(_CONTRACT_A, _CONTRACT_B),
            handlers=(_HANDLER_A,),
            owned_command_topics=frozenset({"onex.cmd.auth.login.v1"}),
            subscribed_event_topics=frozenset({"onex.evt.auth.logged_in.v1"}),
            started_at=_NOW,
        )
        assert manifest.runtime_profile == "prod"
        assert len(manifest.contracts) == 2
        assert len(manifest.handlers) == 1
        assert "onex.cmd.auth.login.v1" in manifest.owned_command_topics

    def test_manifest_immutable(self) -> None:
        manifest = _ModelRuntimeManifest(runtime_profile="prod", started_at=_NOW)
        with pytest.raises(Exception):
            manifest.runtime_profile = "staging"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 2: Evidence bundle flow
# ---------------------------------------------------------------------------


class TestEvidenceBundleFlow:
    def _run_manifest(
        self, *expected: str, correlation_id: str = "corr-001"
    ) -> _ModelStandardRunManifest:
        return _ModelStandardRunManifest(
            correlation_id=correlation_id,
            runner="test_runner",
            started_at=_NOW,
            expected_artifacts=tuple(expected),
        )

    def test_complete_bundle(self) -> None:
        run_manifest = self._run_manifest(
            "run_manifest.json", "artifact_manifest.json", "input.json", "output.json"
        )
        bundle = _ModelStandardEvidenceBundle(
            correlation_id="corr-001",
            run_manifest=run_manifest,
            artifact_manifest=_ModelArtifactManifest(
                correlation_id="corr-001", artifacts=()
            ),
            input_data={"key": "val"},
            output_data={"result": "ok"},
        )
        assert bundle.is_complete is True

    def test_incomplete_bundle_missing_output(self) -> None:
        run_manifest = self._run_manifest("output.json")
        bundle = _ModelStandardEvidenceBundle(
            correlation_id="corr-002",
            run_manifest=run_manifest,
            output_data=None,
        )
        assert bundle.is_complete is False

    def test_artifact_manifest_bundle_hash_stable(self) -> None:
        entry_a = _ModelArtifactEntry(
            filename="run_manifest.json", sha256="sha_a", write_order=0
        )
        entry_b = _ModelArtifactEntry(
            filename="output.json", sha256="sha_b", write_order=1
        )
        manifest_ab = _ModelArtifactManifest(
            correlation_id="c1", artifacts=(entry_a, entry_b)
        )
        manifest_ba = _ModelArtifactManifest(
            correlation_id="c1", artifacts=(entry_b, entry_a)
        )
        assert manifest_ab.bundle_hash == manifest_ba.bundle_hash

    def test_bundle_hash_computed_from_sorted_sha256(self) -> None:
        entry = _ModelArtifactEntry(
            filename="run_manifest.json", sha256="deadbeef", write_order=0
        )
        manifest = _ModelArtifactManifest(correlation_id="c2", artifacts=(entry,))
        expected = hashlib.sha256(b"deadbeef").hexdigest()
        assert manifest.bundle_hash == expected

    def test_write_order_does_not_affect_bundle_hash(self) -> None:
        sha = "cafebabe"
        entry_order0 = _ModelArtifactEntry(filename="a.json", sha256=sha, write_order=0)
        entry_order5 = _ModelArtifactEntry(filename="a.json", sha256=sha, write_order=5)
        m0 = _ModelArtifactManifest(correlation_id="c3", artifacts=(entry_order0,))
        m5 = _ModelArtifactManifest(correlation_id="c3", artifacts=(entry_order5,))
        assert m0.bundle_hash == m5.bundle_hash


# ---------------------------------------------------------------------------
# Test 3: Projection contract + freshness
# ---------------------------------------------------------------------------


class TestProjectionContractFreshness:
    def _cursor(self) -> _ModelCursorContract:
        return _ModelCursorContract(cursor_type="offset", supports_replay=True)

    def _valid_contract(self, **overrides: object) -> _ModelProjectionContract:
        defaults: dict[str, object] = {
            "projection_name": "contract_registry",
            "source_topics": ("onex.evt.contract.registered.v1",),
            "schema_model": "omnibase_infra.models.projection.model_contract_projection.ModelContractProjection",
            "freshness_sla_seconds": 5,
            "freshness_field": "last_seen_at",
            "freshness_source_table": "onex_contracts",
            "degraded_semantics": EnumDegradedBehavior.SERVE_STALE_WITH_WARNING,
            "cursor": self._cursor(),
        }
        defaults.update(overrides)
        return _ModelProjectionContract(**defaults)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        contract = self._valid_contract()
        assert contract.freshness_sla_seconds == 5
        assert contract.freshness_field == "last_seen_at"
        assert contract.freshness_source_table == "onex_contracts"

    def test_degraded_semantics_required_no_default(self) -> None:
        with pytest.raises(ValidationError):
            _ModelProjectionContract(
                projection_name="p",
                source_topics=("t",),
                schema_model="m.M",
                freshness_sla_seconds=5,
                freshness_field="f",
                freshness_source_table="t",
                cursor=self._cursor(),
                # degraded_semantics intentionally omitted
            )

    def test_freshness_field_required(self) -> None:
        with pytest.raises(ValidationError):
            _ModelProjectionContract(
                projection_name="p",
                source_topics=("t",),
                schema_model="m.M",
                freshness_sla_seconds=5,
                freshness_field="",
                freshness_source_table="t",
                degraded_semantics=EnumDegradedBehavior.FAIL_CLOSED,
                cursor=self._cursor(),
            )

    def test_freshness_source_table_required(self) -> None:
        with pytest.raises(ValidationError):
            _ModelProjectionContract(
                projection_name="p",
                source_topics=("t",),
                schema_model="m.M",
                freshness_sla_seconds=5,
                freshness_field="updated_at",
                freshness_source_table="",
                degraded_semantics=EnumDegradedBehavior.RETURN_EMPTY,
                cursor=self._cursor(),
            )

    def test_zero_sla_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_contract(freshness_sla_seconds=0)

    def test_negative_sla_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_contract(freshness_sla_seconds=-1)

    def test_all_degraded_semantics_values_accepted(self) -> None:
        for behavior in EnumDegradedBehavior:
            c = self._valid_contract(degraded_semantics=behavior)
            assert c.degraded_semantics == behavior


# ---------------------------------------------------------------------------
# Test 4: Data provenance validation
# ---------------------------------------------------------------------------


class TestDataProvenanceValidation:
    def test_valid_provenance_values_round_trip(self) -> None:
        for provenance in EnumDataProvenance:
            assert EnumDataProvenance(provenance.value) == provenance

    def test_invalid_provenance_raises(self) -> None:
        with pytest.raises(ValueError):
            EnumDataProvenance("totally_unknown_value")

    def test_missing_normalizes_to_unknown_via_default(self) -> None:
        class _Carrier(BaseModel):
            provenance: EnumDataProvenance = EnumDataProvenance.UNKNOWN

        carrier = _Carrier()
        assert carrier.provenance == EnumDataProvenance.UNKNOWN

    def test_absent_provenance_defaults_to_unknown(self) -> None:
        class _ProvenanceCarrier(BaseModel):
            label: str
            provenance: EnumDataProvenance = EnumDataProvenance.UNKNOWN

        obj = _ProvenanceCarrier(label="test")
        assert obj.provenance == EnumDataProvenance.UNKNOWN

    def test_demo_seeded_distinct_from_measured(self) -> None:
        assert EnumDataProvenance.DEMO_SEEDED != EnumDataProvenance.MEASURED

    def test_all_five_members_present(self) -> None:
        values = {e.value for e in EnumDataProvenance}
        assert values == {
            "demo_seeded",
            "demo_projected_shortcut",
            "measured",
            "estimated",
            "unknown",
        }
