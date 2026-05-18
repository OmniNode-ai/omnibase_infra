# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests replaying golden event chains for dynamic registration.

Each test loads a golden chain fixture, exercises the in-process components
(KafkaContractSource cache/discover), and asserts the chain shape matches
the fixture at every checkpoint.

No Kafka, no .201, no network. The golden chain IS the specification.

Golden chain fixture files live under tests/fixtures/golden_chains/ and are
the single source of truth for both unit and integration tests.

Related: OMN-11249
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

pytestmark = pytest.mark.unit

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "golden_chains"


def _load_golden_chain(name: str) -> list[dict]:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text())


# Minimal valid contract that KafkaContractSource can parse.
# Uses the effect archetype since that requires the fewest optional fields.
_VALID_CONTRACT_YAML = """\
handler_id: "proto.golden_chain_test"
name: "node_golden_chain_test"
contract_version:
  major: 1
  minor: 0
  patch: 0
description: "Golden chain unit test handler"
descriptor:
  node_archetype: "effect"
input_model: "omnibase_infra.models.types.JsonDict"
output_model: "omnibase_core.models.dispatch.ModelHandlerOutput"
metadata:
  handler_class: "omnibase_infra.handlers.handler_http.HandlerHttp"
"""

# Intentionally invalid YAML to trigger parse_failure at checkpoint 1.
_MALFORMED_CONTRACT_YAML = "this: [is: not: valid: {{"


class TestGoldenChainSuccess:
    """Replay the 7-checkpoint success golden chain in-process.

    Checkpoints tested:
        0 — contract_registration_requested (simulated: caller sends to source)
        3 — contract_cached (on_contract_registered returns True)
        4 — contract_materialized (discover_handlers returns the descriptor)
        6 — dispatch_routable (descriptor available for routing)

    Checkpoints 1, 2, 5 (omnimarket and topology-delta) are validated by the
    integration test against the live runtime.
    """

    def test_fixture_has_seven_checkpoints(self) -> None:
        golden = _load_golden_chain("dynamic_registration_success")
        assert len(golden) == 7
        sequences = [e["sequence"] for e in golden]
        assert sequences == list(range(7))

    def test_fixture_event_types_match_spec(self) -> None:
        golden = _load_golden_chain("dynamic_registration_success")
        expected_types = [
            "contract_registration_requested",
            "contract_validated",
            "contract_registration_published",
            "contract_cached",
            "contract_materialized",
            "topology_manifest_delta",
            "dispatch_routable",
        ]
        actual_types = [e["event_type"] for e in golden]
        assert actual_types == expected_types

    def test_checkpoint_3_contract_cached(self) -> None:
        """on_contract_registered returns True — contract is cached."""
        golden = _load_golden_chain("dynamic_registration_success")
        cached_checkpoint = next(
            e for e in golden if e["event_type"] == "contract_cached"
        )
        assert cached_checkpoint["cached"] is True

        source = KafkaContractSource(environment="test")
        result = source.on_contract_registered(
            node_name="node_golden_chain_test",
            contract_yaml=_VALID_CONTRACT_YAML,
            correlation_id=uuid4(),
        )
        assert result is True
        assert source.cached_count == 1

    @pytest.mark.asyncio
    async def test_checkpoint_4_contract_materialized(self) -> None:
        """discover_handlers returns the cached descriptor — contract is materialized."""
        golden = _load_golden_chain("dynamic_registration_success")
        materialized_checkpoint = next(
            e for e in golden if e["event_type"] == "contract_materialized"
        )
        assert materialized_checkpoint["status"] == "materialized"

        source = KafkaContractSource(environment="test")
        source.on_contract_registered(
            node_name="node_golden_chain_test",
            contract_yaml=_VALID_CONTRACT_YAML,
            correlation_id=uuid4(),
        )
        discovery = await source.discover_handlers()
        assert len(discovery.descriptors) == 1
        descriptor = discovery.descriptors[0]
        assert descriptor.handler_id == "proto.golden_chain_test"
        assert descriptor.name == "node_golden_chain_test"

    @pytest.mark.asyncio
    async def test_checkpoint_6_dispatch_routable(self) -> None:
        """Descriptor retrieved from cache proves dispatch routing is possible."""
        golden = _load_golden_chain("dynamic_registration_success")
        routable_checkpoint = next(
            e for e in golden if e["event_type"] == "dispatch_routable"
        )
        assert routable_checkpoint["reachable"] is True

        source = KafkaContractSource(environment="test")
        source.on_contract_registered(
            node_name="node_golden_chain_test",
            contract_yaml=_VALID_CONTRACT_YAML,
            correlation_id=uuid4(),
        )
        # A descriptor in the cache is routable: the dispatch engine can find it.
        descriptor = source.get_cached_descriptor("node_golden_chain_test")
        assert descriptor is not None
        assert descriptor.handler_class is not None

    @pytest.mark.asyncio
    async def test_full_chain_observed_shape_matches_fixture(self) -> None:
        """Walk all in-process checkpoints and assert shape matches the fixture."""
        golden = _load_golden_chain("dynamic_registration_success")
        correlation_id = uuid4()

        source = KafkaContractSource(environment="test")
        observed: list[dict] = []

        # Checkpoint 0: registration requested (caller sends; we record the intent)
        observed.append(
            {"sequence": 0, "event_type": "contract_registration_requested"}
        )

        # Checkpoint 3: contract cached
        cached = source.on_contract_registered(
            node_name="node_golden_chain_test",
            contract_yaml=_VALID_CONTRACT_YAML,
            correlation_id=correlation_id,
        )
        assert cached is True
        observed.append(
            {"sequence": 3, "event_type": "contract_cached", "cached": True}
        )

        # Checkpoint 4: contract materialized
        discovery = await source.discover_handlers()
        assert len(discovery.descriptors) == 1
        observed.append(
            {
                "sequence": 4,
                "event_type": "contract_materialized",
                "status": "materialized",
            }
        )

        # Checkpoint 6: dispatch routable
        descriptor = source.get_cached_descriptor("node_golden_chain_test")
        assert descriptor is not None
        observed.append(
            {"sequence": 6, "event_type": "dispatch_routable", "reachable": True}
        )

        # Every observed checkpoint must match the corresponding fixture entry by event_type.
        fixture_by_type = {e["event_type"]: e for e in golden}
        for obs in observed:
            assert obs["event_type"] in fixture_by_type, (
                f"Observed event_type '{obs['event_type']}' not in golden fixture"
            )
            fixture_entry = fixture_by_type[obs["event_type"]]
            assert obs["sequence"] == fixture_entry["sequence"], (
                f"Sequence mismatch for '{obs['event_type']}': "
                f"observed={obs['sequence']}, fixture={fixture_entry['sequence']}"
            )


class TestGoldenChainMalformedYaml:
    """Replay the malformed-YAML error chain.

    Checkpoint 0 — contract_registration_requested (simulated)
    Checkpoint 1 — contract_registration_rejected (parse_failure)
    """

    def test_fixture_has_two_checkpoints(self) -> None:
        golden = _load_golden_chain("dynamic_registration_malformed_yaml")
        assert len(golden) == 2
        assert golden[1]["reason"] == "parse_failure"

    def test_parse_failure_returns_false_in_graceful_mode(self) -> None:
        golden = _load_golden_chain("dynamic_registration_malformed_yaml")
        rejection_checkpoint = next(
            e for e in golden if e["event_type"] == "contract_registration_rejected"
        )
        assert rejection_checkpoint["reason"] == "parse_failure"
        assert rejection_checkpoint["status"] == "rejected"

        source = KafkaContractSource(environment="test", graceful_mode=True)
        result = source.on_contract_registered(
            node_name="node_bad_yaml",
            contract_yaml=_MALFORMED_CONTRACT_YAML,
            correlation_id=uuid4(),
        )
        # Graceful mode: returns False, does not raise.
        assert result is False
        assert source.cached_count == 0
        assert source.pending_error_count == 1

    @pytest.mark.asyncio
    async def test_parse_failure_error_surfaces_in_discovery(self) -> None:
        source = KafkaContractSource(environment="test", graceful_mode=True)
        source.on_contract_registered(
            node_name="node_bad_yaml",
            contract_yaml=_MALFORMED_CONTRACT_YAML,
            correlation_id=uuid4(),
        )
        discovery = await source.discover_handlers()
        assert len(discovery.descriptors) == 0
        assert len(discovery.validation_errors) == 1


class TestGoldenChainVersionConflict:
    """Replay the version-conflict error chain.

    Two registrations of the same node_name: first succeeds (cached),
    second is a conflict because the hash differs.

    The fixture documents what omnimarket should reject. In-process, the
    KafkaContractSource itself does NOT reject re-registration — it overwrites
    the cache. The version-conflict rejection is enforced by node_contract_registry
    (omnimarket), which is tested in the integration test.

    This unit test validates the fixture shape only.
    """

    def test_fixture_has_two_checkpoints(self) -> None:
        golden = _load_golden_chain("dynamic_registration_version_conflict")
        assert len(golden) == 2
        assert golden[1]["reason"] == "version_conflict"

    def test_fixture_rejection_at_sequence_1(self) -> None:
        golden = _load_golden_chain("dynamic_registration_version_conflict")
        assert golden[1]["sequence"] == 1
        assert golden[1]["event_type"] == "contract_registration_rejected"
        assert golden[1]["status"] == "rejected"

    def test_first_registration_cached_successfully(self) -> None:
        """The first registration (sequence 0 → 3) always succeeds."""
        source = KafkaContractSource(environment="test")
        result = source.on_contract_registered(
            node_name="node_versioned",
            contract_yaml=_VALID_CONTRACT_YAML,
            correlation_id=uuid4(),
        )
        assert result is True
        assert source.cached_count == 1


class TestGoldenChainHashMismatch:
    """Validate hash_mismatch error chain fixture shape."""

    def test_fixture_has_two_checkpoints(self) -> None:
        golden = _load_golden_chain("dynamic_registration_hash_mismatch")
        assert len(golden) == 2
        assert golden[1]["reason"] == "hash_mismatch"
        assert golden[1]["sequence"] == 1
        assert golden[1]["status"] == "rejected"


class TestGoldenChainProfileMismatch:
    """Validate profile_mismatch error chain fixture shape."""

    def test_fixture_has_two_checkpoints(self) -> None:
        golden = _load_golden_chain("dynamic_registration_profile_mismatch")
        assert len(golden) == 2
        assert golden[1]["reason"] == "profile_mismatch"
        assert golden[1]["sequence"] == 1
        assert golden[1]["status"] == "rejected"


class TestGoldenChainHandlerAllowlist:
    """Validate handler_allowlist error chain fixture shape."""

    def test_fixture_has_two_checkpoints(self) -> None:
        golden = _load_golden_chain("dynamic_registration_handler_allowlist")
        assert len(golden) == 2
        assert golden[1]["reason"] == "handler_allowlist"
        assert golden[1]["sequence"] == 1
        assert golden[1]["status"] == "rejected"


class TestGoldenChainFixtureConsistency:
    """Cross-fixture consistency checks.

    All error chains must share the same sequence-0 event_type as the success chain.
    Error chains must diverge at sequence 1 with contract_registration_rejected.
    """

    _ERROR_CHAINS = [
        "dynamic_registration_version_conflict",
        "dynamic_registration_hash_mismatch",
        "dynamic_registration_profile_mismatch",
        "dynamic_registration_handler_allowlist",
        "dynamic_registration_malformed_yaml",
    ]

    def test_all_chains_start_with_registration_requested(self) -> None:
        success = _load_golden_chain("dynamic_registration_success")
        seq0_type = success[0]["event_type"]

        for chain_name in self._ERROR_CHAINS:
            chain = _load_golden_chain(chain_name)
            assert chain[0]["event_type"] == seq0_type, (
                f"Chain '{chain_name}' sequence 0 event_type mismatch: "
                f"expected '{seq0_type}', got '{chain[0]['event_type']}'"
            )

    def test_all_error_chains_diverge_at_sequence_1(self) -> None:
        for chain_name in self._ERROR_CHAINS:
            chain = _load_golden_chain(chain_name)
            assert len(chain) == 2, (
                f"Chain '{chain_name}' should have 2 events, got {len(chain)}"
            )
            assert chain[1]["sequence"] == 1
            assert chain[1]["event_type"] == "contract_registration_rejected"
            assert chain[1]["status"] == "rejected"
            assert "reason" in chain[1], (
                f"Chain '{chain_name}' missing 'reason' at sequence 1"
            )

    def test_error_reasons_are_distinct(self) -> None:
        reasons = []
        for chain_name in self._ERROR_CHAINS:
            chain = _load_golden_chain(chain_name)
            reasons.append(chain[1]["reason"])
        assert len(reasons) == len(set(reasons)), (
            f"Duplicate rejection reasons across error chains: {reasons}"
        )
