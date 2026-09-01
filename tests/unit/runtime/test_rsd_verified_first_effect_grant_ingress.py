# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exact RSD verifier-to-ledger ingress checks using the public package."""

from __future__ import annotations

import ast
import asyncio
import json
import runpy
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from omninode_grant_verifier import signed_executable_grant_v2_vectors

from omnibase_infra.runtime import first_effect_ledger
from omnibase_infra.runtime.first_effect_ledger.composition import (
    FirstEffectGrantIngressRejectedError,
    build_rsd_verified_first_effect_grant_ingress,
)
from omnibase_infra.runtime.first_effect_ledger.enum_verified_first_effect_grant_state import (
    EnumVerifiedFirstEffectGrantState,
)
from omnibase_infra.runtime.first_effect_ledger.model_expected_output_pin import (
    ModelExpectedOutputPin,
)
from omnibase_infra.runtime.first_effect_ledger.model_verified_first_effect_grant_record import (
    ModelVerifiedFirstEffectGrantRecord,
)

_NOW = datetime(2030, 1, 1, tzinfo=UTC)
_EXPECTED_OUTPUT_PIN = ModelExpectedOutputPin(
    topic="events.public.grant.completed.v2",
    event_class="ModelPublicGrantCompleted",
    index=0,
)
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"
_COMPOSITION_PATH = _SOURCE_ROOT / "runtime" / "first_effect_ledger" / "composition.py"
_BOUNDARY_VALIDATOR = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "scripts/validation/validate_verified_grant_ingress_boundary.py"
    )
)["_violations"]


def _valid_wire() -> dict[str, object]:
    vectors = signed_executable_grant_v2_vectors()
    wire = vectors["base_wire"]
    assert type(wire) is dict
    return wire


def _set_path(document: dict[str, object], path: str, value: object) -> None:
    current = document
    for part in path.split(".")[:-1]:
        nested = current[part]
        assert type(nested) is dict
        current = nested
    current[path.rsplit(".", maxsplit=1)[-1]] = value


def _wire_for_vector(
    vector: dict[str, object], base: dict[str, object]
) -> dict[str, object]:
    wire = vector.get("wire")
    if wire is not None:
        assert type(wire) is dict
        return deepcopy(wire)
    result = deepcopy(base)
    patches = vector["patch"]
    assert type(patches) is list
    for patch in patches:
        assert type(patch) is list and len(patch) == 2 and isinstance(patch[0], str)
        _set_path(result, patch[0], patch[1])
    return result


def _verification_now(case: dict[str, object], vectors: dict[str, object]) -> datetime:
    now_text = case.get("verification_now", vectors["now"])
    assert isinstance(now_text, str)
    return datetime.fromisoformat(now_text.replace("Z", "+00:00")).astimezone(UTC)


def _raw_projection_bypass_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and (
            node.func.id == "_ModelVerifiedFirstEffectGrantProjection"
        ):
            lines.append(node.lineno)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_record_verified_grant"
        ):
            lines.append(node.lineno)
    return lines


class _FakeAcquire:
    def __init__(self, connection: _RecordingPool) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RecordingPool:
        return self._connection

    async def __aexit__(self, *_: object) -> None:
        return None


class _RecordingPool:
    record_count = 0
    parameters: tuple[object, ...] | None = None
    closed = False

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self)

    async def close(self) -> None:
        self.closed = True

    async def fetchrow(self, _: str, *parameters: object) -> dict[str, object]:
        self.record_count += 1
        self.parameters = parameters
        keys = (
            "authorization_digest",
            "grant_id",
            "grant_envelope_id",
            "nonce_digest",
            "request_digest",
            "correlation_id",
            "tenant_id",
            "backend_id",
            "rendered_contract_sha256",
            "issuer_key_fingerprint_sha256",
            "retry_disposition",
            "expected_output_topic",
            "expected_output_event_class",
            "expected_output_event_index",
        )
        row = dict(zip(keys, parameters, strict=True))
        row.update(
            state=EnumVerifiedFirstEffectGrantState.VERIFIED,
            outbox_envelope_id=None,
            outbox_body_sha256=None,
            outbox_topic=None,
            outbox_event_class=None,
            outbox_event_index=None,
            workflow_version=None,
            verified_at=_NOW,
            staged_at=None,
            publishing_at=None,
            claimed_at=None,
            published_unknown_at=None,
            terminal_at=None,
            updated_at=_NOW,
            version=0,
        )
        return row


@pytest.mark.unit
def test_production_source_has_no_raw_projection_or_recorder_capability() -> None:
    violations: dict[str, list[int]] = {}
    for source in _SOURCE_ROOT.rglob("*.py"):
        if source == _COMPOSITION_PATH:
            continue
        found = _raw_projection_bypass_calls(source)
        if found:
            violations[str(source.relative_to(_SOURCE_ROOT))] = found
    assert violations == {}


@pytest.mark.unit
def test_rsd_ingress_verifies_then_projects_every_signed_causal_pin() -> None:
    async def scenario() -> None:
        pool = _RecordingPool()

        async def factory() -> _RecordingPool:
            return pool

        ingress = build_rsd_verified_first_effect_grant_ingress(
            pool_factory=factory, expected_output_pin=_EXPECTED_OUTPUT_PIN
        )
        assert not hasattr(first_effect_ledger, "_PostgresVerifiedFirstEffectRecorder")
        assert not hasattr(
            first_effect_ledger, "_ModelVerifiedFirstEffectGrantProjection"
        )
        assert not hasattr(
            first_effect_ledger, "ProtocolVerifiedFirstEffectGrantRecorder"
        )
        assert not hasattr(pool, "verify_and_record")
        recorded = await ingress.verify_and_record(json.dumps(_valid_wire()), now=_NOW)
        material = _valid_wire()["authorization_material"]
        assert type(material) is dict
        lifecycle = material["grant"]
        assert type(lifecycle) is dict
        pins = lifecycle["pins"]
        assert type(pins) is dict
        posture = lifecycle["posture"]
        assert type(posture) is dict
        assert recorded.state is EnumVerifiedFirstEffectGrantState.VERIFIED
        assert recorded.authorization_digest == _valid_wire()["authorization_digest"]
        assert str(recorded.grant_id) == lifecycle["grant_id"]
        assert str(recorded.grant_envelope_id) == lifecycle["envelope_id"]
        assert recorded.nonce_digest == lifecycle["nonce_sha256"]
        assert recorded.request_digest == pins["request_sha256"]
        assert str(recorded.correlation_id) == lifecycle["correlation_id"]
        assert recorded.tenant_id == lifecycle["tenant_id"]
        assert recorded.backend_id == lifecycle["backend_id"]
        assert recorded.rendered_contract_sha256 == pins["rendered_contract_sha256"]
        assert (
            recorded.issuer_key_fingerprint_sha256
            == material["issuer_key_fingerprint_sha256"]
        )
        assert recorded.retry_disposition == posture["retry_disposition"]
        assert recorded.expected_output_topic == lifecycle["expected_output_topic"]
        assert (
            recorded.expected_output_event_class
            == lifecycle["expected_output_event_class"]
        )
        assert (
            recorded.expected_output_event_index
            == lifecycle["expected_output_event_index"]
        )

    asyncio.run(scenario())


@pytest.mark.unit
def test_rsd_ingress_rejects_every_hostile_public_vector_before_authority_recording() -> (
    None
):
    async def scenario() -> None:
        vectors = signed_executable_grant_v2_vectors()
        base = vectors["base_wire"]
        cases = vectors["vectors"]
        assert type(base) is dict and type(cases) is list
        for case in cases:
            assert type(case) is dict
            expected = case["expect"]
            assert type(expected) is dict
            if expected["stage"] == "verified":
                continue
            pool = _RecordingPool()

            async def factory(pool: _RecordingPool = pool) -> _RecordingPool:
                return pool

            ingress = build_rsd_verified_first_effect_grant_ingress(
                pool_factory=factory, expected_output_pin=_EXPECTED_OUTPUT_PIN
            )
            wire = _wire_for_vector(case, base)
            with pytest.raises(FirstEffectGrantIngressRejectedError):
                await ingress.verify_and_record(
                    json.dumps(wire), now=_verification_now(case, vectors)
                )
            assert pool.record_count == 0

    asyncio.run(scenario())


@pytest.mark.unit
def test_rsd_ingress_rejects_a_constructed_bad_signature_without_recording() -> None:
    async def scenario() -> None:
        pool = _RecordingPool()

        async def factory() -> _RecordingPool:
            return pool

        wire = _valid_wire()
        signature = wire["signature_octets"]
        assert type(signature) is list
        assert all(type(value) is int for value in signature)
        octets = cast("list[int]", signature)
        wire["signature_octets"] = [octets[0] ^ 1, *octets[1:]]
        ingress = build_rsd_verified_first_effect_grant_ingress(
            pool_factory=factory, expected_output_pin=_EXPECTED_OUTPUT_PIN
        )
        with pytest.raises(FirstEffectGrantIngressRejectedError):
            await ingress.verify_and_record(json.dumps(wire), now=_NOW)
        assert pool.record_count == 0

    asyncio.run(scenario())


@pytest.mark.unit
def test_rsd_ingress_rejects_an_unsupported_wire_type_without_recording() -> None:
    async def scenario() -> None:
        pool = _RecordingPool()

        async def factory() -> _RecordingPool:
            return pool

        ingress = build_rsd_verified_first_effect_grant_ingress(
            pool_factory=factory, expected_output_pin=_EXPECTED_OUTPUT_PIN
        )
        with pytest.raises(FirstEffectGrantIngressRejectedError, match="unsupported"):
            await ingress.verify_and_record(object(), now=_NOW)
        assert pool.record_count == 0

    asyncio.run(scenario())


@pytest.mark.unit
def test_rsd_ingress_rejects_a_signed_output_pin_that_contradicts_deployment() -> None:
    async def scenario() -> None:
        pool = _RecordingPool()

        async def factory() -> _RecordingPool:
            return pool

        ingress = build_rsd_verified_first_effect_grant_ingress(
            pool_factory=factory,
            expected_output_pin=ModelExpectedOutputPin(
                topic="events.public.grant.denied.v2",
                event_class="ModelPublicGrantDenied",
                index=0,
            ),
        )
        with pytest.raises(FirstEffectGrantIngressRejectedError, match="deployment"):
            await ingress.verify_and_record(json.dumps(_valid_wire()), now=_NOW)
        assert pool.record_count == 0

    asyncio.run(scenario())


@pytest.mark.unit
def test_rsd_ingress_closes_only_an_explicitly_owned_pool() -> None:
    async def scenario() -> None:
        shared = _RecordingPool()
        owned = _RecordingPool()

        async def shared_factory() -> _RecordingPool:
            return shared

        async def owned_factory() -> _RecordingPool:
            return owned

        shared_ingress = build_rsd_verified_first_effect_grant_ingress(
            pool_factory=shared_factory, expected_output_pin=_EXPECTED_OUTPUT_PIN
        )
        owned_ingress = build_rsd_verified_first_effect_grant_ingress(
            pool_factory=owned_factory,
            expected_output_pin=_EXPECTED_OUTPUT_PIN,
            owns_pool=True,
        )
        await shared_ingress.verify_and_record(json.dumps(_valid_wire()), now=_NOW)
        await owned_ingress.verify_and_record(json.dumps(_valid_wire()), now=_NOW)
        await shared_ingress.close()
        await owned_ingress.close()
        assert not shared.closed
        assert owned.closed

    asyncio.run(scenario())
