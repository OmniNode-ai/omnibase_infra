# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18930 (K3 of OMN-18925): delegation runs are compared only inside one cohort.

Two delegation runs are comparable only when every dimension that can change
their outcome is equal: the prompt, the resolved task type, the response
contract, the lane, the runtime build, the consumer that actually handled the
command, the first inference hop, the provider policy, the effective deadline
and the declared retry bounds. Before this ticket the evidence recorded some of
those facts and not others, so a comparison could quietly mix builds, policies
or consumers and report the difference as a model or routing effect.

The key model is ``ModelDelegationCohortKey``. The assertions here are the
ticket's falsifiers and the plan's K3 row (E11 completion mapping):

* removing ANY dimension refuses construction; a response contract is either an
  explicit hash or an explicit ``None``, never omitted;
* changing the build or the provider policy refuses the comparison as
  cross-cohort, and the refusal names the dimension;
* a caller lane label cannot stand in for the typed consumer identity;
* the retained ``.201`` partition-0/offset-489 capture is an INCOMPLETE-key
  negative control: it is refused, and the refusal names every dimension the
  capture never proved. Nothing is filled in from current state;
* a key is assembled only from a terminal receipt plus a runtime readback that
  names the same correlation and lane and was taken within the plan's 300 s
  freshness bound of the terminal.

The model and the offset-489 fixture were first written by a Codex session in
omnibase_infra#3951 (branch codex/omn-18925-capture-port-integration, commit
76d817671); this file carries them forward and adds the typed build identity,
the explicit policy sources and the assembler.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from omnibase_infra.models.delegation import (
    ModelDelegationBuildIdentity,
    ModelDelegationCohortKey,
    ModelDelegationFirstInferenceIdentity,
    ModelDelegationProviderPolicy,
    ModelDelegationRetryBounds,
    ModelDelegationTierRetryBound,
)
from omnibase_infra.models.model_node_identity import ModelNodeIdentity

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATE = _REPO_ROOT / "scripts" / "validate_delegation_cohort_keys.py"
_ASSEMBLE = _REPO_ROOT / "scripts" / "assemble_delegation_cohort_key.py"
_OFFSET489 = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "delegation"
    / "offset489_incomplete_cohort_key.json"
)

_PROMPT_SHA256 = "a" * 64
_RESPONSE_CONTRACT_SHA256 = "c" * 64
_OFFSET489_TERMINAL_SHA256 = (
    "e9cf02241b4cdc303ae66cf661835409cd8b17c5a7dc2cfe39a0f66b8645f9c4"
)


def _policy(**overrides: str | None) -> ModelDelegationProviderPolicy:
    fields: dict[str, str | None] = {
        "routing_tiers_sha256": "b" * 64,
        "backend_config_sha256": "d" * 64,
        "task_class_contracts_sha256": "e" * 64,
        "overlay_sha256": None,
    }
    fields.update(overrides)
    return ModelDelegationProviderPolicy.model_validate(fields)


def _build(**overrides: str) -> ModelDelegationBuildIdentity:
    fields = {
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "image_digest": "sha256:" + "1" * 64,
        "build_provenance_sha256": "2" * 64,
    }
    fields.update(overrides)
    return ModelDelegationBuildIdentity.model_validate(fields)


def _synthetic_complete_key() -> ModelDelegationCohortKey:
    """A synthetic complete key. It is not live-lane evidence."""
    return ModelDelegationCohortKey(
        prompt_sha256=_PROMPT_SHA256,
        resolved_task_type="document",
        response_contract_sha256=_RESPONSE_CONTRACT_SHA256,
        lane="isolated-lab",
        build_identity=_build(),
        consumer_identity=ModelNodeIdentity(
            env="isolated-lab",
            service="omnimarket",
            node_name="node_delegate_skill_orchestrator",
            version="1.3.0",
        ),
        first_hop_identity=ModelDelegationFirstInferenceIdentity(
            backend="test-backend",
            model="test-model",
            tier="local",
            provider="test-provider",
        ),
        provider_policy=_policy(),
        deadline_seconds=240.0,
        retry_bounds=ModelDelegationRetryBounds(
            per_tier=(
                ModelDelegationTierRetryBound(tier="local", max_retries=1),
                ModelDelegationTierRetryBound(tier="remote", max_retries=2),
            ),
            max_escalations=3,
        ),
    )


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        check=False,
        text=True,
    )


# --------------------------------------------------------------------------
# Construction: every dimension is required
# --------------------------------------------------------------------------


def test_complete_synthetic_key_is_stable_and_matches_itself() -> None:
    key = _synthetic_complete_key()

    key.require_same_cohort(_synthetic_complete_key())

    assert key.key_sha256 == _synthetic_complete_key().key_sha256
    assert key.changed_dimensions(_synthetic_complete_key()) == ()


def test_the_key_declares_exactly_the_ticket_dimensions() -> None:
    assert set(ModelDelegationCohortKey.model_fields) == {
        "prompt_sha256",
        "resolved_task_type",
        "response_contract_sha256",
        "lane",
        "build_identity",
        "consumer_identity",
        "first_hop_identity",
        "provider_policy",
        "deadline_seconds",
        "retry_bounds",
    }


def test_every_required_dimension_must_be_present() -> None:
    complete = _synthetic_complete_key().model_dump(mode="python")

    for field_name in ModelDelegationCohortKey.model_fields:
        incomplete = deepcopy(complete)
        incomplete.pop(field_name)
        with pytest.raises(ValidationError, match=field_name):
            ModelDelegationCohortKey.model_validate(incomplete)


@pytest.mark.parametrize(
    ("sub_model", "field_name"),
    [
        ("build_identity", "source_revision"),
        ("build_identity", "image_digest"),
        ("build_identity", "build_provenance_sha256"),
        ("provider_policy", "routing_tiers_sha256"),
        ("provider_policy", "backend_config_sha256"),
        ("provider_policy", "task_class_contracts_sha256"),
        ("provider_policy", "overlay_sha256"),
        ("first_hop_identity", "provider"),
        ("retry_bounds", "max_escalations"),
    ],
)
def test_every_nested_dimension_must_be_present(
    sub_model: str, field_name: str
) -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields[sub_model].pop(field_name)

    with pytest.raises(ValidationError, match=field_name):
        ModelDelegationCohortKey.model_validate(fields)


def test_absent_response_contract_is_explicit_not_omitted() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["response_contract_sha256"] = None
    key = ModelDelegationCohortKey.model_validate(fields)

    assert key.response_contract_sha256 is None
    fields.pop("response_contract_sha256")
    with pytest.raises(ValidationError):
        ModelDelegationCohortKey.model_validate(fields)


def test_lane_label_cannot_substitute_for_typed_consumer_identity() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["consumer_identity"] = fields["lane"]

    with pytest.raises(ValidationError, match="consumer_identity"):
        ModelDelegationCohortKey.model_validate(fields)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("source_revision", "main"),
        ("source_revision", "0123456"),
        ("source_revision", "0123456789ABCDEF0123456789ABCDEF01234567"),
        ("image_digest", "1" * 64),
        ("image_digest", "omnibase-infra-runtime-effects:latest"),
        ("build_provenance_sha256", "unknown"),
    ],
)
def test_build_identity_refuses_labels_that_are_not_content_identities(
    field_name: str, bad_value: str
) -> None:
    fields = _build().model_dump(mode="python")
    fields[field_name] = bad_value

    with pytest.raises(ValidationError, match=field_name):
        ModelDelegationBuildIdentity.model_validate(fields)


def test_invalid_digest_and_unbounded_retry_values_are_rejected() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["prompt_sha256"] = "not-a-digest"

    with pytest.raises(ValidationError, match="prompt_sha256"):
        ModelDelegationCohortKey.model_validate(fields)

    with pytest.raises(ValidationError):
        ModelDelegationRetryBounds(
            per_tier=(ModelDelegationTierRetryBound(tier="local", max_retries=-1),),
            max_escalations=0,
        )
    with pytest.raises(ValidationError):
        ModelDelegationRetryBounds(
            per_tier=(ModelDelegationTierRetryBound(tier="local", max_retries=0),),
            max_escalations=-1,
        )
    with pytest.raises(ValidationError, match="unique"):
        ModelDelegationRetryBounds(
            per_tier=(
                ModelDelegationTierRetryBound(tier="local", max_retries=0),
                ModelDelegationTierRetryBound(tier="local", max_retries=1),
            ),
            max_escalations=0,
        )
    with pytest.raises(ValidationError):
        _policy(routing_tiers_sha256="Z" * 64)
    with pytest.raises(ValidationError):
        _policy(overlay_sha256="not-a-digest")


def test_zero_or_negative_deadline_is_refused() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    for bad in (0.0, -1.0):
        fields["deadline_seconds"] = bad
        with pytest.raises(ValidationError, match="deadline_seconds"):
            ModelDelegationCohortKey.model_validate(fields)


# --------------------------------------------------------------------------
# Comparison: any differing dimension is cross-cohort
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("build_identity", _build(source_revision="f" * 40)),
        ("build_identity", _build(image_digest="sha256:" + "9" * 64)),
        ("build_identity", _build(build_provenance_sha256="8" * 64)),
        ("provider_policy", _policy(routing_tiers_sha256="7" * 64)),
        ("provider_policy", _policy(backend_config_sha256="7" * 64)),
        ("provider_policy", _policy(task_class_contracts_sha256="7" * 64)),
        ("provider_policy", _policy(overlay_sha256="7" * 64)),
    ],
)
def test_build_or_provider_policy_change_rejects_comparison(
    field_name: str, changed_value: object
) -> None:
    baseline = _synthetic_complete_key()
    changed = baseline.model_copy(update={field_name: changed_value})

    assert baseline.changed_dimensions(changed) == (field_name,)
    with pytest.raises(ValueError, match=field_name):
        baseline.require_same_cohort(changed)


def test_changed_dimensions_names_every_difference_and_nothing_else() -> None:
    baseline = _synthetic_complete_key()
    changed = baseline.model_copy(
        update={
            "deadline_seconds": 120.0,
            "lane": "other-lab",
            "first_hop_identity": baseline.first_hop_identity.model_copy(
                update={"backend": "backend-second"}
            ),
        }
    )

    assert baseline.changed_dimensions(changed) == (
        "lane",
        "first_hop_identity",
        "deadline_seconds",
    )
    assert baseline.key_sha256 != changed.key_sha256


def test_policy_hash_is_canonical_and_retry_order_does_not_change_the_key() -> None:
    policy = _policy()
    reordered = ModelDelegationProviderPolicy.model_validate(
        dict(reversed(list(policy.model_dump(mode="python").items())))
    )
    assert policy.sha256 == reordered.sha256

    bounds = _synthetic_complete_key().retry_bounds
    reversed_key = _synthetic_complete_key().model_copy(
        update={
            "retry_bounds": ModelDelegationRetryBounds(
                per_tier=tuple(reversed(bounds.per_tier)), max_escalations=3
            )
        }
    )
    assert _synthetic_complete_key().key_sha256 == reversed_key.key_sha256
    assert bounds.model_copy(update={"max_escalations": 4}) != bounds


# --------------------------------------------------------------------------
# The offset-489 capture is an incomplete-key negative control
# --------------------------------------------------------------------------


def test_offset489_capture_is_an_incomplete_key_negative_control() -> None:
    fixture = json.loads(_OFFSET489.read_text(encoding="utf-8"))

    assert fixture["evidence"]["terminal_envelope_sha256"] == (
        _OFFSET489_TERMINAL_SHA256
    )
    assert fixture["evidence"]["partition"] == 0
    assert fixture["evidence"]["topic_offset"] == 489
    with pytest.raises(ValidationError) as error:
        ModelDelegationCohortKey.model_validate(fixture["observed_key_fields"])

    assert set(fixture["unproven_key_dimensions"]) == {
        "build_identity",
        "consumer_identity",
        "provider_policy",
        "deadline_seconds",
        "retry_bounds",
    }
    # The refusal names exactly the dimensions the capture never proved: the
    # ones it did prove (prompt, task type, explicit no-contract, the first
    # hop) validate, so nothing about them is in the error.
    refused = {str(item["loc"][0]) for item in error.value.errors()}
    assert refused == set(fixture["unproven_key_dimensions"])


def test_validator_cli_refuses_the_offset489_capture() -> None:
    result = _run(_VALIDATE, str(_OFFSET489))

    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith("INVALID_COHORT_KEY: ")
    for field_name in ("build_identity", "consumer_identity", "provider_policy"):
        assert field_name in result.stderr


# --------------------------------------------------------------------------
# The captured-key entry point
# --------------------------------------------------------------------------


def test_validator_cli_accepts_one_complete_key(tmp_path: Path) -> None:
    key = _synthetic_complete_key()
    path = tmp_path / "key.json"
    path.write_text(key.model_dump_json(), encoding="utf-8")

    result = _run(_VALIDATE, str(path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"VALID_COHORT_KEY: {key.key_sha256}\n"


def test_validator_cli_rejects_cross_cohort(tmp_path: Path) -> None:
    baseline = _synthetic_complete_key()
    candidate = baseline.model_copy(
        update={"provider_policy": _policy(routing_tiers_sha256="7" * 64)}
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    result = _run(_VALIDATE, str(baseline_path), str(candidate_path))

    assert result.returncode == 1
    assert result.stdout == "CROSS_COHORT: provider_policy\n"


def test_validator_cli_accepts_same_cohort_inside_a_capture_record(
    tmp_path: Path,
) -> None:
    key = _synthetic_complete_key()
    record = {"cohort_key": key.model_dump(mode="json"), "run_id": "one"}
    other = {"cohort_key": key.model_dump(mode="json"), "run_id": "two"}
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(record), encoding="utf-8")
    b.write_text(json.dumps(other), encoding="utf-8")

    result = _run(_VALIDATE, str(a), str(b))

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"SAME_COHORT: {key.key_sha256}\n"


# --------------------------------------------------------------------------
# Assembly: a key comes only from a terminal plus a contemporaneous readback
# --------------------------------------------------------------------------

_CORRELATION = "ca153777-68fa-4889-8fd6-49ed52f1b8b7"
_TERMINAL_AT = datetime(2026, 9, 24, 6, 23, 7, tzinfo=UTC)
_PROMPT = "Classify this changelog entry and summarize it in one sentence."


def _receipt(**payload_overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correlation_id": _CORRELATION,
        "prompt_text": _PROMPT,
        "task_type": "summarization",
        "response_contract_evidence": {
            "contract_sha256": _RESPONSE_CONTRACT_SHA256,
            "conveyed": True,
        },
        "attempts": [
            {"backend_id": "backend-1", "model_id": "model-1", "tier": "local"}
        ],
        "budget_evidence": {"execution_timeout_seconds": 240},
    }
    payload.update(payload_overrides)
    return {
        "lane": "dev",
        "correlation_id": _CORRELATION,
        "receipt": {
            "result": {
                "terminal_payload": {
                    "correlation_id": _CORRELATION,
                    "envelope_timestamp": _TERMINAL_AT.isoformat(),
                    "payload": payload,
                }
            }
        },
    }


def _readback(**overrides: Any) -> dict[str, Any]:
    readback: dict[str, Any] = {
        "correlation_id": _CORRELATION,
        "lane": "dev",
        "captured_at": (_TERMINAL_AT + timedelta(seconds=30)).isoformat(),
        "consumer_identity": {
            "env": "local",
            "service": "omnimarket",
            "node_name": "node_delegate_skill_orchestrator",
            "version": "1.3.0",
        },
        "build_identity": _build().model_dump(mode="json"),
        "provider_policy": _policy().model_dump(mode="json"),
        "retry_bounds": {
            "per_tier": [{"tier": "local", "max_retries": 1}],
            "max_escalations": 2,
        },
        "first_hop_provider": None,
    }
    readback.update(overrides)
    return readback


def _assemble(
    tmp_path: Path, receipt: dict[str, Any], readback: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    receipt_path = tmp_path / "receipt.json"
    readback_path = tmp_path / "readback.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    readback_path.write_text(json.dumps(readback), encoding="utf-8")
    return _run(
        _ASSEMBLE,
        "--receipt",
        str(receipt_path),
        "--runtime-readback",
        str(readback_path),
    )


def test_assembler_builds_a_complete_key_from_terminal_and_readback(
    tmp_path: Path,
) -> None:
    result = _assemble(tmp_path, _receipt(), _readback())

    assert result.returncode == 0, result.stderr
    key = ModelDelegationCohortKey.model_validate_json(result.stdout)
    assert key.prompt_sha256 == hashlib.sha256(_PROMPT.encode()).hexdigest()
    assert key.resolved_task_type == "summarization"
    assert key.response_contract_sha256 == _RESPONSE_CONTRACT_SHA256
    assert key.lane == "dev"
    assert key.deadline_seconds == 240.0
    assert key.first_hop_identity == ModelDelegationFirstInferenceIdentity(
        backend="backend-1", model="model-1", tier="local", provider=None
    )
    assert key.build_identity == _build()
    assert key.consumer_identity.node_name == "node_delegate_skill_orchestrator"


def test_assembler_keeps_an_explicit_no_contract_as_none(tmp_path: Path) -> None:
    receipt = _receipt(
        response_contract_evidence={"contract_sha256": None, "conveyed": False}
    )

    result = _assemble(tmp_path, receipt, _readback())

    assert result.returncode == 0, result.stderr
    key = ModelDelegationCohortKey.model_validate_json(result.stdout)
    assert key.response_contract_sha256 is None


@pytest.mark.parametrize(
    ("receipt_overrides", "readback_overrides", "reason"),
    [
        ({"response_contract_evidence": None}, {}, "response_contract_evidence"),
        ({"attempts": []}, {}, "attempts"),
        ({"budget_evidence": None}, {}, "execution_timeout_seconds"),
        ({}, {"correlation_id": "0" * 8}, "correlation_id"),
        ({}, {"lane": "stability-test"}, "lane"),
        (
            {},
            {"captured_at": (_TERMINAL_AT + timedelta(seconds=301)).isoformat()},
            "captured_at",
        ),
        (
            {},
            {"captured_at": (_TERMINAL_AT - timedelta(seconds=301)).isoformat()},
            "captured_at",
        ),
        ({}, {"consumer_identity": "dev"}, "consumer_identity"),
        ({}, {"build_identity": None}, "build_identity"),
        ({}, {"provider_policy": None}, "provider_policy"),
        ({}, {"retry_bounds": None}, "retry_bounds"),
    ],
)
def test_assembler_refuses_incomplete_or_mismatched_evidence(
    tmp_path: Path,
    receipt_overrides: dict[str, Any],
    readback_overrides: dict[str, Any],
    reason: str,
) -> None:
    result = _assemble(
        tmp_path, _receipt(**receipt_overrides), _readback(**readback_overrides)
    )

    assert result.returncode == 2, result.stdout
    assert result.stdout == ""
    assert result.stderr.startswith("COHORT_KEY_REFUSED: ")
    assert reason in result.stderr


def test_assembler_refuses_a_malformed_retry_tier_without_a_traceback(
    tmp_path: Path,
) -> None:
    readback = _readback(
        retry_bounds={
            "per_tier": [
                {"tier": None, "max_retries": 1},
                {"tier": "a", "max_retries": 1},
            ],
            "max_escalations": 1,
        }
    )

    result = _assemble(tmp_path, _receipt(), readback)

    assert result.returncode == 2, result.stdout
    assert result.stderr.startswith("COHORT_KEY_REFUSED: ")
    assert "Traceback" not in result.stderr


def test_validator_cli_refuses_a_key_file_that_is_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "key.json"
    path.write_bytes(b"\xff\xfe\x00not json")

    result = _run(_VALIDATE, str(path))

    assert result.returncode == 2
    assert result.stderr.startswith("INVALID_COHORT_KEY: ")
    assert "Traceback" not in result.stderr
