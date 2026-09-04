# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the sanitized model-review capability evidence canary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.ci.model_review_capability_canary import (
    _CanaryFixture,
    build_sanitized_evidence,
)

REPO_ROOT = Path(__file__).parents[4]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "model_review_capability_canary.json"


def test_canary_emits_only_allowlisted_sanitized_evidence() -> None:
    fixture = _CanaryFixture.model_validate_json(FIXTURE.read_text(encoding="utf-8"))

    evidence = build_sanitized_evidence(
        fixture,
        run_id="12345",
        now=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    )

    assert evidence == {
        "schema": "model-review-capability-evidence/v1",
        "run_id": "12345",
        "readiness": "not_ready",
        "execution_scope": "offline_contract_fixture",
        "selection": "not_observed",
        "runner_group": "omnibase-ci",
        "runner_labels": ["linux", "model-review", "omnibase-ci", "self-hosted", "x64"],
        "opaque_reference_ids": [
            "2672472a-bac9-4344-8c8c-79da6cb604ae",
            "b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7",
            "dc9565c8-7f13-46dc-bd89-9694c13e1d2f",
        ],
        "observation_provenance": "runner-local-model-review-preflight",
        "attestation": "unverified",
        "verdict_completion": "not_run",
        "canary_kind": "contract_fixture",
    }
    assert "secret" not in json.dumps(evidence).lower()
    assert "endpoint" not in json.dumps(evidence).lower()


def test_canary_rejects_unallowlisted_runner_label() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["runner_labels"].append("candidate-runner-name")
    fixture = _CanaryFixture.model_validate(raw)

    with pytest.raises(ValueError, match="runner label"):
        build_sanitized_evidence(
            fixture,
            run_id="12345",
            now=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        )


def test_canary_rejects_live_verdict_fields_in_offline_fixture() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["review_legs"] = {"first": "complete", "second": "complete"}

    with pytest.raises(ValueError, match="extra"):
        _CanaryFixture.model_validate(raw)


def test_canary_rejects_stale_observation() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["observation_age_seconds"] = 301
    fixture = _CanaryFixture.model_validate(raw)

    with pytest.raises(ValueError, match="observation_stale"):
        build_sanitized_evidence(
            fixture,
            run_id="12345",
            now=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        )


def test_canary_rejects_partial_reference_health() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["healthy_reference_ids"] = raw["healthy_reference_ids"][:-1]
    fixture = _CanaryFixture.model_validate(raw)

    with pytest.raises(ValueError, match="every contract reference"):
        build_sanitized_evidence(
            fixture,
            run_id="12345",
            now=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        )


def test_canary_rejects_unsanitized_run_id() -> None:
    fixture = _CanaryFixture.model_validate_json(FIXTURE.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="run_id"):
        build_sanitized_evidence(
            fixture,
            run_id="runner/secret-token",
            now=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        )
