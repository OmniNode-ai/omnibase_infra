#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the private, sanitized model-review capability contract canary.

The default fixture mode exercises the typed contract without contacting a
runner, model endpoint, secret store, or event bus. A later operator can run
the same evidence shape from the candidate runner; this tool deliberately
prints only non-sensitive readiness evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.collect_model_review_capability import (
    ModelModelReviewReferenceProbe,
    collect_model_review_capability_observation,
)
from omnibase_infra.observability.runner_health.enum_model_review_capability_failure import (
    EnumModelReviewCapabilityFailure,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_observation import (
    MODEL_REVIEW_OBSERVATION_PROVENANCE,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from omnibase_infra.observability.runner_health.preflight_model_review_capability import (
    preflight_model_review_capability,
)


class _CanaryFixture(BaseModel):
    """Sanitized facts used by the contract-only canary fixture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runner_labels: frozenset[str]
    runner_groups: frozenset[str]
    present_reference_ids: frozenset[UUID]
    healthy_reference_ids: frozenset[UUID]
    observation_age_seconds: int = Field(default=0, ge=0)
    reviewer_cli_available: Literal[True]


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_LABELS = frozenset({"self-hosted", "omnibase-ci", "linux", "x64", "model-review"})


def build_sanitized_evidence(
    fixture: _CanaryFixture,
    *,
    run_id: str,
    now: datetime,
) -> dict[str, object]:
    """Validate an offline fixture and return a non-live evidence projection."""
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    fleet_config = load_runner_fleet_config()
    if fleet_config.model_review is None:
        raise ValueError("model_review capability config is absent")
    config = fleet_config.model_review.model_copy(update={"active": True})
    if fixture.runner_groups != {config.runner_group}:
        raise ValueError("fixture runner group does not match the contract")
    if not fixture.runner_labels <= _SAFE_LABELS:
        raise ValueError("fixture contains an unallowlisted runner label")
    required_reference_ids = {
        config.credential_reference_id,
        config.endpoint_reference_id,
        config.healthcheck_reference_id,
    }
    if fixture.present_reference_ids != required_reference_ids:
        raise ValueError("fixture reference set does not match the contract")
    if fixture.healthy_reference_ids != required_reference_ids:
        raise ValueError("fixture must assert health for every contract reference")

    observed_at = now - timedelta(seconds=fixture.observation_age_seconds)
    # This exercises the same collector/preflight contract as the runner-local
    # probe, but fixture facts are not an attestation source. Its output is
    # explicitly marked offline and cannot claim runner selection or review
    # execution.
    observation = collect_model_review_capability_observation(
        config,
        runner_labels=fixture.runner_labels,
        runner_groups=fixture.runner_groups,
        probe_reference=lambda reference_id: ModelModelReviewReferenceProbe(
            present=reference_id in fixture.present_reference_ids,
            healthy=reference_id in fixture.healthy_reference_ids,
        ),
        probe_reviewer_cli=lambda: fixture.reviewer_cli_available,
        now=observed_at,
    )
    preflight = preflight_model_review_capability(config, observation, now=now)
    unexpected_failures = set(preflight.failures) - {
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE
    }
    if unexpected_failures:
        raise ValueError(
            "model-review capability preflight failed: "
            + ",".join(failure.value for failure in preflight.failures)
        )
    return {
        "schema": "model-review-capability-evidence/v1",
        "run_id": run_id,
        "readiness": "not_ready",
        "execution_scope": "offline_contract_fixture",
        "selection": "not_observed",
        "runner_group": config.runner_group,
        "runner_labels": sorted(fixture.runner_labels),
        "opaque_reference_ids": sorted(
            str(reference) for reference in required_reference_ids
        ),
        "observation_provenance": MODEL_REVIEW_OBSERVATION_PROVENANCE,
        "attestation": "unverified",
        "verdict_completion": "not_run",
        "canary_kind": "contract_fixture",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args()
    fixture = _CanaryFixture.model_validate_json(
        args.fixture.read_text(encoding="utf-8")
    )
    evidence = build_sanitized_evidence(
        fixture,
        run_id=args.run_id,
        now=datetime.now(UTC),
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
