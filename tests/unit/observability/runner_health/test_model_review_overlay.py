# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static guards for the inactive model-review runner overlay."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[4]
OVERLAY = REPO_ROOT / "docker" / "docker-compose.model-review-canary.yml"
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
OVERRIDES = REPO_ROOT / "docker" / "compose-overrides.list"
DEPLOY = REPO_ROOT / "scripts" / "deploy-runners.sh"
HEALTHCHECK = REPO_ROOT / "docker" / "runners" / "model-review-healthcheck.sh"


def test_overlay_is_one_runner_and_inactive_by_default() -> None:
    overlay = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_COMPOSE.read_text(encoding="utf-8"))
    assert set(overlay["services"]) == {"omninode-runner-1"}
    assert set(overlay["services"]) <= set(base["services"])

    service = overlay["services"]["omninode-runner-1"]
    env = service["environment"]
    assert (
        env["MODEL_REVIEW_CAPABILITY_ACTIVE"] == "${MODEL_REVIEW_CAPABILITY_ACTIVE:-0}"
    )
    assert env["MODEL_REVIEW_CONFIG_ACTIVE"] == "${MODEL_REVIEW_CONFIG_ACTIVE:-0}"
    assert "MODEL_REVIEW_RUNNER_LABELS" in env["RUNNER_LABELS"]
    assert "self-hosted,omnibase-ci,linux,x64" in env["RUNNER_LABELS"]
    assert "model-review-healthcheck.sh" in " ".join(service["volumes"])
    assert (
        "healthcheck.sh && /usr/local/bin/model-review-healthcheck.sh"
        in service["healthcheck"]["test"][1]
    )


def test_overlay_is_preserved_by_repair_and_deployment_sync() -> None:
    overrides = {
        line.split("#", 1)[0].strip()
        for line in OVERRIDES.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }
    assert "docker-compose.model-review-canary.yml" in overrides

    deploy = DEPLOY.read_text(encoding="utf-8")
    for path in (
        "docker/compose-overrides.list",
        "docker/docker-compose.model-review-canary.yml",
        "docker/runners/model-review-healthcheck.sh",
    ):
        assert f'"{path}"' in deploy
    assert (
        "-f ${RUNNER_HOST_DIR}/docker/docker-compose.model-review-canary.yml" in deploy
    )

    monitor = (REPO_ROOT / "docker" / "runners" / "runner-monitor.sh").read_text(
        encoding="utf-8"
    )
    assert "COMPOSE_OVERRIDES_LIST" in monitor
    assert "COMPOSE_FILE_ARGS" in monitor


def test_missing_required_overlay_blocks_before_any_compose_recreate() -> None:
    monitor = (REPO_ROOT / "docker" / "runners" / "runner-monitor.sh").read_text(
        encoding="utf-8"
    )
    assert "REQUIRED_MODEL_REVIEW_OVERLAY" in monitor
    assert "REQUIRED_OVERRIDE_MISSING=true" in monitor
    missing_check = monitor.index('if [[ "${REQUIRED_OVERRIDE_MISSING}" == true ]]')
    compose_config = monitor.index('docker compose "${COMPOSE_FILE_ARGS[@]}" config -q')
    assert missing_check < compose_config
    repair_guard = monitor.index('if [[ "${COMPOSE_INTERPOLATION_OK}" != true ]]')
    repair_call = monitor.index(
        'docker compose "${COMPOSE_FILE_ARGS[@]}" up -d --force-recreate'
    )
    assert repair_guard < repair_call


def test_healthcheck_is_shell_valid_and_contains_no_topology_literals() -> None:
    text = HEALTHCHECK.read_text(encoding="utf-8")
    assert ".env" not in text
    assert "192.168." not in text
    assert "tailnet" not in text.lower()


def test_shell_reference_projection_matches_typed_contract() -> None:
    fleet = yaml.safe_load(
        (REPO_ROOT / "config" / "runner_fleet.yaml").read_text(encoding="utf-8")
    )
    refs = fleet["model_review"]
    healthcheck = HEALTHCHECK.read_text(encoding="utf-8")
    for key in (
        "credential_reference_id",
        "endpoint_reference_id",
        "healthcheck_reference_id",
    ):
        assert refs[key] in healthcheck


def test_healthcheck_stays_inert_without_candidate_activation(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_REVIEW_CAPABILITY_ACTIVE": "0",
            "MODEL_REVIEW_OBSERVATION_PATH": str(tmp_path / "missing.json"),
        }
    )
    result = subprocess.run(
        ["bash", str(HEALTHCHECK)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0


def test_healthcheck_rejects_stale_candidate_observation(tmp_path: Path) -> None:
    observed_at = datetime.now(UTC) - timedelta(seconds=301)
    observation = {
        "provenance": "runner-local-model-review-preflight",
        "attestation_id": "f3df2b7f-e8d5-41f0-8d6b-df65ea5c8ae4",
        "attestation_verified": True,
        "observed_at": observed_at.isoformat(),
        "reviewer_cli_available": True,
        "present_reference_ids": [
            "dc9565c8-7f13-46dc-bd89-9694c13e1d2f",
            "b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7",
            "2672472a-bac9-4344-8c8c-79da6cb604ae",
        ],
        "healthy_reference_ids": ["2672472a-bac9-4344-8c8c-79da6cb604ae"],
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "MODEL_REVIEW_CAPABILITY_ACTIVE": "1",
            "MODEL_REVIEW_CONFIG_ACTIVE": "1",
            "MODEL_REVIEW_PREFLIGHT_VERIFIED": "1",
            "RUNNER_GROUP": "omnibase-ci",
            "RUNNER_LABELS": "self-hosted,omnibase-ci,model-review,linux,x64",
            "MODEL_REVIEW_OBSERVATION_PATH": str(observation_path),
        }
    )
    result = subprocess.run(
        ["bash", str(HEALTHCHECK)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "sanctioned live attestation verifier" in result.stderr


def test_active_healthcheck_has_no_verifier_bypass_in_json_or_environment() -> None:
    healthcheck = HEALTHCHECK.read_text(encoding="utf-8")
    assert "MODEL_REVIEW_PREFLIGHT_VERIFIED" not in healthcheck
    assert "attestation_verified" not in healthcheck
    assert "/usr/local/bin/model-review-attestation-verifier" in healthcheck
