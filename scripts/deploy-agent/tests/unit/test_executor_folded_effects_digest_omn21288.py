# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Folded effects health and stability digests resolve to the main runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import (
    _LANE_CONFIGS,
    DeployExecutor,
    ProdStabilityDigestMissingError,
    assert_prod_request_has_stability_digest,
    lane_config_for,
    lane_folded_profiles,
)

pytestmark = pytest.mark.unit
DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64
FOLDED_KEY = "STABILITY_TEST_RUNTIME_MAIN_FOLDED_PROFILES"
# The lane's own unfolded health targets name its containers and ports, so no
# lane container name is restated here.
(MAIN, MAIN_PORT), (EFFECTS, EFFECTS_PORT) = _LANE_CONFIGS[
    EnumRuntimeLane.STABILITY_TEST
].runtime_health_targets
PROD_EFFECTS = _LANE_CONFIGS[EnumRuntimeLane.PROD].runtime_health_targets[1][0]


@pytest.fixture
def inspected(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    containers: list[str] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        assert cmd[:2] == ["docker", "inspect"]
        containers.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout=DIGEST + "\n", stderr="")

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {FOLDED_KEY: "effects"},
    )
    return containers


@pytest.mark.parametrize(
    ("profiles", "container"),
    [
        ("effects", MAIN),
        ("", EFFECTS),
    ],
)
def test_stability_effects_digest_uses_fold_policy(
    monkeypatch: pytest.MonkeyPatch, inspected: list[str], profiles: str, container: str
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env", lambda: {FOLDED_KEY: profiles}
    )
    assert DeployExecutor().resolve_stability_ready_digest("runtime-effects") == DIGEST
    assert inspected == [container]


def test_folded_health_targets_keep_both_ports(inspected: list[str]) -> None:
    assert lane_config_for(EnumRuntimeLane.STABILITY_TEST).runtime_health_targets == (
        (MAIN, MAIN_PORT),
        (MAIN, EFFECTS_PORT),
    )


def test_prod_effects_digest_keeps_own_container(inspected: list[str]) -> None:
    DeployExecutor().verify_running_image_digest(
        lane=EnumRuntimeLane.PROD, expected_digest=DIGEST, service="runtime-effects"
    )
    assert inspected == [PROD_EFFECTS]


def test_folded_stability_digest_still_rejects_mismatch(inspected: list[str]) -> None:
    stability_digest = DeployExecutor().resolve_stability_ready_digest(
        "runtime-effects"
    )
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        services=["runtime-effects"],
        image_digest=OTHER_DIGEST,
    )
    with pytest.raises(ProdStabilityDigestMissingError, match="digest"):
        assert_prod_request_has_stability_digest(
            cmd, stability_ready_digest=stability_digest
        )
    assert inspected == [MAIN]


def test_policy_normalization_and_missing_key() -> None:
    assert lane_folded_profiles(
        EnumRuntimeLane.STABILITY_TEST, {FOLDED_KEY: " Effects, , WORKER, effects "}
    ) == frozenset({"effects", "worker"})
    assert lane_folded_profiles(EnumRuntimeLane.STABILITY_TEST, {}) == frozenset()
    assert (
        lane_folded_profiles(EnumRuntimeLane.DEV, {FOLDED_KEY: "effects"})
        == frozenset()
    )


def test_unfolded_config_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deploy_agent.executor._load_runtime_policy_env", dict)
    first = lane_config_for(EnumRuntimeLane.STABILITY_TEST)
    assert lane_config_for(EnumRuntimeLane.STABILITY_TEST) is first
    assert first.runtime_health_targets[1] == (EFFECTS, EFFECTS_PORT)


def test_missing_policy_file_folds_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", tmp_path / "missing.env"
    )
    assert lane_folded_profiles(EnumRuntimeLane.STABILITY_TEST) == frozenset()
