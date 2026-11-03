# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev boot guard never pairs a pre-fold image with the post-fold overlay (OMN-21288).

A fold changes the dev overlay and retires the folded containers in the same
deploy. A boot-good snapshot of the pre-fold family restored under the fold
overlay would run a main that carries none of the folded roles' contracts while
their containers are gone, so projections would silently stop. The guard
refuses that pairing; the way back across the fold is a rollback by revision,
which moves the image and the overlay together.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Scope
from deploy_agent.executor import DeployExecutor

from tests.unit import test_dev_boot_guard_omn21279 as boot_tests

guard_mod = executor_mod.load_dev_boot_guard()
# The shared Docker double of the boot guard tests, requested by name.
docker = boot_tests.docker

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]
REPO_ROOT = Path(__file__).resolve().parents[4]
PRE_FOLD = ""  # a revision whose dev overlay folds nothing into main
FOLDED_SERVICES = {"tenant-projection-writer"}


def _policy(monkeypatch: pytest.MonkeyPatch, folded: str) -> None:
    """Read this checkout's fold overlay with the policy of the revision on disk."""
    monkeypatch.setattr(
        executor_mod, "RUNTIME_POLICY_ENV_FILE", REPO_ROOT / "docker/runtime-policy.env"
    )
    monkeypatch.setattr(
        executor_mod,
        "_load_runtime_policy_env",
        lambda: {"DEV_RUNTIME_MAIN_FOLDED_PROFILES": folded},
    )


def _committed_fold() -> str:
    for line in (REPO_ROOT / "docker/runtime-policy.env").read_text().splitlines():
        if line.startswith("DEV_RUNTIME_MAIN_FOLDED_PROFILES="):
            return line.split("=", 1)[1]
    raise AssertionError("committed policy declares no dev folded profiles")


def test_planted_pre_fold_boot_good_image_is_refused_before_anything_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    docker: dict,
) -> None:
    _policy(monkeypatch, PRE_FOLD)
    executor_mod.DevBootGuard(
        DeployExecutor(), tmp_path
    ).prepare()  # boot-good, pre-fold
    # The deploy root is now at the fold and the live family is not healthy,
    # so the stored boot-good snapshot is the only candidate.
    _policy(monkeypatch, _committed_fold())
    docker["containers"] = boot_tests._containers(healthy=False)
    with pytest.raises(guard_mod.DevBootFailedError, match="pair its image"):
        executor_mod.DevBootGuard(DeployExecutor(), tmp_path).prepare()
    assert not any("up" in argv for argv in docker["commands"])


def test_failed_fold_deploy_never_restores_the_pre_fold_image_under_the_fold_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    docker: dict,
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    _policy(monkeypatch, PRE_FOLD)
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    _policy(monkeypatch, _committed_fold())  # git_pull brought the fold overlay
    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("fold-job", "b" * 40, Mock()):
            docker["containers"] = boot_tests._containers(healthy=False)
            raise RuntimeError("compose up failed")
    assert not any("up" in argv for argv in docker["commands"])
    incident = json.loads(next(tmp_path.rglob("incident-*.json")).read_text())
    assert incident["recovery_verified"] is False
    assert "pair its image" in incident["recovery_error"]
    assert incident["restored_revision"] == boot_tests.GOOD_COMMIT
    executor.verify.assert_not_called()


def test_a_snapshot_of_the_same_fold_generation_still_restores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    docker: dict,
) -> None:
    """Positive control: the refusal is about the fold, not about every restore."""
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    _policy(monkeypatch, _committed_fold())
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("same-fold", "c" * 40, Mock()):
            docker["containers"] = boot_tests._containers(healthy=False)
            raise RuntimeError("compose up failed")
    assert any("up" in argv for argv in docker["commands"])
    executor.verify.assert_called_once()


def test_rollback_by_revision_returns_the_folded_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deploying the pre-fold revision (its overlay and image together) starts the folded services again."""
    _policy(monkeypatch, _committed_fold())
    folded = executor_mod._requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
    )
    assert not FOLDED_SERVICES & set(folded)
    disabled = executor_mod.lane_config_for(EnumRuntimeLane.DEV).disabled_services
    assert disabled >= FOLDED_SERVICES
    _policy(monkeypatch, PRE_FOLD)
    rolled_back = executor_mod._requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
    )
    assert set(rolled_back) >= FOLDED_SERVICES
    disabled = executor_mod.lane_config_for(EnumRuntimeLane.DEV).disabled_services
    assert not FOLDED_SERVICES & disabled
