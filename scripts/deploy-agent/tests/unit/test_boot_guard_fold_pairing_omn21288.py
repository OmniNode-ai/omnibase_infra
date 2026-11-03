# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev boot guard never pairs a pre-fold image with the post-fold overlay (OMN-21288).

The five-writer fold changes the dev overlay and retires five containers in
the same deploy. A boot-good snapshot of the pre-fold family restored under the
fold overlay would run a main that carries none of the writers' contracts while
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
PRE_FOLD = "tenant-projection"
WRITERS = {
    "projection-live-events-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-tenant-registry-writer",
}


def _policy(monkeypatch: pytest.MonkeyPatch, folded: str) -> None:
    """Read this checkout's fold overlay with the policy of the revision on disk."""
    monkeypatch.setattr(
        executor_mod, "RUNTIME_POLICY_ENV_FILE", REPO_ROOT / "docker/runtime-policy.env"
    )
    policy = executor_mod._load_dotenv_file(REPO_ROOT / "docker/runtime-policy.env")
    monkeypatch.setattr(
        executor_mod,
        "_load_runtime_policy_env",
        lambda: {**policy, "DEV_RUNTIME_MAIN_FOLDED_PROFILES": folded},
    )


def _committed_fold() -> str:
    for line in (REPO_ROOT / "docker/runtime-policy.env").read_text().splitlines():
        if line.startswith("DEV_RUNTIME_MAIN_FOLDED_PROFILES="):
            return line.split("=", 1)[1]
    raise AssertionError("committed policy declares no dev folded profiles")


def test_a_forward_deploy_across_the_fold_is_not_refused_by_a_pre_fold_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    docker: dict,
) -> None:
    """The deadlock measured on dev 2026-11-03: every fold deploy refused at prepare.

    The deploy root is at the fold and the live family is not healthy, so the
    stored pre-fold boot-good snapshot is the only candidate. Refusing here left
    no forward deploy and no rollback able to repair the lane; the snapshot is
    held, the deploy proceeds, and nothing is replaced by prepare itself.
    """
    _policy(monkeypatch, PRE_FOLD)
    executor_mod.DevBootGuard(
        DeployExecutor(), tmp_path
    ).prepare()  # boot-good, pre-fold
    _policy(monkeypatch, _committed_fold())
    docker["containers"] = boot_tests._containers(healthy=False)
    guard = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    guard.prepare()
    assert guard.point.current is not None
    assert guard.point.current.folded_profiles == (PRE_FOLD,)
    assert not any("up" in argv for argv in docker["commands"])


def test_a_legacy_snapshot_without_the_fold_field_does_not_block_the_deploy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    docker: dict,
) -> None:
    """The live snapshot on dev predates folded_profiles and reads as zero folds."""
    _policy(monkeypatch, PRE_FOLD)
    guard = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    guard.prepare()
    legacy = json.loads(guard.snapshot_path.read_text())
    del legacy["folded_profiles"]
    guard.snapshot_path.chmod(0o600)
    guard.snapshot_path.write_text(json.dumps(legacy))
    _policy(monkeypatch, _committed_fold())
    docker["containers"] = boot_tests._containers(healthy=False)
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


def test_rollback_by_revision_returns_the_five_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deploying the pre-fold revision (its overlay and image together) starts the five again."""
    _policy(monkeypatch, _committed_fold())
    folded = executor_mod._requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
    )
    assert not WRITERS & set(folded)
    disabled = executor_mod.lane_config_for(EnumRuntimeLane.DEV).disabled_services
    assert disabled >= WRITERS
    _policy(monkeypatch, PRE_FOLD)
    rolled_back = executor_mod._requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
    )
    assert set(rolled_back) >= WRITERS
    disabled = executor_mod.lane_config_for(EnumRuntimeLane.DEV).disabled_services
    assert not WRITERS & disabled
    # Nothing in the rolled-back list is retired by the pre-fold overlay.
    assert not WRITERS & executor_mod._folded_dev_services(EnumRuntimeLane.DEV)
