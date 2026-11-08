# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A declared branch pin keeps the dev deploy agent's code on a branch (OMN-23039).

THE FAILURE
-----------
``self_update`` fetched ``origin/<DEPLOY_AGENT_TRACKING_REF>`` and rebuilt the clone
onto it at every idle heartbeat. A fix to the agent's own code (the boot guard, the
executor) therefore could not run before its merge: put the clone on the branch and
the next heartbeat put it back on ``dev``. omnibase_infra#5772 could not be shown
working on the h201 dev agent for that reason, and its hand-off was refused for want
of an installed revision.

THE ROUTE
---------
``deploy_agent.code_ref_pin`` registers a record in the agent's state directory that
names the branch whose code the agent loads, with an expiry. ``self_update`` follows
that branch while the record is live and returns to the declared tracking ref when
it is released or expires. The lane's tracking ref (``DEPLOY_AGENT_TRACKING_REF``)
is untouched: it still names the sibling-repo fallback and the ref fence.

These tests drive ``self_update`` over real git clones (a bare origin, the agent's
clone), because the defect is what git does to the clone, not which commands ran.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor, load_dev_boot_guard

pytestmark = pytest.mark.unit

# The agent loads its host effects from its own checkout by file path (OMN-21279), so the
# tests load the pin handler and model the same way.
code_ref_pin = load_dev_boot_guard("handler_dev_lane_code_ref_pin")
ModelCodeRefPin = sys.modules[
    "omnibase_infra.models.model_dev_lane_code_ref_pin"
].ModelDevLaneCodeRefPin
_RUNTIME_PATH = (
    Path(code_ref_pin.__file__).resolve().parents[1]
    / "nodes/node_dev_lane_code_ref_pin_effect/runtime_code_ref_pin.py"
)
_spec = importlib.util.spec_from_file_location(
    "omnibase_infra.nodes.node_dev_lane_code_ref_pin_effect.runtime_code_ref_pin",
    _RUNTIME_PATH,
)
assert _spec is not None and _spec.loader is not None
runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runtime)
ENV_STATE_DIR = code_ref_pin.ENV_STATE_DIR
PIN_FILE_NAME = code_ref_pin.PIN_FILE_NAME


def load_active_pin(state_dir: Path, *, now: datetime):
    return code_ref_pin.read_active_pin(state_dir, now)


_BOUNDARY = EnumSelfUpdateBoundary.POST_TERMINAL
_NOW = datetime(2026, 11, 8, 0, 0, tzinfo=UTC)
_PINNED_BRANCH = "omn-22930-dev-effects-fold"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(cwd),
        },
    )
    return result.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "--quiet", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


class _Fixture:
    """A bare origin with ``dev`` and a branch ahead of it, and the agent's clone on ``dev``."""

    def __init__(self, root: Path) -> None:
        self.origin = root / "origin.git"
        seed = root / "seed"
        self.clone = root / "agent-clone"
        self.state = root / "state"
        self.state.mkdir()
        subprocess.run(
            ["git", "init", "--quiet", "--bare", "-b", "dev", str(self.origin)],
            check=True,
        )
        subprocess.run(
            ["git", "init", "--quiet", "-b", "dev", str(seed)],
            check=True,
        )
        _git(seed, "remote", "add", "origin", str(self.origin))
        self.dev_sha = _commit(seed, "base.txt")
        _git(seed, "push", "--quiet", "origin", "dev")
        _git(seed, "checkout", "--quiet", "-b", _PINNED_BRANCH)
        self.branch_sha = _commit(seed, "fix.txt")
        _git(seed, "push", "--quiet", "origin", _PINNED_BRANCH)
        subprocess.run(
            ["git", "clone", "--quiet", "-b", "dev", str(self.origin), str(self.clone)],
            check=True,
        )

    def clone_head(self) -> str:
        return _git(self.clone, "rev-parse", "HEAD")

    def clone_branch(self) -> str:
        return _git(self.clone, "symbolic-ref", "--short", "HEAD")

    def pin(self, *, expires_in: timedelta = timedelta(hours=48)) -> ModelCodeRefPin:
        record = ModelCodeRefPin(
            branch=_PINNED_BRANCH,
            head=self.branch_sha,
            pr=5772,
            ticket="OMN-23039",
            registered_by="lane-under-test",
            registered_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + expires_in,
        )
        (self.state / PIN_FILE_NAME).write_text(
            record.model_dump_json(), encoding="utf-8"
        )
        return record


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    fixture = _Fixture(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(fixture.clone))
    monkeypatch.setenv(ENV_STATE_DIR, str(fixture.state))
    return fixture


def _heartbeat(declare_loaded_code_sha, loaded: str) -> list[object]:
    """One idle self-update, with only the process image and the dependency sync stubbed."""
    declare_loaded_code_sha(loaded)
    executor = DeployExecutor()
    with (
        patch("os.execv") as execv,
        patch("deploy_agent.executor._uv_sync_after_pull"),
        patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=None),
    ):
        executor.self_update(boundary=_BOUNDARY)
    return list(execv.call_args_list)


class TestPinnedBranchIsLoadedAndHeld:
    def test_pinned_branch_is_loaded_then_not_reset_to_dev_at_the_next_heartbeat(
        self, world: _Fixture, declare_loaded_code_sha
    ) -> None:
        world.pin()

        first = _heartbeat(declare_loaded_code_sha, world.dev_sha)
        assert world.clone_head() == world.branch_sha
        assert world.clone_branch() == _PINNED_BRANCH
        assert len(first) == 1, "the agent re-execs onto the pinned branch's code"

        # The next idle heartbeat: the process now runs the branch code.
        second = _heartbeat(declare_loaded_code_sha, world.branch_sha)
        assert world.clone_head() == world.branch_sha, "the clone was reset to dev"
        assert world.clone_branch() == _PINNED_BRANCH
        assert second == [], "nothing to re-exec: the pinned code is the loaded code"

    def test_a_branch_that_moves_forward_is_followed(
        self, world: _Fixture, declare_loaded_code_sha
    ) -> None:
        world.pin()
        _heartbeat(declare_loaded_code_sha, world.dev_sha)

        seed = world.clone.parent / "seed"
        _git(seed, "checkout", "--quiet", _PINNED_BRANCH)
        newer = _commit(seed, "fix2.txt")
        _git(seed, "push", "--quiet", "origin", _PINNED_BRANCH)

        _heartbeat(declare_loaded_code_sha, world.branch_sha)
        assert world.clone_head() == newer

    def test_control_without_a_pin_the_clone_returns_to_dev(
        self, world: _Fixture, declare_loaded_code_sha
    ) -> None:
        """The failing control: the same clone parked on the branch is reset to dev."""
        _git(world.clone, "checkout", "--quiet", "-B", _PINNED_BRANCH, world.branch_sha)

        _heartbeat(declare_loaded_code_sha, world.branch_sha)

        assert world.clone_head() == world.dev_sha
        assert world.clone_branch() == "dev"

    def test_an_expired_pin_returns_the_clone_to_dev(
        self, world: _Fixture, declare_loaded_code_sha
    ) -> None:
        world.pin()
        _heartbeat(declare_loaded_code_sha, world.dev_sha)
        record = json.loads((world.state / PIN_FILE_NAME).read_text(encoding="utf-8"))
        record["expires_at"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        (world.state / PIN_FILE_NAME).write_text(json.dumps(record), encoding="utf-8")

        _heartbeat(declare_loaded_code_sha, world.branch_sha)

        assert world.clone_head() == world.dev_sha
        assert world.clone_branch() == "dev"

    def test_a_released_pin_returns_the_clone_to_dev(
        self, world: _Fixture, declare_loaded_code_sha, capsys: pytest.CaptureFixture
    ) -> None:
        world.pin()
        _heartbeat(declare_loaded_code_sha, world.dev_sha)

        assert runtime.main(["release", "--state-dir", str(world.state)]) == 0
        assert not (world.state / PIN_FILE_NAME).exists()

        _heartbeat(declare_loaded_code_sha, world.branch_sha)
        assert world.clone_head() == world.dev_sha

    def test_an_unreadable_pin_is_ignored_loudly_not_followed(
        self,
        world: _Fixture,
        declare_loaded_code_sha,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (world.state / PIN_FILE_NAME).write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            _heartbeat(declare_loaded_code_sha, world.dev_sha)
        assert world.clone_branch() == "dev"
        assert "code_ref_pin_unreadable" in caplog.text

    def test_the_lane_tracking_ref_is_not_replaced_by_the_pin(
        self, world: _Fixture, declare_loaded_code_sha, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Builds fall back to the declared tracking ref for siblings, never the pinned branch."""
        from deploy_agent.tracking_ref import load_tracking_ref_from_env

        world.pin()
        _heartbeat(declare_loaded_code_sha, world.dev_sha)
        assert load_tracking_ref_from_env() == "dev"


class TestPinRecord:
    def _fields(self, **override: object) -> dict[str, object]:
        base: dict[str, object] = {
            "branch": _PINNED_BRANCH,
            "head": "a" * 40,
            "pr": 5772,
            "ticket": "OMN-23039",
            "registered_by": "oneruntime-t8-2340",
            "registered_at": _NOW,
            "expires_at": _NOW + timedelta(hours=48),
        }
        base.update(override)
        return base

    def test_a_valid_record_round_trips(self) -> None:
        record = ModelCodeRefPin(**self._fields())
        assert ModelCodeRefPin.model_validate_json(record.model_dump_json()) == record

    @pytest.mark.parametrize("branch", ["origin/dev", "main", "master", "", "dev"])
    def test_a_branch_the_agent_must_not_follow_is_refused(self, branch: str) -> None:
        with pytest.raises(ValueError):
            ModelCodeRefPin(**self._fields(branch=branch))

    def test_a_head_that_is_not_a_full_sha_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelCodeRefPin(**self._fields(head="58b95d2b"))

    def test_a_ticket_that_is_not_an_omn_id_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelCodeRefPin(**self._fields(ticket="23039"))

    def test_an_expiry_more_than_seven_days_out_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelCodeRefPin(**self._fields(expires_at=_NOW + timedelta(days=8)))

    def test_an_expiry_not_after_registration_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelCodeRefPin(**self._fields(expires_at=_NOW))

    def test_extra_fields_are_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelCodeRefPin(**self._fields(unexpected="x"))


class TestLoadActivePin:
    def test_no_state_dir_declared_means_no_branch_change(self) -> None:
        assert code_ref_pin.followed_branch("dev", {}, "post_terminal") == "dev"

    def test_missing_file_means_no_pin(self, tmp_path: Path) -> None:
        assert load_active_pin(tmp_path, now=_NOW) is None

    def test_expired_record_means_no_pin(self, tmp_path: Path) -> None:
        record = ModelCodeRefPin(
            branch=_PINNED_BRANCH,
            head="b" * 40,
            pr=5772,
            ticket="OMN-23039",
            registered_by="lane",
            registered_at=_NOW - timedelta(hours=3),
            expires_at=_NOW - timedelta(hours=1),
        )
        (tmp_path / PIN_FILE_NAME).write_text(record.model_dump_json())
        assert load_active_pin(tmp_path, now=_NOW) is None

    def test_live_record_is_returned(self, tmp_path: Path) -> None:
        record = ModelCodeRefPin(
            branch=_PINNED_BRANCH,
            head="b" * 40,
            pr=5772,
            ticket="OMN-23039",
            registered_by="lane",
            registered_at=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        )
        (tmp_path / PIN_FILE_NAME).write_text(record.model_dump_json())
        assert load_active_pin(tmp_path, now=_NOW) == record


class TestRegisterCommand:
    def _argv(self, state: Path, **override: str) -> list[str]:
        values = {
            "--branch": _PINNED_BRANCH,
            "--head": "c" * 40,
            "--pr": "5772",
            "--ticket": "OMN-23039",
            "--registered-by": "oneruntime-t8-2340",
            "--expires-in-hours": "48",
        }
        values.update(override)
        flat = [part for pair in values.items() for part in pair]
        return ["register", "--state-dir", str(state), *flat]

    def test_register_writes_a_record_the_agent_reads(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert runtime.main(self._argv(tmp_path)) == 0
        pin = load_active_pin(tmp_path, now=datetime.now(UTC))
        assert pin is not None
        assert pin.branch == _PINNED_BRANCH
        assert pin.pr == 5772
        assert "code-ref-pin registered" in capsys.readouterr().out

    def test_register_refuses_a_bad_record_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit):
            runtime.main(self._argv(tmp_path, **{"--branch": "origin/dev"}))
        assert not (tmp_path / PIN_FILE_NAME).exists()

    def test_show_reads_the_live_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        runtime.main(self._argv(tmp_path))
        capsys.readouterr()
        assert runtime.main(["show", "--state-dir", str(tmp_path)]) == 0
        assert _PINNED_BRANCH in capsys.readouterr().out

    def test_show_with_no_record_exits_nonzero(self, tmp_path: Path) -> None:
        assert runtime.main(["show", "--state-dir", str(tmp_path)]) == 1
