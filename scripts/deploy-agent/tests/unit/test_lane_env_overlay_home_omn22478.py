# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overlay home's runtime_lane_env overlay as a second source of a lane's env (OMN-22634).

The deploy agent reads the dev lane's env from two places: the deployment fragment (OMN-20867) and
the overlay home's ``overlays/runtime_lane_env/overlay.yaml``, read from the clone its unit names in
``DEPLOY_AGENT_OVERLAY_HOME_CLONE`` at ``origin/main``. Both go through the same allowlist, and a key
the two set differently is refused.

Every lane, host and value here is synthetic: the real ones live in the private overlay and the
routing table, never in a test.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.instance_lanes import (
    LANE_ENV_FRAGMENT_DIR,
    LANE_ENV_FRAGMENT_REF,
    LANE_ENV_FRAGMENT_SUFFIX,
    LaneEnvFragmentError,
    parse_lane_env_fragment,
)

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

# The standalone agent loads this module by path; the tests load it the same way.
overlay_home = executor_mod.load_dev_boot_guard(
    "nodes.node_setup_local_provision_effect.runtime_lane_env_overlay"
)
# Where and how the overlay is read is the node contract's declaration, not code.
DECLARED = overlay_home.declaration(executor_mod)
ENV_OVERLAY_HOME_CLONE = str(DECLARED["clone_env"])
OVERLAY_HOME_LANE_ENV_REF = str(DECLARED["ref"])
OVERLAY_HOME_LANE_ENV_RELPATH = str(DECLARED["overlay_path"])
OverlayHomeLaneEnvError = overlay_home.OverlayHomeLaneEnvError
# What refuses a malformed overlay: the document's own shape, or the fragment parser's allowlist.
REFUSALS = (OverlayHomeLaneEnvError, LaneEnvFragmentError)


def overlay_home_clone(environ: Mapping[str, str]) -> Path | None:
    clone: Path | None = overlay_home.overlay_home_clone(environ, DECLARED)
    return clone


def parse_home(text: str, *, lane_id: str, source: str) -> dict[str, str]:
    env: dict[str, str] = overlay_home.parse_overlay_home_lane_env(
        text, lane_id=lane_id, source=source, parse_fragment=parse_lane_env_fragment
    )
    return env


def read_home(clone: Path, lane_id: str) -> dict[str, str]:
    env: dict[str, str] = overlay_home.read_overlay_home_lane_env(
        clone, lane_id, declared=DECLARED, parse_fragment=parse_lane_env_fragment
    )
    return env


SCHEMA = "runtime_lane_env.v1"
LANE = "compose-lane-a"
SOURCE = {
    "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": "web-front-end",
    "COST_ESTIMATOR_WEB_SOURCE_URL": "https://example.invalid/web-front-end.git",
}
MODE = {"ONEX_PR_LANDING_GITHUB_MODE": "dry_run"}
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
DEV_UNIT = SCRIPTS_DIR / "deploy" / "deploy-agent-dev.service"


def _document(env: Mapping[str, object], lane: str = LANE, **overrides: object) -> str:
    body: dict[str, object] = {
        "schema_version": SCHEMA,
        "lanes": {lane: {"description": "synthetic", "env": env}},
    }
    body.update(overrides)
    return yaml.safe_dump(body)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )


def _clone(root: Path, files: dict[str, str], ref: str) -> Path:
    """A git repo whose ``ref`` (``origin/<branch>``) holds ``files``."""
    repo = root / "clone"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "README").write_text("clone\n", encoding="utf-8")
    for relpath, text in files.items():
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "clone")
    _git(repo, "update-ref", f"refs/remotes/{ref}", "HEAD")
    return repo


def _home(tmp_path: Path, env: Mapping[str, object], **kwargs: object) -> Path:
    return _clone(
        tmp_path / "home",
        {OVERLAY_HOME_LANE_ENV_RELPATH: _document(env, **kwargs)},
        OVERLAY_HOME_LANE_ENV_REF,
    )


def _fragment_clone(tmp_path: Path, env: Mapping[str, object]) -> Path:
    relpath = f"{LANE_ENV_FRAGMENT_DIR}/{LANE}{LANE_ENV_FRAGMENT_SUFFIX}"
    text = json.dumps(
        {"schema_version": SCHEMA, "lane_id": LANE, "description": "", "env": env}
    )
    return _clone(tmp_path / "fragment", {relpath: text}, LANE_ENV_FRAGMENT_REF)


# -- parsing the document ---------------------------------------------------


@pytest.mark.unit
def test_golden_a_lane_gets_exactly_its_env() -> None:
    other = _document({**SOURCE}, lane="compose-lane-b")
    assert parse_home(other, lane_id=LANE, source="t") == {}
    assert parse_home(_document(SOURCE), lane_id=LANE, source="t") == SOURCE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (_document({"POSTGRES_PASSWORD": "x"}), "POSTGRES_PASSWORD"),
        (_document({**SOURCE, "GITHUB_TOKEN": "x"}), "GITHUB_TOKEN"),
        (_document({"ONEX_PR_LANDING_GITHUB_MODE": ""}), "empty"),
        (_document({"ONEX_PR_LANDING_GITHUB_MODE": "a\nb"}), "line"),
        (_document({"ONEX_PR_LANDING_GITHUB_MODE": False}), "env"),
        (_document(MODE, schema_version="runtime_lane_env.v0"), "schema_version"),
        (_document(MODE, lane="other", schema_version="runtime_lane_env.v0"), "schema"),
        (_document(MODE, extra="x"), "extra"),
        ("lanes: [", "YAML"),
        ("- not a mapping", "document"),
    ],
)
def test_error_a_malformed_overlay_is_refused(text: str, reason: str) -> None:
    with pytest.raises(REFUSALS, match=reason):
        parse_home(text, lane_id=LANE, source="t")


# -- reading the overlay home's clone ---------------------------------------


@pytest.mark.unit
def test_golden_the_overlay_at_the_ref_is_read(tmp_path: Path) -> None:
    clone = _home(tmp_path, SOURCE)
    assert read_home(clone, LANE) == SOURCE


@pytest.mark.unit
def test_golden_no_overlay_at_the_ref_changes_nothing(tmp_path: Path) -> None:
    clone = _clone(tmp_path / "home", {}, OVERLAY_HOME_LANE_ENV_REF)
    assert read_home(clone, LANE) == {}


@pytest.mark.unit
def test_the_overlay_is_read_at_the_ref_not_the_working_tree(tmp_path: Path) -> None:
    clone = _clone(tmp_path / "home", {}, OVERLAY_HOME_LANE_ENV_REF)
    working = clone / OVERLAY_HOME_LANE_ENV_RELPATH
    working.parent.mkdir(parents=True)
    working.write_text(_document(SOURCE), encoding="utf-8")
    assert read_home(clone, LANE) == {}


@pytest.mark.unit
def test_the_overlay_is_read_at_origin_main_not_origin_dev(tmp_path: Path) -> None:
    clone = _clone(
        tmp_path / "home",
        {OVERLAY_HOME_LANE_ENV_RELPATH: _document(SOURCE)},
        LANE_ENV_FRAGMENT_REF,
    )
    assert OVERLAY_HOME_LANE_ENV_REF == "origin/main"
    with pytest.raises(OverlayHomeLaneEnvError, match="origin/main"):
        read_home(clone, LANE)


@pytest.mark.unit
def test_error_a_directory_that_is_not_a_clone_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(OverlayHomeLaneEnvError, match="origin/main"):
        read_home(plain, LANE)


@pytest.mark.unit
def test_error_a_clone_the_unit_names_but_is_absent_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OverlayHomeLaneEnvError, match="not a directory"):
        read_home(tmp_path / "absent", LANE)


@pytest.mark.unit
def test_the_variable_names_no_clone_unless_set() -> None:
    assert overlay_home_clone({}) is None
    assert overlay_home_clone({ENV_OVERLAY_HOME_CLONE: "  "}) is None
    assert overlay_home_clone({ENV_OVERLAY_HOME_CLONE: "/srv/x"}) == Path("/srv/x")


@pytest.mark.unit
def test_error_a_relative_clone_path_is_refused() -> None:
    with pytest.raises(OverlayHomeLaneEnvError, match="absolute"):
        overlay_home_clone({ENV_OVERLAY_HOME_CLONE: "overlays"})


# -- merging the two sources ------------------------------------------------


@pytest.mark.unit
def test_golden_disjoint_sources_merge_and_equal_values_are_accepted() -> None:
    assert overlay_home.merge_lane_env_sources(MODE, SOURCE) == {**MODE, **SOURCE}
    assert overlay_home.merge_lane_env_sources(SOURCE, SOURCE) == SOURCE
    assert overlay_home.merge_lane_env_sources({}, SOURCE) == SOURCE
    assert overlay_home.merge_lane_env_sources(SOURCE, {}) == SOURCE


@pytest.mark.unit
def test_error_a_key_set_differently_in_both_is_refused_by_name_only() -> None:
    key = "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY"
    with pytest.raises(OverlayHomeLaneEnvError) as excinfo:
        overlay_home.merge_lane_env_sources({key: "first-value"}, {key: "second-value"})
    assert key in str(excinfo.value)
    assert "first-value" not in str(excinfo.value)
    assert "second-value" not in str(excinfo.value)


# -- the deploy agent's compose env -----------------------------------------


@pytest.fixture
def dev_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dev-lane agent whose deployment fragment holds a non-source key."""
    fragment = _fragment_clone(tmp_path, MODE)
    monkeypatch.setattr(
        executor_mod, "REPO_DIR", str(Path(__file__).resolve().parents[4])
    )
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(tmp_path))
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", fragment)
    monkeypatch.setattr(
        executor_mod,
        "DEV_INSTANCE_RUNTIME_LANES",
        {executor_mod.active_dev_instance(): LANE},
    )
    monkeypatch.delenv(ENV_OVERLAY_HOME_CLONE, raising=False)
    # Nothing the host's own operator store or process environment holds may reach these tests.
    monkeypatch.setattr(executor_mod, "operator_store_overlay", dict)
    monkeypatch.setattr(executor_mod, "_load_runtime_policy_env", dict)
    for key in (
        *MODE,
        *SOURCE,
        "ESTIMATOR_GATE_PASSWORD",
        "FEATURE_ESTIMATOR_GATE_ENABLED",
        "COST_ESTIMATOR_WEB_IMAGE",
    ):
        monkeypatch.delenv(key, raising=False)
    return fragment


@pytest.mark.unit
def test_golden_compose_env_carries_the_overlay_home_settings(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    without = executor_mod._compose_env()
    assert not set(SOURCE) & set(without)
    assert without["ONEX_PR_LANDING_GITHUB_MODE"] == "dry_run"
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    env = executor_mod._compose_env()
    assert {k: env[k] for k in SOURCE} == SOURCE
    assert env["ONEX_PR_LANDING_GITHUB_MODE"] == "dry_run"


@pytest.mark.unit
def test_golden_the_same_value_in_both_sources_is_accepted(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    both = _fragment_clone(tmp_path / "both", {**MODE, **SOURCE})
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", both)
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    env = executor_mod._compose_env()
    assert {k: env[k] for k in SOURCE} == SOURCE


@pytest.mark.unit
def test_error_compose_env_refuses_a_key_the_sources_set_differently(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY"
    differing = _fragment_clone(tmp_path / "differing", {key: "another-front-end"})
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", differing)
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    with pytest.raises(OverlayHomeLaneEnvError, match=key):
        executor_mod._compose_env()


@pytest.mark.unit
def test_error_compose_env_refuses_an_overlay_key_outside_the_allowlist(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, {"GITHUB_TOKEN": "x"}))
    )
    with pytest.raises(LaneEnvFragmentError, match="GITHUB_TOKEN"):
        executor_mod._compose_env()


@pytest.mark.unit
def test_error_compose_env_refuses_an_unreadable_overlay_home(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(tmp_path / "absent"))
    with pytest.raises(OverlayHomeLaneEnvError, match="not a directory"):
        executor_mod._compose_env()


@pytest.mark.unit
def test_compose_env_of_a_non_dev_agent_reads_no_overlay_home(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", EnumRuntimeLane.STABILITY_TEST.value
    )
    assert not set(SOURCE) & set(executor_mod._compose_env())


@pytest.mark.unit
def test_compose_env_of_a_stability_job_reads_no_overlay_home(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", f"dev,{EnumRuntimeLane.STABILITY_TEST.value}"
    )
    with executor_mod.active_deploy_lane(EnumRuntimeLane.STABILITY_TEST):
        assert not set(SOURCE) & set(executor_mod._compose_env())
    with executor_mod.active_deploy_lane(EnumRuntimeLane.DEV):
        env = executor_mod._compose_env()
    assert {k: env[k] for k in SOURCE} == SOURCE


# -- the settings check is not weakened --------------------------------------


@pytest.fixture
def settings_guard(monkeypatch: pytest.MonkeyPatch) -> Any:
    # The instance serves the estimator page, whatever the shipped routing table says of a test's.
    monkeypatch.setattr(
        executor_mod,
        "lane_config_for",
        lambda lane: SimpleNamespace(disabled_services=(), compose_files=()),
    )
    return executor_mod.load_dev_boot_guard(
        "nodes.node_setup_local_provision_effect.runtime_cost_estimator_build"
    )


def _check(guard: Any) -> tuple[Any, dict[str, str]]:
    with executor_mod.active_deploy_lane(EnumRuntimeLane.DEV):
        return guard.dev_lane_build_settings(executor_mod, EnumRuntimeLane.DEV)


@pytest.mark.unit
def test_the_settings_check_refuses_only_for_the_password_with_the_overlay_home(
    dev_agent: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings_guard: Any,
) -> None:
    monkeypatch.delenv("ESTIMATOR_GATE_PASSWORD", raising=False)
    with pytest.raises(RuntimeError) as before:
        _check(settings_guard)
    assert "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY" in str(before.value)
    assert "COST_ESTIMATOR_WEB_SOURCE_URL" in str(before.value)
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    with pytest.raises(RuntimeError) as after:
        _check(settings_guard)
    message = str(after.value).split(";")[0]
    assert "ESTIMATOR_GATE_PASSWORD" in message
    assert "COST_ESTIMATOR_WEB_SOURCE" not in message


@pytest.mark.unit
def test_the_settings_check_passes_once_the_password_is_present(
    dev_agent: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings_guard: Any,
) -> None:
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    monkeypatch.setenv("ESTIMATOR_GATE_PASSWORD", "synthetic-not-a-secret")
    recipe, _env = _check(settings_guard)
    assert recipe.repository == SOURCE["COST_ESTIMATOR_WEB_SOURCE_REPOSITORY"]
    assert recipe.repository_url == SOURCE["COST_ESTIMATOR_WEB_SOURCE_URL"]


@pytest.mark.unit
def test_the_settings_check_still_refuses_an_invalid_gate_switch(
    dev_agent: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings_guard: Any,
) -> None:
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(_home(tmp_path, SOURCE)))
    monkeypatch.setenv("ESTIMATOR_GATE_PASSWORD", "synthetic-not-a-secret")
    monkeypatch.setenv("FEATURE_ESTIMATOR_GATE_ENABLED", "off")
    with pytest.raises(RuntimeError, match="FEATURE_ESTIMATOR_GATE_ENABLED"):
        _check(settings_guard)


# -- the unit ---------------------------------------------------------------


@pytest.mark.unit
def test_the_dev_unit_declares_and_protects_the_variable() -> None:
    text = DEV_UNIT.read_text(encoding="utf-8")
    declared = [
        line
        for line in text.splitlines()
        if line.startswith(f"Environment={ENV_OVERLAY_HOME_CLONE}=")
    ]
    assert len(declared) == 1
    protected_line = next(
        line
        for line in text.splitlines()
        if line.startswith('Environment="DEPLOY_AGENT_ENV_PROTECTED=')
    )
    protected = protected_line.split("=", 2)[2].rstrip('"').split()
    assert ENV_OVERLAY_HOME_CLONE in protected
