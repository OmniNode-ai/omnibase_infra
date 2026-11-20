# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane's env fragment from the private deployment overlay (OMN-20867).

The deploy agent's compose env (``executor._compose_env``) and the dev lane's
warm refresh (``scripts/runtime_build/refresh_dev_lane.sh``) overlay the
operator env store with a per-lane fragment read from the host's deployment
overlay clone, at the clone's ``origin/dev``. The fragment may set only the
keys its schema version declares; anything else is refused, loud.

Every lane and host name here is synthetic: the real ones live in the private
overlay and the routing table, never in a test.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.instance_lanes import (
    LANE_ENV_FRAGMENT_DIR,
    LANE_ENV_FRAGMENT_REF,
    LANE_ENV_FRAGMENT_SCHEMA_KEYS,
    LANE_ENV_FRAGMENT_SUFFIX,
    LaneEnvFragmentError,
    default_instance_lane_env_exports,
    discover_lane_overlay_clone,
    instance_runtime_lanes_from_text,
    lane_env_shell_exports,
    parse_lane_env_fragment,
    read_lane_env_fragment,
)
from deploy_agent.routing import RoutingTableError

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

SCHEMA = "runtime_lane_env.v1"
LANE = "compose-lane-a"
NON_ACTING = {
    "ONEX_PR_LANDING_GITHUB_MODE": "dry_run",
    "ONEX_PR_LANDING_ARM_ACTION_MODE": "report_only",
    "ONEX_CI_RED_TRIAGE_ACT_RUNNER": "false",
    "ONEX_CI_RED_TRIAGE_ACT_PR_OWN": "false",
    "ONEX_CI_RED_TRIAGE_ACT_SHARED_CAUSE": "false",
    "ONEX_CI_RED_TRIAGE_ACT_DEV_HEAD": "false",
}
# The lane class an instance's runtime borrows contracts from (OMN-20867):
# read by the runtime's auto-wiring ownership filter, set by the overlay.
LANE_CLASS = {
    "ONEX_RUNTIME_LANE_CLASS": "compose-lane",
    "ONEX_RUNTIME_LANE_CLASS_CONTRACTS": "node_a,node_b",
}

# The deployment's provenance overlay (OMN-23981, OMN-25116): the five variables
# the envelope publish seam stamps events from. It stamps when LANE, HOST,
# INSTALL_SOURCE and REVISION are all set; LANE_KIND is unset for lane dev or
# stability and one of mac_unit, lane_worktree, hand_run for lane other; any
# other LANE_KIND is refused and stamps nothing. A fragment may set all five.
PROVENANCE = {
    "ONEX_PROVENANCE_LANE": "other",
    "ONEX_PROVENANCE_LANE_KIND": "lane_worktree",
    "ONEX_PROVENANCE_HOST": "lab-host",
    "ONEX_PROVENANCE_INSTALL_SOURCE": "worktree",
    "ONEX_PROVENANCE_REVISION": "0123456789ab",
}
# A dev lane's fragment: the four required values, LANE_KIND left unset.
PROVENANCE_DEV = {
    "ONEX_PROVENANCE_LANE": "dev",
    "ONEX_PROVENANCE_HOST": "lab-host",
    "ONEX_PROVENANCE_INSTALL_SOURCE": "tag",
    "ONEX_PROVENANCE_REVISION": "0123456789ab",
}
# The values the stamp accepts, as literals because the agent's venv pins a
# published omnibase_core that need not carry the provenance enums; the stamp's
# own rule is pinned in tests/ci/test_runtime_env_anchor.py.
STAMP_LANES_WITHOUT_KIND = frozenset({"dev", "stability"})
STAMP_LANE_KINDS = frozenset({"mac_unit", "lane_worktree", "hand_run"})
STAMP_INSTALL_SOURCES = frozenset({"tag", "branch", "worktree", "hand"})

ROUTING_TEXT = """\
default_instance: inst-a
instances:
  inst-a:
    hostnames: [host-a]
    consumer_group: group-a
    verify:
      receipt_lane: compose-lane-a
  inst-b:
    hostnames: [host-b]
    consumer_group: group-b
    lane:
      receipt_lane: compose-lane-b
    verify:
      receipt_lane: compose-lane-b
"""


def _fragment(env: Mapping[str, object], **overrides: object) -> str:
    body: dict[str, object] = {
        "schema_version": SCHEMA,
        "lane_id": LANE,
        "description": "synthetic",
        "env": env,
    }
    body.update(overrides)
    return json.dumps(body)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )


def _overlay_clone(tmp_path: Path, files: dict[str, str]) -> Path:
    """A git repo whose ``origin/dev`` holds ``files`` (relative path -> text)."""
    repo = tmp_path / "overlay-clone"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "README").write_text("overlay\n", encoding="utf-8")
    for relpath, text in files.items():
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", f"refs/remotes/{LANE_ENV_FRAGMENT_REF}", "HEAD")
    return repo


def _fragment_path(lane_id: str = LANE) -> str:
    return f"{LANE_ENV_FRAGMENT_DIR}/{lane_id}{LANE_ENV_FRAGMENT_SUFFIX}"


# -- schema -----------------------------------------------------------------


@pytest.mark.unit
def test_schema_declares_exactly_the_act_switches_and_the_lane_class() -> None:
    assert {
        SCHEMA: frozenset(NON_ACTING)
        | frozenset(LANE_CLASS)
        | frozenset(PROVENANCE)
        | frozenset(
            {"COST_ESTIMATOR_WEB_SOURCE_REPOSITORY", "COST_ESTIMATOR_WEB_SOURCE_URL"}
        )
        | frozenset({"ESTIMATOR_GATE_OPEN_SOURCES"})
    } == LANE_ENV_FRAGMENT_SCHEMA_KEYS


@pytest.mark.unit
def test_a_fragment_may_select_a_lane_class() -> None:
    env = {**NON_ACTING, **LANE_CLASS}
    assert parse_lane_env_fragment(_fragment(env), lane_id=LANE, source="t") == env


@pytest.mark.unit
@pytest.mark.parametrize("env", [PROVENANCE, PROVENANCE_DEV], ids=["other", "dev"])
def test_a_fragment_may_set_the_provenance_overlay(env: Mapping[str, str]) -> None:
    assert parse_lane_env_fragment(_fragment(env), lane_id=LANE, source="t") == dict(
        env
    )


@pytest.mark.unit
@pytest.mark.parametrize("env", [PROVENANCE, PROVENANCE_DEV], ids=["other", "dev"])
def test_the_provenance_fixtures_are_overlays_the_stamp_accepts(
    env: Mapping[str, str],
) -> None:
    lane = env["ONEX_PROVENANCE_LANE"]
    kind = env.get("ONEX_PROVENANCE_LANE_KIND", "")
    for name in (
        "ONEX_PROVENANCE_LANE",
        "ONEX_PROVENANCE_HOST",
        "ONEX_PROVENANCE_INSTALL_SOURCE",
        "ONEX_PROVENANCE_REVISION",
    ):
        assert env[name].strip(), name
    assert env["ONEX_PROVENANCE_INSTALL_SOURCE"] in STAMP_INSTALL_SOURCES
    if lane in STAMP_LANES_WITHOUT_KIND:
        assert kind == ""
    else:
        assert lane == "other"
        assert kind in STAMP_LANE_KINDS


@pytest.mark.unit
def test_provenance_allowlist_literals_are_the_names_the_stamp_reads() -> None:
    # The allowlist names them as literals because the agent's own venv pins a
    # published omnibase_infra that need not carry the model, so the names are
    # read from the model's source file in the checkout, never imported.
    # Only the VARIABLES assignment is read, so a name in a comment or a
    # docstring of the model file does not count.
    source = (
        Path(__file__).resolve().parents[4]
        / "src/omnibase_infra/models/event_bus/model_event_provenance_overlay.py"
    ).read_text(encoding="utf-8")
    read = frozenset(name for _, name in _overlay_variables(source))
    allowed = LANE_ENV_FRAGMENT_SCHEMA_KEYS[SCHEMA]
    assert read == frozenset(PROVENANCE)
    assert {k for k in allowed if k.startswith("ONEX_PROVENANCE_")} == read


def _overlay_variables(source: str) -> tuple[tuple[str, str], ...]:
    """The literal ``VARIABLES`` of ``ModelEventProvenanceOverlay`` in ``source``."""
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "ModelEventProvenanceOverlay"
        ):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "VARIABLES"
                    and stmt.value is not None
                ):
                    value: tuple[tuple[str, str], ...] = ast.literal_eval(stmt.value)
                    return value
    raise AssertionError("ModelEventProvenanceOverlay.VARIABLES not found")


@pytest.mark.unit
def test_the_drift_read_ignores_names_outside_variables() -> None:
    source = (
        "class ModelEventProvenanceOverlay:\n"
        '    """Mentions "ONEX_PROVENANCE_DOC" in prose."""\n'
        '    # "ONEX_PROVENANCE_COMMENT"\n'
        '    VARIABLES: ClassVar = (("lane", "ONEX_PROVENANCE_LANE"),)\n'
        '    OTHER = "ONEX_PROVENANCE_OTHER"\n'
    )
    assert _overlay_variables(source) == (("lane", "ONEX_PROVENANCE_LANE"),)


@pytest.mark.unit
def test_a_fragment_may_not_set_a_provenance_key_the_stamp_does_not_read() -> None:
    with pytest.raises(LaneEnvFragmentError, match="ONEX_PROVENANCE_EXTRA"):
        parse_lane_env_fragment(
            _fragment({**PROVENANCE, "ONEX_PROVENANCE_EXTRA": "x"}),
            lane_id=LANE,
            source="t",
        )


@pytest.mark.unit
def test_a_fragment_sets_exactly_its_keys() -> None:
    subset = {"ONEX_PR_LANDING_GITHUB_MODE": "dry_run"}
    assert parse_lane_env_fragment(_fragment(subset), lane_id=LANE, source="t") == (
        subset
    )
    assert (
        parse_lane_env_fragment(_fragment(NON_ACTING), lane_id=LANE, source="t")
        == NON_ACTING
    )


@pytest.mark.unit
def test_a_key_outside_the_schema_allowlist_is_refused_by_name() -> None:
    with pytest.raises(LaneEnvFragmentError, match="POSTGRES_PASSWORD"):
        parse_lane_env_fragment(
            _fragment({**NON_ACTING, "POSTGRES_PASSWORD": "x"}),
            lane_id=LANE,
            source="t",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (_fragment({"ONEX_PR_LANDING_GITHUB_MODE": ""}), "empty"),
        (_fragment({"ONEX_PR_LANDING_GITHUB_MODE": "dry\nrun"}), "line"),
        (_fragment({"ONEX_PR_LANDING_GITHUB_MODE": False}), "env"),
        (_fragment(NON_ACTING, schema_version="runtime_lane_env.v0"), "schema"),
        (_fragment(NON_ACTING, lane_id="compose-lane-z"), "lane_id"),
        (_fragment(NON_ACTING, extra="x"), "extra"),
        ("{not json", "JSON"),
    ],
)
def test_a_malformed_fragment_is_refused(text: str, reason: str) -> None:
    with pytest.raises(LaneEnvFragmentError, match=reason):
        parse_lane_env_fragment(text, lane_id=LANE, source="t")


# -- reading the overlay clone ----------------------------------------------


@pytest.mark.unit
def test_no_clone_changes_nothing(tmp_path: Path) -> None:
    assert read_lane_env_fragment(tmp_path / "absent", LANE) == {}


@pytest.mark.unit
def test_no_fragment_at_the_ref_changes_nothing(tmp_path: Path) -> None:
    clone = _overlay_clone(tmp_path, {_fragment_path("compose-lane-b"): "{}"})
    assert read_lane_env_fragment(clone, LANE) == {}


@pytest.mark.unit
def test_a_fragment_at_the_ref_is_read(tmp_path: Path) -> None:
    clone = _overlay_clone(tmp_path, {_fragment_path(): _fragment(NON_ACTING)})
    assert read_lane_env_fragment(clone, LANE) == NON_ACTING


@pytest.mark.unit
def test_the_fragment_is_read_at_the_ref_not_the_working_tree(tmp_path: Path) -> None:
    clone = _overlay_clone(tmp_path, {})
    working = clone / _fragment_path()
    working.parent.mkdir(parents=True)
    working.write_text(_fragment(NON_ACTING), encoding="utf-8")
    assert read_lane_env_fragment(clone, LANE) == {}


@pytest.mark.unit
def test_a_clone_without_the_ref_is_refused(tmp_path: Path) -> None:
    clone = tmp_path / "bare-dir"
    clone.mkdir()
    with pytest.raises(LaneEnvFragmentError, match=LANE_ENV_FRAGMENT_REF):
        read_lane_env_fragment(clone, LANE)


# -- lane identity ------------------------------------------------------------


@pytest.mark.unit
def test_each_instance_runtime_lane_is_its_receipt_lane() -> None:
    assert instance_runtime_lanes_from_text(ROUTING_TEXT) == {
        "inst-a": "compose-lane-a",
        "inst-b": "compose-lane-b",
    }


@pytest.mark.unit
def test_an_instance_without_a_receipt_lane_is_refused() -> None:
    text = ROUTING_TEXT.replace("      receipt_lane: compose-lane-a\n", "")
    with pytest.raises(RoutingTableError, match="inst-a"):
        instance_runtime_lanes_from_text(text)


@pytest.mark.unit
def test_the_shipped_table_names_a_runtime_lane_for_every_dev_instance() -> None:
    assert set(executor_mod.DEV_INSTANCE_RUNTIME_LANES) == set(
        executor_mod.dev_instance_lane_configs()
    )


# -- the deploy agent's compose env -----------------------------------------


@pytest.fixture
def dev_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dev-lane agent process whose overlay clone holds the non-acting fragment."""
    clone = _overlay_clone(tmp_path, {_fragment_path(): _fragment(NON_ACTING)})
    # Compose also reads broker policy from its deploy source (OMN-20275).
    monkeypatch.setattr(
        executor_mod,
        "deploy_repo_dir",
        lambda: str(Path(__file__).resolve().parents[4]),
    )
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(tmp_path))
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", clone)
    monkeypatch.setattr(
        executor_mod,
        "DEV_INSTANCE_RUNTIME_LANES",
        {executor_mod.active_dev_instance(): LANE},
    )
    for key in NON_ACTING:
        monkeypatch.delenv(key, raising=False)
    return clone


@pytest.mark.unit
def test_compose_env_overlays_exactly_the_fragment_keys(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator store's value for a listed key is overlaid, not kept.
    monkeypatch.setenv("ONEX_PR_LANDING_GITHUB_MODE", "enforce")
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", tmp_path / "absent")
    without = executor_mod._compose_env()
    assert without["ONEX_PR_LANDING_GITHUB_MODE"] == "enforce"
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", dev_agent)
    env = executor_mod._compose_env()
    assert {k: env[k] for k in NON_ACTING} == NON_ACTING
    changed = {k for k in env if env.get(k) != without.get(k)}
    assert changed == set(NON_ACTING)


@pytest.mark.unit
def test_compose_env_without_a_fragment_changes_nothing(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", tmp_path / "absent")
    env = executor_mod._compose_env()
    assert not set(NON_ACTING) & set(env)


@pytest.mark.unit
def test_compose_env_of_a_process_the_agent_did_not_start_reads_no_fragment(
    dev_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", None)
    env = executor_mod._compose_env()
    assert not set(NON_ACTING) & set(env)


@pytest.mark.unit
def test_compose_env_of_a_non_dev_agent_reads_no_fragment(
    dev_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", EnumRuntimeLane.STABILITY_TEST.value
    )
    env = executor_mod._compose_env()
    assert not set(NON_ACTING) & set(env)


@pytest.mark.unit
def test_compose_env_of_a_lab_agent_that_also_deploys_stability(
    dev_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-19998: a lab dev agent's fence may admit the stability lane beside dev.

    Its dev jobs, and its dev-lane work between jobs, keep the dev fragment; a
    stability-lane job of the same agent never takes it.
    """
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", f"dev,{EnumRuntimeLane.STABILITY_TEST.value}"
    )
    between_jobs = executor_mod._compose_env()
    assert {k: between_jobs[k] for k in NON_ACTING} == NON_ACTING
    with executor_mod.active_deploy_lane(EnumRuntimeLane.DEV):
        dev_job = executor_mod._compose_env()
    assert {k: dev_job[k] for k in NON_ACTING} == NON_ACTING
    with executor_mod.active_deploy_lane(EnumRuntimeLane.STABILITY_TEST):
        stability_job = executor_mod._compose_env()
    assert not set(NON_ACTING) & set(stability_job)
    assert executor_mod._ActiveDeployLane.lane is None


@pytest.mark.unit
def test_compose_env_refuses_a_fragment_outside_the_allowlist(
    dev_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = _overlay_clone(
        tmp_path / "bad",
        {_fragment_path(): _fragment({"GITHUB_TOKEN": "x"})},
    )
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", bad)
    with pytest.raises(LaneEnvFragmentError, match="GITHUB_TOKEN"):
        executor_mod._compose_env()


# -- the warm refresh -------------------------------------------------------


@pytest.mark.unit
def test_shell_exports_round_trip_through_bash() -> None:
    exports = lane_env_shell_exports(NON_ACTING)
    script = (
        exports + "\n" + "\n".join(f'printf "%s=%s\\n" {k} "${k}"' for k in NON_ACTING)
    )
    out = subprocess.run(
        ["bash", "-c", script], check=True, capture_output=True, text=True
    ).stdout
    assert dict(line.split("=", 1) for line in out.splitlines()) == NON_ACTING


@pytest.mark.unit
def test_refresh_exports_the_default_instance_fragment(tmp_path: Path) -> None:
    repo_root = tmp_path / "infra"
    (repo_root / "config").mkdir(parents=True)
    (repo_root / "config" / "deploy_lane_routing.yaml").write_text(
        ROUTING_TEXT, encoding="utf-8"
    )
    home = tmp_path / "home"
    _overlay_clone(home, {_fragment_path(): _fragment(NON_ACTING)})
    # A sibling clone without fragments is not the overlay clone.
    _overlay_clone(home / "other", {})
    (home / "other" / "overlay-clone").rename(home / "plain-clone")
    assert default_instance_lane_env_exports(repo_root, home) == (
        lane_env_shell_exports(NON_ACTING)
    )
    assert default_instance_lane_env_exports(repo_root, tmp_path / "absent") == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("table", "match"),
    [
        (None, "routing table unreadable"),
        ("instances: [unclosed\n", "routing table is not YAML"),
        ("- a list, not a mapping\n", "routing table declares no instances"),
    ],
    ids=["unreadable", "not-yaml", "not-a-mapping"],
)
def test_refresh_refuses_a_bad_routing_table_as_routing_table_error(
    tmp_path: Path, table: str | None, match: str
) -> None:
    # Before OMN-25116 these raised OSError, yaml.YAMLError and AttributeError;
    # the one caller, refresh_dev_lane.sh, stops on any nonzero exit of its
    # python -c and catches no exception type.
    repo_root = tmp_path / "infra"
    (repo_root / "config").mkdir(parents=True)
    if table is not None:
        (repo_root / "config" / "deploy_lane_routing.yaml").write_text(
            table, encoding="utf-8"
        )
    with pytest.raises(RoutingTableError, match=match):
        default_instance_lane_env_exports(repo_root, tmp_path / "home")


@pytest.mark.unit
def test_two_clones_carrying_fragments_are_refused(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _overlay_clone(home, {_fragment_path(): _fragment(NON_ACTING)})
    _overlay_clone(home / "second", {_fragment_path(): _fragment(NON_ACTING)})
    (home / "second" / "overlay-clone").rename(home / "overlay-copy")
    with pytest.raises(LaneEnvFragmentError, match="more than one clone"):
        discover_lane_overlay_clone(home)


@pytest.mark.unit
def test_refresh_overlays_the_fragment_after_the_operator_env() -> None:
    script = (
        Path(__file__).resolve().parents[3] / "runtime_build" / "refresh_dev_lane.sh"
    ).read_text(encoding="utf-8")
    operator = script.index('source "${OMNIBASE_OPERATOR_ENV_FILE}"')
    overlay = script.index("default_instance_lane_env_exports")
    deploy = script.index("=== Build + bring-up")
    assert operator < overlay < deploy


@pytest.mark.unit
def test_source_overlay_uses_its_own_ref_with_inherited_git_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": "web-front-end",
        "COST_ESTIMATOR_WEB_SOURCE_URL": "https://example.invalid/web-front-end.git",
    }
    clone = _overlay_clone(tmp_path, {_fragment_path(): _fragment(source)})
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "unrelated"))
    assert read_lane_env_fragment(clone, LANE) == source
    assert discover_lane_overlay_clone(tmp_path) == clone
