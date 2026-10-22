# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-merge runtime pool driver against fake lab hosts (OMN-18893).

Operator ruling 2026-09-27: runtime PRs deploy to any free runtime lane on any
lab machine. These tests hold the driver to the parts of that ruling a
misreading would break silently: it never picks an excluded, leased, held or
overloaded host; the lease is taken atomically and only broken once expired;
teardown and release run even when the build fails; and PASS is printed only
when every check and the zero-residue readback agree.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import itertools
import json
import os
import pwd
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.helpers.lane_census_fixtures import NEUTRAL_MANIFEST

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"
PROVE_SH = REPO_ROOT / "scripts" / "runtime_build" / "prepr_pool_prove.sh"

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("prepr_runtime_pool", POOL_PY)
# OMN-20934: a synthetic pool; a lab's own table lives under its workspace config root.
POOL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prepr_runtime_pool.yaml"
CFG = pool.load_pool_config(POOL_FIXTURE)
NOW = dt.datetime(2026, 9, 27, 14, 30, tzinfo=dt.UTC)

GOOD = {
    "snap-pre": "snap-pre lines=50 local-containers=0 listeners=0",
    "clone": "omnibase_infra#4210 fetched head abc expected abc match=yes\nclone done",
    "build": "stage rc=0\nbuild rc=0 2026-09-27T14:25:00Z\nup rc=0 2026-09-27T14:28:00Z",
    "probe": (
        "file omnibase_infra/x.py image=aaa build-tree=aaa match=yes\n"
        "port 8085 HTTP 200 status healthy healthy True failed_handlers 0\n"
        "port 8086 HTTP 200 status healthy healthy True failed_handlers 0\n"
        "/omnibase-infra-local-migration-gate health=healthy restarts=0 running exit=0\n"
        "omnibase-infra-local-omninode-runtime lines=900 autowire-fail=0 dup-dispatcher=0 ERROR=0 Traceback=0\n"
        "omnibase-infra-local-runtime-effects lines=800 autowire-fail=0 dup-dispatcher=0 ERROR=0 Traceback=0\n"
        "failed-contracts omnibase-infra-local-omninode-runtime: \n"
        "failed-contracts omnibase-infra-local-runtime-effects: \n"
    ),
    "tests": "omnibase_infra focused rc=0 2026-09-27T14:40:00Z",
    "teardown": (
        "containers=0 volumes=0 networks=0 images=0 listeners=0 workdir=gone\n"
        "positive control dogfood containers 7 running 7"
    ),
    "snap-post": "snapshot-diff=0\nsnap-post lines=50 local-containers=0 listeners=0",
}


class FakeHost:
    def __init__(
        self,
        cores: int = 12,
        load: float = 2.0,
        slot: int = 0,
        listen: int = 0,
        online: bool = True,
    ) -> None:
        self.cores, self.load, self.slot, self.listen, self.online = (
            cores,
            load,
            slot,
            listen,
            online,
        )
        self.lease: str | None = None
        self.phases = dict(GOOD)
        self.base_phases = dict(GOOD)
        self.ran: list[str] = []


class FakeTransport:
    """Interprets the driver's remote commands against in-memory hosts."""

    def __init__(self, hosts: dict[str, FakeHost]) -> None:
        self.hosts = hosts
        self.puts: list[tuple[str, str]] = []

    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        h = self.hosts.get(host.name)
        if h is None or not h.online:
            return 255, "ssh: connect to host: Operation timed out"
        m = re.search(r"prove\.sh \S+ (\S+)$", command)
        if m:
            if pool.LEASE_LOST in command:
                found = re.search(r'"holder": "([^"]+)"', command)
                assert found is not None
                if h.lease is None or json.loads(h.lease)["holder"] != found.group(1):
                    h.ran.append("lost:" + m.group(1))
                    return 97, "LEASE-LOST " + (h.lease or "")
                body = re.search(r"printf '%s' '(\{.*?\})'", command)
                assert body is not None
                h.lease = body.group(1)
            base = "-base-" in command
            h.ran.append(("base:" if base else "") + m.group(1))
            return 0, (h.base_phases if base else h.phases)[m.group(1)]
        if "CORES=$(getconf" in command:
            return 0, (
                f"CORES={h.cores}\nLOAD={h.load}\nSLOT={h.slot}\nLISTEN={h.listen}\n"
                f"PROJECTS=omnibase-infra-dogfood:7,\nLEASE={h.lease or ''}\n"
            )
        if "RELEASED" in command:  # release
            if h.lease is None:
                return 0, "ABSENT"
            found = re.search(r'"holder": "([^"]+)"', command)
            assert found is not None
            holder = found.group(1)
            if json.loads(h.lease)["holder"] == holder:
                h.lease = None
                return 0, "RELEASED"
            return 0, "FOREIGN"
        if ".break" in command:  # break an expired lease, guarded and compared
            body = re.search(r"printf '%s' '(\{.*?\})'", command)
            assert body is not None
            if h.lease is not None and shlex.quote(h.lease) in command:
                h.lease = body.group(1)
                return 0, "ACQUIRED"
            return 0, "HELD\n" + (h.lease or "")
        if "then printf" in command:  # acquire
            if h.lease is None:
                body = re.search(r"printf '%s' '(\{.*?\})'", command)
                assert body is not None
                h.lease = body.group(1)
                return 0, "ACQUIRED"
            return 0, "HELD\n" + h.lease
        if command.startswith("rm -rf $HOME/"):
            h.lease = None
            return 0, ""
        return 0, ""

    def put(self, host: Any, local: Path, remote: str) -> int:
        self.puts.append((host.name, remote))
        return 0


def _params(tmp_path: Path) -> Path:
    p = tmp_path / "p.env"
    p.write_text(
        "INFRA_PR=4210\nINFRA_HEAD=abc\nID_FILES=omnibase_infra:omnibase_infra/x.py\n"
        "TESTS=omnibase_infra:tests/unit/test_x.py\n",
        encoding="utf-8",
    )
    return p


def _lease(holder: str, until: dt.datetime) -> str:
    return json.dumps(
        {"holder": holder, "until": until.strftime("%Y-%m-%dT%H:%M:%SZ"), "taken": "x"}
    )


# ------------------------------------------------------------------ config


def test_declared_manifest_hosts_reads_the_manifest_host_keys(tmp_path: Path) -> None:
    # That our pool's machines are declared in our manifest is checked against
    # our lab's own pool table on the lab (OMN-20934); here, the reader.
    manifest = tmp_path / "lane-manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"hosts": {m: {} for m in ("pool-a", "pool-b", "pool-c")}}),
        encoding="utf-8",
    )
    declared = pool.declared_manifest_hosts(manifest)
    assert declared == {"pool-a", "pool-b", "pool-c"}
    assert {h.machine for h in CFG.hosts} - declared == {"pool-d", "pool-e"}


def test_the_slot_project_is_never_a_declared_lane_in_the_neutral_manifest(
    tmp_path: Path,
) -> None:
    # OMN-20934: the same invariant over the neutral manifest every checkout
    # carries, so it runs where the lab's private manifest is absent; the test
    # below holds it against the lab's own manifest where that is fetched.
    declared = pool.declared_lane_projects(NEUTRAL_MANIFEST)
    assert declared, "the reader found no lane to compare the slot project with"
    assert CFG.compose_project not in declared
    isolated = [h for h in CFG.hosts if h.kind == "isolated"]
    assert {h.compose_project for h in isolated} == {CFG.compose_project}
    # control: a manifest that does declare the slot project as a lane is seen
    colliding = tmp_path / "lane-manifest.yaml"
    colliding.write_text(
        yaml.safe_dump({"lanes": {"slot": {"compose_project": CFG.compose_project}}}),
        encoding="utf-8",
    )
    assert CFG.compose_project in pool.declared_lane_projects(colliding)


@pytest.mark.lab_overlay
def test_the_slot_project_is_never_a_declared_lane() -> None:
    assert CFG.compose_project not in pool.declared_lane_projects()
    isolated = [h for h in CFG.hosts if h.kind == "isolated"]
    assert {h.compose_project for h in isolated} == {CFG.compose_project}


def test_the_loader_reads_members_and_their_exclusion_reasons() -> None:
    # Which lab machines take part is our lab's policy, checked beside our lab's
    # table under the workspace config root (OMN-20934); here, the loader's reading.
    in_pool = {h.machine for h in CFG.hosts if h.status == "pool"}
    assert in_pool == {"pool-a", "pool-b", "pool-c", "pool-d"}
    assert CFG.host("pool-e").status == "excluded"
    assert CFG.host("pool-d").positive_control == "omnibase-infra-lane-d"
    assert CFG.host(CFG.host("pool-a").ssh_target).name == "pool-a"
    assert all(h.reason for h in CFG.hosts if h.status == "excluded")


def test_without_an_overlay_the_pool_refuses_and_names_how_to_supply_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("OMNI_HOME", raising=False)
    with pytest.raises(
        ValueError, match="not configured: set ONEX_WORKSPACE_CONFIG_ROOT"
    ):
        pool.load_pool_config()
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="does not exist"):
        pool.load_pool_config()
    # the driver turns the refusal into its usage exit and names no lab host
    assert pool.main(["status"]) == pool.EXIT_USAGE


def test_the_pool_resolves_under_the_workspace_config_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = tmp_path / "owner"
    (owner / "config" / "lab").mkdir(parents=True)
    shutil.copyfile(POOL_FIXTURE, owner / "config" / "lab" / "prepr_runtime_pool.yaml")
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(owner))
    assert (
        pool.default_pool_config()
        == (owner / "config" / "lab" / "prepr_runtime_pool.yaml").resolve()
    )
    assert pool.load_pool_config() == CFG
    # with the override unset, the runtime's config owner of $OMNI_HOME owns it
    from omnibase_infra.runtime.service_kernel import workspace_runtime_config_root

    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OMNI_HOME", str(home))
    expected = workspace_runtime_config_root(home) / pool.POOL_CONFIG_RELATIVE
    with pytest.raises(ValueError, match=re.escape(str(expected))):
        pool.load_pool_config()
    expected.parent.mkdir(parents=True)
    shutil.copyfile(POOL_FIXTURE, expected)
    assert pool.load_pool_config() == CFG


def test_201_takes_part_only_through_its_pre_pr_slots() -> None:
    on_201 = [h for h in CFG.hosts if h.machine == "pool-c"]
    assert CFG.host("pool-c").status == "excluded"
    members = sorted((h for h in on_201 if h.status == "pool"), key=lambda h: h.slot)
    assert [h.kind for h in members] == ["prepr-slot", "prepr-slot"]
    assert [h.compose_project for h in members] == [
        "omnibase-infra-prepr-1",
        "omnibase-infra-prepr-2",
    ]
    assert [(h.main_port, h.effects_port) for h in members] == [
        (28085, 28086),
        (38085, 38086),
    ]
    # never a governed .201 lane's project, and never its ports
    policy = pool._slot_policy()
    for h in members:
        assert policy.assert_target_is_a_pool_slot(h.compose_project).slot == h.slot
        assert not {8085, 8086, 5436, 19092, 16379} & set(h.ports)
    assert len({h.lease_dir for h in members} | {CFG.lease_dir}) == 3
    assert len({h.surface for h in CFG.hosts}) == len(CFG.hosts)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        ({"kind": "prepr-slot", "slot": 3}, "not a pre-PR slot"),
        ({"slot": 1}, "names a slot but is not a prepr-slot"),
        ({"kind": "declared-lane"}, "has kind"),
    ],
)
def test_a_member_outside_the_slot_policy_is_refused(
    tmp_path: Path, edit: dict[str, Any], match: str
) -> None:
    raw = yaml.safe_load(POOL_FIXTURE.read_text(encoding="utf-8"))
    raw["hosts"][0].update(edit)
    bad = tmp_path / "pool.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        pool.load_pool_config(bad)


def test_an_exclusion_without_a_reason_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(POOL_FIXTURE.read_text(encoding="utf-8"))
    next(h for h in raw["hosts"] if h["status"] == "excluded").pop("reason")
    bad = tmp_path / "pool.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="states no reason"):
        pool.load_pool_config(bad)


def test_prove_script_touches_only_the_slot_project() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    assert "P=omnibase-infra-local;" in text
    # every destructive docker verb is scoped by the slot project label or an id list from it
    for line in text.splitlines():
        if re.search(r"docker (rm|volume rm|network rm|image rm)", line):
            assert (
                "$C" in line
                or "$V" in line
                or "$N" in line
                or '"$i"' in line
                or '"$v"' in line
            ), line


def test_probe_waits_for_the_runtimes_before_reading_health() -> None:
    # The first live run (omnimarket#3016, 2026-09-27) probed the second `up`
    # returned and read no health at all: the runtimes need minutes to settle.
    text = PROVE_SH.read_text(encoding="utf-8")
    probe = text[text.index("probe)") : text.index("tests)")]
    assert probe.index("health wait") < probe.index("== health")
    assert "health: starting" in probe and "PROBE_WAIT_S" in probe


def test_isolated_docker_config_pins_the_file_store_and_holds_no_credential(
    tmp_path: Path,
) -> None:
    # OMN-19998: on a macOS lab host a config with no auths key made the docker CLI
    # detect the keychain store again, so every isolated build from a
    # non-interactive session failed at "keychain cannot be accessed".
    text = PROVE_SH.read_text(encoding="utf-8")
    branch = text[text.index("= isolated ]; then") :]
    start = branch.index("python3 -c '") + len("python3 -c '")
    snippet = branch[start : branch.index('\' "$DC/config.json"')]
    home = tmp_path / "home"
    (home / ".docker").mkdir(parents=True)
    (home / ".docker" / "config.json").write_text(
        json.dumps(
            {
                "credsStore": "osxkeychain",
                "credHelpers": {"ghcr.io": "osxkeychain"},
                "auths": {"ghcr.io": {"auth": "c2VjcmV0"}},
                "currentContext": "desktop-linux",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "config.json"
    subprocess.run(
        [sys.executable, "-c", snippet, str(out)],
        check=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert "credsStore" not in cfg and "credHelpers" not in cfg
    assert cfg["currentContext"] == "desktop-linux"
    # a non-empty auths keeps the CLI on the plain file store; no entry holds a credential
    assert cfg.get("auths"), cfg
    assert all(not entry for entry in cfg["auths"].values()), cfg


# ------------------------------------------------------------------ ledger


def test_live_surface_holds_respects_until_and_release() -> None:
    rows = [
        "2026-09-27T14:00:00Z | HOLD | lane=a | id=2026-09-27T14:00:00Z-a | surface=a-runtime | until=2026-09-27T15:00:00Z | x",
        "2026-09-27T13:00:00Z | HOLD | lane=b | id=2026-09-27T13:00:00Z-b | surface=e-runtime | until=2026-09-27T14:00:00Z | x",
        "2026-09-27T14:10:00Z | HOLD | lane=c | id=2026-09-27T14:10:00Z-c | surface=b-runtime | until=2026-09-27T16:00:00Z | x",
        "2026-09-27T14:20:00Z | RELEASE | lane=c | re=2026-09-27T14:10:00Z-c | surface=b-runtime | result=PASS | restored=yes",
        "2026-09-27T14:21:00Z | HOLD | lane=d | id=2026-09-27T14:21:00Z-d | to=all | pr=omnimarket#1 | until=2026-09-27T20:00:00Z | not a surface",
    ]
    live = pool.live_surface_holds(rows, NOW)
    assert set(live) == {"a-runtime"}
    assert live["a-runtime"][1] == "a"


# ------------------------------------------------------------------ survey and pick


def _survey(
    hosts: dict[str, FakeHost],
    holds: dict[str, Any] | None = None,
    me: str | None = None,
) -> list[Any]:
    states: list[Any] = pool.survey(CFG, FakeTransport(hosts), holds or {}, NOW, me=me)
    return states


@pytest.mark.unit
def test_survey_probes_concurrently_and_preserves_host_order() -> None:
    class SlowTransport(FakeTransport):
        def __init__(self, hosts: dict[str, FakeHost]) -> None:
            super().__init__(hosts)
            self.lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0
            self.probed: list[str] = []

        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if "CORES=$(getconf" not in command:
                return super().run(host, command, timeout)
            with self.lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                self.probed.append(host.name)
            try:
                time.sleep(0.3)
                return super().run(host, command, timeout)
            finally:
                with self.lock:
                    self.in_flight -= 1

    transport = SlowTransport({host.name: FakeHost() for host in CFG.hosts})
    states = pool.survey(CFG, transport, {}, NOW)

    assert transport.max_in_flight > 1
    assert [state.host.name for state in states] == [host.name for host in CFG.hosts]
    assert sorted(transport.probed) == sorted(
        host.name for host in CFG.hosts if host.status != "excluded"
    )


def test_pick_takes_the_least_loaded_free_host_and_never_an_excluded_one() -> None:
    hosts = {
        "pool-a": FakeHost(load=6.0),
        "pool-b": FakeHost(cores=24, load=3.0),
        "pool-c": FakeHost(),
        "pool-d": FakeHost(cores=10, load=1.0),
    }
    states = _survey(hosts)
    verdicts = {s.host.name: s.verdict for s in states}
    assert verdicts["pool-c"] == "EXCLUDED"
    assert verdicts["pool-e"] == "EXCLUDED"
    assert verdicts["pool-d"] == "FREE"
    assert pool.pick(states).host.name == "pool-d"


def test_offline_overloaded_leased_held_and_occupied_hosts_are_not_free() -> None:
    hosts = {
        "pool-a": FakeHost(),
        "pool-d": FakeHost(online=False),
        "pool-b": FakeHost(cores=24, load=74.0),
    }
    hosts["pool-a"].lease = _lease("other-lane", NOW + dt.timedelta(minutes=30))
    v = {s.host.name: s for s in _survey(hosts)}
    assert v["pool-a"].verdict == "BUSY" and "other-lane" in v["pool-a"].detail
    assert v["pool-d"].verdict == "OFFLINE"
    assert v["pool-b"].verdict == "OVERLOADED"
    assert pool.pick(list(v.values())) is None

    held = {
        "pool-a": FakeHost(),
        "pool-d": FakeHost(slot=3),
        "pool-b": FakeHost(listen=1),
    }
    v2 = {
        s.host.name: s
        for s in _survey(
            held, holds={"a-runtime": ("id1", "peer", "2026-09-27T15:00:00Z")}
        )
    }
    assert v2["pool-a"].verdict == "BUSY" and "HOLD id1" in v2["pool-a"].detail
    assert (
        v2["pool-d"].verdict == "BUSY" and "containers present" in v2["pool-d"].detail
    )
    assert v2["pool-b"].verdict == "BUSY" and "listeners" in v2["pool-b"].detail


def test_own_hold_and_expired_lease_do_not_block() -> None:
    hosts = {"pool-a": FakeHost()}
    hosts["pool-a"].lease = _lease("gone-lane", NOW - dt.timedelta(minutes=1))
    v = {
        s.host.name: s
        for s in _survey(hosts, holds={"a-runtime": ("id", "me", "x")}, me="me")
    }
    assert v["pool-a"].verdict == "FREE"


# ------------------------------------------------------------------ lease


def test_lease_is_exclusive_and_only_its_holder_releases_it() -> None:
    hosts = {"pool-a": FakeHost()}
    tp = FakeTransport(hosts)
    h = CFG.host("pool-a")
    until = NOW + dt.timedelta(minutes=60)
    assert pool.acquire_lease(CFG, tp, h, "me", until, NOW)[0]
    ok, why = pool.acquire_lease(CFG, tp, h, "peer", until, NOW)
    assert not ok and "held by me" in why
    assert not pool.release_lease(CFG, tp, h, "peer")[0]
    assert pool.release_lease(CFG, tp, h, "me")[0]
    assert hosts["pool-a"].lease is None


def test_expired_lease_is_broken() -> None:
    hosts = {"pool-a": FakeHost()}
    hosts["pool-a"].lease = _lease("gone-lane", NOW - dt.timedelta(seconds=1))
    ok, why = pool.acquire_lease(
        CFG,
        FakeTransport(hosts),
        CFG.host("pool-a"),
        "me",
        NOW + dt.timedelta(hours=1),
        NOW,
    )
    assert ok and "gone-lane" in why
    assert json.loads(hosts["pool-a"].lease)["holder"] == "me"


# ------------------------------------------------------------------ judge


def test_judge_pass_only_when_every_check_and_restore_hold() -> None:
    rb = pool.judge(GOOD)
    assert rb.outcome == "PASS" and rb.restored, rb


def test_judge_passes_when_the_pr_changes_no_test_files() -> None:
    rb = pool.judge({**GOOD, "tests": "no-focused-tests: the PR changes no test files"})
    assert rb.checks["focused_tests"] is True
    assert rb.outcome == "PASS"
    assert "focused tests: n/a, the PR changes no test files" in rb.notes


@pytest.mark.parametrize("text", ["", "unrecognized tests output"])
def test_judge_fails_without_focused_test_evidence(text: str) -> None:
    rb = pool.judge({**GOOD, "tests": text})
    assert rb.checks["focused_tests"] is False


def test_judge_no_focused_tests_marker_does_not_override_a_failure() -> None:
    rb = pool.judge(
        {
            **GOOD,
            "tests": (
                "no-focused-tests: the PR changes no test files\n"
                "omnibase_infra focused rc=1"
            ),
        }
    )
    assert rb.checks["focused_tests"] is False
    assert "focused tests: n/a, the PR changes no test files" not in rb.notes


def test_prove_script_uses_the_no_focused_tests_marker() -> None:
    marker = "no-focused-tests: the PR changes no test files"
    assert marker in PROVE_SH.read_text(encoding="utf-8")
    assert marker == pool.NO_FOCUSED_TESTS_MARKER


@pytest.mark.parametrize(
    ("phase", "text", "check"),
    [
        (
            "probe",
            GOOD["probe"].replace(
                "port 8086 HTTP 200 status healthy",
                "port 8086 HTTP 503 status degraded",
            ),
            "health_8086",
        ),
        ("probe", GOOD["probe"].replace("match=yes", "match=NO"), "image_identity"),
        (
            "probe",
            GOOD["probe"].replace("autowire-fail=0 dup", "autowire-fail=2 dup", 1),
            "no_wiring_failures",
        ),
        (
            "probe",
            GOOD["probe"].replace("health=healthy", "health=unhealthy"),
            "migration_gate_healthy",
        ),
        ("tests", "omnibase_infra focused rc=1", "focused_tests"),
        ("clone", "fetched head abd expected abc match=NO", "head_matches"),
    ],
)
def test_judge_fails_on_each_broken_readback(phase: str, text: str, check: str) -> None:
    rb = pool.judge({**GOOD, phase: text})
    assert rb.outcome == "FAIL"
    assert rb.checks[check] is False


def test_judge_inconclusive_when_the_stack_never_built() -> None:
    assert pool.judge({**GOOD, "build": "build rc=1"}).outcome == "INCONCLUSIVE"


def test_restored_needs_zero_residue_a_positive_control_and_an_empty_diff() -> None:
    assert not pool.judge(
        {**GOOD, "teardown": GOOD["teardown"].replace("volumes=0", "volumes=2")}
    ).restored
    assert not pool.judge(
        {**GOOD, "teardown": GOOD["teardown"].replace("containers 7", "containers 0")}
    ).restored
    assert not pool.judge({**GOOD, "snap-post": "snapshot-diff=3"}).restored
    assert not pool.judge({k: v for k, v in GOOD.items() if k != "snap-post"}).restored


# ------------------------------------------------------------------ image identity (OMN-19896)

# The probe identity section of a runtime PR that changes nothing under src/,
# as the fixed prove script prints it: omnibase_infra#4111 changed only
# config/deploy_lane_routing.yaml and deploy-agent tests, so the caller listed
# no ID_FILES and no per-file line exists. Before OMN-19896 this FAILed
# image_identity by construction (ledger RELEASE 2026-09-28T02:32:13Z).
_M = "omnibase-infra-local-omninode-runtime"
_E = "omnibase-infra-local-runtime-effects"
_TREE_OK = (
    "image-revision-label 2fdde86b86d9fc4c4c1c7261ab449cb081737f9e\n"
    "build GIT_SHA 2fdde86b86d9fc4c4c1c7261ab449cb081737f9e market 8db69992fc\n"
    "identity-subject omnibase_infra\n"
    "src-diff omnibase_infra files=0\n"
    f"pkg-tree omnibase_infra {_M} tree-files=4210 image-files=4210 missing=0 differ=0 stale-py=0 extra-other=0 match=yes\n"
    f"pkg-tree omnibase_infra {_E} tree-files=4210 image-files=4210 missing=0 differ=0 stale-py=0 extra-other=0 match=yes\n"
)
NO_SRC_PROBE = _TREE_OK + GOOD["probe"].split("\n", 1)[1]


def test_image_identity_passes_a_runtime_pr_with_no_src_diff() -> None:
    rb = pool.judge({**GOOD, "probe": NO_SRC_PROBE})
    assert rb.checks["image_identity"] is True, rb
    assert rb.outcome == "PASS"
    assert any("changes nothing under src/" in n for n in rb.notes)


@pytest.mark.parametrize(
    "broken",
    [
        # a module the PR deleted still ships in the image
        _TREE_OK.replace(
            f"{_E} tree-files=4210 image-files=4210 missing=0 differ=0 stale-py=0 extra-other=0 match=yes",
            f"{_E} tree-files=4210 image-files=4211 missing=0 differ=0 stale-py=1 extra-other=0 match=NO",
        ),
        # the image runs code the build tree does not hold
        _TREE_OK.replace(
            "differ=0 stale-py=0 extra-other=0 match=yes",
            "differ=3 stale-py=0 extra-other=0 match=NO",
            1,
        ),
        # the subject has no compare line at all (the compare never ran)
        "\n".join(ln for ln in _TREE_OK.splitlines() if not ln.startswith("pkg-tree "))
        + "\n",
    ],
)
def test_image_identity_fails_when_the_image_is_not_the_build_tree(broken: str) -> None:
    probe = broken + GOOD["probe"].split("\n", 1)[1]
    rb = pool.judge({**GOOD, "probe": probe})
    assert rb.checks["image_identity"] is False
    assert rb.outcome == "FAIL"


def test_image_identity_fails_on_a_changed_file_mismatch_even_when_the_package_matches() -> (
    None
):
    probe = (
        _TREE_OK
        + "file omnibase_infra/x.py image=aaa build-tree=bbb match=NO\n"
        + GOOD["probe"].split("\n", 1)[1]
    )
    assert pool.judge({**GOOD, "probe": probe}).checks["image_identity"] is False


def test_image_identity_fails_on_an_unreadable_per_file_line() -> None:
    # the in-container read raised, and its message has spaces: before OMN-19896
    # the line failed to parse and was silently dropped beside a matching one
    probe = GOOD["probe"].replace(
        "file omnibase_infra/x.py image=aaa build-tree=aaa match=yes\n",
        "file omnibase_infra/x.py image=aaa build-tree=aaa match=yes\n"
        "file omnibase_infra/y.py image=FileNotFoundError: [Errno 2] No such file build-tree=bbb match=NO\n",
    )
    assert pool.judge({**GOOD, "probe": probe}).checks["image_identity"] is False


def test_an_older_probe_with_no_subject_keeps_the_per_file_rule() -> None:
    assert pool.judge(GOOD).checks["image_identity"] is True
    no_lines = "\n".join(
        ln for ln in GOOD["probe"].splitlines() if not ln.startswith("file ")
    )
    assert pool.judge({**GOOD, "probe": no_lines}).checks["image_identity"] is False


def _heredoc(name: str) -> str:
    text = PROVE_SH.read_text(encoding="utf-8")
    start = text.index(f"IFS= read -r -d '' {name} <<'PY' || :\n")
    body = text[start:].split("\n", 1)[1]
    return body[: body.index("\nPY\n") + 1]


def _run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False, **kw)


def _fake_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A build tree with a tracked package, and the same package installed."""
    tree = tmp_path / "tree"
    pkg = tree / "src" / "fakepkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n")
    (pkg / "sub" / "mod.py").write_text("Y = 2\n")
    (pkg / "sub" / "contract.yaml").write_text("name: y\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t.invalid",
            "commit",
            "-qm",
            "t",
        ],
    ):
        assert _run(cmd, cwd=tree).returncode == 0
    site = tmp_path / "site"
    shutil.copytree(pkg, site / "fakepkg")
    # bytecode the image carries is not source and never compared
    (site / "fakepkg" / "__pycache__").mkdir()
    (site / "fakepkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\0")
    return tree, site


def _compare(tree: Path, site: Path, tmp_path: Path) -> str:
    listing = tmp_path / "img.txt"
    lst = _run(
        [sys.executable, "-c", _heredoc("IMG_LIST_PY"), "fakepkg"],
        env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"},
    )
    assert lst.returncode == 0, lst.stderr
    listing.write_text(lst.stdout)
    out = _run(
        [
            sys.executable,
            "-c",
            _heredoc("PKG_TREE_PY"),
            str(tree),
            "fakepkg",
            "rt",
            str(listing),
        ]
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_whole_package_compare_reads_an_identical_install_as_a_match(
    tmp_path: Path,
) -> None:
    tree, site = _fake_tree(tmp_path)
    out = _compare(tree, site, tmp_path)
    assert (
        "pkg-tree fakepkg rt tree-files=3 image-files=3 missing=0 differ=0 stale-py=0"
        in out
    )
    assert out.splitlines()[0].endswith("match=yes"), out


@pytest.mark.parametrize(
    ("mutate", "kind"),
    [
        (lambda s: (s / "fakepkg" / "sub" / "mod.py").write_text("Y = 3\n"), "differ"),
        (lambda s: (s / "fakepkg" / "sub" / "mod.py").unlink(), "missing"),
        (lambda s: (s / "fakepkg" / "old.py").write_text("Z = 0\n"), "stale-py"),
    ],
)
def test_the_whole_package_compare_names_what_differs(
    tmp_path: Path, mutate: Any, kind: str
) -> None:
    tree, site = _fake_tree(tmp_path)
    mutate(site)
    out = _compare(tree, site, tmp_path)
    assert out.splitlines()[0].endswith("match=NO"), out
    assert f"pkg-tree-diff fakepkg rt {kind} " in out
    rb = pool.judge(
        {
            **GOOD,
            "probe": "identity-subject fakepkg\n"
            + out
            + GOOD["probe"].split("\n", 1)[1],
        }
    )
    assert rb.checks["image_identity"] is False


def test_a_force_included_resource_is_counted_not_held_against_the_image(
    tmp_path: Path,
) -> None:
    tree, site = _fake_tree(tmp_path)
    (site / "fakepkg" / "config").mkdir()
    (site / "fakepkg" / "config" / "lanes.yaml").write_text("a: 1\n")
    out = _compare(tree, site, tmp_path)
    assert "extra-other=1 match=yes" in out


def test_the_probe_compares_every_repo_under_test_in_both_runtimes() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    probe = text[text.index("probe)") : text.index("tests)")]
    assert 'echo "identity-subject $repo"' in probe
    assert "for c in $M $E; do" in probe[probe.index("identity-subject") :]
    assert '"$IMG_LIST_PY"' in probe and '"$PKG_TREE_PY"' in probe


# ------------------------------------------------------------------ run end to end


def test_run_pass_end_to_end_prints_a_citable_readback(tmp_path: Path) -> None:
    hosts = {
        "pool-a": FakeHost(load=5.0),
        "pool-e": FakeHost(online=False),
        "pool-b": FakeHost(cores=24, load=60.0),
    }
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_PASS, text
    assert text.startswith("LAB PROOF PASS: omnibase_infra#4210 head abc on pool-a")
    assert (
        "restored=yes" in text and "surface=a-runtime result=PASS restored=yes" in text
    )
    assert hosts["pool-a"].ran == list(pool.PHASES)
    assert hosts["pool-a"].lease is None


def test_failed_build_still_tears_down_and_releases(tmp_path: Path) -> None:
    hosts = {"pool-a": FakeHost()}
    hosts["pool-a"].phases["build"] = "build rc=1\nup rc=1"
    tp = FakeTransport(hosts)

    def run(host: Any, command: str, timeout: float) -> tuple[int, str]:
        rc, out = FakeTransport.run(tp, host, command, timeout)
        return (1 if command.endswith(" build") else rc), out

    tp.run = run  # type: ignore[method-assign]
    code, _text = pool.run_proof(
        CFG,
        tp,
        _params(tmp_path),
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_INCONCLUSIVE
    assert hosts["pool-a"].ran == [
        "snap-pre",
        "clone",
        "build",
        "teardown",
        "snap-post",
    ]
    assert hosts["pool-a"].lease is None


def test_run_refuses_when_no_host_is_free(tmp_path: Path) -> None:
    hosts = {
        "pool-a": FakeHost(cores=12, load=40.0),
        "pool-b": FakeHost(slot=9),
    }
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_NO_FREE_HOST
    assert (
        "pool-a OVERLOADED" in text
        and "pool-e EXCLUDED" in text
        and "pool-c EXCLUDED" in text
        and "pool-d OFFLINE" in text
    )


def test_run_refuses_a_host_under_a_peer_ledger_hold(tmp_path: Path) -> None:
    rows = [
        "2026-09-27T14:00:00Z | HOLD | lane=peer | id=2026-09-27T14:00:00Z-peer | surface=a-runtime | until=2026-09-27T15:00:00Z | x"
    ]
    hosts = {"pool-a": FakeHost()}
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "pool-a",
        rows,
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_NO_FREE_HOST and "peer" in text
    assert hosts["pool-a"].ran == []


# ------------------------------------------------------------------ base control

DEV_FAILS = (
    GOOD["probe"]
    .replace(
        "failed-contracts omnibase-infra-local-omninode-runtime: ",
        "failed-contracts omnibase-infra-local-omninode-runtime: projection_baselines projection_traces",
    )
    .replace("port 8085 HTTP 200 status healthy", "port 8085 HTTP 200 status degraded")
)


def test_a_failure_dev_already_has_is_not_the_prs() -> None:
    rb = pool.judge({**GOOD, "probe": DEV_FAILS}, base_probe=DEV_FAILS)
    assert rb.outcome == "PASS", rb
    assert any("fail to wire at dev too" in n for n in rb.notes)
    assert any("status degraded" in n for n in rb.notes)


def test_without_a_base_every_failed_contract_is_the_prs() -> None:
    assert pool.judge({**GOOD, "probe": DEV_FAILS}).outcome == "FAIL"


def test_a_contract_that_wires_at_the_base_and_fails_at_the_head_is_the_prs() -> None:
    head = DEV_FAILS.replace(
        "projection_traces", "projection_traces projection_pr_landing"
    )
    rb = pool.judge({**GOOD, "probe": head}, base_probe=DEV_FAILS)
    assert rb.outcome == "FAIL"
    assert any("projection_pr_landing" in n for n in rb.notes)


def test_a_missing_failed_contracts_line_is_an_unread_log_not_a_clean_one() -> None:
    probe = "\n".join(
        line
        for line in GOOD["probe"].splitlines()
        if not line.startswith("failed-contracts")
    )
    assert pool.judge({**GOOD, "probe": probe}).checks["no_wiring_failures"] is False


def test_base_control_runs_on_the_same_lease_only_when_needed(tmp_path: Path) -> None:
    hosts = {"pool-a": FakeHost()}
    hosts["pool-a"].phases["probe"] = DEV_FAILS
    hosts["pool-a"].base_phases["probe"] = DEV_FAILS
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        90,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        with_base_control=True,
    )
    assert code == pool.EXIT_PASS, text
    ran = hosts["pool-a"].ran
    assert ran[: len(pool.PHASES)] == list(pool.PHASES)
    assert ran[len(pool.PHASES) :] == [
        "base:snap-pre",
        "base:clone",
        "base:build",
        "base:probe",
        "base:teardown",
        "base:snap-post",
    ]
    assert "base control run at dev" in text
    assert hosts["pool-a"].lease is None

    clean = {"pool-a": FakeHost()}
    pool.run_proof(
        CFG,
        FakeTransport(clean),
        _params(tmp_path),
        "me",
        90,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        with_base_control=True,
    )
    assert not any(r.startswith("base:") for r in clean["pool-a"].ran)


def test_params_keep_a_quote_that_belongs_to_the_value(tmp_path: Path) -> None:
    # Found live (omnibase_infra#4154, 2026-09-27): stripping every trailing quote
    # cut the closing quote off a SQL literal and every readback query errored.
    p = tmp_path / "p.env"
    p.write_text(
        "SQL=\"select 1 where x<>'postgres'\"\nA='plain'\nB=bare\n", encoding="utf-8"
    )
    params = pool.read_params(p)
    assert params["SQL"] == "select 1 where x<>'postgres'"
    assert params["A"] == "plain" and params["B"] == "bare"


def _with_reasons(probe: str, reasons: dict[str, str]) -> str:
    lines = [
        f"  failed-contract-reason omnibase-infra-local-omninode-runtime {n}: handler=H{n}: {r}"
        for n, r in reasons.items()
    ]
    return probe + "\n".join(lines) + "\n"


DSN = "ValueError: Projection handler requires topology bindings with configured DSNs: omninode_x"


def test_a_new_contract_failing_for_devs_own_reason_is_the_same_gap() -> None:
    # Found live (omnibase_infra#4154 + omnimarket#2905, 2026-09-27): the PR's new
    # projection failed to wire on the laptop bundle for the same missing-DSN
    # reason 27 projections at dev fail for.
    head = _with_reasons(
        DEV_FAILS.replace(
            "projection_traces", "projection_traces projection_session_content"
        ),
        {
            "projection_baselines": DSN,
            "projection_traces": DSN,
            "projection_session_content": DSN,
        },
    )
    rb = pool.judge({**GOOD, "probe": head}, base_probe=DEV_FAILS)
    assert rb.outcome == "PASS", rb
    assert any(
        "projection_session_content fails to wire for the reason" in n for n in rb.notes
    )


def test_a_new_contract_failing_for_a_different_reason_is_the_prs() -> None:
    head = _with_reasons(
        DEV_FAILS.replace(
            "projection_traces", "projection_traces projection_session_content"
        ),
        {
            "projection_baselines": DSN,
            "projection_traces": DSN,
            "projection_session_content": "ImportError: cannot import name X",
        },
    )
    rb = pool.judge({**GOOD, "probe": head}, base_probe=DEV_FAILS)
    assert rb.outcome == "FAIL"
    assert any(
        "projection_session_content" in n and "base does not have" in n
        for n in rb.notes
    )


def test_an_inconclusive_run_is_released_as_aborted(tmp_path: Path) -> None:
    # The ledger grammar takes PASS, FAIL or ABORTED on a surface RELEASE
    # (MSG 2026-09-27T16:25:02Z-drain-runtime-token-83).
    hosts = {"pool-a": FakeHost()}
    hosts["pool-a"].phases["build"] = "build rc=1"
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_INCONCLUSIVE
    assert "LAB PROOF INCONCLUSIVE" in text
    assert "result=ABORTED" in text and "result=INCONCLUSIVE" not in text


def test_a_failed_build_exits_the_phase_non_zero() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    build = text[text.index("build)") : text.index("probe)")]
    assert 'if [ "$brc" != 0 ]' in build and "exit 1" in build
    assert '[ "$urc" = 0 ] || exit 1' in build


def test_the_keychain_host_builds_with_an_isolated_docker_config(
    tmp_path: Path,
) -> None:
    assert CFG.host("pool-b").docker_config == "isolated"
    assert CFG.host("pool-a").docker_config == "host"
    hosts = {"pool-b": FakeHost(cores=24, load=2.0)}
    pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "pool-b",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    env = next(tmp_path.glob("*.resolved.env")).read_text(encoding="utf-8")
    assert "DOCKER_CONFIG_MODE=isolated" in env


FAILED_TEST = "tests/unit/test_x.py::test_git_trace"


def test_a_focused_failure_that_fails_at_dev_too_is_dev_inherited() -> None:
    # Found live (omnimarket#3016 at 1273db3ac on .101, 2026-09-27): a test from
    # dev asserts a git trace string Apple Git 2.39 does not print.
    tests = (
        f"FAILED {FAILED_TEST} - AssertionError\nomnibase_infra focused rc=1 t\n"
        f"dev-control omnibase_infra {FAILED_TEST} rc=1 at abc\n"
    )
    rb = pool.judge({**GOOD, "tests": tests})
    assert rb.checks["focused_tests"] is True
    assert any("fails at dev too" in n for n in rb.notes)


@pytest.mark.parametrize("dev_rc", ["0", "4", "5", None])
def test_a_focused_failure_dev_does_not_share_is_the_prs(dev_rc: str | None) -> None:
    tests = f"FAILED {FAILED_TEST} - AssertionError\nomnibase_infra focused rc=1 t\n"
    if dev_rc is not None:
        tests += f"dev-control omnibase_infra {FAILED_TEST} rc={dev_rc} at abc\n"
    rb = pool.judge({**GOOD, "tests": tests})
    assert rb.checks["focused_tests"] is False
    assert any("fails at the head only" in n for n in rb.notes)


# ------------------------------------------------------------------ .201 pre-PR slots

SLOT_GOOD = {
    **GOOD,
    "build": ("prepr_verify_lane rc=0 2026-09-27T14:40:00Z\nbuild rc=0\nup rc=0"),
    "probe": (
        "file omnibase_infra/x.py image=aaa build-tree=aaa match=yes\n"
        "port 28085 HTTP 200 status healthy healthy True failed_handlers 0\n"
        "port 28086 HTTP 200 status healthy healthy True failed_handlers 0\n"
        "slot-migration-gate health=healthy\n"
        "omninode-prepr-1-runtime lines=900 autowire-fail=0 dup-dispatcher=0 ERROR=0 Traceback=0\n"
        "omninode-prepr-1-runtime-effects lines=800 autowire-fail=0 dup-dispatcher=0 ERROR=0 Traceback=0\n"
        "failed-contracts omninode-prepr-1-runtime: \n"
        "failed-contracts omninode-prepr-1-runtime-effects: \n"
        "slot-published-ports=23002,28085,28086,28090 outside-block=none\n"
    ),
    "snap-post": (
        "snapshot-diff=0\nneighbour-diff=0 neighbours=40\n"
        "snap-post lines=50 local-containers=0 listeners=0"
    ),
    "teardown": (
        "prepr_teardown_slot rc=0\nslot-teardown verdict=CLEAN\n"
        "containers=0 volumes=0 networks=0 images=0 listeners=0 workdir=gone\n"
        "positive control omnibase-infra containers 26 running 26"
    ),
}


class RecordingTransport(FakeTransport):
    def __init__(self, hosts: dict[str, FakeHost]) -> None:
        super().__init__(hosts)
        self.commands: list[tuple[str, str]] = []

    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        self.commands.append((host.name, command))
        return super().run(host, command, timeout)


def test_a_slot_is_probed_on_its_own_project_ports_and_lease() -> None:
    hosts = {"pool-c-slot-1": FakeHost(cores=32, load=10.0)}
    tp = RecordingTransport(hosts)
    states = pool.survey(CFG, tp, {}, NOW)
    by = {s.host.name: s for s in states}
    assert by["pool-c-slot-1"].verdict == "FREE"
    probe = next(c for n, c in tp.commands if n == "pool-c-slot-1")
    assert "com.docker.compose.project=omnibase-infra-prepr-1 " in probe
    assert ":(28085|28086|28090|23002)" in probe
    assert ".onex-prepr-pool/lease-prepr-1/lease.json" in probe
    assert "8085|" not in probe.replace("28085|", "")

    busy = {"pool-c-slot-2": FakeHost(cores=32, load=10.0, slot=12)}
    s2 = {s.host.name: s for s in pool.survey(CFG, FakeTransport(busy), {}, NOW)}
    assert s2["pool-c-slot-2"].verdict == "BUSY"
    slot2 = CFG.host("pool-c-slot-2")
    assert f"{slot2.compose_project} containers present" in s2[slot2.name].detail


def test_a_slot_run_passes_end_to_end_with_the_slot_ports(tmp_path: Path) -> None:
    hosts = {"pool-c-slot-1": FakeHost(cores=32, load=10.0)}
    hosts["pool-c-slot-1"].phases = dict(SLOT_GOOD)
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        150,
        "pool-c-slot-1",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_PASS, text
    assert "pre-PR slot project omnibase-infra-prepr-1" in text
    assert "ok   health_28085" in text and "ok   health_28086" in text
    assert "surface=c-slot-1 result=PASS restored=yes" in text
    env = next(tmp_path.glob("*.resolved.env")).read_text(encoding="utf-8")
    for line in (
        "SLOT_KIND=prepr-slot",
        "PREPR_SLOT=1",
        "PROJECT=omnibase-infra-prepr-1",
        "MAIN_PORT=28085",
        "EFFECTS_PORT=28086",
        "POSITIVE_CONTROL=omnibase-infra",
    ):
        assert line in env.splitlines(), line
    assert hosts["pool-c-slot-1"].lease is None
    assert (
        "NEIGHBOUR_PROJECTS="
        + shlex.quote(pool.neighbour_projects(CFG.host("pool-c-slot-1")))
        in env.splitlines()
    )
    assert (
        "slot ports: published 23002,28085,28086,28090, all in the reserved block"
        in text
    )
    assert "40 declared-lane containers, none recreated or restarted" in text


def _pool_with_overlays(tmp_path: Path, projects: tuple[str, ...]) -> Any:
    lab = tmp_path / "config" / "lab"
    (lab / "lane-overlays").mkdir(parents=True)
    table = lab / "prepr_runtime_pool.yaml"
    table.write_text(POOL_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    for project in projects:
        (lab / "lane-overlays" / f"{project}.bifrost.yaml").write_text(
            "providers: []\n", encoding="utf-8"
        )
    return pool.load_pool_config(table)


def test_a_slot_run_ships_the_slot_s_own_lane_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OMN-19064, found live 2026-10-19 on a pre-PR slot: lane overlays left the repo
    # (OMN-20932) and the pool's non-login ssh shell names no config root, so the
    # slot entrypoint refused "no lane overlay for the slot's compose project"
    # before any build. The driver now ships the overlay beside the pool table.
    monkeypatch.delenv("ONEX_LANE_OVERLAY_DIR", raising=False)
    slot = CFG.host("pool-c-slot-1")
    cfg = _pool_with_overlays(tmp_path, (slot.compose_project,))
    hosts = {"pool-c-slot-1": FakeHost(cores=32, load=10.0)}
    hosts["pool-c-slot-1"].phases = dict(SLOT_GOOD)
    tp = FakeTransport(hosts)
    work = tmp_path / "work"
    work.mkdir()
    code, text = pool.run_proof(
        cfg,
        tp,
        _params(work),
        "me",
        150,
        "pool-c-slot-1",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_PASS, text
    assert any(r.endswith("/overlay.bifrost.yaml") for _, r in tp.puts), tp.puts
    env = next(work.glob("*.resolved.env")).read_text(encoding="utf-8")
    assert any(ln.startswith("OVERLAY_FILE=") for ln in env.splitlines())


def test_the_slot_overlay_is_read_from_the_pool_table_s_config_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ONEX_LANE_OVERLAY_DIR", raising=False)
    slot = CFG.host("pool-c-slot-1")
    cfg = _pool_with_overlays(tmp_path, (slot.compose_project,))
    found = pool.slot_lane_overlay(cfg, cfg.host("pool-c-slot-1"))
    assert found == (
        tmp_path / "config/lab/lane-overlays" / f"{slot.compose_project}.bifrost.yaml"
    )
    # an isolated member mounts its own stack's overlay, never a slot's
    isolated = next(h for h in cfg.hosts if h.kind != "prepr-slot")
    assert pool.slot_lane_overlay(cfg, isolated) is None
    # a slot with no overlay file ships none; the on-host refusal names it
    other = next(h for h in cfg.hosts if h.kind == "prepr-slot" and h.slot != slot.slot)
    assert pool.slot_lane_overlay(cfg, other) is None
    # ONEX_LANE_OVERLAY_DIR outranks the config root, as in compose_files.sh
    alt = tmp_path / "alt"
    alt.mkdir()
    (alt / f"{other.compose_project}.bifrost.yaml").write_text("x\n", encoding="utf-8")
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(alt))
    assert (
        pool.slot_lane_overlay(cfg, other)
        == alt / f"{other.compose_project}.bifrost.yaml"
    )
    assert pool.slot_lane_overlay(cfg, cfg.host("pool-c-slot-1")) is None


def test_a_slot_run_publishing_outside_its_block_fails(tmp_path: Path) -> None:
    # OMN-19064 AC-2: a slot whose project publishes a port outside its reserved
    # block is a FAIL, even with every other check green
    hosts = {"pool-c-slot-1": FakeHost(cores=32, load=10.0)}
    hosts["pool-c-slot-1"].phases = {
        **SLOT_GOOD,
        "probe": SLOT_GOOD["probe"].replace(
            "outside-block=none", "outside-block=28098"
        ),
    }
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        150,
        "pool-c-slot-1",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_FAIL, text
    assert "published 28098 outside the reserved block" in text


def test_a_params_file_cannot_move_a_run_onto_another_project(tmp_path: Path) -> None:
    p = _params(tmp_path)
    p.write_text(
        p.read_text(encoding="utf-8")
        + "PROJECT=omnibase-infra\nSLOT_KIND=prepr-slot\nPREPR_SLOT=1\nMAIN_PORT=8085\n",
        encoding="utf-8",
    )
    pool.run_proof(
        CFG,
        FakeTransport({"pool-a": FakeHost()}),
        p,
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    env = next(tmp_path.glob("*.resolved.env")).read_text(encoding="utf-8").splitlines()
    assert "PROJECT=omnibase-infra-local" in env and "PROJECT=omnibase-infra" not in env
    assert "SLOT_KIND=isolated" in env and "MAIN_PORT=8085" in env
    assert sum(line.startswith("PROJECT=") for line in env) == 1


def test_a_slot_migration_failure_is_the_prs_finding() -> None:
    build = "prepr_verify_lane rc=9\nbuild rc=0\nslot-migration FAILED\nup rc=9"
    rb = pool.judge({**SLOT_GOOD, "build": build, "probe": ""}, ports=(28085, 28086))
    assert rb.outcome == "FAIL"
    assert rb.checks["stack_built"] and not rb.checks["migration_gate_healthy"]
    assert any("migration failed" in n for n in rb.notes)


def test_a_slot_teardown_that_is_not_clean_is_not_restored() -> None:
    td = SLOT_GOOD["teardown"].replace("verdict=CLEAN", "verdict=RESIDUE")
    rb = pool.judge({**SLOT_GOOD, "teardown": td}, ports=(28085, 28086))
    assert not rb.restored
    assert any("slot teardown verdict RESIDUE" in n for n in rb.notes)
    assert pool.judge(SLOT_GOOD, ports=(28085, 28086)).restored


def test_the_prove_script_brings_a_slot_up_and_down_only_through_the_sanctioned_pair() -> (
    None
):
    text = PROVE_SH.read_text(encoding="utf-8")
    build = text[text.index("build)") : text.index("probe)")]
    slot_build = build[: build.index("export DEPLOY_SOURCE_REFS_OUT")]
    assert "scripts/runtime_build/prepr_verify_lane.sh --slot" in slot_build
    assert "docker compose" not in slot_build and "catalog.cli" not in slot_build
    teardown = text[text.index("teardown)") :]
    slot_td = teardown[: teardown.index("IMGS=")]
    assert "prepr_teardown_slot.sh" in slot_td
    assert not re.search(r"docker (rm|volume rm|network rm|image rm)", slot_td)
    # a slot another holder started is never torn down by this run
    assert "foreign-slot" in slot_build and "foreign-slot" in slot_td


@pytest.mark.parametrize(
    "params",
    [
        "SLOT_KIND=prepr-slot\nPREPR_SLOT=1\nPROJECT=omnibase-infra\n",
        "SLOT_KIND=prepr-slot\nPREPR_SLOT=3\n",
        "SLOT_KIND=isolated\nPROJECT=omnibase-infra-stability-test\n",
    ],
)
def test_the_prove_script_refuses_a_project_it_did_not_derive(
    tmp_path: Path, params: str
) -> None:
    env = tmp_path / "params.env"
    env.write_text(f"TAG=t\nW={tmp_path}/w\n{params}", encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(PROVE_SH), str(env), "snap-pre"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (tmp_path / "w").exists()


def test_the_slot_publishes_the_port_its_effects_runtime_listens_on() -> None:
    # Found live (omnibase_infra#4180 on pool-c-slot-1, 2026-09-27): the slot
    # published 28086 onto container port 8086, where nothing listens, so its
    # effects runtime was healthy inside and unreachable from the host.
    docker = REPO_ROOT / "docker"
    infra = (docker / "docker-compose.infra.yml").read_text(encoding="utf-8")
    prepr = (docker / "docker-compose.prepr.yml").read_text(encoding="utf-8")
    base = re.search(r'"\$\{DEV_RUNTIME_EFFECTS_PORT:[^}]*\}:(\d+)"', infra)
    slot = re.search(r'"\$\{PREPR_RUNTIME_EFFECTS_PORT:[^}]*\}:(\d+)"', prepr)
    main = re.search(r'"\$\{PREPR_RUNTIME_MAIN_PORT:[^}]*\}:(\d+)"', prepr)
    assert base and slot and main
    assert slot.group(1) == base.group(1) == main.group(1) == "8085"


def test_one_holder_on_both_slots_gets_two_work_directories(tmp_path: Path) -> None:
    works = []
    for name in ("pool-c-slot-1", "pool-c-slot-2"):
        hosts = {name: FakeHost(cores=32, load=4.0)}
        hosts[name].phases = dict(SLOT_GOOD)
        d = tmp_path / name
        d.mkdir()
        pool.run_proof(
            CFG,
            FakeTransport(hosts),
            _params(d),
            "me",
            150,
            name,
            [],
            now_fn=lambda: NOW,
            log=lambda s: None,
        )
        env = next(d.glob("*.resolved.env")).read_text(encoding="utf-8")
        works.append(next(x for x in env.splitlines() if x.startswith("W=")))
    assert len(set(works)) == 2, works


def test_a_slot_is_picked_only_when_every_isolated_member_is_taken() -> None:
    hosts = {
        "pool-a": FakeHost(cores=12, load=11.0),
        "pool-c-slot-1": FakeHost(cores=32, load=1.0),
        "pool-c-slot-2": FakeHost(cores=32, load=1.0),
    }
    assert pool.pick(_survey(hosts)).host.name == "pool-a"
    hosts["pool-a"].online = False
    assert pool.pick(_survey(hosts)).host.name == "pool-c-slot-1"


# ------------------------------------------------------------------ group proof

GROUP_CLONE = """omnibase_infra group base cd2cc37bd48852d195eb18d3d6b08b04d973f064 (dev cd2cc37bd48852d195eb18d3d6b08b04d973f064)
omnibase_infra#4134 fetched head fdd93c786c27fd9daf2a878a0d9cd189f4868989 expected fdd93c786c27fd9daf2a878a0d9cd189f4868989 match=yes
group-step omnibase_infra#4134 commit 27629e91ea63223168320f848a371aac38868c27 tree 29a02c42f933477f8210ed83fdcd706a3fe07e30
omnibase_infra#4198 fetched head e812bfd787753ea76eb7f599b7cdbf933ebf534c expected e812bfd787753ea76eb7f599b7cdbf933ebf534c match=yes
group-step omnibase_infra#4198 commit 606e69a0e2ef1d176049fb83ba2f67dfb1932bcc tree bc5de1bd286eee60ef64e5970e120eeecf48f756
omnibase_infra#4214 fetched head 0ba71c956606c2a500b4f0d095bcde22e2ffb29b expected 0ba71c956606c2a500b4f0d095bcde22e2ffb29b match=yes
group-step omnibase_infra#4214 commit 8036b4c7a3b574ddfc29266d991163574a7b7d0c tree bca014d9c20c11139da49d5395d94b87b86b7a28
group-commit omnibase_infra 8036b4c7a3b574ddfc29266d991163574a7b7d0c tree bca014d9c20c11139da49d5395d94b87b86b7a28 base cd2cc37bd48852d195eb18d3d6b08b04d973f064 tree-agrees=yes
"""
GROUP_MEMBERS = [
    ("4134", "fdd93c786c27fd9daf2a878a0d9cd189f4868989"),
    ("4198", "e812bfd787753ea76eb7f599b7cdbf933ebf534c"),
    ("4214", "0ba71c956606c2a500b4f0d095bcde22e2ffb29b"),
]
GROUP_SPEC = " ".join(f"{number}:{head}" for number, head in GROUP_MEMBERS)


@pytest.mark.unit
def test_group_facts_parse_real_host_clone_output() -> None:
    facts = pool.group_facts(GROUP_CLONE)
    assert facts is not None
    assert facts.repo == "omnibase_infra"
    assert facts.members == [(f"omnibase_infra#{n}", h) for n, h in GROUP_MEMBERS]
    assert facts.commit == "8036b4c7a3b574ddfc29266d991163574a7b7d0c"
    assert facts.tree == "bca014d9c20c11139da49d5395d94b87b86b7a28"
    assert facts.base == "cd2cc37bd48852d195eb18d3d6b08b04d973f064"
    assert facts.tree_agrees == "yes"
    assert facts.failures == []
    assert facts.empty == []


@pytest.mark.unit
def test_non_group_clone_has_no_group_facts() -> None:
    assert pool.group_facts(GOOD["clone"]) is None


@pytest.mark.unit
def test_judge_accepts_a_group_built_with_the_planned_tree() -> None:
    rb = pool.judge({**GOOD, "clone": GROUP_CLONE})
    assert rb.checks["group_built"] is True
    assert rb.outcome == "PASS"


@pytest.mark.unit
def test_judge_names_a_group_conflict_without_a_final_commit() -> None:
    conflict = "group-conflict omnibase_infra#4134 CONFLICT (content)"
    clone = "\n".join(GROUP_CLONE.splitlines()[:2]) + "\n" + conflict + "\n"
    rb = pool.judge({**GOOD, "clone": clone})
    assert rb.checks["group_built"] is False
    assert any(conflict in note for note in rb.notes)
    assert rb.group is not None and rb.group.commit is None


@pytest.mark.unit
def test_judge_rejects_a_group_tree_that_disagrees() -> None:
    rb = pool.judge(
        {**GOOD, "clone": GROUP_CLONE.replace("tree-agrees=yes", "tree-agrees=NO")}
    )
    assert rb.checks["group_built"] is False
    assert any("different tree" in note for note in rb.notes)


@pytest.mark.unit
def test_group_members_parse_valid_spec_in_order() -> None:
    assert pool.group_members(GROUP_SPEC) == GROUP_MEMBERS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (f"4134:abc 4198:{'b' * 40}", "40-hex head sha"),
        (f"4134:{'a' * 40}", "at least two members"),
        (f"4134:{'a' * 40} 4134:{'b' * 40}", "names a PR twice"),
    ],
)
def test_group_members_reject_invalid_specs(spec: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        pool.group_members(spec)


@pytest.mark.unit
def test_group_readback_names_every_member_and_group() -> None:
    facts = pool.group_facts(GROUP_CLONE)
    assert facts is not None
    rb = pool.Readback(
        checks={"stack_built": True, "group_built": True},
        notes=[],
        restored=True,
        residue="clean",
        group=facts,
    )
    text = pool.render_readback(
        rb,
        CFG.host("pool-a"),
        {"INFRA_GROUP": GROUP_SPEC},
        "2026-09-28T10:00:00Z",
        "2026-09-28T10:30:00Z",
    )
    lines = text.splitlines()
    assert lines[0].startswith(
        "LAB PROOF PASS: omnibase_infra#4134 head fdd93c786c + "
        "omnibase_infra#4198 head e812bfd787 + "
        "omnibase_infra#4214 head 0ba71c9566 on pool-a"
    )
    assert any(line.startswith("  group: 3 member(s)") for line in lines)


@pytest.mark.unit
def test_market_group_readback_names_every_member() -> None:
    rb = pool.Readback(
        checks={"stack_built": True, "group_built": True},
        notes=[],
        restored=True,
        residue="clean",
    )
    text = pool.render_readback(
        rb,
        CFG.host("pool-a"),
        {"MARKET_GROUP": GROUP_SPEC},
        "2026-09-28T10:00:00Z",
        "2026-09-28T10:30:00Z",
    )
    assert text.splitlines()[0].startswith(
        "LAB PROOF PASS: omnimarket#4134 head fdd93c786c + "
        "omnimarket#4198 head e812bfd787 + "
        "omnimarket#4214 head 0ba71c9566 on pool-a"
    )


class NoHostTransport:
    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        pytest.fail(
            "invalid group params must be rejected before running a host command"
        )

    def put(self, host: Any, local: Path, remote: str) -> int:
        pytest.fail("invalid group params must be rejected before copying to a host")


@pytest.mark.unit
def test_run_rejects_pr_and_group_before_touching_any_host(tmp_path: Path) -> None:
    path = _params(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8") + f"INFRA_GROUP='{GROUP_SPEC}'\n",
        encoding="utf-8",
    )
    code, text = pool.run_proof(
        CFG,
        NoHostTransport(),
        path,
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == 5
    assert "both INFRA_PR and INFRA_GROUP" in text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (f"MARKET_GROUP='{GROUP_SPEC}'\n", "both MARKET_PR and MARKET_GROUP"),
        (
            f"INFRA_GROUP='{GROUP_SPEC}'\nMARKET_GROUP='{GROUP_SPEC}'\n",
            "both INFRA_GROUP and MARKET_GROUP",
        ),
        (f"MARKET_GROUP='{GROUP_SPEC}'\n", "both INFRA_PR and MARKET_GROUP"),
    ],
)
def test_run_rejects_market_group_cross_subjects_before_touching_a_host(
    tmp_path: Path, extra: str, reason: str
) -> None:
    path = _params(tmp_path)
    if "MARKET_PR" in reason:
        path.write_text(
            "MARKET_PR=4210\nMARKET_HEAD=" + "a" * 40 + "\n" + extra,
            encoding="utf-8",
        )
    else:
        path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")
    code, text = pool.run_proof(
        CFG,
        NoHostTransport(),
        path,
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_USAGE
    assert reason in text


@pytest.mark.unit
def test_run_rejects_a_one_member_market_group_before_touching_a_host(
    tmp_path: Path,
) -> None:
    path = _params(tmp_path)
    path.write_text(f"MARKET_GROUP='4134:{'a' * 40}'\n", encoding="utf-8")
    code, text = pool.run_proof(
        CFG,
        NoHostTransport(),
        path,
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_USAGE
    assert "MARKET_GROUP needs at least two members" in text


@pytest.mark.unit
def test_prove_script_fetches_market_groups_and_probes_them_as_subjects() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    clone = text[text.index("clone)") : text.index("build)")]
    assert (
        'fetch_group "$R/omnimarket" omnimarket "$MARKET_GROUP" '
        '"${MARKET_GROUP_BASE:-}" "${MARKET_GROUP_TREE:-}"' in clone
    )
    assert (
        '[ -n "${MARKET_PR:-}${MARKET_GROUP:-}${MARKET_REF_SHA:-}" ] '
        '&& SUBJECTS="$SUBJECTS omnimarket"' in text
    )


# ------------------------------------------------- overlay and post-up command

EXERCISE_OK = (
    "exercise start 2026-10-17T23:00:00Z command: echo counted\n"
    "delegate_skill_failed_records=1\n"
    "exercise rc=0 2026-10-17T23:01:00Z"
)


def _overlay(tmp_path: Path) -> Path:
    path = tmp_path / "held.bifrost.yaml"
    path.write_text("endpoint_url: &model_endpoint 'http://held.invalid:1'\n")
    return path


def _run_with_exercise(
    tmp_path: Path, host: FakeHost, exercise: str, **kw: Any
) -> tuple[int, str, FakeTransport]:
    host.phases["exercise"] = exercise
    tp = FakeTransport({"pool-a": host})
    code, text = pool.run_proof(
        CFG,
        tp,
        _params(tmp_path),
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        **kw,
    )
    return code, text, tp


@pytest.mark.unit
def test_an_overlay_and_a_post_up_command_reach_the_host_and_run_after_probe(
    tmp_path: Path,
) -> None:
    host = FakeHost()
    code, text, tp = _run_with_exercise(
        tmp_path,
        host,
        EXERCISE_OK,
        overlay=_overlay(tmp_path),
        post_up_command="echo counted",
    )
    assert code == pool.EXIT_PASS, text
    assert host.ran == [
        "snap-pre",
        "clone",
        "build",
        "probe",
        "exercise",
        "tests",
        "teardown",
        "snap-post",
    ]
    assert any(r.endswith("/overlay.bifrost.yaml") for _, r in tp.puts)
    env = next(tmp_path.glob("*.resolved.env")).read_text(encoding="utf-8")
    assert "POST_UP_COMMAND='echo counted'" in env.splitlines()
    assert any(line.startswith("OVERLAY_FILE=/tmp/") for line in env.splitlines())
    assert "ok   post_up_command" in text
    assert "note: exercise output: delegate_skill_failed_records=1" in text


@pytest.mark.unit
def test_a_post_up_command_that_exits_non_zero_fails_the_proof(tmp_path: Path) -> None:
    code, text, _ = _run_with_exercise(
        tmp_path,
        FakeHost(),
        "exercise start x\ncount was 2\nexercise rc=3 2026-10-17T23:01:00Z",
        post_up_command="false",
    )
    assert code == pool.EXIT_FAIL, text
    assert text.startswith("LAB PROOF FAIL")
    assert "FAIL post_up_command" in text
    assert "note: exercise output: count was 2" in text


@pytest.mark.unit
def test_a_post_up_command_that_printed_no_exit_line_is_not_a_pass(
    tmp_path: Path,
) -> None:
    code, text, _ = _run_with_exercise(
        tmp_path, FakeHost(), "exercise start x", post_up_command="true"
    )
    assert code == pool.EXIT_FAIL, text
    assert "FAIL post_up_command" in text


@pytest.mark.unit
def test_a_run_without_a_post_up_command_runs_no_exercise_phase(
    tmp_path: Path,
) -> None:
    host = FakeHost()
    code, _, tp = _run_with_exercise(tmp_path, host, EXERCISE_OK)
    assert code == pool.EXIT_PASS
    assert "exercise" not in host.ran
    assert not any(r.endswith("/overlay.bifrost.yaml") for _, r in tp.puts)


@pytest.mark.unit
def test_a_missing_overlay_file_is_a_usage_error_before_any_host_is_touched(
    tmp_path: Path,
) -> None:
    host = FakeHost()
    code, text, _ = _run_with_exercise(
        tmp_path, host, EXERCISE_OK, overlay=tmp_path / "absent.yaml"
    )
    assert code == pool.EXIT_USAGE
    assert "absent.yaml" in text
    assert host.ran == []


@pytest.mark.unit
def test_a_params_file_cannot_set_the_overlay_or_the_command(tmp_path: Path) -> None:
    path = _params(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "OVERLAY_FILE=/etc/passwd\nPOST_UP_COMMAND='rm -rf /'\n",
        encoding="utf-8",
    )
    host = FakeHost()
    pool.run_proof(
        CFG,
        FakeTransport({"pool-a": host}),
        path,
        "me",
        60,
        None,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    env = next(tmp_path.glob("*.resolved.env")).read_text(encoding="utf-8")
    assert "/etc/passwd" not in env and "rm -rf" not in env


@pytest.mark.unit
def test_the_cli_takes_an_overlay_and_a_post_up_command() -> None:
    args = pool.build_parser().parse_args(
        [
            "run",
            "--params",
            "p.env",
            "--holder",
            "me",
            "--overlay",
            "o.yaml",
            "--post-up-command",
            "echo hi",
        ]
    )
    assert args.overlay == Path("o.yaml") and args.post_up_command == "echo hi"


@pytest.mark.unit
def test_the_prove_script_mounts_the_overlay_on_a_slot_and_on_an_isolated_stack() -> (
    None
):
    text = PROVE_SH.read_text(encoding="utf-8")
    build = text[text.index("build)") : text.index("probe)")]
    slot = build[: build.index("export DEPLOY_SOURCE_REFS_OUT")]
    # a slot reads its Bifrost overlay from ONEX_LANE_OVERLAY_DIR/<project>.bifrost.yaml
    assert slot.index('"$W/lane-overlay/$P.bifrost.yaml"') < slot.index(
        "prepr_verify_lane.sh"
    )
    assert slot.index('export ONEX_LANE_OVERLAY_DIR="$W/lane-overlay"') < slot.index(
        "prepr_verify_lane.sh"
    )
    isolated = build[build.index("export DEPLOY_SOURCE_REFS_OUT") :]
    assert isolated.index(
        'cp "$OVERLAY_FILE" "$W/local.bifrost.yaml"'
    ) < isolated.index("catalog.cli generate local")


@pytest.mark.unit
def test_the_exercise_phase_runs_the_command_and_reports_its_exit_code(
    tmp_path: Path,
) -> None:
    work = tmp_path / "w"
    (work / "root" / "omnibase_infra").mkdir(parents=True)
    params = tmp_path / "params.env"
    params.write_text(
        f"TAG=t\nW={work}\nMAIN_PORT=18085\nEFFECTS_PORT=18086\n"
        "POST_UP_COMMAND='echo main=$MAIN_PORT project=$PROJECT; echo marker; exit 3'\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        ["bash", str(PROVE_SH), str(params), "exercise"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "main=18085 project=omnibase-infra-local" in done.stdout, done.stderr
    assert "marker" in done.stdout
    assert re.search(r"^exercise rc=3 ", done.stdout, re.M), done.stdout
    assert done.returncode == 0


# ------------------------------------------------------------------ OMN-20117
# A proof that must act on the running stack (restart a runtime during a live
# call) names one command in the params; the host runs it after the probe.


def _post_params(tmp_path: Path) -> Path:
    p = _params(tmp_path)
    p.write_text(
        p.read_text(encoding="utf-8")
        + "POST_BRINGUP_CMD='docker restart $MAIN_RUNTIME'\n",
        encoding="utf-8",
    )
    return p


def _run_with_post(
    tmp_path: Path, post: str | None, host: str = "pool-a"
) -> tuple[int, str, FakeHost]:
    hosts = {host: FakeHost()}
    if host == "pool-c-slot-1":
        hosts[host].phases = dict(SLOT_GOOD)
    if post is not None:
        hosts[host].phases["post-bringup"] = post
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _post_params(tmp_path),
        "me",
        150,
        host,
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    return code, text, hosts[host]


def test_a_named_post_bringup_command_runs_between_probe_and_tests(
    tmp_path: Path,
) -> None:
    code, text, host = _run_with_post(
        tmp_path,
        "post-bringup cmd: docker restart x\npost-bringup rc=0 2026-09-27T14:35:00Z",
    )
    assert code == pool.EXIT_PASS, text
    assert host.ran == [
        "snap-pre",
        "clone",
        "build",
        "probe",
        "post-bringup",
        "tests",
        "teardown",
        "snap-post",
    ]
    assert "ok   post_bringup" in text


def test_a_post_bringup_command_that_fails_fails_the_proof(tmp_path: Path) -> None:
    code, text, _ = _run_with_post(tmp_path, "post-bringup rc=3 2026-09-27T14:35:00Z")
    assert code == pool.EXIT_FAIL, text
    assert "FAIL post_bringup" in text
    assert "post-bringup command rc=3" in text


def test_a_post_bringup_command_that_never_finished_fails_the_proof(
    tmp_path: Path,
) -> None:
    code, text, _ = _run_with_post(tmp_path, "")
    assert code == pool.EXIT_FAIL, text
    assert "post-bringup command printed no exit status" in text


def test_a_post_bringup_command_runs_on_a_prepr_slot(tmp_path: Path) -> None:
    code, text, host = _run_with_post(
        tmp_path, "post-bringup rc=0 2026-09-27T14:35:00Z", host="pool-c-slot-1"
    )
    assert code == pool.EXIT_PASS, text
    assert "post-bringup" in host.ran
    assert f"pre-PR slot project {CFG.host('pool-c-slot-1').compose_project}" in text


def test_without_a_post_bringup_command_the_phase_and_check_are_absent(
    tmp_path: Path,
) -> None:
    hosts = {"pool-a": FakeHost()}
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_PASS, text
    assert "post-bringup" not in hosts["pool-a"].ran
    assert "post_bringup" not in text


def test_the_post_bringup_receipt_row_names_the_lines_that_decided_it() -> None:
    outputs = {**GOOD, "post-bringup": "post-bringup rc=0 2026-09-27T14:35:00Z"}
    rb = pool.judge(outputs)
    rows = pool.receipt_checks(rb, outputs, (8085, 8086))
    row = next(r for r in rows if r.name == "post_bringup")
    assert row.ok and "post-bringup rc=0" in row.evidence


def test_the_base_control_does_not_run_the_post_bringup_command(
    tmp_path: Path,
) -> None:
    hosts = {"pool-a": FakeHost()}
    hosts["pool-a"].phases["probe"] = GOOD["probe"].replace(
        "healthy True failed_handlers 0", "healthy False failed_handlers 1", 1
    )
    hosts["pool-a"].phases["post-bringup"] = "post-bringup rc=0 x"
    pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _post_params(tmp_path),
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        with_base_control=True,
    )
    assert "base:post-bringup" not in hosts["pool-a"].ran
    assert "base:probe" in hosts["pool-a"].ran


def test_the_prove_script_runs_the_named_command_with_the_members_names(
    tmp_path: Path,
) -> None:
    slot = 2
    work = tmp_path / "w"
    (work / "root" / "omnibase_infra").mkdir(parents=True)
    env = tmp_path / "params.env"
    env.write_text(
        f"TAG=t\nW={work}\nSLOT_KIND=prepr-slot\nPREPR_SLOT={slot}\nMAIN_PORT=38085\n"
        "EFFECTS_PORT=38086\n"
        "POST_BRINGUP_CMD='echo main=$MAIN_RUNTIME effects=$EFFECTS_RUNTIME "
        "project=$PROJECT port=$MAIN_PORT; exit 7'\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(PROVE_SH), str(env), "post-bringup"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        f"main=omninode-prepr-{slot}-runtime "
        f"effects=omninode-prepr-{slot}-runtime-effects "
        f"project=omnibase-infra-prepr-{slot} port=38085"
    ) in proc.stdout
    assert re.search(r"^post-bringup rc=7 ", proc.stdout, re.M)


def test_the_prove_script_without_a_command_runs_nothing(tmp_path: Path) -> None:
    env = tmp_path / "params.env"
    env.write_text(f"TAG=t\nW={tmp_path}/w\n", encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(PROVE_SH), str(env), "post-bringup"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no-post-bringup" in proc.stdout and "post-bringup rc=" not in proc.stdout


# ------------------------------------------------------------------ OMN-17427
# stage_workspace.sh keeps its pinned sibling worktrees under
# ~/.omnibase/state/deploy_source_trees/<build context path>, and they point into
# the run's own clone. The work directory is keyed by holder and host, so a second
# run by the same holder reuses the key; with the first run's clone gone, the
# build refused with "not a git repository" on omnibase_core.


@pytest.mark.unit
def test_the_prove_script_drops_its_source_tree_cache_and_leaves_other_runs_alone(
    tmp_path: Path,
) -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    fn = re.search(r"^drop_source_trees\(\) \{.*?^\}", text, re.M | re.S)
    assert fn is not None, "prepr_pool_prove.sh defines no drop_source_trees"
    work = tmp_path / "prove-pool-me-host"
    (work / "root" / "omnibase_infra").mkdir(parents=True)
    trees = tmp_path / "home" / ".omnibase" / "state" / "deploy_source_trees"
    key = re.sub(
        r"[^A-Za-z0-9._-]", "_", str(work.resolve() / "root" / "omnibase_infra")
    )
    (trees / key / "abc").mkdir(parents=True)
    (trees / "_another_runs_build_context").mkdir(parents=True)
    done = subprocess.run(
        [
            "bash",
            "-c",
            f"set -u; W={work}; HOME={tmp_path / 'home'}; {fn.group(0)}\ndrop_source_trees",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert not (trees / key).exists()
    assert (trees / "_another_runs_build_context").is_dir()


@pytest.mark.unit
def test_the_prove_script_drops_the_cache_before_a_clone_and_before_each_teardown_removes_the_work_dir() -> (
    None
):
    lines = PROVE_SH.read_text(encoding="utf-8").splitlines()
    removals = [i for i, ln in enumerate(lines) if 'rm -rf "$W"' in ln]
    assert len(removals) == 2, removals
    for i in removals:
        assert "drop_source_trees" in lines[i - 1], lines[i - 1 : i + 1]
    start = lines.index("clone)")
    assert "drop_source_trees" in lines[start + 1], lines[start : start + 3]


# ------------------------------------------------------------------ verify (OMN-20085)

SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
SHA2 = "0f1e2d3c4b5a69788796a5b4c3d2e1f001122334"


def _resolver(known: dict[str, str] | None = None) -> Any:
    table = {"feature-x": SHA, SHA: SHA} if known is None else known

    def resolve(repo: str, ref: str) -> str:
        if ref not in table:
            raise pool.RefUnresolvedError(f"{ref!r} is not in the test clone of {repo}")
        return table[ref]

    return resolve


class TimedTransport(FakeTransport):
    """Remembers the timeout each remote command was given."""

    def __init__(self, hosts: dict[str, FakeHost]) -> None:
        super().__init__(hosts)
        self.timed: list[tuple[str, float]] = []

    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        self.timed.append((command, timeout))
        return super().run(host, command, timeout)

    def tests_budgets(self) -> list[float]:
        return [t for c, t in self.timed if c.endswith(" tests")]


def _slot_hosts(*, slot1_busy: bool = False) -> dict[str, FakeHost]:
    one = FakeHost(cores=32, load=10.0, slot=12 if slot1_busy else 0)
    two = FakeHost(cores=32, load=10.0)
    one.phases = dict(SLOT_GOOD)
    # slot 2 serves its own ports
    two.phases = {
        k: v.replace("28085", "38085").replace("28086", "38086")
        for k, v in SLOT_GOOD.items()
    }
    return {"pool-c-slot-1": one, "pool-c-slot-2": two}


def _verify(tp: Any, tmp_path: Path, **over: Any) -> tuple[int, str]:
    args: dict[str, Any] = {
        "repo": "omnibase_infra",
        "ref": "feature-x",
        "holder": "me",
        "tests": ("omnibase_infra:tests/unit/test_x.py",),
        "suite": False,
        "resolver": _resolver(),
        "ledger_lines": [],
        "workdir": tmp_path / "work",
        "now_fn": lambda: NOW,
        "log": lambda s: None,
    }
    args.update(over)
    return pool.run_verify(CFG, tp, **args)


@pytest.mark.parametrize("suite", [False, True])
def test_verify_lease_renew_every_phase_has_host_and_ssh_timeouts(
    tmp_path: Path, suite: bool
) -> None:
    tp = TimedTransport(_slot_hosts())
    code, text = _verify(
        tp,
        tmp_path,
        suite=suite,
        tests=() if suite else ("omnibase_infra:tests/x.py",),
        suite_budget_s=5400,
    )
    assert code == pool.EXIT_PASS, text
    budgets = {
        "snap-pre": 120,
        "clone": 900,
        "build": 7200,
        "probe": 1800,
        "tests": 5400 if suite else 2400,
        "teardown": 600,
        "snap-post": 600,
    }
    commands = [(c, t) for c, t in tp.timed if "/prove.sh " in c]
    assert len(commands) == len(pool.PHASES)
    for (command, ssh_timeout), phase in zip(commands, pool.PHASES, strict=True):
        budget = budgets[phase]
        assert command.endswith(f" {phase}")
        renewal = pool.renew_lease_command(
            CFG.host("pool-c-slot-1"),
            "me",
            NOW + dt.timedelta(seconds=budget + pool.RENEW_MARGIN_S),
            NOW,
        )
        assert command.startswith(f"{renewal} && timeout -k 30 {budget} bash ")
        assert ssh_timeout == budget + 60


def test_verify_lease_renew_until_advances_with_each_phase(tmp_path: Path) -> None:
    current = NOW
    seen: list[tuple[str, dt.datetime, dict[str, str]]] = []

    class AdvancingTransport(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            nonlocal current
            rc, out = super().run(host, command, timeout)
            match = re.search(r"timeout -k 30 (\d+) bash .* (\S+)$", command)
            if match:
                seen.append(
                    (match.group(2), current, json.loads(self.hosts[host.name].lease))
                )
                current += dt.timedelta(seconds=int(match.group(1)))
            return rc, out

    code, text = _verify(
        AdvancingTransport(_slot_hosts()), tmp_path, now_fn=lambda: current
    )
    assert code == pool.EXIT_PASS, text
    assert [phase for phase, _, _ in seen] == list(pool.PHASES)
    expiries = [pool._parse_ts(lease["until"]) for _, _, lease in seen]
    assert all(a < b for a, b in itertools.pairwise(expiries))
    for phase, started, lease in seen:
        assert pool._parse_ts(lease["taken"]) == started
        if phase == "tests":
            assert pool._parse_ts(lease["until"]) >= started + dt.timedelta(
                seconds=2400
            )


@pytest.mark.parametrize(
    ("take_after", "lost_before"),
    [("build", "probe"), ("tests", "teardown"), ("teardown", "snap-post")],
)
def test_verify_lease_renew_peer_takeover_stops_without_teardown(
    tmp_path: Path, take_after: str, lost_before: str
) -> None:
    hosts = _slot_hosts()
    peer_lease = _lease("peer", NOW + dt.timedelta(hours=3))

    class PeerTakesLease(TimedTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            rc, out = super().run(host, command, timeout)
            if command.endswith(f" {take_after}"):
                self.hosts[host.name].lease = peer_lease
            return rc, out

    tp = PeerTakesLease(hosts)
    logs: list[str] = []
    result_out = tmp_path / "result.json"
    code, text = _verify(tp, tmp_path, result_out=result_out, log=logs.append)
    assert code == pool.EXIT_INCONCLUSIVE
    assert text.startswith("LAB PROOF INCONCLUSIVE:")
    assert (
        f"lease lost before {lost_before}: the slot was left to its new holder, not torn down"
        in text
    )
    expected = list(pool.PHASES[: pool.PHASES.index(lost_before)])
    assert hosts["pool-c-slot-1"].ran == [*expected, f"lost:{lost_before}"]
    assert hosts["pool-c-slot-1"].lease == peer_lease
    assert any(c.startswith("rm -rf /tmp/verify-") for c, _ in tp.timed)
    assert sum(line.startswith("lease lost before ") for line in logs) == 1
    result = json.loads(result_out.read_text(encoding="utf-8"))
    assert result["outcome"] == "INCONCLUSIVE"
    assert result["exit_code"] == pool.EXIT_INCONCLUSIVE


def test_verify_lease_renew_run_proof_keeps_original_commands_and_timeouts(
    tmp_path: Path,
) -> None:
    tp = TimedTransport({"pool-a": FakeHost()})
    code, text = pool.run_proof(
        CFG,
        tp,
        _params(tmp_path),
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_PASS, text
    commands = [(c, t) for c, t in tp.timed if "/prove.sh " in c]
    assert len(commands) == len(pool.PHASES)
    assert [t for _, t in commands] == [120, 900, 3600, 1800, 2400, 600, 600]
    for command, _ in commands:
        assert command.startswith("bash /tmp/pool-")
        assert "timeout -k" not in command
        assert pool.LEASE_LOST not in command


def test_verify_ref_builds_the_slot_from_a_branch_with_no_pr(tmp_path: Path) -> None:
    hosts = _slot_hosts()
    code, text = _verify(FakeTransport(hosts), tmp_path)
    assert code == pool.EXIT_PASS, text
    assert text.startswith(f"LAB PROOF PASS: omnibase_infra@{SHA[:10]} (ref feature-x)")
    env = (tmp_path / "work" / "verify.resolved.env").read_text(encoding="utf-8")
    assert f"INFRA_REF_SHA={SHA}" in env.splitlines()
    assert "INFRA_PR" not in env and "MARKET_REF_SHA" not in env
    assert "TESTS='omnibase_infra:tests/unit/test_x.py'" in env.splitlines() or (
        "TESTS=omnibase_infra:tests/unit/test_x.py" in env.splitlines()
    )
    assert hosts["pool-c-slot-1"].ran == list(pool.PHASES)
    assert hosts["pool-c-slot-1"].lease is None


def test_verify_ref_names_the_market_param_for_omnimarket(tmp_path: Path) -> None:
    hosts = _slot_hosts()
    code, text = _verify(
        FakeTransport(hosts),
        tmp_path,
        repo="omnimarket",
        tests=("omnimarket:tests/unit/test_y.py",),
    )
    assert code == pool.EXIT_PASS, text
    env = (tmp_path / "work" / "verify.resolved.env").read_text(encoding="utf-8")
    assert f"MARKET_REF_SHA={SHA}" in env.splitlines()
    assert "INFRA_REF_SHA" not in env


def test_verify_ref_that_does_not_resolve_exits_5_before_any_lease(
    tmp_path: Path,
) -> None:
    code, text = _verify(NoHostTransport(), tmp_path, ref="no-such-branch")
    assert code == pool.EXIT_USAGE
    assert "ref does not resolve, nothing leased" in text


def _git(*args: str, cwd: Path) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return done.stdout.strip()


def test_verify_ref_resolves_a_branch_and_a_sha_in_the_canonical_clone(
    tmp_path: Path,
) -> None:
    origin, clone = tmp_path / "origin", tmp_path / "omnibase_infra"
    origin.mkdir()
    _git("init", "-q", "-b", "dev", cwd=origin)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@lab.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "c",
        cwd=origin,
    )
    _git("branch", "feature-x", cwd=origin)
    head = _git("rev-parse", "HEAD", cwd=origin)
    _git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    assert pool.resolve_ref_in_clone(clone, "feature-x") == head
    assert pool.resolve_ref_in_clone(clone, head) == head
    for bad in ("nope", "0" * 40, "-x", "a..b", "feature-x/", ""):
        with pytest.raises(pool.RefUnresolvedError):
            pool.resolve_ref_in_clone(clone, bad)
    with pytest.raises(pool.RefUnresolvedError):
        pool.resolve_ref_in_clone(tmp_path / "missing", "feature-x")


def test_verify_first_free_slot_takes_only_prepr_slots_in_slot_order(
    tmp_path: Path,
) -> None:
    hosts = {
        **_slot_hosts(),
        "pool-a": FakeHost(load=0.1),
        "pool-d": FakeHost(load=0.1),
    }
    hosts["pool-a"].phases = dict(GOOD)
    code, _ = _verify(FakeTransport(hosts), tmp_path)
    assert code == pool.EXIT_PASS
    assert hosts["pool-c-slot-1"].ran == list(pool.PHASES)
    assert hosts["pool-c-slot-2"].ran == []
    assert hosts["pool-a"].ran == [] and hosts["pool-d"].ran == []


def test_verify_first_free_slot_skips_a_busy_slot(tmp_path: Path) -> None:
    hosts = _slot_hosts(slot1_busy=True)
    code, _ = _verify(FakeTransport(hosts), tmp_path)
    assert code == pool.EXIT_PASS
    assert hosts["pool-c-slot-1"].ran == []
    assert hosts["pool-c-slot-2"].ran == list(pool.PHASES)


def test_verify_first_free_slot_tries_the_next_when_the_first_is_lost_to_a_peer(
    tmp_path: Path,
) -> None:
    hosts = _slot_hosts()

    class PeerTakesSlotOne(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            rc, out = super().run(host, command, timeout)
            if "CORES=$(getconf" in command and host.name == "pool-c-slot-1":
                # the peer leases it right after this survey read it as free
                self.hosts["pool-c-slot-1"].lease = _lease(
                    "peer", NOW + dt.timedelta(hours=1)
                )
            return rc, out

    code, _ = _verify(PeerTakesSlotOne(hosts), tmp_path)
    assert code == pool.EXIT_PASS
    assert hosts["pool-c-slot-1"].ran == []
    assert json.loads(hosts["pool-c-slot-1"].lease)["holder"] == "peer"
    assert hosts["pool-c-slot-2"].ran == list(pool.PHASES)
    assert hosts["pool-c-slot-2"].lease is None


def test_verify_first_free_slot_exits_3_when_no_slot_is_free(tmp_path: Path) -> None:
    hosts = _slot_hosts(slot1_busy=True)
    hosts["pool-c-slot-2"].slot = 12
    hosts["pool-a"] = FakeHost(load=0.1)  # a free isolated member is not offered
    code, text = _verify(FakeTransport(hosts), tmp_path)
    assert code == pool.EXIT_NO_FREE_HOST
    assert "no free pre-PR slot" in text
    assert hosts["pool-a"].ran == []


def test_verify_first_free_slot_exits_3_when_every_free_slot_is_lost(
    tmp_path: Path,
) -> None:
    hosts = _slot_hosts()

    class PeersTakeBoth(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            rc, out = super().run(host, command, timeout)
            if "CORES=$(getconf" in command and host.name.startswith("pool-c-slot"):
                self.hosts[host.name].lease = _lease(
                    "peer", NOW + dt.timedelta(hours=1)
                )
            return rc, out

    code, text = _verify(PeersTakeBoth(hosts), tmp_path)
    assert code == pool.EXIT_NO_FREE_HOST
    assert "taken before its lease" in text


def test_verify_suite_refuses_a_call_with_no_tests_and_no_suite(tmp_path: Path) -> None:
    code, text = _verify(NoHostTransport(), tmp_path, tests=(), suite=False)
    assert code == pool.EXIT_USAGE
    assert "names no tests" in text


@pytest.mark.parametrize(
    "spec",
    [
        "tests/unit/x.py",
        "other_repo:tests/x.py",
        "omnibase_infra:a b",
        "omnibase_infra:$(x)",
    ],
)
def test_verify_suite_refuses_a_malformed_test_spec(tmp_path: Path, spec: str) -> None:
    code, _ = _verify(NoHostTransport(), tmp_path, tests=(spec,))
    assert code == pool.EXIT_USAGE


def test_verify_suite_runs_the_unit_suite_under_its_own_budget(tmp_path: Path) -> None:
    hosts = _slot_hosts()
    hosts["pool-c-slot-1"].phases["tests"] = (
        "omnibase_infra suite rc=0 2026-10-19T01:00:00Z"
    )
    tp = TimedTransport(hosts)
    code, text = _verify(
        tp, tmp_path, tests=(), suite=True, suite_budget_s=5400, suite_workers=6
    )
    assert code == pool.EXIT_PASS, text
    env = (
        (tmp_path / "work" / "verify.resolved.env")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "SUITE_REPO=omnibase_infra" in env and "SUITE_WORKERS=6" in env
    assert not any(line.startswith("TESTS=") for line in env)
    assert tp.tests_budgets() == [5460]


def test_a_focused_run_keeps_the_default_tests_budget(tmp_path: Path) -> None:
    hosts = _slot_hosts()
    tp = TimedTransport(hosts)
    assert _verify(tp, tmp_path)[0] == pool.EXIT_PASS
    assert tp.tests_budgets() == [2460]


def test_verify_suite_judges_a_failing_suite_and_bounds_the_dev_control() -> None:
    failing = (
        "FAILED tests/unit/a.py::t1\nomnibase_infra suite rc=1 t\n"
        "dev-control omnibase_infra tests/unit/a.py::t1 rc=1 at abc\n"
        "dev-control-bound omnibase_infra skipped=3 bound=1"
    )
    rb = pool.judge({**SLOT_GOOD, "tests": failing}, ports=(28085, 28086))
    assert rb.checks["focused_tests"] is True
    rb = pool.judge(
        {
            **SLOT_GOOD,
            "tests": "FAILED tests/unit/b.py::t2\nomnibase_infra suite rc=1 t",
        },
        ports=(28085, 28086),
    )
    assert rb.checks["focused_tests"] is False


def test_the_prove_script_runs_a_suite_and_bounds_its_dev_control() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    assert "SUITE_REPO" in text and "tests/unit" in text
    assert "DEV_CONTROL_MAX:-25" in text
    assert (
        "fetch_ref()" in text and "INFRA_REF_SHA" in text and "MARKET_REF_SHA" in text
    )
    assert 'git -C "$d" fetch -q origin "$sha"' in text
    done = subprocess.run(
        ["bash", "-n", str(PROVE_SH)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_verify_result_is_written_for_every_outcome_with_the_exit_code(
    tmp_path: Path,
) -> None:
    def check(name: str, tp: Any, **over: Any) -> dict[str, Any]:
        out = tmp_path / name / "result.json"
        code, _ = _verify(tp, tmp_path / name, result_out=out, **over)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema"] == "onex.prepr.verify-result.v1"
        assert data["exit_code"] == code
        assert data["repo"] == "omnibase_infra" and data["holder"] == "me"
        return data

    ok = check("pass", FakeTransport(_slot_hosts()))
    assert ok["outcome"] == "PASS" and ok["exit_code"] == 0
    assert ok["sha"] == SHA and ok["slot"] == "pool-c-slot-1" and ok["restored"] is True
    assert ok["checks"]["focused_tests"] is True

    bad = _slot_hosts()
    bad["pool-c-slot-1"].phases["tests"] = (
        "FAILED tests/unit/x.py::t\nomnibase_infra focused rc=1 t"
    )
    failed = check("fail", FakeTransport(bad))
    assert failed["outcome"] == "FAIL" and failed["exit_code"] == 1

    inconclusive = _slot_hosts()
    inconclusive["pool-c-slot-1"].phases["build"] = "build rc=1"
    assert check("inconclusive", FakeTransport(inconclusive))["exit_code"] == 2

    full = _slot_hosts(slot1_busy=True)
    full["pool-c-slot-2"].slot = 12
    nofree = check("nofree", FakeTransport(full))
    assert nofree["outcome"] == "NO_FREE_SLOT" and nofree["exit_code"] == 3
    assert nofree["slot"] == ""

    usage = check("usage", NoHostTransport(), tests=())
    assert usage["outcome"] == "USAGE" and usage["exit_code"] == 5
    unresolved = check("unresolved", NoHostTransport(), ref="nope")
    assert unresolved["exit_code"] == 5


def _canonical_clone(root: Path) -> str:
    """A registry root holding a clone of omnibase_infra with a feature-x branch; its head sha."""
    origin = root / "origin"
    origin.mkdir(parents=True)
    _git("init", "-q", "-b", "dev", cwd=origin)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@lab.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "c",
        cwd=origin,
    )
    _git("branch", "feature-x", cwd=origin)
    _git("clone", "-q", str(origin), str(root / "omnibase_infra"), cwd=root)
    return _git("rev-parse", "HEAD", cwd=origin)


def _verify_argv(tmp_path: Path, out: Path, ref: str) -> list[str]:
    ledger = tmp_path / "ledger.md"
    ledger.write_text("", encoding="utf-8")
    return [
        "--config",
        str(POOL_FIXTURE),
        "--ledger",
        str(ledger),
        "verify",
        "--repo",
        "omnibase_infra",
        "--ref",
        ref,
        "--holder",
        "me",
        "--tests",
        "omnibase_infra:tests/unit/test_x.py",
        "--result-out",
        str(out),
        "--clone-root",
        str(tmp_path / "root"),
    ]


@pytest.mark.parametrize("busy", [False, True])
def test_verify_result_exit_code_equals_the_process_exit_code(
    tmp_path: Path, busy: bool
) -> None:
    head = _canonical_clone(tmp_path / "root")
    out = tmp_path / "result.json"
    hosts = _slot_hosts(slot1_busy=busy)
    if busy:
        hosts["pool-c-slot-2"].slot = 12
    code = pool.main(
        _verify_argv(tmp_path, out, "feature-x"), transport=FakeTransport(hosts)
    )
    assert code == (pool.EXIT_NO_FREE_HOST if busy else pool.EXIT_PASS)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["exit_code"] == code and data["sha"] == head
    code = pool.main(
        _verify_argv(tmp_path, out, "no-such"), transport=NoHostTransport()
    )
    assert code == pool.EXIT_USAGE
    assert json.loads(out.read_text(encoding="utf-8"))["exit_code"] == code


def test_verify_result_for_an_unreadable_ledger_is_still_written(
    tmp_path: Path,
) -> None:
    out = tmp_path / "result.json"
    argv = _verify_argv(tmp_path, out, "x")
    argv[argv.index("--ledger") + 1] = str(tmp_path / "absent.md")
    assert pool.main(argv, transport=NoHostTransport()) == pool.EXIT_USAGE
    assert json.loads(out.read_text(encoding="utf-8"))["exit_code"] == pool.EXIT_USAGE


def test_verify_result_for_a_config_error_is_still_written(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    code = pool.main(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "verify",
            "--repo",
            "omnibase_infra",
            "--ref",
            "x",
            "--holder",
            "me",
            "--suite",
            "--result-out",
            str(out),
        ],
        transport=NoHostTransport(),
    )
    assert code == pool.EXIT_USAGE
    assert json.loads(out.read_text(encoding="utf-8"))["exit_code"] == pool.EXIT_USAGE


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_verify_signal_tears_the_slot_down_releases_the_lease_and_says_interrupted(
    tmp_path: Path, sig: signal.Signals
) -> None:
    hosts = _slot_hosts()

    class SignalledMidBuild(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if command.endswith(" build"):
                os.kill(os.getpid(), sig)
                time.sleep(5)  # the handler raises before this returns
            return super().run(host, command, timeout)

    before = {s: signal.getsignal(s) for s in pool.VERIFY_SIGNALS}
    out = tmp_path / "result.json"
    code, text = _verify(
        SignalledMidBuild(hosts), tmp_path, result_out=out, handle_signals=True
    )
    assert code == 128 + sig.value
    assert "INTERRUPTED" in text
    one = hosts["pool-c-slot-1"]
    assert one.ran[-2:] == ["teardown", "snap-post"], one.ran
    assert one.lease is None
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["outcome"] == "INTERRUPTED" and data["exit_code"] == code
    assert data["restored"] is True
    assert {s: signal.getsignal(s) for s in pool.VERIFY_SIGNALS} == before


def test_verify_signal_during_the_survey_leaves_nothing_leased(tmp_path: Path) -> None:
    hosts = _slot_hosts()

    class SignalledInSurvey(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if "CORES=$(getconf" in command:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
            return super().run(host, command, timeout)

    out = tmp_path / "result.json"
    code, _ = _verify(
        SignalledInSurvey(hosts), tmp_path, result_out=out, handle_signals=True
    )
    assert code == 128 + signal.SIGTERM.value
    assert all(h.lease is None and h.ran == [] for h in hosts.values())
    assert json.loads(out.read_text(encoding="utf-8"))["outcome"] == "INTERRUPTED"


def test_verify_signal_a_second_signal_during_cleanup_does_not_cut_it_short(
    tmp_path: Path,
) -> None:
    hosts = _slot_hosts()

    class TwiceSignalled(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if command.endswith(" build"):
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
            if command.endswith(" teardown"):
                os.kill(os.getpid(), signal.SIGINT)  # ignored: cleanup is running
            return super().run(host, command, timeout)

    code, _ = _verify(TwiceSignalled(hosts), tmp_path, handle_signals=True)
    assert code == 128 + signal.SIGTERM.value
    assert hosts["pool-c-slot-1"].ran[-2:] == ["teardown", "snap-post"]
    assert hosts["pool-c-slot-1"].lease is None


@pytest.mark.parametrize("phase", ["teardown", "snap-post"])
def test_verify_signal_first_reaching_a_cleanup_phase_lets_it_finish(
    tmp_path: Path, phase: str
) -> None:
    # 2026-10-20T15:20Z on pre-PR slot 2: one SIGTERM arrived during the
    # teardown phase, the phase was cut short and the summary said torn down
    hosts = _slot_hosts()

    class SignalledInCleanup(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if command.endswith(f" {phase}"):
                os.kill(os.getpid(), signal.SIGTERM)
            return super().run(host, command, timeout)

    out = tmp_path / "result.json"
    code, text = _verify(
        SignalledInCleanup(hosts), tmp_path, result_out=out, handle_signals=True
    )
    one = hosts["pool-c-slot-1"]
    assert code == 128 + signal.SIGTERM.value
    assert one.ran[-2:] == ["teardown", "snap-post"], one.ran
    assert one.lease is None
    assert "torn down" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["outcome"] == "INTERRUPTED" and data["exit_code"] == code
    assert data["restored"] is True


def test_verify_signal_during_the_lease_release_still_releases(
    tmp_path: Path,
) -> None:
    hosts = _slot_hosts()

    class SignalledInRelease(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if "RELEASED" in command and "prove.sh" not in command:
                os.kill(os.getpid(), signal.SIGTERM)
            return super().run(host, command, timeout)

    code, text = _verify(SignalledInRelease(hosts), tmp_path, handle_signals=True)
    assert code == 128 + signal.SIGTERM.value
    assert hosts["pool-c-slot-1"].lease is None
    assert hosts["pool-c-slot-1"].ran[-2:] == ["teardown", "snap-post"]
    assert "the lease was released" in text


def test_verify_signal_while_the_lease_is_taken_releases_it(tmp_path: Path) -> None:
    # the host took the lease and the signal arrived before the reply was read:
    # the run must still know it holds the slot and give it back
    hosts = _slot_hosts()

    class SignalledInAcquire(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            reply = super().run(host, command, timeout)
            if "then printf" in command and "prove.sh" not in command:
                os.kill(os.getpid(), signal.SIGTERM)
            return reply

    out = tmp_path / "result.json"
    code, _ = _verify(
        SignalledInAcquire(hosts), tmp_path, result_out=out, handle_signals=True
    )
    assert code == 128 + signal.SIGTERM.value
    assert all(h.lease is None for h in hosts.values())
    assert hosts["pool-c-slot-1"].ran == []
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["outcome"] == "INTERRUPTED" and data["slot"] == "pool-c-slot-1"


def test_verify_signal_text_claims_only_the_cleanup_that_ran(tmp_path: Path) -> None:
    hosts = _slot_hosts()

    class SignalledInSurvey(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if "CORES=$(getconf" in command:
                os.kill(os.getpid(), signal.SIGTERM)
            return super().run(host, command, timeout)

    out = tmp_path / "result.json"
    _, text = _verify(
        SignalledInSurvey(hosts), tmp_path, result_out=out, handle_signals=True
    )
    assert "torn down" not in text and "released" not in text
    assert "nothing was leased" in text
    assert (
        "nothing was leased" in json.loads(out.read_text(encoding="utf-8"))["summary"]
    )

    lost = _slot_hosts()

    class LeaseLostThenSignalled(FakeTransport):
        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if command.endswith(" probe"):
                # a peer broke the lease; the run stops without teardown, and a
                # signal then arrives while the driver gives up the slot
                self.hosts[host.name].lease = _lease(
                    "peer", NOW + dt.timedelta(hours=3)
                )
            if "RELEASED" in command and "prove.sh" not in command:
                os.kill(os.getpid(), signal.SIGTERM)
            return super().run(host, command, timeout)

    code, text = _verify(LeaseLostThenSignalled(lost), tmp_path, handle_signals=True)
    assert code == 128 + signal.SIGTERM.value
    assert "teardown" not in lost["pool-c-slot-1"].ran
    assert "torn down" in text and "not torn down" in text


def test_local_transport_is_this_host_loopback() -> None:
    current_user = pwd.getpwuid(os.getuid()).pw_name
    assert pool.is_this_host(f"{current_user}@127.0.0.1") is True


def test_local_transport_is_this_host_other_user() -> None:
    assert pool.is_this_host("someone-else-xyz@127.0.0.1") is False


def test_local_transport_is_this_host_foreign_address() -> None:
    current_user = pwd.getpwuid(os.getuid()).pw_name
    assert pool.is_this_host(f"{current_user}@192.0.2.1") is False
    assert pool.is_this_host("nohost.invalid") is False


def test_local_transport_runs_locally_without_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_run = pool.subprocess.run
    calls: list[list[str]] = []

    def recording_run(
        argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(pool.subprocess, "run", recording_run)
    command = "pwd; echo marker"
    rc, output = pool.SshTransport(local_check=lambda t: True).run(
        CFG.host("pool-a"), command, 30
    )
    assert rc == 0
    assert str(Path.home()) in output
    assert "marker" in output
    assert all(argv[0] != "ssh" for argv in calls)
    assert calls[0] == ["bash", "-c", command]
    assert calls[0][2] == command


def test_local_transport_put_copies_file(tmp_path: Path) -> None:
    local = tmp_path / "source.txt"
    remote = tmp_path / "destination.txt"
    local.write_text("local transport copy\n", encoding="utf-8")
    transport = pool.SshTransport(local_check=lambda t: True)
    host = CFG.host("pool-a")
    assert transport.put(host, local, str(remote)) == 0
    assert remote.read_text(encoding="utf-8") == local.read_text(encoding="utf-8")
    assert transport.put(host, local, str(tmp_path / "missing" / "file.txt")) == 1


def test_local_transport_remote_host_still_uses_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(pool.subprocess, "run", fake_run)
    host = CFG.host("pool-a")
    command = "echo marker"
    assert pool.SshTransport(local_check=lambda t: False).run(host, command, 30) == (
        0,
        "ok",
    )
    assert calls[0][:3] == ["ssh", "-o", "BatchMode=yes"]
    assert calls[0][-2:] == [host.ssh_target, pool.REMOTE_PATH + command]
