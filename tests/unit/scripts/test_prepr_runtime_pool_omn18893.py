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
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

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
CFG = pool.load_pool_config()
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
            base = "-base-" in command
            h.ran.append(("base:" if base else "") + m.group(1))
            return 0, (h.base_phases if base else h.phases)[m.group(1)]
        if "CORES=$(getconf" in command:
            return 0, (
                f"CORES={h.cores}\nLOAD={h.load}\nSLOT={h.slot}\nLISTEN={h.listen}\n"
                f"PROJECTS=omnibase-infra-dogfood:7,\nLEASE={h.lease or ''}\n"
            )
        if "then printf" in command:  # acquire
            if h.lease is None:
                body = re.search(r"printf '%s' '(\{.*?\})'", command)
                assert body is not None
                h.lease = body.group(1)
                return 0, "ACQUIRED"
            return 0, "HELD\n" + h.lease
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


def test_every_pool_host_is_declared_in_the_lane_manifest() -> None:
    declared = pool.declared_manifest_hosts()
    assert {h.machine for h in CFG.hosts} <= declared


def test_the_slot_project_is_never_a_declared_lane() -> None:
    assert CFG.compose_project not in pool.declared_lane_projects()
    isolated = [h for h in CFG.hosts if h.kind == "isolated"]
    assert {h.compose_project for h in isolated} == {CFG.compose_project}


def test_every_lab_machine_takes_part_and_every_exclusion_states_a_reason() -> None:
    # Operator ruling 2026-09-27T20:20:42Z: ".105, .201 and .202 are available,
    # what do you mean only 1 lab host?" Every lab machine has a pool member.
    in_pool = {h.machine for h in CFG.hosts if h.status == "pool"}
    assert in_pool == {"lab-101", "lab-105", "lab-200", "lab-201", "lab-202"}
    assert CFG.host("lab-202").status == "pool"
    assert CFG.host("lab-202").positive_control == "omnibase-infra-dev-202"
    assert all(h.reason for h in CFG.hosts if h.status == "excluded")


def test_201_takes_part_only_through_its_pre_pr_slots() -> None:
    on_201 = [h for h in CFG.hosts if h.machine == "lab-201"]
    assert CFG.host("lab-201").status == "excluded"
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
    raw = yaml.safe_load(pool.POOL_CONFIG.read_text(encoding="utf-8"))
    raw["hosts"][0].update(edit)
    bad = tmp_path / "pool.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        pool.load_pool_config(bad)


def test_model_endpoint_is_not_the_202_server() -> None:
    assert "192.168.86.202" not in CFG.model_endpoint  # onex-allow-internal-ip


def test_an_exclusion_without_a_reason_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(pool.POOL_CONFIG.read_text(encoding="utf-8"))
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


# ------------------------------------------------------------------ ledger


def test_live_surface_holds_respects_until_and_release() -> None:
    rows = [
        "2026-09-27T14:00:00Z | HOLD | lane=a | id=2026-09-27T14:00:00Z-a | surface=101-runtime | until=2026-09-27T15:00:00Z | x",
        "2026-09-27T13:00:00Z | HOLD | lane=b | id=2026-09-27T13:00:00Z-b | surface=105-runtime | until=2026-09-27T14:00:00Z | x",
        "2026-09-27T14:10:00Z | HOLD | lane=c | id=2026-09-27T14:10:00Z-c | surface=200-runtime | until=2026-09-27T16:00:00Z | x",
        "2026-09-27T14:20:00Z | RELEASE | lane=c | re=2026-09-27T14:10:00Z-c | surface=200-runtime | result=PASS | restored=yes",
        "2026-09-27T14:21:00Z | HOLD | lane=d | id=2026-09-27T14:21:00Z-d | to=all | pr=omnimarket#1 | until=2026-09-27T20:00:00Z | not a surface",
    ]
    live = pool.live_surface_holds(rows, NOW)
    assert set(live) == {"101-runtime"}
    assert live["101-runtime"][1] == "a"


# ------------------------------------------------------------------ survey and pick


def _survey(
    hosts: dict[str, FakeHost],
    holds: dict[str, Any] | None = None,
    me: str | None = None,
) -> list[Any]:
    states: list[Any] = pool.survey(CFG, FakeTransport(hosts), holds or {}, NOW, me=me)
    return states


def test_pick_takes_the_least_loaded_free_host_and_never_an_excluded_one() -> None:
    hosts = {
        "lab-101": FakeHost(load=6.0),
        "lab-105": FakeHost(cores=10, load=1.0),
        "lab-200": FakeHost(cores=24, load=3.0),
        "lab-201": FakeHost(),
        "lab-202": FakeHost(),
    }
    states = _survey(hosts)
    verdicts = {s.host.name: s.verdict for s in states}
    assert verdicts["lab-201"] == "EXCLUDED"
    assert verdicts["lab-202"] == "FREE"
    assert pool.pick(states).host.name == "lab-105"


def test_offline_overloaded_leased_held_and_occupied_hosts_are_not_free() -> None:
    hosts = {
        "lab-101": FakeHost(),
        "lab-105": FakeHost(online=False),
        "lab-200": FakeHost(cores=24, load=74.0),
    }
    hosts["lab-101"].lease = _lease("other-lane", NOW + dt.timedelta(minutes=30))
    v = {s.host.name: s for s in _survey(hosts)}
    assert v["lab-101"].verdict == "BUSY" and "other-lane" in v["lab-101"].detail
    assert v["lab-105"].verdict == "OFFLINE"
    assert v["lab-200"].verdict == "OVERLOADED"
    assert pool.pick(list(v.values())) is None

    held = {
        "lab-101": FakeHost(),
        "lab-105": FakeHost(slot=3),
        "lab-200": FakeHost(listen=1),
    }
    v2 = {
        s.host.name: s
        for s in _survey(
            held, holds={"101-runtime": ("id1", "peer", "2026-09-27T15:00:00Z")}
        )
    }
    assert v2["lab-101"].verdict == "BUSY" and "HOLD id1" in v2["lab-101"].detail
    assert (
        v2["lab-105"].verdict == "BUSY" and "containers present" in v2["lab-105"].detail
    )
    assert v2["lab-200"].verdict == "BUSY" and "listeners" in v2["lab-200"].detail


def test_own_hold_and_expired_lease_do_not_block() -> None:
    hosts = {"lab-101": FakeHost()}
    hosts["lab-101"].lease = _lease("gone-lane", NOW - dt.timedelta(minutes=1))
    v = {
        s.host.name: s
        for s in _survey(hosts, holds={"101-runtime": ("id", "me", "x")}, me="me")
    }
    assert v["lab-101"].verdict == "FREE"


# ------------------------------------------------------------------ lease


def test_lease_is_exclusive_and_only_its_holder_releases_it() -> None:
    hosts = {"lab-101": FakeHost()}
    tp = FakeTransport(hosts)
    h = CFG.host("lab-101")
    until = NOW + dt.timedelta(minutes=60)
    assert pool.acquire_lease(CFG, tp, h, "me", until, NOW)[0]
    ok, why = pool.acquire_lease(CFG, tp, h, "peer", until, NOW)
    assert not ok and "held by me" in why
    assert not pool.release_lease(CFG, tp, h, "peer")[0]
    assert pool.release_lease(CFG, tp, h, "me")[0]
    assert hosts["lab-101"].lease is None


def test_expired_lease_is_broken() -> None:
    hosts = {"lab-101": FakeHost()}
    hosts["lab-101"].lease = _lease("gone-lane", NOW - dt.timedelta(seconds=1))
    ok, why = pool.acquire_lease(
        CFG,
        FakeTransport(hosts),
        CFG.host("lab-101"),
        "me",
        NOW + dt.timedelta(hours=1),
        NOW,
    )
    assert ok and "gone-lane" in why
    assert json.loads(hosts["lab-101"].lease)["holder"] == "me"


# ------------------------------------------------------------------ judge


def test_judge_pass_only_when_every_check_and_restore_hold() -> None:
    rb = pool.judge(GOOD)
    assert rb.outcome == "PASS" and rb.restored, rb


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


# ------------------------------------------------------------------ run end to end


def test_run_pass_end_to_end_prints_a_citable_readback(tmp_path: Path) -> None:
    hosts = {
        "lab-101": FakeHost(load=5.0),
        "lab-105": FakeHost(online=False),
        "lab-200": FakeHost(cores=24, load=60.0),
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
    assert text.startswith("LAB PROOF PASS: omnibase_infra#4210 head abc on lab-101")
    assert (
        "restored=yes" in text
        and "surface=101-runtime result=PASS restored=yes" in text
    )
    assert hosts["lab-101"].ran == list(pool.PHASES)
    assert hosts["lab-101"].lease is None


def test_failed_build_still_tears_down_and_releases(tmp_path: Path) -> None:
    hosts = {"lab-101": FakeHost()}
    hosts["lab-101"].phases["build"] = "build rc=1\nup rc=1"
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
        "lab-101",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_INCONCLUSIVE
    assert hosts["lab-101"].ran == [
        "snap-pre",
        "clone",
        "build",
        "teardown",
        "snap-post",
    ]
    assert hosts["lab-101"].lease is None


def test_run_refuses_when_no_host_is_free(tmp_path: Path) -> None:
    hosts = {
        "lab-101": FakeHost(cores=12, load=40.0),
        "lab-105": FakeHost(online=False),
        "lab-200": FakeHost(slot=9),
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
        "lab-101 OVERLOADED" in text
        and "lab-105 OFFLINE" in text
        and "lab-201 EXCLUDED" in text
        and "lab-202 OFFLINE" in text
    )


def test_run_refuses_a_host_under_a_peer_ledger_hold(tmp_path: Path) -> None:
    rows = [
        "2026-09-27T14:00:00Z | HOLD | lane=peer | id=2026-09-27T14:00:00Z-peer | surface=101-runtime | until=2026-09-27T15:00:00Z | x"
    ]
    hosts = {"lab-101": FakeHost()}
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "lab-101",
        rows,
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_NO_FREE_HOST and "peer" in text
    assert hosts["lab-101"].ran == []


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
    hosts = {"lab-101": FakeHost()}
    hosts["lab-101"].phases["probe"] = DEV_FAILS
    hosts["lab-101"].base_phases["probe"] = DEV_FAILS
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        90,
        "lab-101",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        with_base_control=True,
    )
    assert code == pool.EXIT_PASS, text
    ran = hosts["lab-101"].ran
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
    assert hosts["lab-101"].lease is None

    clean = {"lab-101": FakeHost()}
    pool.run_proof(
        CFG,
        FakeTransport(clean),
        _params(tmp_path),
        "me",
        90,
        "lab-101",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        with_base_control=True,
    )
    assert not any(r.startswith("base:") for r in clean["lab-101"].ran)


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
    hosts = {"lab-101": FakeHost()}
    hosts["lab-101"].phases["build"] = "build rc=1"
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "lab-101",
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
    assert CFG.host("lab-200").docker_config == "isolated"
    assert CFG.host("lab-101").docker_config == "host"
    hosts = {"lab-200": FakeHost(cores=24, load=2.0)}
    pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        60,
        "lab-200",
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
    hosts = {"lab-201-prepr-1": FakeHost(cores=32, load=10.0)}
    tp = RecordingTransport(hosts)
    states = pool.survey(CFG, tp, {}, NOW)
    by = {s.host.name: s for s in states}
    assert by["lab-201-prepr-1"].verdict == "FREE"
    probe = next(c for n, c in tp.commands if n == "lab-201-prepr-1")
    assert "com.docker.compose.project=omnibase-infra-prepr-1 " in probe
    assert ":(28085|28086|28090|23002)" in probe
    assert ".onex-prepr-pool/lease-prepr-1/lease.json" in probe
    assert "8085|" not in probe.replace("28085|", "")

    busy = {"lab-201-prepr-2": FakeHost(cores=32, load=10.0, slot=12)}
    s2 = {s.host.name: s for s in pool.survey(CFG, FakeTransport(busy), {}, NOW)}
    assert s2["lab-201-prepr-2"].verdict == "BUSY"
    assert "omnibase-infra-prepr-2 containers present" in s2["lab-201-prepr-2"].detail


def test_a_slot_run_passes_end_to_end_with_the_slot_ports(tmp_path: Path) -> None:
    hosts = {"lab-201-prepr-1": FakeHost(cores=32, load=10.0)}
    hosts["lab-201-prepr-1"].phases = dict(SLOT_GOOD)
    code, text = pool.run_proof(
        CFG,
        FakeTransport(hosts),
        _params(tmp_path),
        "me",
        150,
        "lab-201-prepr-1",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
    )
    assert code == pool.EXIT_PASS, text
    assert "pre-PR slot project omnibase-infra-prepr-1" in text
    assert "ok   health_28085" in text and "ok   health_28086" in text
    assert "surface=prepr-1-201 result=PASS restored=yes" in text
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
    assert hosts["lab-201-prepr-1"].lease is None


def test_a_params_file_cannot_move_a_run_onto_another_project(tmp_path: Path) -> None:
    p = _params(tmp_path)
    p.write_text(
        p.read_text(encoding="utf-8")
        + "PROJECT=omnibase-infra\nSLOT_KIND=prepr-slot\nPREPR_SLOT=1\nMAIN_PORT=8085\n",
        encoding="utf-8",
    )
    pool.run_proof(
        CFG,
        FakeTransport({"lab-101": FakeHost()}),
        p,
        "me",
        60,
        "lab-101",
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
