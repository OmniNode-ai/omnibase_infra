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
import shutil
import subprocess
import sys
import threading
import time
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
    # Operator 2026-10-01: .105 is out of the mix (taken to tech week).
    assert in_pool == {"lab-101", "lab-200", "lab-201", "lab-202"}
    assert CFG.host("lab-105").status == "excluded"
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
        "lab-101": FakeHost(load=6.0),
        "lab-200": FakeHost(cores=24, load=3.0),
        "lab-201": FakeHost(),
        "lab-202": FakeHost(cores=10, load=1.0),
    }
    states = _survey(hosts)
    verdicts = {s.host.name: s.verdict for s in states}
    assert verdicts["lab-201"] == "EXCLUDED"
    assert verdicts["lab-105"] == "EXCLUDED"
    assert verdicts["lab-202"] == "FREE"
    assert pool.pick(states).host.name == "lab-202"


def test_offline_overloaded_leased_held_and_occupied_hosts_are_not_free() -> None:
    hosts = {
        "lab-101": FakeHost(),
        "lab-202": FakeHost(online=False),
        "lab-200": FakeHost(cores=24, load=74.0),
    }
    hosts["lab-101"].lease = _lease("other-lane", NOW + dt.timedelta(minutes=30))
    v = {s.host.name: s for s in _survey(hosts)}
    assert v["lab-101"].verdict == "BUSY" and "other-lane" in v["lab-101"].detail
    assert v["lab-202"].verdict == "OFFLINE"
    assert v["lab-200"].verdict == "OVERLOADED"
    assert pool.pick(list(v.values())) is None

    held = {
        "lab-101": FakeHost(),
        "lab-202": FakeHost(slot=3),
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
        v2["lab-202"].verdict == "BUSY" and "containers present" in v2["lab-202"].detail
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
        and "lab-105 EXCLUDED" in text
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


def test_the_slot_publishes_the_port_its_effects_runtime_listens_on() -> None:
    # Found live (omnibase_infra#4180 on lab-201-prepr-1, 2026-09-27): the slot
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
    for name in ("lab-201-prepr-1", "lab-201-prepr-2"):
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
        "lab-101": FakeHost(cores=12, load=11.0),
        "lab-201-prepr-1": FakeHost(cores=32, load=1.0),
        "lab-201-prepr-2": FakeHost(cores=32, load=1.0),
    }
    assert pool.pick(_survey(hosts)).host.name == "lab-101"
    hosts["lab-101"].online = False
    assert pool.pick(_survey(hosts)).host.name == "lab-201-prepr-1"


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
        CFG.host("lab-101"),
        {"INFRA_GROUP": GROUP_SPEC},
        "2026-09-28T10:00:00Z",
        "2026-09-28T10:30:00Z",
    )
    lines = text.splitlines()
    assert lines[0].startswith(
        "LAB PROOF PASS: omnibase_infra#4134 head fdd93c786c + "
        "omnibase_infra#4198 head e812bfd787 + "
        "omnibase_infra#4214 head 0ba71c9566 on lab-101"
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
        CFG.host("lab-101"),
        {"MARKET_GROUP": GROUP_SPEC},
        "2026-09-28T10:00:00Z",
        "2026-09-28T10:30:00Z",
    )
    assert text.splitlines()[0].startswith(
        "LAB PROOF PASS: omnimarket#4134 head fdd93c786c + "
        "omnimarket#4198 head e812bfd787 + "
        "omnimarket#4214 head 0ba71c9566 on lab-101"
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
        '[ -n "${MARKET_PR:-}${MARKET_GROUP:-}" ] && SUBJECTS="$SUBJECTS omnimarket"'
        in text
    )
