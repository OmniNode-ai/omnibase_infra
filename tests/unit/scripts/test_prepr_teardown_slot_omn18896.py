# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task 7 of epic OMN-18888, the teardown half: destroy one pre-PR slot.

Until this module existed a slot was brought up by a sanctioned entrypoint and
torn down by hand, five times on 2026-09-24, each lane re-deriving the list of
what a slot leaves behind on the SHARED dev-lane servers. A hand teardown that
deletes by a loose pattern can delete the dev lane's own topics, groups or
Valkey keys, so every selector here is asserted against listings that carry the
dev lane's names beside the slot's, and the dev names must survive.

The reaper and the heartbeat lease (the other half of Task 7) are not here.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
TEARDOWN_PY = RUNTIME_BUILD / "prepr_teardown_slot.py"
TEARDOWN_SH = RUNTIME_BUILD / "prepr_teardown_slot.sh"
POLICY_PATH = RUNTIME_BUILD / "prepr_slot_policy.py"

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = _load("prepr_slot_policy", POLICY_PATH)
teardown = _load("prepr_teardown_slot", TEARDOWN_PY)

SLOT1 = policy.SLOTS[1]

# Listings in the exact column shape rpk prints, dev names beside slot names.
TOPIC_LISTING = """NAME                                   PARTITIONS  REPLICAS
onex.evt.platform.node-heartbeat.v1    6           1
onex.cmd.deploy.rebuild-requested.v1   1           1
prepr1.onex.evt.platform.node-heartbeat.v1  6      1
prepr1.onex.cmd.omnimarket.x.v1        1           1
prepr10.onex.evt.platform.node-heartbeat.v1 6      1
prepr2.onex.evt.platform.node-heartbeat.v1  6      1
prepr1x.onex.evt.not-a-slot.v1         1           1
agent-actions                          1           1
"""

GROUP_LISTING = """BROKER  GROUP                                   STATE
0       agent-observability-postgres            Stable
0       local.omninode-runtime.node.consume.v1  Stable
0       prepr1.omninode-runtime.node.consume.v1 Empty
0       prepr1.omnimarket.writer.project.v1     Empty
0       prepr10.omninode-runtime.node.consume.v1 Empty
0       prepr2.omninode-runtime.node.consume.v1 Empty
"""


# ---------------------------------------------------------------------------
# Selectors: the slot's names and nothing else
# ---------------------------------------------------------------------------


def test_topic_selection_takes_only_the_slot_prefix() -> None:
    sel = teardown.selectors_for(SLOT1)
    names = teardown.parse_rpk_column(TOPIC_LISTING, "NAME")
    assert teardown.select_slot_names(names, sel.name_prefixes) == [
        "prepr1.onex.cmd.omnimarket.x.v1",
        "prepr1.onex.evt.platform.node-heartbeat.v1",
    ]


def test_group_selection_takes_only_the_slot_prefix() -> None:
    sel = teardown.selectors_for(SLOT1)
    names = teardown.parse_rpk_column(GROUP_LISTING, "GROUP")
    assert teardown.select_slot_names(names, sel.name_prefixes) == [
        "prepr1.omnimarket.writer.project.v1",
        "prepr1.omninode-runtime.node.consume.v1",
    ]


def test_the_rpk_parser_reads_the_named_column_not_a_position() -> None:
    names = teardown.parse_rpk_column(GROUP_LISTING, "GROUP")
    assert "local.omninode-runtime.node.consume.v1" in names
    assert "0" not in names and "Stable" not in names


def test_the_rpk_parser_refuses_a_listing_without_its_header() -> None:
    """A listing that is an error message must not parse as an empty pool."""
    with pytest.raises(ValueError):
        teardown.parse_rpk_column(
            "unable to request metadata: is SASL missing?\n", "NAME"
        )


@pytest.mark.parametrize(
    "bad_token", ["", "onex", "local", "prepr", "prepr0", "PREPR1", "prepr1.", "p"]
)
def test_a_token_that_is_not_a_slot_token_is_refused(bad_token: str) -> None:
    """An empty or dev-shaped prefix would select the dev lane's own names."""
    forged = SLOT1._replace(topic_namespace=bad_token)
    with pytest.raises(policy.RefusalError) as excinfo:
        teardown.selectors_for(forged)
    assert excinfo.value.code == policy.EXIT_USAGE


def test_the_dev_valkey_index_is_refused() -> None:
    forged = SLOT1._replace(valkey_db_index=0)
    with pytest.raises(policy.RefusalError):
        teardown.selectors_for(forged)


def test_a_declared_lane_project_is_refused_by_name() -> None:
    forged = SLOT1._replace(compose_project="omnibase-infra")
    with pytest.raises(policy.RefusalError) as excinfo:
        teardown.selectors_for(forged)
    assert excinfo.value.code == policy.EXIT_REFUSED_DECLARED_LANE


@pytest.mark.parametrize("slot", sorted(policy.SLOTS))
def test_every_pool_slot_has_valid_selectors(slot: int) -> None:
    sel = teardown.selectors_for(policy.SLOTS[slot])
    assert all(p.endswith(".") for p in sel.name_prefixes)
    assert sel.valkey_db_index != 0


def test_the_staging_root_must_sit_under_the_pool_parent(tmp_path: Path) -> None:
    parent = teardown.STAGING_PARENT
    assert teardown.resolve_staging_root(SLOT1, None) == parent / "slot-1"
    assert teardown.resolve_staging_root(SLOT1, str(parent / "x")) == parent / "x"
    for bad in ("/", str(parent.parent), str(parent), str(tmp_path), "/home"):
        with pytest.raises(policy.RefusalError):
            teardown.resolve_staging_root(SLOT1, bad)
    with pytest.raises(policy.RefusalError):
        teardown.resolve_staging_root(SLOT1, str(parent / "slot-1" / ".." / ".." / "x"))


# ---------------------------------------------------------------------------
# The whole run against a fake host
# ---------------------------------------------------------------------------


class FakeHost:
    """Answers the docker commands the teardown issues, and records them.

    State is a dict of sets so a delete really removes the name and the
    readback sees the result, rather than the test asserting a call was made.
    """

    def __init__(self, *, leave_topic: bool = False, dev_topics: int = 2) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.leave_topic = leave_topic
        self.containers = {
            "omnibase-infra-prepr-1": {"c1", "c2"},
            "omnibase-infra": {"d1", "d2"},
        }
        self.volumes = {"omnibase-infra-prepr-1": {"v1"}, "omnibase-infra": {"dv"}}
        self.images = {"omnibase-infra-prepr-1": {"i1", "i2"}, "omnibase-infra": {"di"}}
        self.topics = {f"onex.dev.t{i}" for i in range(dev_topics)} | {
            "prepr1.onex.a",
            "prepr1.onex.b",
            "prepr2.onex.a",
        }
        self.groups = {"local.g", "prepr1.g", "prepr2.g"}
        self.valkey = {0: 5, 1: 3, 2: 1}
        self.databases = {
            "omnibase_infra",
            "omnibase_infra_prepr1",
            "omniintelligence_prepr1",
        }
        self.roles = {"postgres", "role_omnibase_prepr1"}
        self.dropped_slot: str | None = None

    def _ok(self, out: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, out, "")

    def __call__(
        self, argv: list[str], env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        self.envs.append(env)
        joined = " ".join(argv)
        if argv[:2] == ["bash", str(teardown.PROVISION_DB)]:
            assert env is not None
            slot = env["ONEX_DB_SLOT"]
            self.dropped_slot = slot
            self.databases = {d for d in self.databases if not d.endswith("_" + slot)}
            self.roles = {r for r in self.roles if not r.endswith("_" + slot)}
            return self._ok()
        label = next(
            (a.rsplit("=", 1)[1] for a in argv if a.startswith("label=")), None
        )
        if argv[:3] == ["docker", "ps", "-a"]:
            return self._ok("\n".join(sorted(self.containers.get(label, set()))))
        if argv[:3] == ["docker", "rm", "-f"]:
            for p in self.containers.values():
                p.difference_update(argv[3:])
            return self._ok()
        if argv[:3] == ["docker", "volume", "ls"]:
            return self._ok("\n".join(sorted(self.volumes.get(label, set()))))
        if argv[:3] == ["docker", "volume", "rm"]:
            for p in self.volumes.values():
                p.difference_update(argv[3:])
            return self._ok()
        if argv[:2] == ["docker", "images"]:
            return self._ok("\n".join(sorted(self.images.get(label, set()))))
        if argv[:3] == ["docker", "rmi", "-f"]:
            for p in self.images.values():
                p.difference_update(argv[3:])
            return self._ok()
        if "rpk" in argv and "topic" in argv and "list" in argv:
            rows = "\n".join(f"{t} 1 1" for t in sorted(self.topics))
            return self._ok("NAME PARTITIONS REPLICAS\n" + rows + "\n")
        if "rpk" in argv and "topic" in argv and "delete" in argv:
            doomed = set(argv[argv.index("delete") + 1 :])
            if self.leave_topic:
                doomed = set(sorted(doomed)[1:])
            self.topics -= doomed
            return self._ok()
        if "rpk" in argv and "group" in argv and "list" in argv:
            rows = "\n".join(f"0 {g} Empty" for g in sorted(self.groups))
            return self._ok("BROKER GROUP STATE\n" + rows + "\n")
        if "rpk" in argv and "group" in argv and "delete" in argv:
            self.groups -= set(argv[argv.index("delete") + 1 :])
            return self._ok()
        if "valkey-cli" in argv:
            idx = int(argv[argv.index("-n") + 1])
            if argv[-1] == "FLUSHDB":
                self.valkey[idx] = 0
                return self._ok("OK")
            return self._ok(str(self.valkey[idx]))
        if "psql" in argv:
            sql = argv[-1]
            m = re.search(r"LIKE '%\\_(\w+)'", sql)
            if "pg_database" in sql and m:
                return self._ok(
                    str(sum(d.endswith("_" + m.group(1)) for d in self.databases))
                )
            if "pg_roles" in sql and m:
                return self._ok(
                    str(sum(r.endswith("_" + m.group(1)) for r in self.roles))
                )
            if "pg_database" in sql:
                return self._ok(str(len(self.databases)))
            raise AssertionError(f"unexpected psql: {sql}")
        raise AssertionError(f"unexpected command: {joined}")


def _run(
    host: FakeHost, tmp_path: Path, *, plan_only: bool = False
) -> tuple[int, dict[str, Any]]:
    staging = tmp_path / "slot-1"
    (staging / "repo").mkdir(parents=True)
    report_path = tmp_path / "report.json"
    code = teardown.run_teardown(
        SLOT1,
        runner=host,
        staging_root=staging,
        base_env={
            "POSTGRES_USER": "postgres",
            "RPK_USER": "u",
            "RPK_PASS": "p",
            "RPK_SASL_MECHANISM": "SCRAM-SHA-256",
            "REDISCLI_AUTH": "x",
        },
        plan_only=plan_only,
        report_path=report_path,
    )
    return code, json.loads(report_path.read_text(encoding="utf-8"))


def test_a_full_teardown_leaves_zero_slot_residue_and_the_dev_lane_intact(
    tmp_path: Path,
) -> None:
    host = FakeHost()
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_OK, report
    assert report["verdict"] == "CLEAN"
    assert all(v == 0 for v in report["residue"].values()), report["residue"]
    assert all(v > 0 for v in report["controls"].values()), report["controls"]
    # The dev lane and the other slot are untouched.
    assert host.containers["omnibase-infra"] == {"d1", "d2"}
    assert host.volumes["omnibase-infra"] == {"dv"}
    assert host.images["omnibase-infra"] == {"di"}
    assert {"onex.dev.t0", "onex.dev.t1", "prepr2.onex.a"} <= host.topics
    assert {"local.g", "prepr2.g"} <= host.groups
    assert host.valkey[0] == 5 and host.valkey[2] == 1
    assert "omnibase_infra" in host.databases
    assert host.dropped_slot == "prepr1"
    assert not (tmp_path / "slot-1").exists()


def test_no_delete_command_ever_names_a_non_slot_object(tmp_path: Path) -> None:
    host = FakeHost()
    _run(host, tmp_path)
    for argv in host.calls:
        if "delete" in argv:
            targets = argv[argv.index("delete") + 1 :]
            assert targets and all(t.startswith("prepr1.") for t in targets), argv
        if argv[:3] in (
            ["docker", "rm", "-f"],
            ["docker", "rmi", "-f"],
            ["docker", "volume", "rm"],
        ):
            assert not ({"d1", "d2", "dv", "di"} & set(argv)), argv
        if "valkey-cli" in argv and argv[-1] == "FLUSHDB":
            assert argv[argv.index("-n") + 1] == "1", argv


def test_credentials_travel_by_environment_never_on_the_command_line(
    tmp_path: Path,
) -> None:
    host = FakeHost()
    _run(host, tmp_path)
    for argv in host.calls:
        joined = " ".join(argv)
        assert "pass=" not in joined and "user=" not in joined, argv
        if "valkey-cli" in argv:
            assert "-a" not in argv, argv


def test_a_residue_the_readback_finds_fails_the_run(tmp_path: Path) -> None:
    host = FakeHost(leave_topic=True)
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["verdict"] == "INCOMPLETE"
    assert report["residue"]["topics"] == 1


def test_a_zero_with_no_positive_control_is_not_believed(tmp_path: Path) -> None:
    """Rule 16: a broker listing with no dev topics is an unproven zero."""
    host = FakeHost(dev_topics=0)
    host.topics.discard("prepr2.onex.a")
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["verdict"] == "UNPROVEN"


def test_plan_only_deletes_nothing(tmp_path: Path) -> None:
    host = FakeHost()
    code, report = _run(host, tmp_path, plan_only=True)
    assert code == policy.EXIT_OK
    assert report["verdict"] == "PLAN"
    assert report["planned"]["topics"] == ["prepr1.onex.a", "prepr1.onex.b"]
    mutating = [
        a
        for a in host.calls
        if "delete" in a
        or a[:3]
        in (["docker", "rm", "-f"], ["docker", "rmi", "-f"], ["docker", "volume", "rm"])
        or a[-1:] == ["FLUSHDB"]
        or a[:1] == ["bash"]
    ]
    assert mutating == []
    assert (tmp_path / "slot-1").exists()


# ---------------------------------------------------------------------------
# The shell wrapper
# ---------------------------------------------------------------------------


def test_the_wrapper_declares_no_lane_project_force_or_skip_option() -> None:
    text = TEARDOWN_SH.read_text(encoding="utf-8")
    options = set(re.findall(r"^\s+(--[a-z-]+)\)", text, re.M))
    assert options, "the option parser was not found"
    assert not options & {"--lane", "--compose-project", "--force", "--skip"}
    assert {"--slot", "--reason"} <= options


def test_the_wrapper_takes_the_same_slot_lock_as_the_bring_up() -> None:
    text = TEARDOWN_SH.read_text(encoding="utf-8")
    assert "lane_lock_acquire" in text
    assert re.search(r"trap\s+\S*release\S*\s+EXIT|trap\s+cleanup\s+EXIT", text)


def test_the_wrapper_exit_code_literal_matches_the_policy_module() -> None:
    text = TEARDOWN_SH.read_text(encoding="utf-8")
    literals = dict(re.findall(r"^(EXIT_[A-Z_]+)=(\d+)$", text, re.M))
    assert literals, "no EXIT_ literals found"
    for name, value in literals.items():
        assert getattr(policy, name) == int(value), name


def test_the_wrapper_refuses_a_missing_reason_before_touching_anything(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(TEARDOWN_SH), "--slot", "1"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        check=False,
    )
    assert result.returncode == policy.EXIT_REFUSED_ATTRIBUTION, result.stderr


def test_the_wrapper_refuses_a_slot_outside_the_pool(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(TEARDOWN_SH), "--slot", "3", "--reason", "test"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        check=False,
    )
    assert result.returncode == policy.EXIT_USAGE, result.stderr
