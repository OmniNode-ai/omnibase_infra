# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task 7 of epic OMN-18888, the teardown half: destroy one pre-PR slot.

Until this module existed a slot was brought up by a sanctioned entrypoint and
torn down by hand, five times on 2026-09-24, each lane re-deriving the list of
what a slot leaves behind on the SHARED dev-lane servers. A hand teardown that
deletes by a loose pattern can delete the dev lane's own topics, groups or
Valkey keys, so every selector here is asserted against listings that carry the
dev lane's names beside the slot's, and the dev names must survive.

The reaper and heartbeat lease are exercised in test_prepr_slot_reaper_omn18896.py.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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

    def __init__(
        self,
        *,
        leave_topic: bool = False,
        dev_topics: int = 2,
        servers: dict[str, list[str]] | None = None,
        refuse_rm: bool = False,
        slot_topics: int = 0,
        unhealthy_after_topic_deletes: int | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.unhealthy_after_topic_deletes = unhealthy_after_topic_deletes
        self.topic_deletes = 0
        self.envs: list[dict[str, str] | None] = []
        self.leave_topic = leave_topic
        self.servers: dict[str, list[str]] = {
            "redpanda": ["omnibase-infra-redpanda"],
            "valkey": ["omnibase-infra-valkey"],
            "postgres": ["omnibase-infra-postgres"],
        }
        if servers is not None:
            self.servers.update(servers)
        self.refuse_rm = refuse_rm
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
        self.topics |= {f"prepr1.onex.bulk{i:04d}" for i in range(slot_topics)}
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
        if argv[:2] == ["docker", "exec"]:
            container_index = 2
            while argv[container_index] == "-e":
                container_index += 2
            container = argv[container_index]
            if not any(container in names for names in self.servers.values()):
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    "",
                    f"Error response from daemon: No such container: {container}",
                )
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
        if argv[:3] == ["docker", "ps", "--filter"]:
            service = next(
                a.split("=", 2)[2]
                for a in argv
                if a.startswith("label=com.docker.compose.service=")
            )
            assert argv == [
                "docker",
                "ps",
                "--filter",
                "label=com.docker.compose.project=omnibase-infra",
                "--filter",
                f"label=com.docker.compose.service={service}",
                "--filter",
                "status=running",
                "--format",
                "{{.Names}}",
            ]
            return self._ok("\n".join(self.servers.get(service, [])))
        if argv[:3] == ["docker", "ps", "-a"]:
            return self._ok("\n".join(sorted(self.containers.get(label, set()))))
        if argv[:3] == ["docker", "rm", "-f"]:
            if self.refuse_rm:
                return subprocess.CompletedProcess(
                    argv, 64, "", "LANDING-COMPOSE-GUARD refused: test"
                )
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
        if "rpk" in argv and "cluster" in argv and "health" in argv:
            bound = self.unhealthy_after_topic_deletes
            healthy = bound is None or self.topic_deletes < bound
            return self._ok(
                "CLUSTER HEALTH OVERVIEW\n=======================\n"
                f"Healthy:                          {str(healthy).lower()}\n"
            )
        if "rpk" in argv and "topic" in argv and "delete" in argv:
            self.topic_deletes += 1
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


def test_a_renamed_broker_is_found_by_its_compose_labels(tmp_path: Path) -> None:
    broker = "e2fc03e50cd0_omnibase-infra-redpanda"
    host = FakeHost(servers={"redpanda": [broker]})
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_OK, report
    assert report["verdict"] == "CLEAN"
    assert all(v == 0 for v in report["residue"].values()), report["residue"]
    assert host.topics == {"onex.dev.t0", "onex.dev.t1", "prepr2.onex.a"}
    assert host.groups == {"local.g", "prepr2.g"}
    rpk_calls = [argv for argv in host.calls if "rpk" in argv]
    assert rpk_calls
    assert all(argv[argv.index("rpk") - 1] == broker for argv in rpk_calls)


def test_a_renamed_postgres_and_valkey_are_found_by_their_compose_labels(
    tmp_path: Path,
) -> None:
    postgres = "e2fc03e50cd0_omnibase-infra-postgres"
    valkey = "e2fc03e50cd0_omnibase-infra-valkey"
    host = FakeHost(servers={"postgres": [postgres], "valkey": [valkey]})
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_OK, report
    assert report["verdict"] == "CLEAN"
    assert all(v == 0 for v in report["residue"].values()), report["residue"]
    assert host.databases == {"omnibase_infra"}
    assert host.roles == {"postgres"}
    assert host.valkey == {0: 5, 1: 0, 2: 1}
    psql_calls = [argv for argv in host.calls if "psql" in argv]
    valkey_calls = [argv for argv in host.calls if "valkey-cli" in argv]
    assert psql_calls and valkey_calls
    assert all(argv[argv.index("psql") - 1] == postgres for argv in psql_calls)
    assert all(argv[argv.index("valkey-cli") - 1] == valkey for argv in valkey_calls)


def test_no_running_broker_refuses_before_any_delete(tmp_path: Path) -> None:
    host = FakeHost(servers={"redpanda": []})
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["verdict"] == "INCOMPLETE"
    assert any(
        "no running redpanda container in compose project omnibase-infra" in error
        for error in report["errors"]
    )
    assert not any(
        "delete" in argv
        or argv[:3] == ["docker", "rm", "-f"]
        or "FLUSHDB" in argv
        or argv[:1] == ["bash"]
        for argv in host.calls
    )
    assert (tmp_path / "slot-1").exists()


def test_two_running_brokers_are_ambiguous_and_refused(tmp_path: Path) -> None:
    brokers = ["omnibase-infra-redpanda", "e2fc03e50cd0_omnibase-infra-redpanda"]
    host = FakeHost(servers={"redpanda": brokers})
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["verdict"] == "INCOMPLETE"
    assert any(
        "redpanda" in error and all(name in error for name in brokers)
        for error in report["errors"]
    )
    assert not any(
        "delete" in argv
        or argv[:3] == ["docker", "rm", "-f"]
        or "FLUSHDB" in argv
        or argv[:1] == ["bash"]
        for argv in host.calls
    )
    assert (tmp_path / "slot-1").exists()


def test_each_server_is_resolved_once_per_run(tmp_path: Path) -> None:
    host = FakeHost()
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_OK, report
    for service in ("redpanda", "valkey", "postgres"):
        assert (
            sum(
                argv[:3] == ["docker", "ps", "--filter"]
                and f"label=com.docker.compose.service={service}" in argv
                for argv in host.calls
            )
            == 1
        )


def test_a_refused_container_removal_stops_before_anything_else_is_deleted(
    tmp_path: Path,
) -> None:
    host = FakeHost(refuse_rm=True)
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["verdict"] == "INCOMPLETE"
    assert host.containers[teardown.selectors_for(SLOT1).compose_project] == {
        "c1",
        "c2",
    }
    assert not any("delete" in argv for argv in host.calls)
    assert not any("FLUSHDB" in argv for argv in host.calls)
    assert not any(
        argv[:2] == ["bash", str(teardown.PROVISION_DB)] for argv in host.calls
    )
    assert not any(
        argv[:3] in (["docker", "volume", "rm"], ["docker", "rmi", "-f"])
        for argv in host.calls
    )
    assert (tmp_path / "slot-1" / "repo").exists()


def test_a_staging_root_that_cannot_be_removed_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_rmtree(path: Path, *, ignore_errors: bool) -> None:
        raise OSError(39, "Directory not empty")

    monkeypatch.setattr(teardown, "shutil", SimpleNamespace(rmtree=refuse_rmtree))
    code, report = _run(FakeHost(), tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["verdict"] == "INCOMPLETE"
    assert report["residue"]["staging_root"] == 1
    assert any(error.startswith("remove staging root:") for error in report["errors"])


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


@pytest.mark.parametrize("settles", [True, False])
def test_consumer_groups_are_retried_for_one_bounded_session_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settles: bool
) -> None:
    class DrainingHost(FakeHost):
        attempts = 0

        def __call__(
            self, argv: list[str], env: dict[str, str] | None = None
        ) -> subprocess.CompletedProcess[str]:
            if "rpk" in argv and "group" in argv and "delete" in argv:
                self.attempts += 1
                if self.attempts < 3 or not settles:
                    self.calls.append(argv)
                    return subprocess.CompletedProcess(
                        argv, 1, "", "group is still active"
                    )
            return super().__call__(argv, env)

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    host = DrainingHost()
    code, report = _run(host, tmp_path)
    assert host.attempts >= 3
    assert clock[0] <= teardown.GROUP_DRAIN_TIMEOUT_S
    assert code == (policy.EXIT_OK if settles else policy.EXIT_TEARDOWN_INCOMPLETE)
    assert report["residue"]["groups"] == (0 if settles else 1)
    assert {"local.g", "prepr2.g"} <= host.groups


def test_group_retry_does_not_treat_an_unreadable_listing_as_empty(
    tmp_path: Path,
) -> None:
    class UnreadableHost(FakeHost):
        def __call__(
            self, argv: list[str], env: dict[str, str] | None = None
        ) -> subprocess.CompletedProcess[str]:
            if (
                "rpk" in argv
                and "group" in argv
                and "list" in argv
                and not self.containers[teardown.policy.SLOTS[1].compose_project]
            ):
                return subprocess.CompletedProcess(argv, 1, "", "broker unavailable")
            return super().__call__(argv, env)

    code, report = _run(UnreadableHost(), tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE
    assert report["residue"]["groups"] == -1


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


# ---------------------------------------------------------------------------
# OMN-17427: a slot's topic deletes are paced by the shared broker's health.
#
# 2026-10-20 15:18Z: slot 2's teardown deleted ~1845 prepr2 topics from the
# SHARED dev broker in back-to-back chunks of 200. Partition removal stalled,
# the broker (Redpanda v24.2.7, smp 1) crashed at ~15:19:15Z and asserted on
# every restart after, taking the dev bus down. A teardown must never push the
# dev lane's own broker past the point where it stops answering.
# ---------------------------------------------------------------------------


def _topic_delete_calls(host: FakeHost) -> list[list[str]]:
    return [c for c in host.calls if "rpk" in c and "topic" in c and "delete" in c]


def test_topic_deletes_are_small_chunks_each_followed_by_a_broker_health_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(teardown, "BROKER_HEALTH_POLL_SECONDS", 0.0)
    host = FakeHost(slot_topics=120)
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_OK, report
    deletes = _topic_delete_calls(host)
    assert len(deletes) > 1
    for call in deletes:
        names = call[call.index("delete") + 1 :]
        assert 0 < len(names) <= teardown.TOPIC_DELETE_CHUNK
    for i, call in enumerate(host.calls):
        if call in deletes:
            nxt = host.calls[i + 1]
            assert "cluster" in nxt and "health" in nxt, (call, nxt)
    assert not any(t.startswith("prepr1.") for t in host.topics)


def test_a_broker_that_stops_reporting_healthy_halts_the_topic_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(teardown, "BROKER_HEALTH_POLL_SECONDS", 0.0)
    host = FakeHost(slot_topics=120, unhealthy_after_topic_deletes=1)
    code, report = _run(host, tmp_path)
    assert code == policy.EXIT_TEARDOWN_INCOMPLETE, report
    assert report["verdict"] == "INCOMPLETE"
    assert len(_topic_delete_calls(host)) == 1
    assert report["residue"]["topics"] > 0
    assert {"onex.dev.t0", "onex.dev.t1", "prepr2.onex.a"} <= host.topics
