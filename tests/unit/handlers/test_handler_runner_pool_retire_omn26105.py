# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26105: a declared verb retires a whole runner pool on any registry host, idle only, one runner at a time.

The installer refused every pool-wide retire on a non-primary host ("--soft, --retire-surplus and --add require the
primary host"), so h202's CI pool could not be drained. The verb selects containers by the exact pool prefix followed
by ``-<digits>``: the customer-plane and verify runners share h202 and the name stem with the CI pool, and a prefix
match on the stem is a defect. Host and pool names are read from the fleet declaration, as the installer reads them.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_runner_pool_retire import HandlerRunnerPoolRetire
from omnibase_infra.models.model_runner_pool_retire_request import (
    ModelRunnerPoolRetireRequest,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FLEET = ROOT / "config/runner_fleet.yaml"
INSTALLER = ROOT / "scripts/deploy-runners.sh"


def _secondary_ci_host() -> tuple[str, str, str, str]:
    """The fleet's address-keyed host, its CI pool prefix, and the prefixes of its customer-plane and verify pools."""
    fleet = load_runner_fleet_config(FLEET)
    host = next(
        row
        for row in fleet.hosts
        if row.host[0].isdigit()
        and any(pool.runner_name_prefix.endswith("-ci-runner") for pool in row.pools)
    )
    ci = next(p for p in host.pools if p.runner_name_prefix.endswith("-ci-runner"))
    customer = next(p for p in host.pools if "customer-plane" in p.classes)
    return (
        host.host,
        ci.runner_name_prefix,
        customer.runner_name_prefix,
        host.runner_name_prefix,
    )


H202, POOL, CUSTOMER_PLANE_POOL, VERIFY_POOL = _secondary_ci_host()
STEM = POOL.split("-")[0]
CUSTOMER_PLANE = f"{CUSTOMER_PLANE_POOL}-1"
VERIFY = f"{VERIFY_POOL}-1"
DESTRUCTIVE = ("docker stop", "docker rm", "-X DELETE", "docker kill", "docker restart")


class _Host:
    """A runner host and the organization registry as the handler's two seams see them."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        handler: HandlerRunnerPoolRetire,
        containers: dict[str, dict[str, object]],
        *,
        registry: bool = True,
    ) -> None:
        self.containers = containers
        self.events: list[tuple[str, str]] = []
        self.scripts: list[str] = []
        self.commands: list[tuple[str, ...]] = []
        self.registry_readable = registry
        self.registered: dict[str, int] = {
            name: index
            for index, name in enumerate(containers, start=100)
            if containers[name].get("registered", True)
        }
        monkeypatch.setattr(handler, "_remote", self.remote)
        monkeypatch.setattr(handler, "_command", self.command)
        monkeypatch.setattr(handler, "_pause", self.pause)

    async def pause(self) -> None:
        self.events.append(("pause", ""))

    def _name_of(self, script: str) -> str:
        found = [name for name in self.containers if name in script]
        ids = [
            name
            for name, state in self.containers.items()
            if str(state.get("id", "")) and str(state["id"]) in script
        ]
        return (found or ids or [""])[0]

    async def remote(
        self, _host: str, script: str, *, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        self.scripts.append(script)
        if "docker ps" in script:
            self.events.append(("list", ""))
            return 0, "\n".join(
                f"{state.get('id', name + '-id')} {name} {state.get('state', 'running')} "
                f"{state.get('project', POOL)} {state.get('service', name)}"
                for name, state in self.containers.items()
            )
        name = self._name_of(script)
        state = self.containers.get(name, {})
        if "docker stop" in script:
            self.events.append(("stop", name))
            if state.get("busy"):
                return 42, ""
            state["state"] = "exited"
            return 0, ""
        if "docker rm" in script:
            self.events.append(("remove", name))
            self.containers.pop(name, None)
            return 0, ""
        # the idle probe
        self.events.append(("probe", name))
        if state.get("unreadable"):
            return 1, ""
        return 0, "busy" if state.get("busy") else "idle"

    async def command(
        self, *argv: str, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        self.commands.append(argv)
        assert argv[0] == "gh"
        if "DELETE" in argv:
            runner_id = int(argv[-1].rsplit("/", 1)[1])
            name = next(n for n, i in self.registered.items() if i == runner_id)
            self.events.append(("deregister", name))
            self.registered.pop(name)
            return 0, ""
        self.events.append(("registry", ""))
        if not self.registry_readable:
            return 1, ""
        rows = [
            {
                "id": runner_id,
                "name": name,
                "status": (
                    "online"
                    if self.containers.get(name, {}).get("state", "running")
                    == "running"
                    else "offline"
                ),
                "busy": bool(self.containers.get(name, {}).get("busy", False)),
            }
            for name, runner_id in self.registered.items()
        ]
        return 0, json.dumps([{"runners": rows}])

    def names(self, kind: str) -> list[str]:
        return [name for event, name in self.events if event == kind]


def _pool(count: int = 3, **extra: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        f"{POOL}-{i}": dict(extra.get(f"{POOL}-{i}", {})) for i in range(1, count + 1)
    }


def _request(**fields: object) -> ModelRunnerPoolRetireRequest:
    values: dict[str, object] = {
        "host": H202,
        "prefix": POOL,
        "fleet_config": FLEET,
        **fields,
    }
    return ModelRunnerPoolRetireRequest(**values)


# ----------------------------------------------------------------- selection


@pytest.mark.asyncio
async def test_retire_pool_selects_the_exact_prefix_and_never_a_neighbour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    containers = {
        **_pool(3),
        f"{POOL}-10": {},
        CUSTOMER_PLANE: {"project": CUSTOMER_PLANE_POOL},
        VERIFY: {"project": VERIFY_POOL},
        # Names that carry the prefix or the stem without being `<prefix>-<digits>`.
        f"{POOL}-extra": {},
        f"{POOL}-1-sidecar": {},
        f"x{POOL}-4": {},
        f"{STEM}-5": {},
        # A pool name whose compose project says it belongs to another pool.
        f"{POOL}-11": {"project": VERIFY_POOL},
    }
    host = _Host(monkeypatch, handler, containers)
    result = await handler.handle(_request(dry_run=True))
    assert result.success is True
    assert result.planned == (f"{POOL}-1", f"{POOL}-2", f"{POOL}-3", f"{POOL}-10")
    assert CUSTOMER_PLANE not in result.planned and VERIFY not in result.planned
    assert not [s for s in host.scripts if any(w in s for w in DESTRUCTIVE)]


@pytest.mark.asyncio
async def test_retire_pool_secondary_host_never_selects_customer_plane_or_verify_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    # Even when every neighbour reports the pool's own compose project, the name decides.
    containers = {
        **_pool(2),
        CUSTOMER_PLANE: {"project": POOL},
        VERIFY: {"project": POOL},
    }
    host = _Host(monkeypatch, handler, containers)
    result = await handler.handle(_request())
    assert result.success is True
    assert result.retired == (f"{POOL}-1", f"{POOL}-2")
    assert set(host.containers) == {CUSTOMER_PLANE, VERIFY}
    assert not [e for e in host.events if e[1] in {CUSTOMER_PLANE, VERIFY}]


# ------------------------------------------------------------------- dry run


@pytest.mark.asyncio
async def test_retire_pool_dry_run_prints_the_plan_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(
        monkeypatch,
        handler,
        _pool(3, **{f"{POOL}-2": {"busy": True}}),
    )
    result = await handler.handle(_request(dry_run=True))
    assert result.success is True
    assert result.planned == (f"{POOL}-1", f"{POOL}-2", f"{POOL}-3")
    assert result.busy == (f"{POOL}-2",)
    assert result.retired == ()
    assert set(host.containers) == set(_pool(3))
    assert not [s for s in host.scripts if any(w in s for w in DESTRUCTIVE)]
    assert not [c for c in host.commands if "DELETE" in c]
    # A dry run reads container evidence only: it needs no organization registry access.
    assert host.commands == []


@pytest.mark.asyncio
async def test_retire_pool_dry_run_honours_limit_and_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    _Host(monkeypatch, handler, _pool(4))
    limited = await handler.handle(_request(dry_run=True, limit=2))
    assert limited.planned == (f"{POOL}-1", f"{POOL}-2")
    only = await handler.handle(_request(dry_run=True, only=f"{POOL}-3"))
    assert only.planned == (f"{POOL}-3",)
    refused = await handler.handle(_request(dry_run=True, only=CUSTOMER_PLANE))
    assert refused.success is False
    assert refused.planned == ()


# ------------------------------------------------------------- idle only / drain


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [{"busy": True}, {"unreadable": True}], ids=["busy", "unreadable"]
)
async def test_retire_pool_never_stops_removes_or_deregisters_a_busy_runner(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, object]
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, _pool(2, **{f"{POOL}-1": state}))
    result = await handler.handle(_request())
    assert result.retired == (f"{POOL}-2",)
    assert result.success is False
    assert f"{POOL}-1" in (result.error_message or "")
    assert f"{POOL}-1" not in host.names("stop")
    assert f"{POOL}-1" not in host.names("remove")
    assert f"{POOL}-1" not in host.names("deregister")
    assert f"{POOL}-1" in host.containers


@pytest.mark.asyncio
async def test_retire_pool_unreadable_registry_counts_as_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, _pool(2), registry=False)
    result = await handler.handle(_request())
    assert result.success is False
    assert result.retired == ()
    assert host.names("stop") == [] and host.names("remove") == []


@pytest.mark.asyncio
async def test_retire_pool_online_busy_in_the_registry_is_not_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The container shows no worker, yet GitHub says the runner holds a job: the second signal wins.
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, _pool(1))
    original = host.command

    async def registry_busy(
        *argv: str, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        code, output = await original(*argv, stdin=stdin, timeout=timeout)
        if "DELETE" not in argv and output:
            rows = json.loads(output)
            for row in rows[0]["runners"]:
                row["busy"] = True
            output = json.dumps(rows)
        return code, output

    monkeypatch.setattr(handler, "_command", registry_busy)
    result = await handler.handle(_request())
    assert result.retired == ()
    assert host.names("stop") == []


@pytest.mark.asyncio
async def test_retire_pool_drains_one_runner_at_a_time_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, _pool(3))
    result = await handler.handle(_request())
    assert result.success is True
    assert result.retired == (f"{POOL}-1", f"{POOL}-2", f"{POOL}-3")
    assert result.deregistered == result.retired
    steps = [
        (event, name)
        for event, name in host.events
        if event in {"stop", "deregister", "remove"}
    ]
    # stop new jobs, then deregister, then remove the container; the next runner starts after the last one is gone.
    assert steps == [
        (event, f"{POOL}-{i}")
        for i in (1, 2, 3)
        for event in ("stop", "deregister", "remove")
    ]
    assert host.containers == {}


@pytest.mark.asyncio
async def test_retire_pool_limit_stops_after_n_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, _pool(3))
    result = await handler.handle(_request(limit=1))
    assert result.success is True
    assert result.retired == (f"{POOL}-1",)
    assert set(host.containers) == {f"{POOL}-2", f"{POOL}-3"}


@pytest.mark.asyncio
async def test_retire_pool_stop_that_finds_a_job_leaves_the_runner_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Idle at the first read, busy by the time of the stop: the remote check refuses (exit 42) and nothing is removed.
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, _pool(1))
    seen = {"probes": 0}
    original = host.remote

    async def remote(
        h: str, script: str, *, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        if "docker stop" in script:
            host.containers[f"{POOL}-1"]["busy"] = True
        elif "docker ps" not in script:
            seen["probes"] += 1
        return await original(h, script, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(handler, "_remote", remote)
    result = await handler.handle(_request())
    assert result.retired == ()
    assert host.names("remove") == [] and host.names("deregister") == []
    assert f"{POOL}-1" in host.containers


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["stop", "rm"])
async def test_retire_pool_destructive_scripts_name_one_container_and_no_pattern(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, {**_pool(2), CUSTOMER_PLANE: {}})
    await handler.handle(_request())
    scripts = [s for s in host.scripts if f"docker {command} " in s]
    assert len(scripts) == 2
    for script in scripts:
        assert "docker ps" not in script and "xargs" not in script
        assert " -v" not in script.replace("-t 60", "")
        assert CUSTOMER_PLANE not in script and VERIFY not in script
        assert len(re.findall(rf"{POOL}-[0-9]+-id", script)) >= 1
        assert not re.search(r"docker (stop|rm)[^\n]*[*?]", script)


# ------------------------------------------------- secondary host and declared pools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_name", "prefix"),
    [
        (H202, POOL),
        ("h051", "hp051-omnibase-ci-runner"),
        ("h053", "hp053-ci-runner"),
        ("h053", "hp053-ci-mq-runner"),
    ],
    ids=["h202", "h051", "h053", "h053-mq"],
)
async def test_retire_pool_secondary_host_accepts_a_declared_ci_pool(
    monkeypatch: pytest.MonkeyPatch, host_name: str, prefix: str
) -> None:
    handler = HandlerRunnerPoolRetire()
    containers = {
        f"{prefix}-{i}": {"project": prefix, "service": f"{prefix}-{i}"} for i in (1, 2)
    }
    _Host(monkeypatch, handler, containers)
    result = await handler.handle(_request(host=host_name, prefix=prefix, dry_run=True))
    assert result.success is True, result.error_message
    assert result.planned == (f"{prefix}-1", f"{prefix}-2")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_name", "prefix"),
    [
        (H202, CUSTOMER_PLANE_POOL),
        (H202, VERIFY_POOL),
        (H202, f"{STEM}-"),
        (H202, STEM),
        (H202, "hp053-ci-runner"),
        ("h053", POOL),
        ("unknown-host", POOL),
    ],
    ids=[
        "customer-plane",
        "verify",
        "stem-dash",
        "stem",
        "pool-of-another-host",
        "pool-on-the-wrong-host",
        "undeclared-host",
    ],
)
async def test_retire_pool_secondary_host_refuses_what_is_not_a_declared_ci_pool(
    monkeypatch: pytest.MonkeyPatch, host_name: str, prefix: str
) -> None:
    handler = HandlerRunnerPoolRetire()
    host = _Host(monkeypatch, handler, {**_pool(2), CUSTOMER_PLANE: {}, VERIFY: {}})
    result = await handler.handle(_request(host=host_name, prefix=prefix, dry_run=True))
    assert result.success is False
    assert result.planned == ()
    # Refused from the local fleet declaration, before a single remote read.
    assert host.scripts == [] and host.commands == []


# ------------------------------------------------------------- the installer flag


def _installer(
    tmp_path: Path, *args: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the real installer with ssh, rsync, docker and gh replaced by a recorder."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    probe = tmp_path / "effects"
    listing = "\n".join(
        f"{n}-id {n} running {p} {n}"
        for n, p in [
            *((f"{POOL}-{i}", POOL) for i in (1, 2)),
            (CUSTOMER_PLANE, CUSTOMER_PLANE_POOL),
            (VERIFY, VERIFY_POOL),
        ]
    )
    for command in ("rsync", "docker", "gh"):
        stub = bin_dir / command
        stub.write_text(
            '#!/bin/sh\nprintf "%s\\n" "${0##*/}" >> "$ROLL_TEST_PROBE"\nexit 89\n'
        )
        stub.chmod(0o755)
    ssh = bin_dir / "ssh"
    ssh.write_text(
        '#!/bin/sh\nargs="$*"\ncase "$args" in\n'
        '*"docker stop"*|*"docker rm"*|*"-X DELETE"*) printf "write\\n" >> "$ROLL_TEST_PROBE"; exit 89 ;;\n'
        '*"docker ps"*) printf "%s\\n" "$ROLL_TEST_LISTING" ;;\n'
        "*) echo idle ;;\nesac\n"
    )
    ssh.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\nshift 4\nshift\nexec "$ROLL_TEST_PYTHON" "$@"\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUNNER_FLEET_CONFIG_PATH": str(FLEET),
            "ROLL_TEST_PROBE": str(probe),
            "ROLL_TEST_LISTING": listing,
            "ROLL_TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(ROOT / "src"),
            "ONEX_STATE_DIR": str(tmp_path / "state"),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result, probe.read_text() if probe.exists() else ""


def test_retire_pool_secondary_host_flag_plans_the_pool_without_a_token_or_a_write(
    tmp_path: Path,
) -> None:
    result, effects = _installer(
        tmp_path, "--retire-pool", f"--host={H202}", f"--pool={POOL}", "--dry-run"
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"{POOL}-1" in output and f"{POOL}-2" in output
    assert CUSTOMER_PLANE not in output and VERIFY not in output
    # No registration token, no registry read, no sync and no write reached the host.
    assert effects == ""


@pytest.mark.parametrize(
    "other", ["--rolling", "--soft", "--retire-surplus", "--relabel", "--add=svc"]
)
def test_retire_pool_secondary_host_flag_refuses_every_other_mode(
    tmp_path: Path, other: str
) -> None:
    result, effects = _installer(
        tmp_path,
        "--retire-pool",
        f"--host={H202}",
        f"--pool={POOL}",
        other,
        "--dry-run",
    )
    assert result.returncode != 0
    assert "--retire-pool" in result.stdout + result.stderr
    assert effects == ""


def test_retire_pool_secondary_host_flag_refuses_the_primary_host(
    tmp_path: Path,
) -> None:
    result, effects = _installer(tmp_path, "--retire-pool", "--dry-run")
    assert result.returncode != 0
    assert "--retire-pool" in result.stdout + result.stderr
    assert effects == ""


def test_retire_pool_secondary_host_flag_refuses_a_neighbour_pool_before_any_remote_read(
    tmp_path: Path,
) -> None:
    result, effects = _installer(
        tmp_path,
        "--retire-pool",
        f"--host={H202}",
        f"--pool={CUSTOMER_PLANE_POOL}",
        "--dry-run",
    )
    assert result.returncode != 0
    assert effects == ""
    assert shlex.quote("write") not in effects
