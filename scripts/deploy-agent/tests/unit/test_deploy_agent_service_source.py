# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Systemd unit must execute deploy-agent code from the canonical repo copy."""

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"


def test_service_uses_canonical_repo_source() -> None:
    service = _DEPLOY_DIR / "deploy-agent.service"
    text = service.read_text()

    assert "WorkingDirectory=/data/omninode/omnibase_infra/scripts/deploy-agent" in text
    assert (
        "Environment=DEPLOY_AGENT_DIR=/data/omninode/omnibase_infra/scripts/deploy-agent"
        in text
    )
    assert "WorkingDirectory=/data/omninode/deploy-agent" not in text

    # ExecStart must invoke the interpreter from a venv rooted under the
    # canonical repo copy, not the legacy standalone `/data/omninode/deploy-agent`
    # install. A prior drift shipped WorkingDirectory pointed at the canonical
    # path while ExecStart still hardcoded the legacy venv's python -- systemd
    # ignores WorkingDirectory for resolving the ExecStart binary path, so that
    # combination silently ran the stale legacy venv/interpreter.
    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=")
    ]
    assert exec_start_lines, "service file must declare ExecStart"
    assert all(
        "/data/omninode/omnibase_infra/scripts/deploy-agent/.venv/bin/python" in line
        for line in exec_start_lines
    ), exec_start_lines
    assert all(
        "/data/omninode/deploy-agent/venv" not in line for line in exec_start_lines
    )


def test_service_declares_no_watchdog() -> None:
    """OMN-13760: the base unit must not arm WatchdogSec.

    The agent runs minutes-long synchronous rebuilds that block the event loop,
    so a systemd liveness watchdog SIGABRTs it mid-rebuild. Restart=on-failure
    covers genuine crashes instead.
    """
    text = (_DEPLOY_DIR / "deploy-agent.service").read_text()
    unit_lines = [ln.strip() for ln in text.splitlines()]
    assert not any(ln.startswith("WatchdogSec=") for ln in unit_lines), (
        "deploy-agent.service must not declare WatchdogSec (see OMN-13760)"
    )


def test_override_drop_in_disables_watchdog() -> None:
    """The tracked .201 drop-in must explicitly disable the watchdog.

    Belt-and-suspenders in case an older base unit carrying WatchdogSec=30 is
    still installed on the host.
    """
    override = _DEPLOY_DIR / "deploy-agent.service.d" / "override.conf"
    text = override.read_text()
    assert "WatchdogSec=0" in text


def test_override_drop_in_is_wired_to_prod_broker() -> None:
    """OMN-15181 Finding 3: the single live deploy-agent must consume prod.

    Live /proc/<pid>/environ readback (2026-07-26) showed the only running
    deploy-agent process wired to KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:39092 /
    KAFKA_ENVIRONMENT=stability-test -- zero consumer presence on the prod
    broker, so a real gated prod redeploy command would sit unconsumed even
    after the network (Finding 1) and resolver (Finding 2) gaps are fixed.
    This locks the repoint to the prod broker's host-mapped external
    listener (docker/docker-compose.prod.yml,
    ${PROD_REDPANDA_EXTERNAL_PORT:-49092}:19092, advertised externally as
    192.168.86.201:49092 -- reachable from omninode-pc itself) so a future
    edit can't silently drift the only live instance back to stability-test
    (or leave it there) without failing this test.
    """
    override = _DEPLOY_DIR / "deploy-agent.service.d" / "override.conf"
    text = override.read_text()

    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=/usr/bin/env")
    ]
    assert exec_start_lines, "override.conf must declare an ExecStart override"
    exec_start = exec_start_lines[0]

    assert "KAFKA_BOOTSTRAP_SERVERS=192.168.86.201:49092" in exec_start
    assert "KAFKA_ENVIRONMENT=prod" in exec_start
    assert "127.0.0.1:39092" not in exec_start
    assert "stability-test" not in exec_start

    env_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("Environment=KAFKA_ENVIRONMENT=")
    ]
    assert env_lines == ["Environment=KAFKA_ENVIRONMENT=prod"]


def test_dev_lane_unit_exists_and_is_fenced_to_dev() -> None:
    """OMN-16939: the dev-lane instance must be a tracked unit, not a hand-edit.

    The prod drop-in pins deploy-agent.service to 192.168.86.201:49092, so
    before this unit existed the DEV broker's rebuild-requested topic had zero
    consumer groups and four signed commands sat on it unconsumed.
    """
    text = (_DEPLOY_DIR / "deploy-agent-dev.service").read_text()

    assert "Environment=DEPLOY_AGENT_ALLOWED_LANES=dev" in text
    # The assertion that keeps the dev instance off the prod bus: the literal
    # dev broker address is the point of the test, not a fallback.
    dev_broker_line = (
        "Environment=KAFKA_BOOTSTRAP_SERVERS=192.168.86.201:19092"  # kafka-fallback-ok
    )
    assert dev_broker_line in text
    # must not collide with the prod instance's port or state dir
    assert "Environment=DEPLOY_AGENT_PORT=8098" in text
    assert (
        "Environment=DEPLOY_AGENT_STATE_DIR=/data/omninode/deploy-agent/state/jobs-dev"
        in text
    )
    # canonical repo copy for both cwd and interpreter (OMN-13760).
    #
    # OMN-18073 moved the interpreter off ExecStart: the unit now execs
    # deploy/deploy-agent-launch.sh, which bash-`source`s the operator env store
    # (systemd's EnvironmentFile= parser mangles its ANSI-C-quoted value) and
    # then execs DEPLOY_AGENT_PYTHON. Both halves must still resolve inside the
    # canonical repo copy, so the OMN-13760 invariant is asserted on both.
    assert "WorkingDirectory=/data/omninode/omnibase_infra/scripts/deploy-agent" in text
    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=")
    ]
    assert exec_start_lines
    assert all(
        line.startswith("ExecStart=/data/omninode/omnibase_infra/scripts/deploy-agent/")
        for line in exec_start_lines
    ), exec_start_lines
    assert (
        "Environment=DEPLOY_AGENT_PYTHON=/data/omninode/omnibase_infra/"
        "scripts/deploy-agent/.venv/bin/python" in text
    )
    assert "/data/omninode/deploy-agent/venv" not in text
    assert not any(ln.strip().startswith("WatchdogSec=") for ln in text.splitlines())


def test_dev_unit_does_not_touch_the_prod_bus() -> None:
    """Negative control for the test above: the dev unit must never name the
    prod broker port, and the prod drop-in must never name the dev one."""
    dev = (_DEPLOY_DIR / "deploy-agent-dev.service").read_text()
    prod_override = (
        _DEPLOY_DIR / "deploy-agent.service.d" / "override.conf"
    ).read_text()

    dev_directives = [
        ln for ln in dev.splitlines() if ln and not ln.lstrip().startswith("#")
    ]
    assert not any("49092" in ln for ln in dev_directives), dev_directives
    assert "192.168.86.201:49092" in prod_override
    assert "Environment=DEPLOY_AGENT_ALLOWED_LANES=prod" in prod_override


def test_no_unit_kills_the_port_holder() -> None:
    """OMN-16939: no unit may kill whatever already holds its port.

    ``ExecStartPre=/bin/sh -c 'fuser -k <port>/tcp || true'`` shipped in #3262
    (hostile-reviewer MAJOR fp=0c64d40e5e65, undisclosed at merge). It is a
    blind ``kill -9`` of *whatever* is listening: the peer lane's agent, a
    still-draining rebuild subprocess, or an unrelated process that happened to
    bind the port. ``|| true`` then swallows the outcome, so the unit starts
    reporting success either way and the kill leaves no trace.

    A port collision is a fail-fast error, not a licence to kill. The
    preflight names the holder and refuses; a human decides what to stop.
    """
    kill_verbs = (
        "fuser -k",
        "fuser --kill",
        "pkill",
        "killall",
        "kill -9",
        "kill -KILL",
    )
    for unit in sorted(_DEPLOY_DIR.glob("*.service")) + sorted(
        _DEPLOY_DIR.glob("*.service.d/*.conf")
    ):
        text = unit.read_text()
        for directive in (
            "ExecStartPre=",
            "ExecStart=",
            "ExecStartPost=",
            "ExecStop=",
            "ExecStopPost=",
            "ExecReload=",
        ):
            for line in text.splitlines():
                if not line.startswith(directive):
                    continue
                for verb in kill_verbs:
                    assert verb not in line, (
                        f"{unit.name}: {directive} must not kill the port holder "
                        f"(found {verb!r} in {line!r})"
                    )


def test_units_preflight_their_port_fail_closed() -> None:
    """Each agent unit must fail-fast on a busy port via the shared preflight."""
    expected = {
        "deploy-agent.service": "8099",
        "deploy-agent-dev.service": "8098",
    }
    for unit_name, port in expected.items():
        text = (_DEPLOY_DIR / unit_name).read_text()
        pre_lines = [ln for ln in text.splitlines() if ln.startswith("ExecStartPre=")]
        assert pre_lines, f"{unit_name} must declare an ExecStartPre port preflight"
        assert any(
            "preflight_port_free.sh" in ln and ln.rstrip().endswith(f" {port}")
            for ln in pre_lines
        ), pre_lines
        # No `|| true` anywhere on the preflight line: swallowing the exit
        # status is what made the fuser form report success unconditionally.
        assert all("|| true" not in ln for ln in pre_lines), pre_lines


def test_preflight_script_is_executable_and_never_kills() -> None:
    script = _DEPLOY_DIR / "preflight_port_free.sh"
    assert script.is_file(), f"{script} must exist"
    assert script.stat().st_mode & 0o111, f"{script} must be executable"
    body = script.read_text()
    for verb in ("fuser", "pkill", "killall", "kill "):
        assert verb not in body, f"preflight must never kill: found {verb!r}"


_PREFLIGHT = _DEPLOY_DIR / "preflight_port_free.sh"


def _run_preflight(port: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_PREFLIGHT), str(port)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("bad", ["", "0", "70000", "80a", "-1"])
def test_preflight_rejects_bad_port_fail_closed(bad: str) -> None:
    """An unusable argument is exit 2 (unprovable), never a silent pass."""
    assert _run_preflight(bad).returncode == 2


def test_preflight_passes_when_port_is_free() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    # Socket closed: nothing is listening on free_port now.
    result = _run_preflight(free_port)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    shutil.which("ss") is None and shutil.which("lsof") is None,
    reason="no socket-inspection tool on this host",
)
def test_preflight_refuses_and_names_the_holder_without_stopping_it() -> None:
    """A held port is exit 1, the holder is named, and it is still alive after."""
    with socket.socket() as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]

        result = _run_preflight(port)

        assert result.returncode == 1, (result.returncode, result.stderr)
        assert f"port {port} is already in use" in result.stderr
        assert "Refusing to start" in result.stderr
        # "Names the holder" is the whole point of replacing the kill-first
        # form, so assert the identity actually reaches stderr rather than
        # only that *some* refusal was printed. The holder here is this very
        # pytest process, so an unprivileged probe can always attribute it.
        assert str(os.getpid()) in result.stderr, result.stderr
        # The holder must still be listening: the preflight observes, never acts.
        holder.setblocking(False)
        with socket.socket() as client:
            client.settimeout(2)
            client.connect(("127.0.0.1", port))


def _write_stub(directory: Path, name: str, body: str) -> None:
    stub = directory / name
    stub.write_text(body)
    stub.chmod(0o755)


def test_preflight_fails_closed_when_the_probe_tool_errors(tmp_path: Path) -> None:
    """A probe that fails to RUN is exit 2, never a silent "port is free".

    Both probes were previously invoked as ``$(... 2>/dev/null || true)``, which
    maps *every* tool failure -- unsupported filter syntax on an older
    iproute2, a netlink error, /proc/net unreadable in a restricted namespace
    -- onto an empty holder list and therefore onto exit 0. That is fail-OPEN,
    and it silently contradicted the docblock's fail-closed contract. Shim both
    tools onto failing stubs and assert the refusal (hostile-reviewer MAJOR
    fp=b71ceadd2319).
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub(bindir, "ss", '#!/bin/sh\necho "ss: netlink error" >&2\nexit 1\n')
    # lsof exit 1 means "ran, matched nothing"; 2 is a genuine failure.
    _write_stub(bindir, "lsof", '#!/bin/sh\necho "lsof: fatal" >&2\nexit 2\n')

    result = subprocess.run(
        [str(_PREFLIGHT), "8098"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": str(bindir)},
    )
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "unproven" in result.stderr, result.stderr


def test_preflight_fails_closed_when_no_probe_tool_exists(tmp_path: Path) -> None:
    """No `ss` and no `lsof` is unprovable, so it refuses (exit 2)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [str(_PREFLIGHT), "8098"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": str(empty)},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "neither 'ss' nor 'lsof'" in result.stderr


def test_lsof_no_match_exit_1_is_free_not_an_error(tmp_path: Path) -> None:
    """Negative control for the test above: lsof's exit 1 must stay "free".

    `lsof` exits 1 when the query ran and matched nothing. Treating every
    non-zero probe status as a failure would turn the common "port is free"
    case into a permanent refusal, so the lsof branch must special-case it.
    Only `lsof` is on PATH here, which forces the fallback branch.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub(bindir, "lsof", "#!/bin/sh\nexit 1\n")

    result = subprocess.run(
        [str(_PREFLIGHT), "8098"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": str(bindir)},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_units_bound_the_fail_fast_restart_loop() -> None:
    """A refusal must stop, not re-refuse every RestartSec forever.

    Restart=on-failure with RestartSec=5 spaces five restarts over ~25s, which
    never trips systemd's default start limit of 5 starts in 10s -- so before
    this the unit would re-run the refusing preflight every five seconds for as
    long as the collision lasted (hostile-reviewer MINOR fp=b58479fbe9d6).
    """
    for unit_name in ("deploy-agent.service", "deploy-agent-dev.service"):
        text = (_DEPLOY_DIR / unit_name).read_text()
        directives = [ln.strip() for ln in text.splitlines()]
        assert "StartLimitIntervalSec=300" in directives, unit_name
        assert "StartLimitBurst=5" in directives, unit_name


def test_dev_unit_declares_its_sasl_scram_control_bus_transport() -> None:
    """OMN-18012: the dev unit is the dev agent's lane declaration surface.

    The dev-lane Redpanda external listener requires SCRAM-SHA-256 over
    PLAINTEXT. deploy_agent.kafka_config refuses to start on an undeclared
    transport, so the three names below are what make this unit startable at
    all -- and declaring them here, rather than letting the loader infer a
    protocol from the presence of credentials, is the whole fix.
    """
    text = (_DEPLOY_DIR / "deploy-agent-dev.service").read_text()

    assert "Environment=KAFKA_SECURITY_PROTOCOL=SASL_PLAINTEXT" in text
    assert "Environment=KAFKA_SASL_MECHANISM=SCRAM-SHA-256" in text
    # The lane's SCRAM principal stays in the operator env file under its
    # DEV_ prefix; a unit file must never carry the credential itself.
    assert "Environment=KAFKA_SASL_ENV_PREFIX=DEV_" in text
    assert "KAFKA_SASL_PASSWORD=" not in text
    assert "KAFKA_SASL_USERNAME=" not in text


def test_prod_drop_in_declares_its_plaintext_control_bus_transport() -> None:
    """The prod broker configures no SASL; PLAINTEXT is declared, not defaulted."""
    text = (_DEPLOY_DIR / "deploy-agent.service.d" / "override.conf").read_text()

    assert "Environment=KAFKA_SECURITY_PROTOCOL=PLAINTEXT" in text
    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=/usr/bin/env")
    ]
    assert exec_start_lines, "drop-in must re-declare ExecStart"
    assert all(
        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT" in line for line in exec_start_lines
    ), exec_start_lines
