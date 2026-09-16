# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the additive `--add=<service>` deploy mode (OMN-18408).

Standing up `omninode-verify-runner-1` needed a path none of the three existing
modes provides:

- the DEFAULT path force-recreates the whole compose project -- roughly three
  hours serialised across the 88-runner fleet (OMN-18188), during which the
  monitor's auto-bounce cron races anything left offline past 300s;
- ``--soft`` never creates a container at all, only updates entrypoints in ones
  already running;
- ``--rolling --only=NAME`` (OMN-18415) refuses this by design, twice over.
  ``fleet_services()`` enumerates only ``omninode-runner-N``, so a non-pool name
  fails closed as "not a fleet service", and ``roll_one_runner`` requires the
  target to be online and idle before it will touch it -- which a container that
  does not exist yet can never be.

Those refusals are correct and are not relaxed here. ``--add`` is a separate
verb: create or converge a DECLARED, non-general-pool service, additively.

Why not widen ``--only``: it already means "narrow the rolling recreate to this
runner". Giving one flag both "recreate this existing pool runner" and "create
this new non-pool runner" puts two very different blast radii behind one string
an operator types under time pressure, on a script whose default mode takes the
fleet down for three hours.

Assertions run the real script in ``--dry-run`` and read the rendered command
line, which is the actual string ``run_ssh`` would execute.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

VERIFY_SERVICE = "omninode-verify-runner-1"
POOL_SERVICE = "omninode-runner-7"
UNDECLARED_SERVICE = "omninode-not-a-real-service"


def _write_exec(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def stub_bin(tmp_path: Path) -> Path:
    """`gh`, `ssh` and `rsync` stubs.

    A dry run should never invoke them. The stubs are what make that a proven
    property rather than an assumption: were a future edit to start shelling out
    during a dry run, the stub records it instead of reaching the live org or
    the live lab host.
    """
    if shutil.which("bash") is None:  # pragma: no cover - environment guard
        pytest.skip("bash not available")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.log"
    for tool in ("gh", "ssh", "rsync", "scp"):
        _write_exec(
            bindir / tool,
            f"""\
            printf '{tool} %s\\n' "$*" >> "{calls}"
            exit 0
            """,
        )
    return bindir


def _run(bindir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _compose_up_line(stdout: str) -> str:
    for line in stdout.splitlines():
        if "compose" in line and "up -d" in line:
            return line
    raise AssertionError(f"no compose up line in dry-run output:\n{stdout}")


def test_the_script_still_parses() -> None:
    """Control for everything below: a broken script cannot be trusted to have
    refused anything on purpose."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_default_path_still_force_recreates_everything(stub_bin: Path) -> None:
    """Positive control.

    Every assertion in the additive tests is the ABSENCE of a flag. Without
    this, a typo dropping the flags from BOTH paths would read as `--add`
    working perfectly.
    """
    result = _run(stub_bin, "--dry-run", "--skip-build")
    assert result.returncode == 0, result.stderr
    line = _compose_up_line(result.stdout)
    assert "--force-recreate" in line
    assert "--remove-orphans" in line


def test_add_names_only_the_requested_service(stub_bin: Path) -> None:
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={VERIFY_SERVICE}")
    assert result.returncode == 0, result.stderr
    assert VERIFY_SERVICE in _compose_up_line(result.stdout)


def test_add_never_force_recreates(stub_bin: Path) -> None:
    """Additive means additive. A converged container is not restarted, so
    running `--add` against a healthy runner is a no-op rather than an outage.
    """
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={VERIFY_SERVICE}")
    assert result.returncode == 0, result.stderr
    assert "--force-recreate" not in _compose_up_line(result.stdout)


def test_add_never_removes_orphans(stub_bin: Path) -> None:
    """`--remove-orphans` is evaluated against the WHOLE project even when
    services are named: it deletes every container the compose file does not
    declare. It must never ride a one-service call."""
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={VERIFY_SERVICE}")
    assert result.returncode == 0, result.stderr
    assert "--remove-orphans" not in _compose_up_line(result.stdout)


def test_add_refuses_a_general_pool_runner(stub_bin: Path) -> None:
    """The general pool belongs to the fleet and rolling paths.

    Creating one outside them skips the toolcache seeding the fleet procedure
    brackets every recreate with, and bypasses the busy check that keeps a roll
    from killing somebody's running job.
    """
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={POOL_SERVICE}")
    assert result.returncode != 0
    assert POOL_SERVICE in result.stderr
    assert "--rolling" in result.stderr, "the refusal must name the right path"


def test_add_refuses_an_undeclared_service(stub_bin: Path) -> None:
    """Fail closed on a typo.

    `docker compose up -d <unknown>` errors on the host, but only after the
    rsync has already run; refusing locally keeps a typo from touching the host
    at all, and matches the rolling mode's own refusal for an unknown name.
    """
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={UNDECLARED_SERVICE}")
    assert result.returncode != 0
    assert UNDECLARED_SERVICE in result.stderr


def test_add_refuses_to_combine_with_soft_or_rolling(stub_bin: Path) -> None:
    for other in ("--soft", "--rolling"):
        result = _run(
            stub_bin, "--dry-run", "--skip-build", other, f"--add={VERIFY_SERVICE}"
        )
        assert result.returncode != 0, f"--add with {other} was accepted"


def test_add_does_not_wait_for_the_whole_fleet(stub_bin: Path) -> None:
    """The fleet poll waits for `expected_count` runners in the group.

    Inheriting it would block for the full poll window and then warn on every
    run, on a fleet this mode never touched -- training the operator to ignore
    the one signal that says the new runner did not come up.
    """
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={VERIFY_SERVICE}")
    assert result.returncode == 0, result.stderr
    assert "Polling GitHub API for 88 online runners" not in result.stdout
    assert VERIFY_SERVICE in result.stdout


def test_add_does_not_reinstall_the_fleet_crons(stub_bin: Path) -> None:
    """Cron installs rewrite the operator's crontab. They are fleet-level and
    idempotent, but a one-container deploy has no business touching them."""
    result = _run(stub_bin, "--dry-run", "--skip-build", f"--add={VERIFY_SERVICE}")
    assert result.returncode == 0, result.stderr
    assert "prune cron" not in result.stdout
    assert "monitor cron" not in result.stdout


def test_add_exports_the_deploy_runner_registration_variable() -> None:
    """A brand-new container has no cached registration to restore from.

    The non-pool services interpolate DEPLOY_RUNNER_TOKEN rather than
    RUNNER_TOKEN, so without this the container starts and then fails at
    registration -- which looks like a broken image, not a missing handle.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    assert "DEPLOY_RUNNER_TOKEN" in text
    # Control: the rolling path deliberately exports an EMPTY RUNNER_TOKEN,
    # because a recreate restores cached creds. If that disappears, this
    # assertion is reading a different script than the one documented.
    assert "export RUNNER_TOKEN=''" in text


def test_help_documents_the_additive_mode(stub_bin: Path) -> None:
    result = _run(stub_bin, "--help")
    assert result.returncode == 0
    assert "--add" in result.stdout
