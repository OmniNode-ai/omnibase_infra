# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The runner registration token never appears in a command line (OMN-21765).

``deploy-runners.sh`` used to hand the token to the runner host inside the ssh
command string::

    RUNNER_TOKEN=$(echo '<base64 of the token>' | base64 -d)

sshd starts the remote shell with that string as its argv, so a base64 of the
token (which is the same secret) sat in ``ps`` for every login on the host for
the length of the image build and the compose up. The token now travels on ssh
stdin and is read into the remote shell's environment.

These tests run the real code. ``ssh`` is a stub that records its argv and its
stdin and then runs the remote script locally, exactly as the remote shell
would, with a stub ``docker`` on PATH. The stub ``docker`` takes a ``ps``
listing at the moment compose would run, which is the moment the leak was
visible, and records the environment it received. Every assertion is on what
that listing and those logs hold, not on the source text, except the one
scanner test, which has its own positive control.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

# Not a credential: a marker long and distinctive enough that a substring hit is
# a real hit.
TOKEN = "AAAFAKEREGTOKENOMN21765MARKER0123456789"
TOKEN_B64 = base64.b64encode(TOKEN.encode()).decode()
SERVICE = "omninode-verify-runner-1"

FAKE_SSH = """\
#!/usr/bin/env bash
# Record argv, then behave like the remote end. Only a call that reads the token
# from stdin is executed (locally, under bash -c, as sshd would run it); every
# other call is recorded and answered with success.
log_dir="${FAKE_LOG_DIR}"
{ printf 'ARGV:'; printf ' [%s]' "$@"; printf '\\n'; } >> "${log_dir}/ssh_argv.log"
script="${2:-}"
if [[ "${script}" == *'$(cat)'* ]]; then
    stdin_copy="$(cat)"
    printf '%s' "${stdin_copy}" >> "${log_dir}/ssh_stdin.log"
    printf '%s' "${stdin_copy}" | bash -c "${script}"
    exit $?
fi
exit 0
"""

FAKE_DOCKER = """\
#!/usr/bin/env bash
log_dir="${FAKE_LOG_DIR}"
printf '%s\\n' "$*" >> "${log_dir}/docker.log"
if [[ "$*" == *" up "* ]]; then
    ps -A -ww -o args= >> "${log_dir}/ps.log" 2>&1
    printf '%s' "${RUNNER_TOKEN:-}" >> "${log_dir}/env_runner_token.log"
    printf '%s' "${DEPLOY_RUNNER_TOKEN:-}" >> "${log_dir}/env_deploy_runner_token.log"
    : > "${log_dir}/online"
fi
exit 0
"""

FAKE_GH = """\
#!/usr/bin/env bash
log_dir="${FAKE_LOG_DIR}"
if [[ "$*" == *"registration-token"* ]]; then
    printf '%s\\n' "${FAKE_TOKEN}"
    exit 0
fi
if [[ "$*" == *"actions/runners"* ]]; then
    if [[ -e "${log_dir}/online" ]]; then
        printf '{"runners":[{"name":"%s","status":"online","busy":false}]}\\n' "${FAKE_SERVICE}"
    else
        echo '{"runners":[]}'
    fi
    exit 0
fi
exit 0
"""


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def harness(tmp_path: Path) -> dict[str, Path]:
    if shutil.which("bash") is None or shutil.which("jq") is None:
        pytest.skip("bash and jq are required")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    host_dir = tmp_path / "host_runners"
    host_dir.mkdir()
    _write_exec(bindir / "ssh", FAKE_SSH)
    _write_exec(bindir / "docker", FAKE_DOCKER)
    _write_exec(bindir / "gh", FAKE_GH)
    for tool in ("rsync", "scp"):
        _write_exec(bindir / tool, "#!/usr/bin/env bash\nexit 0\n")
    return {"bin": bindir, "logs": logs, "host_dir": host_dir}


def _env(h: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{h['bin']}:{env.get('PATH', '')}"
    env["FAKE_LOG_DIR"] = str(h["logs"])
    env["FAKE_TOKEN"] = TOKEN
    env["FAKE_SERVICE"] = SERVICE
    env["ROLL_ONLINE_INTERVAL_SECONDS"] = "0"
    env["ROLL_ONLINE_MAX_SECONDS"] = "1"
    env.pop("DEPLOY_RUNNER_TOKEN", None)
    return env


def _read(h: dict[str, Path], name: str) -> str:
    path = h["logs"] / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _extract_function(script_text: str, name: str) -> str:
    marker = f"\n{name}() {{"
    assert marker in script_text, f"{name}() is not defined in deploy-runners.sh"
    start = script_text.index(marker) + 1
    depth = 0
    for i in range(start + len(marker) - 2, len(script_text)):
        ch = script_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return script_text[start : i + 1]
    raise AssertionError(f"unbalanced braces extracting {name}()")


def _extract_assignment(script_text: str, name: str) -> str:
    match = re.search(rf"^{name}='.*'$", script_text, flags=re.MULTILINE)
    assert match, f"{name}= is not defined in deploy-runners.sh"
    return match.group(0)


def _assert_token_stayed_out_of_every_command_line(h: dict[str, Path]) -> None:
    argv = _read(h, "ssh_argv.log")
    ps = _read(h, "ps.log")
    assert argv, "the ssh stub was never called: the test proved nothing"
    assert ps, "compose never ran: the ps listing was never taken"
    for secret in (TOKEN, TOKEN_B64):
        assert secret not in argv, "the token is in an ssh command string"
        assert secret not in ps, "the token is in a ps listing during compose up"
    # The remote shell was running while ps was taken; prove the listing is the
    # kind that would have shown the old command string.
    assert "bash -c" in ps or "docker" in ps


def _assert_token_reached_compose_by_stdin_and_environment(
    h: dict[str, Path], *, deploy_variable: bool = False
) -> None:
    assert TOKEN in _read(h, "ssh_stdin.log"), "the token never crossed on stdin"
    assert _read(h, "env_runner_token.log").strip() == TOKEN
    if deploy_variable:
        assert _read(h, "env_deploy_runner_token.log").strip() == TOKEN


# --- the detector has a positive control --------------------------------------


def test_the_ps_check_catches_the_old_command_string(tmp_path: Path) -> None:
    """Without this, a ps check that cannot see argv would pass for any script."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    listing = tmp_path / "ps.log"
    old_remote_script = (
        f"RUNNER_TOKEN=$(echo '{TOKEN_B64}' | base64 -d); export RUNNER_TOKEN; "
        f"ps -A -ww -o args= > {listing}"
    )
    subprocess.run(["bash", "-c", old_remote_script], check=True, timeout=30)
    assert TOKEN_B64 in listing.read_text(encoding="utf-8")


# --- the four ssh sites -------------------------------------------------------


def test_the_script_still_parses() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_default_deploy_keeps_the_token_out_of_argv(
    harness: dict[str, Path],
) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    functions = "\n".join(
        [
            _extract_assignment(text, "REMOTE_READ_TOKEN"),
            _extract_function(text, "ssh_token_stdin"),
            _extract_function(text, "deploy_runners"),
        ]
    )
    driver = textwrap.dedent(
        f"""\
        set -euo pipefail
        log() {{ :; }}
        DRY_RUN=false
        SKIP_BUILD=true
        RUNNER_HOST=fake-host
        RUNNER_HOST_DIR="{harness["host_dir"]}"
        {functions}
        deploy_runners "${{TEST_REGISTRATION_TOKEN}}"
        """
    )
    # The driver is itself a process with an argv: the token reaches it by
    # environment, or the test would leak the token it is checking for.
    env = _env(harness)
    env["TEST_REGISTRATION_TOKEN"] = TOKEN
    result = subprocess.run(
        ["bash", "-c", driver],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    _assert_token_stayed_out_of_every_command_line(harness)
    _assert_token_reached_compose_by_stdin_and_environment(harness)
    assert TOKEN not in result.stdout + result.stderr


def test_the_rolling_recreate_keeps_the_token_out_of_argv(
    harness: dict[str, Path], tmp_path: Path
) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    functions = "\n".join(
        [
            _extract_assignment(text, "REMOTE_READ_TOKEN"),
            _extract_function(text, "ssh_token_stdin"),
            _extract_function(text, "roll_one_runner"),
        ]
    )
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    token_file.chmod(0o600)
    driver = textwrap.dedent(
        f"""\
        set -euo pipefail
        log() {{ :; }}
        warn() {{ echo "$*" >&2; }}
        runner_is_idle() {{ return 0; }}
        runner_container_state() {{ echo running; }}
        _runner_cache_key_for_service() {{ echo deadbeefcafe0000; }}
        _runner_cache_ready() {{ return 0; }}
        wait_for_runner_online() {{ return 0; }}
        DRY_RUN=false
        RUNNER_HOST=fake-host
        RUNNER_HOST_DIR="{harness["host_dir"]}"
        TOKEN_FILE="{token_file}"
        {functions}
        roll_one_runner omninode-runner-41
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver],
        env=_env(harness),
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    _assert_token_stayed_out_of_every_command_line(harness)
    _assert_token_reached_compose_by_stdin_and_environment(harness)
    assert TOKEN not in result.stdout + result.stderr


def test_the_additive_mode_keeps_the_token_out_of_argv(
    harness: dict[str, Path],
) -> None:
    """The whole script, not a function: --add mints a token and runs compose."""
    env = _env(harness)
    result = subprocess.run(
        ["bash", str(SCRIPT), f"--add={SERVICE}", "--skip-build"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr[-2000:])
    _assert_token_stayed_out_of_every_command_line(harness)
    _assert_token_reached_compose_by_stdin_and_environment(
        harness, deploy_variable=True
    )
    assert TOKEN not in result.stdout + result.stderr


def test_the_dry_run_prints_no_token_and_says_where_it_would_go(
    harness: dict[str, Path],
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), f"--add={SERVICE}", "--skip-build", "--dry-run"],
        cwd=REPO_ROOT,
        env=_env(harness),
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr[-2000:])
    assert "registration token on stdin" in result.stdout
    assert TOKEN not in result.stdout + result.stderr
    assert "base64" not in result.stdout


def test_an_empty_stdin_makes_the_remote_script_refuse_before_compose(
    harness: dict[str, Path],
) -> None:
    """A token that failed to arrive must stop the call, not recreate a runner
    with an empty registration token (the OMN-18877 crash loop)."""
    text = SCRIPT.read_text(encoding="utf-8")
    driver = textwrap.dedent(
        f"""\
        set -u
        {_extract_assignment(text, "REMOTE_READ_TOKEN")}
        {_extract_function(text, "ssh_token_stdin")}
        log() {{ :; }}
        DRY_RUN=false
        RUNNER_HOST=fake-host
        ssh_token_stdin '' "set -euo pipefail; ${{REMOTE_READ_TOKEN}}; docker compose up -d"
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver],
        env=_env(harness),
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 64, (result.stdout, result.stderr)
    assert "docker" not in _read(harness, "docker.log")


# --- the scanner, for the paths the harness does not run ----------------------


def _token_in_a_command_string(script_text: str) -> list[str]:
    """Lines that build a token into text that becomes an argv.

    Flags a base64 round trip, and an ``echo``/``printf`` of a token variable
    inside a double-quoted remote script.
    """
    hits = []
    for line in script_text.splitlines():
        code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        if re.search(r"\|\s*base64\s+-d", code) or re.search(
            r"""echo\s+'?\$\{?(remote_)?(reg_handle|token|mig_token)[a-z_0-9]*\}?'?\s*\|""",
            code,
        ):
            hits.append(line)
    return hits


def test_the_scanner_flags_the_old_pattern() -> None:
    old = "        RUNNER_TOKEN=\\$(echo '${remote_token_b64}' | base64 -d)\n"
    assert _token_in_a_command_string(old)


def test_no_remote_script_in_deploy_runners_carries_a_token() -> None:
    assert _token_in_a_command_string(SCRIPT.read_text(encoding="utf-8")) == []


# --- inside the container: config.sh -------------------------------------------

ENTRYPOINT = REPO_ROOT / "docker" / "runners" / "entrypoint.sh"

FAKE_CONFIG_SH = """\
#!/usr/bin/env bash
{ printf 'ARGV:'; printf ' [%s]' "$@"; printf '\\n'; } >> "${FAKE_LOG_DIR}/config_argv.log"
ps -A -ww -o args= >> "${FAKE_LOG_DIR}/config_ps.log" 2>&1
printf '%s\\n' "${ACTIONS_RUNNER_INPUT_TOKEN:-}" >> "${FAKE_LOG_DIR}/config_env_token.log"
exit 0
"""


def test_the_entrypoint_hands_config_sh_the_token_in_its_environment(
    harness: dict[str, Path], tmp_path: Path
) -> None:
    """A container process is listed in the HOST's ps with its whole argv, so
    ``config.sh --token <value>`` showed the token for as long as registration
    took. The runner reads ACTIONS_RUNNER_INPUT_TOKEN for any input not given as
    a flag (measured on runner 2.336.0: with it config.sh attempts the
    registration, without it it stops at "Invalid configuration provided for
    token")."""
    runner_home = tmp_path / "runner_home"
    runner_home.mkdir()
    _write_exec(runner_home / "config.sh", FAKE_CONFIG_SH)
    text = ENTRYPOINT.read_text(encoding="utf-8")
    driver = textwrap.dedent(
        f"""\
        set -euo pipefail
        _as_runner() {{ "$@"; }}
        RUNNER_HOME="{runner_home}"
        RUNNER_NAME=omninode-runner-1
        GITHUB_ORG_URL=https://github.com/OmniNode-ai
        RUNNER_LABELS=self-hosted
        RUNNER_GROUP=
        RUNNER_WORK_DIR=/tmp/work
        {_extract_function(text, "_register")}
        {_extract_function(text, "_deregister")}
        _register
        _deregister
        # The variable must not outlive the call, or the listener inherits it.
        [ -z "${{ACTIONS_RUNNER_INPUT_TOKEN:-}}" ]
        """
    )
    env = _env(harness)
    env["RUNNER_TOKEN"] = TOKEN
    result = subprocess.run(
        ["bash", "-c", driver],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    argv = _read(harness, "config_argv.log")
    assert argv.count("ARGV:") == 2, "config.sh should have run for register and remove"
    assert "--token" not in argv
    assert TOKEN not in argv
    assert TOKEN not in _read(harness, "config_ps.log")
    assert _read(harness, "config_env_token.log").split() == [TOKEN, TOKEN]
    assert TOKEN not in result.stdout + result.stderr
