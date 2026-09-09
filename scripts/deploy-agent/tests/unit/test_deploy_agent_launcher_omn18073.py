# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev unit must reach its environment through bash, not EnvironmentFile=.

OMN-18073. systemd's env-file parser does not implement bash ANSI-C ``$'...'``
quoting: it keeps the literal ``$'`` / ``'`` wrapper and drops every backslash
escape, so each ``\\n`` collapses to the bare letter ``n``. The operator env
store carries ``ONEXBOT_OCC_PRIVATE_KEY`` in exactly that form, and the agent
hands ``dict(os.environ)`` to ``docker compose``, whose interpolation wrote the
mangled key into every dev-lane container it created.

``scripts/deploy-runtime.sh`` bash-``source``s the very same file on the very
same host and decodes it correctly. These tests pin the unit onto that same
transport and prove, with a throwaway generated key, that the two parsers really
do disagree -- the positive control without which "the launcher works" is an
untested assertion.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_DEV_UNIT = _DEPLOY_DIR / "deploy-agent-dev.service"
_LAUNCHER = _DEPLOY_DIR / "deploy-agent-launch.sh"

_OPERATOR_ENV_STORE = "/home/jonah/.omnibase/.env"

_ENVIRONMENT_RE = re.compile(r'^Environment="?([A-Za-z_][A-Za-z0-9_]*)=')


def _directive_lines(unit: Path) -> list[str]:
    return [
        line
        for line in unit.read_text().splitlines()
        if line and not line.lstrip().startswith("#")
    ]


def _unit_value(unit: Path, name: str) -> str:
    """Return the value of one ``Environment=NAME=...`` directive."""
    for line in _directive_lines(unit):
        for prefix in (f"Environment={name}=", f'Environment="{name}='):
            if line.startswith(prefix):
                return line[len(prefix) :].rstrip('"')
    raise AssertionError(f"{unit.name} does not declare Environment={name}=")


def test_dev_unit_declares_no_environment_file_on_the_operator_store() -> None:
    """The RED control for this whole ticket: no EnvironmentFile= at all.

    Not "does not point at the store" -- none. A second EnvironmentFile= line
    would reintroduce systemd's parser for whatever it names, and the store is
    the only env file this unit ever had.
    """
    offenders = [
        line
        for line in _directive_lines(_DEV_UNIT)
        if line.startswith("EnvironmentFile=")
    ]
    assert offenders == [], (
        "deploy-agent-dev.service must not use EnvironmentFile= -- systemd's "
        "parser mangles the store's ANSI-C-quoted value (OMN-18073). Source it "
        f"through {_LAUNCHER.name} instead. Found: {offenders}"
    )
    # The store must still be reachable, by the name the launcher requires.
    assert _unit_value(_DEV_UNIT, "DEPLOY_AGENT_ENV_FILE") == _OPERATOR_ENV_STORE


def test_dev_unit_exec_start_goes_through_the_launcher() -> None:
    """ExecStart is the launcher, and the canonical interpreter moves with it.

    The interpreter assertion is inherited from OMN-13760: a prior drift shipped
    WorkingDirectory pointed at the canonical repo copy while ExecStart still
    named the legacy ``/data/omninode/deploy-agent/venv`` python, and systemd
    does not use WorkingDirectory to resolve the ExecStart binary. That
    invariant now lives on DEPLOY_AGENT_PYTHON, so it must be asserted there.
    """
    exec_start = [
        line for line in _directive_lines(_DEV_UNIT) if line.startswith("ExecStart=")
    ]
    assert exec_start == [
        "ExecStart=/data/omninode/omnibase_infra/scripts/deploy-agent/deploy/"
        "deploy-agent-launch.sh"
    ], exec_start

    interpreter = _unit_value(_DEV_UNIT, "DEPLOY_AGENT_PYTHON")
    assert (
        interpreter
        == "/data/omninode/omnibase_infra/scripts/deploy-agent/.venv/bin/python"
    )
    assert "/data/omninode/deploy-agent/venv" not in interpreter


def test_launcher_exists_and_is_an_executable_bash_script() -> None:
    """POSIX ``sh`` does not implement ``$'...'``; the shebang must be bash."""
    assert _LAUNCHER.is_file(), f"{_LAUNCHER} must exist"
    assert _LAUNCHER.stat().st_mode & 0o111, f"{_LAUNCHER} must be executable"
    assert _LAUNCHER.read_text().splitlines()[0].endswith("bash")


def test_protected_list_covers_every_name_the_unit_declares() -> None:
    """The invariant that keeps the launcher faithful instead of merely different.

    systemd applies directives in file order, so the ``Environment=`` lines that
    followed ``EnvironmentFile=`` used to WIN over the store. A plain
    ``set -a; source`` inverts that. Measured on the lab host 2026-09-09, the
    store defines both KAFKA_BOOTSTRAP_SERVERS and KAFKA_ENVIRONMENT, and its
    KAFKA_ENVIRONMENT is ``local`` while this unit declares ``dev`` -- so an
    unprotected name is a silently relabelled lane, not a cosmetic difference.

    Deriving the expected set from the unit itself is the point: adding a new
    ``Environment=`` line without extending DEPLOY_AGENT_ENV_PROTECTED fails
    here rather than in production.
    """
    declared = {
        match.group(1)
        for line in _directive_lines(_DEV_UNIT)
        if (match := _ENVIRONMENT_RE.match(line))
    }
    protected = set(_unit_value(_DEV_UNIT, "DEPLOY_AGENT_ENV_PROTECTED").split())

    assert declared, "unit declares no Environment= directives -- parser is wrong"
    missing = sorted(declared - protected)
    assert missing == [], (
        "every name the unit declares must be in DEPLOY_AGENT_ENV_PROTECTED, or "
        f"the operator env store silently overrides it. Missing: {missing}"
    )


# --------------------------------------------------------------------------
# Behavioural proof: the two parsers disagree, and the launcher takes bash's.
# --------------------------------------------------------------------------


def _throwaway_ansi_c_line(name: str) -> tuple[str, str]:
    """Return (env-file line, the value bash should decode it to).

    A throwaway, generated, non-secret PEM-shaped body of the same SHAPE as the
    real credential -- multi-line, base64-ish, and containing a literal ``n`` in
    the payload. The ``n`` is the whole reason the executor guard refuses rather
    than repairs: once the backslashes are gone nothing can tell a newline's
    ``n`` from one that belongs to the value.
    """
    decoded = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "bm90YXJlYWxrZXluZXZlcnVzZWRhbnl3aGVyZQ==\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    line = f"{name}=$'" + decoded.replace("\n", "\\n") + "'"
    return line, decoded


def _run_launcher(tmp_path: Path, env_file: Path, protected: str) -> dict[str, str]:
    """Run the launcher with a stub interpreter that dumps its environment."""
    dump = tmp_path / "env.dump"
    stub = tmp_path / "python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"open({str(dump)!r}, 'w').write(repr(dict(os.environ)))\n"
    )
    stub.chmod(0o755)

    result = subprocess.run(
        [str(_LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "DEPLOY_AGENT_ENV_FILE": str(env_file),
            "DEPLOY_AGENT_PYTHON": str(stub),
            "DEPLOY_AGENT_ENV_PROTECTED": protected,
            **{
                name: "unit-owned" for name in protected.split() if name.startswith("K")
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    return dict(eval(dump.read_text()))  # noqa: S307 - our own stub's output


def test_systemd_environmentfile_mangles_what_bash_decodes(tmp_path: Path) -> None:
    """The positive control. Without this the launcher fix is an assertion.

    Same file, two parsers. bash yields real newlines; systemd's documented
    env-file grammar -- which is what ``EnvironmentFile=`` implements, and which
    has no ``$'...'`` production -- keeps the wrapper and drops the backslashes.
    Reproduced here with a throwaway key so the disagreement is proven in CI
    rather than cited from a lab readback.
    """
    line, decoded = _throwaway_ansi_c_line("THROWAWAY_KEY")
    env_file = tmp_path / "store.env"
    env_file.write_text(line + "\n")

    bash_value = subprocess.run(
        ["bash", "-c", f'set -a; . "{env_file}"; set +a; printf "%s" "$THROWAWAY_KEY"'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert bash_value == decoded
    assert "\n" in bash_value
    assert not bash_value.startswith("$'")

    # systemd's grammar, applied to the same bytes: the wrapper is data and a
    # backslash escape is not a production, so `\n` is the two characters that
    # were written and the leading `$'` survives into the value.
    raw = env_file.read_text().split("=", 1)[1].strip()
    systemd_value = raw.replace("\\n", "n")
    assert systemd_value.startswith("$'")
    assert "\n" not in systemd_value
    assert systemd_value != bash_value


def test_launcher_hands_the_agent_the_bash_decoded_value(tmp_path: Path) -> None:
    """GREEN: through the launcher, the process env carries the real value."""
    line, decoded = _throwaway_ansi_c_line("THROWAWAY_KEY")
    env_file = tmp_path / "store.env"
    env_file.write_text(line + "\n")

    env = _run_launcher(tmp_path, env_file, "KAFKA_ENVIRONMENT")

    assert env["THROWAWAY_KEY"] == decoded
    assert not env["THROWAWAY_KEY"].startswith("$'")
    # The executor's own guard is what production checks; assert the launched
    # environment would pass it rather than restating the shape by hand.
    assert [
        name
        for name, value in env.items()
        if value.startswith("$'") and value.endswith("'") and len(value) >= 3
    ] == []


def test_launcher_keeps_the_unit_declaration_winning_over_the_store(
    tmp_path: Path,
) -> None:
    """The lane fence. A plain `set -a; source` would fail this test.

    The real store sets KAFKA_ENVIRONMENT=local; the dev unit declares dev.
    """
    env_file = tmp_path / "store.env"
    env_file.write_text("KAFKA_ENVIRONMENT=local\nSTORE_ONLY_NAME=from-store\n")

    env = _run_launcher(tmp_path, env_file, "KAFKA_ENVIRONMENT")

    assert env["KAFKA_ENVIRONMENT"] == "unit-owned"
    # Negative control: an unprotected name still comes from the store, so the
    # test above is proving precedence rather than proving the source was a
    # no-op.
    assert env["STORE_ONLY_NAME"] == "from-store"


def test_launcher_exports_its_own_path_for_the_self_update_re_exec(
    tmp_path: Path,
) -> None:
    """DEPLOY_AGENT_LAUNCHER is the handle executor.self_update() re-execs."""
    env_file = tmp_path / "store.env"
    env_file.write_text("STORE_ONLY_NAME=from-store\n")

    env = _run_launcher(tmp_path, env_file, "KAFKA_ENVIRONMENT")

    assert Path(env["DEPLOY_AGENT_LAUNCHER"]).resolve() == _LAUNCHER.resolve()


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("DEPLOY_AGENT_ENV_FILE", "DEPLOY_AGENT_ENV_FILE is required"),
        ("DEPLOY_AGENT_PYTHON", "DEPLOY_AGENT_PYTHON is required"),
        ("DEPLOY_AGENT_ENV_PROTECTED", "DEPLOY_AGENT_ENV_PROTECTED is required"),
    ],
)
def test_launcher_refuses_a_missing_required_input(
    tmp_path: Path, missing: str, expected: str
) -> None:
    """Fail-closed: no default, and never a fallback to an un-sourced env.

    Falling back would start the agent with exactly the environment this script
    exists to replace, and it would start successfully.
    """
    env_file = tmp_path / "store.env"
    env_file.write_text("STORE_ONLY_NAME=from-store\n")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "DEPLOY_AGENT_ENV_FILE": str(env_file),
        "DEPLOY_AGENT_PYTHON": sys.executable,
        "DEPLOY_AGENT_ENV_PROTECTED": "KAFKA_ENVIRONMENT",
    }
    del env[missing]

    result = subprocess.run(
        [str(_LAUNCHER)], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert expected in result.stderr


def test_launcher_refuses_an_unreadable_env_file(tmp_path: Path) -> None:
    """An absent store is a refusal, not a silent start on the ambient env."""
    result = subprocess.run(
        [str(_LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "DEPLOY_AGENT_ENV_FILE": str(tmp_path / "does-not-exist.env"),
            "DEPLOY_AGENT_PYTHON": sys.executable,
            "DEPLOY_AGENT_ENV_PROTECTED": "KAFKA_ENVIRONMENT",
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "env file is not readable" in result.stderr


def test_launcher_refuses_a_non_identifier_protected_name(tmp_path: Path) -> None:
    """The protected list reaches ``eval``; a malformed unit must refuse."""
    env_file = tmp_path / "store.env"
    env_file.write_text("STORE_ONLY_NAME=from-store\n")
    result = subprocess.run(
        [str(_LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "DEPLOY_AGENT_ENV_FILE": str(env_file),
            "DEPLOY_AGENT_PYTHON": sys.executable,
            "DEPLOY_AGENT_ENV_PROTECTED": "KAFKA_ENVIRONMENT rm$(id)",
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "non-identifier name" in result.stderr
