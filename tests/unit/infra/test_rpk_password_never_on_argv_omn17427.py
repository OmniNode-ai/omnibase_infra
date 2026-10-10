# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: no committed source puts a broker password on an ``rpk`` command line.

``ps`` on a shared host lists the full argv of every process, including the ones inside
containers. A ``-X pass=<value>`` or ``--password <value>`` flag, or a
``docker exec -e *PASS*=<value>``, keeps the live SASL password on that table for as long
as the call runs, and a consumer that never exits (a killed ``docker exec`` client leaves
its in-container ``rpk`` behind) keeps it there for days. A shell expansion such as
``-X pass="$DEV_KAFKA_SASL_PASSWORD"`` is no safer: the shell expands it before it execs
``rpk``, so the value is in the ``rpk`` argv even though the wrapper's argv shows only the
variable name.

The shape this repo uses instead is the environment ``rpk`` reads natively:
``RPK_USER``, ``RPK_PASS`` and ``RPK_SASL_MECHANISM`` set on the one call (or exported
inside the one-shot's shell), with no credential flag at all.

The scan is static, so it fires before the command is ever run. Each pattern has a positive
control (a synthetic bad line the scanner must flag) and the safe shape has a negative
control, because a scanner that flags everything or nothing is equally useless.

Not covered: ``rpk security user create <name> -p <password>``. That flag sets the password
of the principal being created and has no environment form; the three one-shots that use it
live for seconds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCANNED_ROOTS = ("docker", "scripts", "src")
_SCANNED_SUFFIXES = frozenset({".sh", ".py", ".yml", ".yaml"})

# `-X pass=<value>`: the value is a literal or a shell expansion. A bare `-X pass=` (the flag
# named in prose, nothing after the `=`) is not a leak.
_X_PASS_FLAG = re.compile(r"""-X[ \t]+["']?pass=["']?[^\s"'`]""")
# The argv-list form used from Python: ["-X", "pass=..."] / ["-X", f"pass={...}"].
_X_PASS_ARGV_ITEM = re.compile(r"""["']-X["']\s*,\s*f?["']pass=""")
_LONG_PASSWORD_FLAG = re.compile(r"--password(?:=|[ \t]+)[^\s-]")
# `docker exec -e SOME_PASS=...`: host-side argv even though rpk never sees it as a flag.
_DOCKER_EXEC_ENV_PASSWORD = re.compile(r"docker\s+exec\b[^\n]*[ \t]-e[ \t]+\w*PASS\w*=")

_PATTERNS = (
    _X_PASS_FLAG,
    _X_PASS_ARGV_ITEM,
    _LONG_PASSWORD_FLAG,
    _DOCKER_EXEC_ENV_PASSWORD,
)


def _find_violations(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            hits.append(f"line {line_no}: {match.group(0)!r}")
    return hits


def _scanned_files() -> list[Path]:
    return sorted(
        path
        for root in _SCANNED_ROOTS
        for path in (_REPO_ROOT / root).rglob("*")
        if path.is_file() and path.suffix in _SCANNED_SUFFIXES
    )


@pytest.mark.parametrize(
    "bad_line",
    [
        'rpk topic list -X user="$U" -X pass="$P"',
        "rpk topic list -X pass=hunter2",
        'AUTH="-X sasl.mechanism=SCRAM-SHA-256 -X pass=$$DEV_KAFKA_SASL_PASSWORD"',
        'rpk "$@" -X pass="$DEV_KAFKA_SASL_PASSWORD" -X sasl.mechanism=SCRAM-SHA-256',
        "rpk topic list --password hunter2",
        "rpk topic list --password=hunter2",
        'cmd = ["rpk", "topic", "list", "-X", "pass=" + password]',
        'cmd = ["rpk", "topic", "list", "-X", f"pass={password}"]',
        'docker exec -e RPK_PASS="$P" redpanda rpk topic list',
    ],
)
def test_scanner_flags_a_password_on_the_command_line(bad_line: str) -> None:
    assert _find_violations(bad_line + "\n"), f"scanner missed: {bad_line}"


@pytest.mark.parametrize(
    "good_line",
    [
        # The environment form, set on one call.
        'RPK_USER="$U" RPK_PASS="$P" RPK_SASL_MECHANISM=SCRAM-SHA-256 rpk topic list',
        # The environment form, exported inside the one-shot's shell.
        'export RPK_USER="$U" RPK_PASS="$P" RPK_SASL_MECHANISM=SCRAM-SHA-256',
        # The flag named in prose, with no value after the `=`.
        "pass `-X user= -X pass= -X sasl.mechanism=SCRAM-SHA-256` (never a value here)",
        "# a -X pass= flag would put the password on argv",
        # A non-secret -X setting.
        'rpk topic list -X brokers="redpanda:9092"',
        # Confluent client config key, not an rpk flag.
        'config["sasl.password"] = password',
    ],
)
def test_scanner_passes_the_environment_shape_and_prose(good_line: str) -> None:
    assert not _find_violations(good_line + "\n"), f"scanner flagged: {good_line}"


def test_scan_covers_the_files_that_run_rpk() -> None:
    names = {path.name for path in _scanned_files()}
    assert "docker-compose.dev-lane.yml" in names
    assert "docker-compose.ci-bus.yml" in names
    assert "_consumer_flow_lane.py" in names
    assert "c28_consumer_flow_probe.py" in names


def test_no_committed_source_puts_a_password_on_an_rpk_command_line() -> None:
    offenders = {
        str(path.relative_to(_REPO_ROOT)): hits
        for path in _scanned_files()
        if (hits := _find_violations(path.read_text(encoding="utf-8")))
    }
    assert not offenders, (
        "a password reaches an rpk command line (readable by every user through `ps`) in: "
        f"{offenders}. Set RPK_USER / RPK_PASS / RPK_SASL_MECHANISM in the environment of the "
        "call instead of passing -X pass= or --password."
    )
