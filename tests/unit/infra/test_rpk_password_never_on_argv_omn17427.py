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

OMN-19513: the scan walks every tracked directory of the repository (the ``Makefile``,
``.github`` workflows, ``contracts`` whose DoD ``check_value`` commands run on hosts, and the
``tests`` harnesses that drive a broker), not only ``docker``, ``scripts`` and ``src``. A batch
of 249 ``sh -c 'rpk -X user=$DEV_KAFKA_SASL_USERNAME -X pass=<value> ... "$@"'`` consumers sat on
the omni-201 process table for 59 hours with the password in plain view; no committed text
contains that launcher, so the scan must reach every place the shape can be written down.
The one exclusion is ``tests/unit/**/test_*.py``: a unit test asserts about command text and
cannot start a broker, so the strings in it are data.

Not covered: ``rpk security user create <name> -p <password>``. That flag sets the password
of the principal being created and has no environment form; the three one-shots that use it
live for seconds.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCANNED_SUFFIXES = frozenset({".sh", ".py", ".yml", ".yaml", ".mk"})
_SCANNED_NAMES = frozenset({"Makefile"})
_PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".onex_state",
        ".tox",
        "build",
        "dist",
    }
)

# `-X pass=<value>`: the value is a literal or a shell expansion. A bare `-X pass=` (the flag
# named in prose, nothing after the `=`) is not a leak.
_X_PASS_FLAG = re.compile(r"""-X[ \t]+["']?pass=["']?[^\s"'`]""")
# The argv-list form used from Python: ["-X", "pass=..."] / ["-X", f"pass={...}"].
_X_PASS_ARGV_ITEM = re.compile(r"""["']-X["']\s*,\s*f?["']pass=""")
_LONG_PASSWORD_FLAG = re.compile(r"--password(?:=|[ \t]+)[^\s-]")
# `docker exec -e SOME_PASS=...`: host-side argv even though rpk never sees it as a flag.
# libpq's `PGPASSWORD` is a Postgres credential, not the broker's SASL password this guard
# is about, so a name starting `PG` is not matched.
_DOCKER_EXEC_ENV_PASSWORD = re.compile(
    r"docker\s+exec\b[^\n]*[ \t]-e[ \t]+(?!PG)\w*PASS\w*="
)

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


def _is_unit_test_module(relative: Path) -> bool:
    """A unit test asserts about command text and cannot start a broker."""
    return (
        relative.parts[:2] == ("tests", "unit")
        and relative.name.startswith("test_")
        and relative.suffix == ".py"
    )


def _scanned_files(root: Path = _REPO_ROOT) -> list[Path]:
    found: list[Path] = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in _PRUNED_DIRS)
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix not in _SCANNED_SUFFIXES and name not in _SCANNED_NAMES:
                continue
            if _is_unit_test_module(path.relative_to(root)):
                continue
            found.append(path)
    return found


def _offenders(root: Path = _REPO_ROOT) -> dict[str, list[str]]:
    return {
        str(path.relative_to(root)): hits
        for path in _scanned_files(root)
        if (hits := _find_violations(path.read_text(encoding="utf-8")))
    }


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
        "docker exec -e KAFKA_SASL_PASSWORD=hunter2 redpanda rpk topic list",
        # OMN-19513: the observed omni-201 launcher. The user stays a container-side
        # variable, the password is expanded by the host shell before the container sees it.
        "docker exec omnibase-infra-redpanda sh -c 'rpk -X user=$DEV_KAFKA_SASL_USERNAME "
        '-X pass=hunter2 -X sasl.mechanism=SCRAM-SHA-256 "$@"\' rpk topic consume '
        "onex.cmd.example.v1 -o @-24h -n 1 -f %o",
        'docker exec "$C" sh -c "rpk -X user=\\$DEV_KAFKA_SASL_USERNAME -X pass=$PW \\"\\$@\\"" rpk topic consume t',
        # The same flag as a Makefile recipe line, and with the value quoted.
        "\trpk -X user=$(U) -X pass=$(P) topic list",
        "rpk -X pass='hunter2' topic list",
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
        # OMN-19513: the credential on stdin, read inside the one-shot's shell
        # (the lab-rebuild-triage probe fix), with user and mechanism as non-secret -e values.
        "docker exec -i -e RPK_USER=svc -e RPK_SASL_MECHANISM=SCRAM-SHA-256 "
        '"$C" sh -c \'IFS= read -r RPK_PASS; export RPK_PASS; exec rpk "$@"\' rpk '
        "topic consume onex.cmd.example.v1 -o @-24h -n 1 -f %o",
        # A password-less `docker exec -e` that inherits the value from the caller.
        'RPK_PASS="$P" docker exec -e RPK_PASS "$C" rpk topic list',
        # A Postgres credential is not the broker credential this guard covers.
        'docker exec -e PGPASSWORD=test-password "$C" psql -U postgres',
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


def test_scan_reaches_the_places_the_launcher_shape_can_hide() -> None:
    """OMN-19513: a Makefile, a workflow, a DoD contract and a test harness all run commands."""
    relative = {str(path.relative_to(_REPO_ROOT)) for path in _scanned_files()}
    assert "Makefile" in relative
    assert ".github/workflows/reusable-runtime-boot.yml" in relative
    assert "contracts/OMN-19513.yaml" in relative
    assert "tests/integration/customer_path/redpanda_sasl_harness.py" in relative
    # The one exclusion: unit-test modules, whose strings are assertions about text.
    assert (
        "tests/unit/infra/test_rpk_password_never_on_argv_omn17427.py" not in relative
    )


_BAD_LAUNCHER = (
    "docker exec c sh -c 'rpk -X user=$U -X pass=hunter2 \"$@\"' rpk topic consume t\n"
)


@pytest.mark.parametrize(
    "relative_path",
    [
        "Makefile",
        ".github/workflows/probe.yml",
        "contracts/OMN-1.yaml",
        "tests/integration/probe_harness.py",
        "tests/lab/probe.sh",
        "deploy/probe.sh",
    ],
)
def test_tree_scan_flags_the_launcher_in_every_root_it_can_hide_in(
    tmp_path: Path, relative_path: str
) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_BAD_LAUNCHER, encoding="utf-8")
    assert list(_offenders(tmp_path)) == [relative_path]


def test_tree_scan_passes_the_stdin_shape_and_skips_pruned_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "Makefile").write_text(
        "probe:\n\tdocker exec -i -e RPK_USER=u c sh -c 'IFS= read -r RPK_PASS; "
        'export RPK_PASS; exec rpk "$@"\' rpk topic list\n',
        encoding="utf-8",
    )
    pruned = tmp_path / ".venv" / "lib"
    pruned.mkdir(parents=True)
    (pruned / "vendored.py").write_text(_BAD_LAUNCHER, encoding="utf-8")
    unit = tmp_path / "tests" / "unit" / "x"
    unit.mkdir(parents=True)
    (unit / "test_asserts_text.py").write_text(_BAD_LAUNCHER, encoding="utf-8")
    assert _offenders(tmp_path) == {}


def test_no_committed_source_puts_a_password_on_an_rpk_command_line() -> None:
    offenders = _offenders()
    assert not offenders, (
        "a password reaches an rpk command line (readable by every user through `ps`) in: "
        f"{offenders}. Set RPK_USER / RPK_PASS / RPK_SASL_MECHANISM in the environment of the "
        "call instead of passing -X pass= or --password."
    )
