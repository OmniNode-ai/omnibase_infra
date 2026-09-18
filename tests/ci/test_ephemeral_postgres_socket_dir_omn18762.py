# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18762 — an ephemeral test cluster must not use the packaged socket dir.

WHY THIS EXISTS
---------------
Two fixtures started their throwaway PostgreSQL 16 cluster with no
``unix_socket_directories`` override, so the server used its compile-time
default. On the pgdg Ubuntu build the fleet runner carries that default is
``/var/run/postgresql``, owned by the ``postgres`` system user and unwritable
by the runner user, so the postmaster died at startup with::

    FATAL:  could not create lock file "/var/run/postgresql/.s.PGSQL.<port>.lock": Permission denied

Both fixtures treated that as ``pytest.skip``. 91 migration proofs across 8
modules therefore reported SKIPPED on every full-matrix CI run and the split
reported green — a false green that nothing surfaced, because the OMN-14172
silent-skip guard reads only the curated ``integration-guard`` junit and never
the ``test-parallel`` splits.

Two things are pinned here, and they are different claims:

``test_every_ephemeral_cluster_pins_its_socket_directory``
    the STRUCTURAL ratchet — no start invocation anywhere under ``tests/`` may
    omit the override again. It parses each invocation rather than grepping the
    file, so a fixture that merely mentions the flag in a comment does not pass.

``test_a_cluster_that_cannot_start_fails_rather_than_skips``
    the BEHAVIOURAL proof for the two fixtures this ticket repaired — it drives
    each fixture with a start that is guaranteed to fail and asserts the run
    goes red. A structural check alone would not catch a future edit that
    restores the silent path while keeping the flag.
"""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.integration.migrations import (
    test_application_migration_ledger_omn15413 as ledger_module,
)
from tests.integration.migrations.cutover import (
    test_cutover_receipts_postgres16 as cutover_module,
)

pytestmark = [pytest.mark.unit]

TESTS_ROOT = Path(__file__).resolve().parents[1]

# The two spellings PostgreSQL accepts for "put the socket somewhere I chose":
# ``pg_ctl -o "-k <dir>"`` and the long-form GUC.
SOCKET_DIR_TOKENS = ("-k ", "unix_socket_directories")


def _option_strings_of_start_invocations(
    source: str,
) -> Iterator[tuple[int, str | None]]:
    """Yield ``(lineno, -o option source)`` for each pg_ctl *start* argv list.

    A list literal counts as a pg_ctl start when its first element's source
    names ``pg_ctl`` (covering ``str(bin_dir / "pg_ctl")``, a ``_PG_CTL``
    module constant and a bare ``"pg_ctl"`` on PATH) and the list carries the
    ``start`` subcommand. The ``-o`` value is yielded as its own SOURCE text so
    an f-string is judged on what it interpolates, and ``None`` is yielded when
    the invocation passes no ``-o`` at all — an omission is the same defect as
    a wrong value.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        head = ast.get_source_segment(source, node.elts[0]) or ""
        if "pg_ctl" not in head:
            continue
        literals = [
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        if "start" not in literals:
            continue
        option_source: str | None = None
        for index, element in enumerate(node.elts[:-1]):
            if (
                isinstance(element, ast.Constant)
                and element.value == "-o"
                and index + 1 < len(node.elts)
            ):
                option_source = ast.get_source_segment(source, node.elts[index + 1])
        yield node.lineno, option_source


def test_every_ephemeral_cluster_pins_its_socket_directory() -> None:
    """No pg_ctl start under tests/ may rely on the packaged socket directory.

    RED proof for this ratchet: drop ``-k {root}`` from the option string in
    ``tests/integration/migrations/cutover/test_cutover_receipts_postgres16.py``
    and this test names that file and line.
    """
    offenders: list[str] = []
    invocations = 0
    for path in sorted(TESTS_ROOT.rglob("test_*.py")) + sorted(
        TESTS_ROOT.rglob("conftest.py")
    ):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - unreadable file
            continue
        if "pg_ctl" not in source:
            continue
        for lineno, option_source in _option_strings_of_start_invocations(source):
            invocations += 1
            if option_source is None:
                offenders.append(
                    f"{path.relative_to(TESTS_ROOT.parent)}:{lineno} passes no -o "
                    "server options at all, so the cluster uses the packaged "
                    "unix_socket_directories default"
                )
                continue
            if not any(token in option_source for token in SOCKET_DIR_TOKENS):
                offenders.append(
                    f"{path.relative_to(TESTS_ROOT.parent)}:{lineno} starts a "
                    f"cluster with {option_source!r}, which pins no socket "
                    "directory"
                )

    # Positive control: a zero-offender verdict is only meaningful if the parser
    # actually found the invocations. An empty sweep reads exactly like a clean
    # one, which is the class of false green this whole module exists to remove.
    assert invocations >= 8, (
        f"parsed only {invocations} pg_ctl start invocations under {TESTS_ROOT} — "
        "the parser matched almost nothing, so its zero-offender verdict proves "
        "nothing. Fix the parser, do not trust the result."
    )
    assert not offenders, (
        "ephemeral PostgreSQL clusters must pin their socket directory "
        "(pg_ctl -o '-k <writable dir>' or -c unix_socket_directories=<dir>); "
        "the packaged default is unwritable on the CI runner and the cluster "
        "dies before it binds TCP:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    ("module", "fixture_name"),
    [
        (cutover_module, "postgres_dsn"),
        (ledger_module, "pg16"),
    ],
    ids=["cutover_receipts", "migration_ledger"],
)
def test_a_cluster_that_cannot_start_fails_rather_than_skips(
    module: Any,
    fixture_name: str,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster that will not start is RED, never a silent skip.

    ``initdb`` is allowed to succeed and ``pg_ctl start`` is forced to fail,
    which is exactly the shape the fleet runner produced. The fixture must
    raise pytest's Failed, not its Skipped.
    """
    if module._postgres_bin_dir() is None:
        pytest.skip("PostgreSQL 16 binaries are unavailable on this host")

    real_run = subprocess.run

    def fake_run(
        argv: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if argv and str(argv[0]).endswith("pg_ctl") and "start" in argv:
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="pg_ctl: could not start server\n"
            )
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    fixture = getattr(module, fixture_name)
    function = getattr(fixture, "__wrapped__", fixture)
    generator = function(tmp_path_factory)

    # BaseException, not Exception: pytest's Failed and Skipped both derive
    # from BaseException, and catching only Exception would let BOTH escape —
    # including the skip this test exists to refuse.
    with pytest.raises(BaseException) as raised:
        next(generator)

    outcome = type(raised.value).__name__
    assert outcome == "Failed", (
        f"{module.__name__}.{fixture_name} raised {outcome} when the ephemeral "
        "cluster could not start. A skip here is a false green: the module's "
        "proofs report SKIPPED and the CI split reports success."
    )
