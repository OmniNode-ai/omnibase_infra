# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ledger test-write guard: a test process never writes the canonical ledger (OMN-19513).

Operator ruling 2026-10-01: "it should be impossible to like fake test like that". A lane on a
lab host ran a suite that inherited its shell's ledger and bus variables, and 17 fixture rows
reached the ledger of record. Test isolation is a convention a test can forget; this guard sits in
the write path itself (``scripts/ledger_lock.py`` calls it before the lock and before any write).

THE RULE. A write is refused when BOTH hold:

1. The process runs under a test runner: ``PYTEST_CURRENT_TEST`` is set, ``pytest`` or
   ``unittest`` is imported, or ``ONEX_TEST_CONTEXT`` is set to any non-empty value. The
   environment signals are inherited, so a ``ledger_lock.py`` subprocess a test starts is in test
   context too. No value of ``ONEX_TEST_CONTEXT`` removes a signal.
2. The target is canonical:

   - a ledger FILE outside the temporary directory, or inside a ``$OMNI_HOME`` that is not itself
     under the temporary directory (a test cannot tell the ledger of record, a lane's snapshot
     and an archive split apart, so all are canonical);
   - a canonical work-ledger TOPIC (``onex.{evt,cmd}.omnimarket.work-ledger-*``);
   - a projection DSN whose host is not loopback or a local socket.

A test that needs a ledger uses a ``tmp_path`` file, a non-canonical topic or a loopback DSN.
There is no bypass flag and no allowlist. The refusal is :class:`LedgerTestWriteRefusedError`; the
command exit is :data:`EXIT_TEST_WRITE_REFUSED`. Stdlib only, so the script loads it by path. The module reads no environment: the caller passes
its environment mapping, so this repo's "no new environment reads" gate has nothing to flag.
"""

from __future__ import annotations

import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

GUARD_NAME = "ledger-test-write-guard"
TEST_CONTEXT_ENV = "ONEX_TEST_CONTEXT"
EXIT_TEST_WRITE_REFUSED = 79
CANONICAL_TOPIC = re.compile(
    r"^onex\.(?:evt|cmd)\.omnimarket\.work-ledger-[a-z0-9-]+\.v\d+$"
)
_LOOPBACK_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1"})
_RUNNER_MODULES = ("pytest", "unittest")


class LedgerTestWriteRefusedError(Exception):
    """A test process tried to write a canonical ledger target. The message names the guard."""

    def __init__(self, signal: str, target_kind: str, target: str) -> None:
        self.signal = signal
        self.target_kind = target_kind
        self.target = target
        super().__init__(
            f"{GUARD_NAME} REFUSED -- a test process ({signal}) tried to write the canonical "
            f"{target_kind} {target}. Nothing was written. A test writes a tmp_path ledger, a "
            "non-canonical topic or a loopback DSN "
            "(omnibase_infra.handlers.handler_ledger_write_guard, OMN-19513)."
        )


def test_context(environ: Mapping[str, str]) -> str | None:
    """The first test-runner signal present, named, or None outside a test."""
    if environ.get("PYTEST_CURRENT_TEST"):
        return "PYTEST_CURRENT_TEST is set"
    if environ.get(TEST_CONTEXT_ENV, "").strip():
        return f"{TEST_CONTEXT_ENV} is set"
    for name in _RUNNER_MODULES:
        if name in sys.modules:
            return f"{name} is imported"
    return None


def _temp_root() -> Path:
    return Path(tempfile.gettempdir()).resolve()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def file_is_canonical(path: Path, environ: Mapping[str, str]) -> bool:
    """A ledger file is a test's own only under the temporary directory, and never inside a
    registry root (``OMNI_HOME``) that is not itself a test's scratch directory."""
    resolved = path.expanduser().resolve()
    temp_root = _temp_root()
    omni_home = environ.get("OMNI_HOME", "").strip()
    if omni_home:
        home = Path(omni_home).expanduser().resolve()
        if not _inside(home, temp_root) and _inside(resolved, home):
            return True
    return not _inside(resolved, temp_root)


def topic_is_canonical(topic: str) -> bool:
    return CANONICAL_TOPIC.match(topic.strip()) is not None


def dsn_is_canonical(dsn: str) -> bool:
    """A DSN reaches a shared database unless its host is loopback or a local socket."""
    text = dsn.strip()
    if "://" in text:
        host = urlsplit(text).hostname or ""
    else:
        match = re.search(r"(?:^|\s)host=(\S+)", text)
        host = match.group(1) if match else ""
    return not (host.lower() in _LOOPBACK_HOSTS or host.startswith("/"))


def _refuse_if_test(environ: Mapping[str, str], target_kind: str, target: str) -> None:
    signal = test_context(environ)
    if signal is not None:
        raise LedgerTestWriteRefusedError(signal, target_kind, target)


def check_file(path: Path, environ: Mapping[str, str]) -> None:
    """Refuse a test's write to a canonical ledger file."""
    if file_is_canonical(path, environ):
        _refuse_if_test(environ, "ledger file", str(path))


def check_topic(topic: str, environ: Mapping[str, str]) -> None:
    """Refuse a test's publish of a canonical work-ledger topic."""
    if topic_is_canonical(topic):
        _refuse_if_test(environ, "ledger topic", topic)


def check_dsn(dsn: str, environ: Mapping[str, str]) -> None:
    """Refuse a test's write to a projection DSN that is not loopback."""
    if dsn_is_canonical(dsn):
        _refuse_if_test(
            environ, "projection DSN", urlsplit(dsn).hostname or "<non-loopback host>"
        )


__all__ = [
    "CANONICAL_TOPIC",
    "EXIT_TEST_WRITE_REFUSED",
    "GUARD_NAME",
    "TEST_CONTEXT_ENV",
    "LedgerTestWriteRefusedError",
    "check_dsn",
    "check_file",
    "check_topic",
    "dsn_is_canonical",
    "file_is_canonical",
    "test_context",
    "topic_is_canonical",
]
