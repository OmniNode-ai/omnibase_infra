# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The open-claims read crosses the real process boundary to the ledger reader node (OMN-20547).

The runtime command of ``node_ledger_open_claims_effect`` runs as its own process and reaches the
reader through a stand-in interpreter that answers ``-m omnimarket.nodes.node_work_ledger_reader_effect``
from canned rows, exactly where ``ONEX_LEDGER_BUS_APPEND_COMMAND`` points a lab host at the real one. The
ledger file is never named: a reader that cannot answer is a refusal (exit 3), never a read of the file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.scripts.ledger_reader_stub import LedgerReaderStub

RUNTIME = (
    "omnibase_infra.nodes.node_ledger_open_claims_effect.runtime_ledger_open_claims"
)
CLAIM = (
    "2026-11-20T10:00:00Z",
    "lane-a",
    "OMN-1",
    "repo_x docs/a.md on jonah/omn-1-x",
    True,
)
CLOSED = ("2026-11-20T11:00:00Z", "lane-b", "OMN-2", "repo_x docs/closed.md", False)


def _run(stub: LedgerReaderStub, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", RUNTIME, *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**stub.env(), "PATH": "/usr/bin:/bin"},
    )


def _read_args(stub: LedgerReaderStub) -> list[str]:
    return [
        "--reader-command",
        stub.env()["ONEX_LEDGER_READER_COMMAND"],
        "--reader",
        "git-maintenance",
        "--window-hours",
        "72",
    ]


def test_snapshot_then_match_names_only_the_open_claim(tmp_path: Path) -> None:
    stub = LedgerReaderStub(tmp_path / "stub")
    stub.set_claims(CLAIM, CLOSED)
    snap = tmp_path / "snap.jsonl"
    done = _run(stub, "snapshot", *_read_args(stub), "--out", str(snap))
    assert done.returncode == 0, done.stderr
    partial = _run(stub, "match", str(snap), "omn-1", "docs/closed.md")
    assert (
        partial.stdout.splitlines() == []
    )  # a key is a whole token: omn-1 is not omn-1-x
    hit = _run(stub, "match", str(snap), "jonah/omn-1-x")
    assert hit.stdout.splitlines() == ["2026-11-20T10:00:00Z lane=lane-a"]
    assert not _run(
        stub, "match", str(snap), "docs/closed.md"
    ).stdout  # closed: never reported
    queries = [x.split("--query ")[1].split()[0] for x in stub.calls()]
    assert queries == [
        "open-claims",
        "rows",
    ]  # one open-claims read, one CLAIM rows read: the bus, not a file


def test_owner_names_the_lane_holding_the_path(tmp_path: Path) -> None:
    stub = LedgerReaderStub(tmp_path / "stub")
    stub.set_claims(CLAIM)
    done = _run(
        stub, "owner", *_read_args(stub), "--repo", "repo_x", "--path", "docs/a.md"
    )
    assert (done.returncode, done.stdout.strip()) == (0, "lane-a")
    done = _run(
        stub, "owner", *_read_args(stub), "--repo", "repo_x", "--path", "docs/none.md"
    )
    assert (done.returncode, done.stdout.strip()) == (0, "unowned")


def test_a_reader_that_cannot_answer_is_a_refusal_never_a_file_read(
    tmp_path: Path,
) -> None:
    stub = LedgerReaderStub(tmp_path / "stub")
    stub.set_claims(CLAIM)
    for code in (3, 69):
        stub.refuse_with(code)
        done = _run(
            stub, "snapshot", *_read_args(stub), "--out", str(tmp_path / "never.jsonl")
        )
        assert done.returncode == 3 and "REFUSED" in done.stderr, done.stderr
        assert not (tmp_path / "never.jsonl").exists()
