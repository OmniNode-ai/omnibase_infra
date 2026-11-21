# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerLedgerOpenClaims reads open CLAIMs through the reader node only (OMN-20547)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_ledger_open_claims as handler_module
from omnibase_infra.handlers.handler_ledger_open_claims import (
    HandlerLedgerOpenClaims,
    LedgerReadRefusedError,
    matching,
    owner_of,
)
from omnibase_infra.models.model_ledger_open_claim import ModelLedgerOpenClaim
from omnibase_infra.models.model_ledger_open_claims_request import (
    ModelLedgerOpenClaimsRequest,
)
from omnibase_infra.nodes.node_ledger_open_claims_effect.runtime_ledger_open_claims import (
    main,
)

pytestmark = pytest.mark.unit

REQUEST = ModelLedgerOpenClaimsRequest(
    reader_command="/venv/bin/python -m reader_module",
    reader="git-maintenance",
    window_hours=72,
)
ROW_A = "2026-11-20T10:00:00Z | CLAIM | lane=lane-a | ticket=OMN-1 | repo_x docs/a.md on jonah/omn-1-x"
ROW_B = "2026-11-21T10:00:00Z | CLAIM | lane=lane-b | ticket=OMN-2 | repo_x docs/a.md"
OPEN_A = "OPEN-CLAIM lane=lane-a claim=2026-11-20T10:00:00Z db:1 ticket=OMN-1 host=h1"
OPEN_B = "OPEN-CLAIM lane=lane-b claim=2026-11-21T10:00:00Z db:2 ticket=OMN-2 host=h1"


class Reader:
    """A subprocess.run stand-in answering the reader node: stdout by query, or one exit for every call."""

    def __init__(self, rows: list[str], opened: list[str], code: int = 0) -> None:
        self.rows, self.opened, self.code = rows, opened, code
        self.argvs: list[list[str]] = []

    def __call__(
        self, argv: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.argvs.append(argv)
        query = argv[argv.index("--query") + 1]
        out = self.rows if query == "rows" else self.opened
        return subprocess.CompletedProcess(
            argv,
            self.code,
            stdout="".join(f"{x}\n" for x in out),
            stderr="READER reader=git-maintenance status=refused\n",
        )


def _reads_through(monkeypatch: pytest.MonkeyPatch, runner: object) -> None:
    monkeypatch.setattr(handler_module.subprocess, "run", runner)


def test_each_open_claim_is_joined_to_its_row_and_closed_claims_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = (
        "2026-11-19T10:00:00Z | CLAIM | lane=done | ticket=OMN-9 | repo_x docs/a.md"
    )
    reader = Reader(
        [closed, ROW_B, ROW_A], ["open-claims since=x open=2", OPEN_B, OPEN_A]
    )
    _reads_through(monkeypatch, reader)
    result = HandlerLedgerOpenClaims().handle(REQUEST)
    assert [c.lane for c in result.claims] == ["lane-a", "lane-b"]
    assert result.claims[0].row == ROW_A
    queries = [a[a.index("--query") + 1] for a in reader.argvs]
    assert queries == ["open-claims", "rows"]
    rows_call = reader.argvs[1]
    assert rows_call[:4] == ["/venv/bin/python", "-m", "reader_module", "--reader"]
    assert rows_call[rows_call.index("--reader") + 1] == "git-maintenance"
    assert rows_call[rows_call.index("--window-hours") + 1] == "72"
    assert rows_call[rows_call.index("--kind") + 1] == "CLAIM"


def test_a_claim_off_the_rows_window_keeps_its_open_claim_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reads_through(monkeypatch, Reader([ROW_B], [OPEN_A, OPEN_B]))
    result = HandlerLedgerOpenClaims().handle(REQUEST)
    by_lane = {c.lane: c.row for c in result.claims}
    assert by_lane["lane-a"] == OPEN_A
    assert "ticket=OMN-1" in by_lane["lane-a"]


@pytest.mark.parametrize("code", [1, 3, 69])
def test_a_reader_that_does_not_answer_is_a_refusal_never_an_empty_answer(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    _reads_through(monkeypatch, Reader([ROW_A], [OPEN_A], code=code))
    with pytest.raises(LedgerReadRefusedError, match=f"ledger reader exit {code}"):
        HandlerLedgerOpenClaims().handle(REQUEST)


def test_a_reader_that_cannot_be_run_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    _reads_through(monkeypatch, broken)
    with pytest.raises(LedgerReadRefusedError, match="not run"):
        HandlerLedgerOpenClaims().handle(REQUEST)


def test_matching_takes_a_key_as_a_whole_token_only() -> None:
    claims = [ModelLedgerOpenClaim(stamp="s", lane="l", row=ROW_A)]
    assert matching(claims, ["OMN-1"]) == claims
    assert matching(claims, ["jonah/omn-1-x"]) == claims
    assert matching(claims, ["OMN-12"]) == []
    assert matching(claims, ["omn-1-x"]) == claims
    assert matching(claims, ["omn-1"]) == []
    assert matching(claims, ["docs/a"]) == []


def test_the_owner_is_the_newest_claim_naming_the_repo_and_a_path() -> None:
    claims = [
        ModelLedgerOpenClaim(stamp="2026-11-20T10:00:00Z", lane="lane-a", row=ROW_A),
        ModelLedgerOpenClaim(stamp="2026-11-21T10:00:00Z", lane="lane-b", row=ROW_B),
    ]
    assert owner_of(claims, "repo_x", ["docs/a.md"]) == "lane-b"
    assert owner_of(claims, "repo_x", ["README.md"]) is None
    assert owner_of(claims, "repo_y", ["docs/a.md"]) is None


READ_ARGS = [
    "--reader-command",
    "/venv/bin/python -m reader_module",
    "--reader",
    "git-maintenance",
    "--window-hours",
    "72",
]


def test_snapshot_then_match_prints_stamp_and_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "snap.jsonl"
    _reads_through(monkeypatch, Reader([ROW_A, ROW_B], [OPEN_A, OPEN_B]))
    assert main(["snapshot", *READ_ARGS, "--out", str(out)]) == 0
    assert main(["match", str(out), "--", "OMN-2"]) == 0
    assert capsys.readouterr().out == "2026-11-21T10:00:00Z lane=lane-b\n"


def test_owner_prints_the_lane_or_unowned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _reads_through(monkeypatch, Reader([ROW_A, ROW_B], [OPEN_A, OPEN_B]))
    assert main(["owner", *READ_ARGS, "--repo", "repo_x", "--path", "docs/a.md"]) == 0
    assert main(["owner", *READ_ARGS, "--repo", "repo_x", "--path", "other"]) == 0
    assert capsys.readouterr().out == "lane-b\nunowned\n"


def test_snapshot_refuses_with_exit_3_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "snap.jsonl"
    _reads_through(monkeypatch, Reader([], [], code=3))
    assert main(["snapshot", *READ_ARGS, "--out", str(out)]) == 3
    assert "ledger reader exit 3" in capsys.readouterr().err
    assert not out.exists()
