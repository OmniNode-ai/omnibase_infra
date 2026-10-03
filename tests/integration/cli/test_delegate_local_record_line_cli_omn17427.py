# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` on the tier-0 in-memory bus says where its record went (OMN-17427).

The unit tests pin the wording. This pins the path a caller takes: the CLI entry
resolves the shipped tier-0 default bus, logs the stderr record line naming the
local database file and table, and still reaches receipt mode. Only receipt mode
and the packaged contract lookup are stubbed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import run_delegate
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.integration


def test_in_memory_run_logs_where_the_local_record_went(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reached: list[dict[str, object]] = []

    def _fake_receipt_mode(**kwargs: object) -> int:
        reached.append(kwargs)
        return 0

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _n: tmp_path / "c.yaml"
    )
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)

    with caplog.at_level(logging.WARNING, logger=cli_delegate.logger.name):
        code = run_delegate(
            prompt="probe",
            task_type="document",
            max_tokens=None,
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=5,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            omni_home=None,
        )

    assert code == 0
    assert len(reached) == 1
    lines = [
        r.getMessage()
        for r in caplog.records
        if "using inmemory event bus" in r.getMessage()
    ]
    assert len(lines) == 1
    assert "table delegation_events" in lines[0]
    assert "nothing is sent to the shared lab projection" in lines[0]
