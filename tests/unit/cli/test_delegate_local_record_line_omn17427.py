# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The in-memory-bus warning says where the record went (OMN-17427).

THE DEFECT. A fresh install's ``onex delegate`` warned that its evidence went to
"the local SQLite fallback, NOT the shared delegation_events projection", yet the
local table it writes is itself named ``delegation_events``. A new user could not
tell where the record went. The line now names the local database file and table
and says what that means: local to this machine, not the shared lab projection,
and a table that merely shares the shared one's name.

Machine-readable output is untouched: the line is a stderr log record, and the
transport line, the receipt and the stdout result are not asserted to change here
because nothing they carry was edited.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    describe_local_evidence_destination,
    run_delegate,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

_EVIDENCE_MODULE = "omnimarket.projection.sqlite_database"


def _fake_evidence_module(db_path: Path) -> SimpleNamespace:
    return SimpleNamespace(default_evidence_db_path=lambda: db_path)


def _resolve_evidence_module_to(
    monkeypatch: pytest.MonkeyPatch, module: SimpleNamespace | None
) -> None:
    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None) -> object:
        if name == _EVIDENCE_MODULE:
            if module is None:
                raise ImportError(name)
            return module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module)


class TestDescribeLocalEvidenceDestination:
    def test_names_the_database_file_and_the_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / ".omninode" / "delegation" / "delegation.sqlite"
        _resolve_evidence_module_to(monkeypatch, _fake_evidence_module(db_path))
        assert describe_local_evidence_destination() == (
            f"database file {db_path}, table delegation_events"
        )

    def test_without_omnimarket_names_only_the_directory_never_a_guessed_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _resolve_evidence_module_to(monkeypatch, None)
        assert describe_local_evidence_destination() == (
            "a local database file under ~/.omninode/delegation/, "
            "table delegation_events"
        )


class TestInmemoryBusWarning:
    """The warning ``run_delegate`` logs when the tier-0 default answers."""

    def _warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        db_path: Path,
    ) -> str:
        _resolve_evidence_module_to(monkeypatch, _fake_evidence_module(db_path))
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", lambda **_kw: 0)
        with caplog.at_level(logging.WARNING, logger=cli_delegate.logger.name):
            assert (
                run_delegate(
                    prompt="document the router",
                    task_type="document",
                    max_tokens=None,
                    locus=EnumDelegateLocus.IN_PROCESS,
                    state_root=tmp_path / "state",
                    timeout=60,
                    verbose=False,
                    emit_socket=tmp_path / "no-daemon.sock",
                    omni_home=None,
                )
                == 0
            )
        lines = [
            record.getMessage()
            for record in caplog.records
            if "using inmemory event bus" in record.getMessage()
        ]
        assert len(lines) == 1
        return lines[0]

    def test_the_line_states_where_the_record_went_and_what_that_means(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        db_path = tmp_path / "delegation.sqlite"
        line = self._warning(tmp_path, monkeypatch, caplog, db_path)
        # Where: the file and the table.
        assert str(db_path) in line
        assert "table delegation_events" in line
        # What it means: this machine only, and not the shared projection.
        assert "only on this machine" in line
        assert "nothing is sent to the shared lab projection" in line
        # The shared-name collision is named, not hidden, and the local table is
        # not claimed to be the projection.
        assert "shares its name with the shared delegation_events projection" in line
        assert "but is not it" in line

    def test_the_superseded_contradictory_wording_is_gone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        line = self._warning(
            tmp_path, monkeypatch, caplog, tmp_path / "delegation.sqlite"
        )
        assert "local SQLite fallback" not in line
        assert "NOT the shared delegation_events projection" not in line
