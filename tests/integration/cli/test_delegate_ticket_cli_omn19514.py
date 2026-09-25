# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for ``onex delegate --ticket`` (OMN-19514).

The unit module beside this one drives ``resolve_delegate_ticket`` and
``_write_payload`` directly. This one goes through ``click`` -- the real
command, the real option parsing, and a real (fixture-backed) dispatched run
-- because the feature this ticket adds is a request-metadata field written
by ``run_delegate``, and a payload-writer test cannot observe whether the
flag threads from the command line into the request that actually gets
dispatched.

``check-integration-tests`` (``integration-test-check.yml``, hard gate since
2026-04-13) flagged the PR that introduced ``--ticket`` for shipping no test
under ``tests/integration/`` or ``tests/e2e/`` -- every existing coverage for
this change was at the unit level. This file closes that gap.

The dispatched-run tests use the same stand-in task-class contract and
in-memory, in-process technique as
``tests/integration/cli/test_delegate_pre_publish_failure_omn19131.py``, and
the same ``HandlerCorrelatedNoop`` fixture handler, so a completed run needs
no live broker and no co-installed ``omnimarket``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    DELEGATE_TICKET_METADATA_KEY,
    delegate_command,
)
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
)

pytestmark = pytest.mark.integration

_STAND_IN_TASK_CLASS_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18305"
    / "task_class_contracts_vocabulary.yaml"
)

_MODEL_IMPORT_PATH = "tests.fixtures.handler_correlated_noop.ModelCorrelatedNoopRequest"
_HANDLER_IMPORT_PATH = "tests.fixtures.handler_correlated_noop.HandlerCorrelatedNoop"

# Sanity: the stand-in classes really do live at the paths the fixture
# contract below names by string (yaml has no import to fail at collection
# time if this ever drifts).
assert ModelCorrelatedNoopRequest.__module__ + ".ModelCorrelatedNoopRequest" == (
    _MODEL_IMPORT_PATH
)
assert HandlerCorrelatedNoop.__module__ + ".HandlerCorrelatedNoop" == (
    _HANDLER_IMPORT_PATH
)

_NOOP_CONTRACT = (
    "---\n"
    "name: correlated_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
    f"input_model: {_MODEL_IMPORT_PATH}\n"
    "handler:\n"
    f"  module: {_HANDLER_IMPORT_PATH.rsplit('.', 1)[0]}\n"
    "  class: HandlerCorrelatedNoop\n"
    f"  input_model: {_MODEL_IMPORT_PATH}\n"
    "handler_routing:\n"
    f"  default_handler: {_HANDLER_IMPORT_PATH.rsplit('.', 1)[0]}:HandlerCorrelatedNoop\n"
)


@pytest.fixture(autouse=True)
def stand_in_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route ``run_delegate`` at a fixture contract, offline and co-install-free."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_delegate,
        "resolve_task_class_contract_path",
        lambda: _STAND_IN_TASK_CLASS_CONTRACT,
    )
    contract_path = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract_path.parent.mkdir()
    contract_path.write_text(_NOOP_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    # This suite itself runs from an ``omni_worktrees/<TICKET>/`` checkout
    # (Operating Rule 9), so the real process cwd would satisfy the
    # worktree-path resolver on every test, including the ones that assert
    # "no ticket named". Ground every test in a plain scratch directory and
    # let the two worktree-cwd tests below `chdir` explicitly from there.
    plain_cwd = tmp_path / "cwd"
    plain_cwd.mkdir()
    monkeypatch.chdir(plain_cwd)
    return contract_path


def _invoke(*extra: str, cwd: Path | None = None) -> Result:
    return CliRunner().invoke(delegate_command, list(extra), catch_exceptions=False)


def _dispatch(tmp_path: Path, *extra: str) -> tuple[Result, Path]:
    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly the word READY",
            "--task-type",
            "summarization",
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
            *extra,
        ],
        catch_exceptions=False,
    )
    return result, state_root


def _sole_payload(state_root: Path) -> dict[str, object]:
    payloads = sorted((state_root / "tmp").glob("delegate-input-*.json"))
    assert len(payloads) == 1, f"expected exactly one payload, got {payloads}"
    loaded = json.loads(payloads[0].read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


class TestTheFlagIsReachableFromACommandLine:
    """Parser-level coverage: the flag must exist and refuse before anything runs."""

    def test_the_ticket_flag_parses_rather_than_being_an_unknown_option(self) -> None:
        result = _invoke(
            "summarise this", "--task-type", "summarization", "--ticket", "OMN-19514"
        )
        assert "No such option" not in result.output

    def test_the_flag_is_listed_in_the_commands_own_help(self) -> None:
        result = _invoke("--help")
        assert result.exit_code == 0
        assert "--ticket" in result.output

    def test_a_malformed_ticket_is_refused_before_anything_runs(
        self, tmp_path: Path
    ) -> None:
        """The refusal must fire before the omnimarket drift guard.

        No ``--state-root`` is passed a valid contract or bus here; if this
        ever ran past parsing it would fail for an unrelated reason, and the
        assertion on the message would be the only thing telling the two
        failures apart.
        """
        result = _invoke("summarise this", "--ticket", "omn-1")

        assert result.exit_code == 2
        assert "--ticket 'omn-1' is not a ticket identifier" in result.output


class TestAnExplicitTicketReachesTheDispatchedRequest:
    """The feature itself: the flag threads into the request that gets sent."""

    def test_the_ticket_is_written_into_the_dispatched_requests_metadata(
        self, tmp_path: Path
    ) -> None:
        result, state_root = _dispatch(tmp_path, "--ticket", "OMN-19514")

        assert result.exit_code == 0, result.stderr
        assert "ticket: OMN-19514 (explicit)" in result.stderr
        payload = _sole_payload(state_root)
        assert payload["metadata"] == {DELEGATE_TICKET_METADATA_KEY: "OMN-19514"}

    def test_no_ticket_omits_the_metadata_key_entirely(self, tmp_path: Path) -> None:
        """A caller who never named a ticket gets the pre-OMN-19514 payload shape."""
        result, state_root = _dispatch(tmp_path)

        assert result.exit_code == 0, result.stderr
        assert "ticket: none (none)" in result.stderr
        payload = _sole_payload(state_root)
        assert "metadata" not in payload

    def test_an_omitted_flag_is_resolved_from_a_ticket_worktree_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operating Rule 9's ``omni_worktrees/<TICKET>/`` shape names the ticket."""
        worktree = tmp_path / "omni_worktrees" / "OMN-19514" / "omnibase_infra"
        worktree.mkdir(parents=True)
        monkeypatch.chdir(worktree)

        result, state_root = _dispatch(tmp_path)

        assert result.exit_code == 0, result.stderr
        assert "ticket: OMN-19514 (worktree path)" in result.stderr
        payload = _sole_payload(state_root)
        assert payload["metadata"] == {DELEGATE_TICKET_METADATA_KEY: "OMN-19514"}

    def test_an_explicit_ticket_wins_over_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worktree = tmp_path / "omni_worktrees" / "OMN-1" / "omnibase_infra"
        worktree.mkdir(parents=True)
        monkeypatch.chdir(worktree)

        result, state_root = _dispatch(tmp_path, "--ticket", "OMN-19514")

        assert result.exit_code == 0, result.stderr
        assert "ticket: OMN-19514 (explicit)" in result.stderr
        payload = _sole_payload(state_root)
        assert payload["metadata"] == {DELEGATE_TICKET_METADATA_KEY: "OMN-19514"}
