# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for ``onex delegate --session-model`` (OMN-20557).

The unit module drives the payload builder and a stubbed receipt call. This one
goes through ``click`` and a real (fixture-backed) dispatched run, so it
observes the session model in the request that actually gets dispatched, and
its absence when the flag is not given.

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
from omnibase_infra.cli.cli_delegate import delegate_command
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
)

pytestmark = pytest.mark.integration

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


class TestTheSessionModelReachesTheDispatchedRequest:
    def test_the_flag_is_listed_in_the_commands_own_help(self) -> None:
        result = _invoke("--help")
        assert result.exit_code == 0
        assert "--session-model" in result.output

    def test_a_malformed_session_model_is_refused_before_anything_runs(
        self,
    ) -> None:
        result = _invoke("summarise this", "--session-model", "claude opus")
        assert result.exit_code == 2
        assert "--session-model 'claude opus' is not a model id" in result.output

    def test_the_session_model_is_written_into_the_dispatched_metadata(
        self, tmp_path: Path
    ) -> None:
        result, state_root = _dispatch(tmp_path, "--session-model", "claude-opus-5-5")

        assert result.exit_code == 0, result.stderr
        assert "session model: claude-opus-5-5" in result.stderr
        metadata = _sole_payload(state_root)["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["session_model"] == "claude-opus-5-5"

    def test_no_flag_writes_no_session_model_key(self, tmp_path: Path) -> None:
        result, state_root = _dispatch(tmp_path)

        assert result.exit_code == 0, result.stderr
        assert "session model: none" in result.stderr
        metadata = _sole_payload(state_root).get("metadata", {})
        assert isinstance(metadata, dict)
        assert "session_model" not in metadata
