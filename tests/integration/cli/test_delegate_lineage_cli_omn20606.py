# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for ``onex delegate`` lineage flags (OMN-20606).

The unit module ``tests/unit/cli/test_cli_delegate_lineage.py`` drives
``ModelDelegateLineage`` and ``_write_payload`` directly. This one goes through
``click`` with a real (fixture-backed) in-process dispatched run, because the
feature is request metadata written by ``run_delegate``: only a dispatched run
shows the three flags reaching the request that is actually sent.

Same stand-in contract and in-memory technique as
``test_delegate_ticket_cli_omn19514.py``; no live broker, no omnimarket.
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

_PARENT = "5b0c8a52-9a43-4f3c-a9f1-0d6f2b4e7c11"

_MODEL_IMPORT_PATH = "tests.fixtures.handler_correlated_noop.ModelCorrelatedNoopRequest"
_HANDLER_IMPORT_PATH = "tests.fixtures.handler_correlated_noop.HandlerCorrelatedNoop"

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


def _sole_metadata(state_root: Path) -> dict[str, object]:
    payloads = sorted((state_root / "tmp").glob("delegate-input-*.json"))
    assert len(payloads) == 1, f"expected exactly one payload, got {payloads}"
    loaded = json.loads(payloads[0].read_text(encoding="utf-8"))
    metadata = loaded["metadata"]
    assert isinstance(metadata, dict)
    return metadata


class TestLineageReachesTheDispatchedRequest:
    def test_the_three_flags_are_written_into_the_requests_metadata(
        self, tmp_path: Path
    ) -> None:
        result, state_root = _dispatch(
            tmp_path,
            "--parent-correlation-id",
            _PARENT,
            "--lineage-kind",
            "fallback",
            "--parent-failure-cause",
            "provider_quota_exhausted",
        )

        assert result.exit_code == 0, result.stderr
        assert "lineage: " in result.stderr
        metadata = _sole_metadata(state_root)
        assert metadata["parent_correlation_id"] == _PARENT
        assert metadata["lineage_kind"] == "fallback"
        assert metadata["parent_failure_cause"] == "provider_quota_exhausted"

    def test_no_lineage_flags_write_no_lineage_keys(self, tmp_path: Path) -> None:
        result, state_root = _dispatch(tmp_path)

        assert result.exit_code == 0, result.stderr
        assert "lineage: " not in result.stderr
        metadata = _sole_metadata(state_root)
        assert not {
            "parent_correlation_id",
            "lineage_kind",
            "parent_failure_cause",
        } & set(metadata)

    def test_a_partial_lineage_is_refused_before_anything_dispatches(
        self, tmp_path: Path
    ) -> None:
        result, state_root = _dispatch(tmp_path, "--parent-correlation-id", _PARENT)

        assert result.exit_code == 2
        assert "--lineage-kind" in result.output
        assert not (state_root / "tmp").exists() or not list(
            (state_root / "tmp").glob("delegate-input-*.json")
        )
