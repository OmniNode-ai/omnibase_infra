# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegate CLI reports env configuration provenance across a real run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration

_FIXTURE_MODULE = "tests.fixtures.handler_correlated_noop"
_NOOP_CONTRACT = (
    "---\n"
    "name: correlated_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
    f"input_model: {_FIXTURE_MODULE}.ModelCorrelatedNoopRequest\n"
    "handler:\n"
    f"  module: {_FIXTURE_MODULE}\n"
    "  class: HandlerCorrelatedNoop\n"
    f"  input_model: {_FIXTURE_MODULE}.ModelCorrelatedNoopRequest\n"
    "handler_routing:\n"
    f"  default_handler: {_FIXTURE_MODULE}:HandlerCorrelatedNoop\n"
)
_CONFIG_KEYS = (
    "BIFROST_CONTRACT_PATH",
    "BIFROST_OVERLAY_PATH",
    "DELEGATION_ROUTING_TIERS_PATH",
)


@pytest.fixture(autouse=True)
def _offline_delegate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the shipped fixture handler and an isolated in-memory runtime."""
    for key in (*_CONFIG_KEYS, "OMNI_HOME", "ONEX_CONTRACTS_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text(_NOOP_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    plain_cwd = tmp_path / "cwd"
    plain_cwd.mkdir()
    monkeypatch.chdir(plain_cwd)


def _run(tmp_path: Path) -> tuple[Result, dict[str, object]]:
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
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
    receipts = list(state_root.glob("runs/*/receipt.json"))
    assert len(receipts) == 1
    parsed: object = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return result, cast("dict[str, object]", parsed)


def test_all_env_overrides_reach_stderr_and_persisted_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {key: str(tmp_path / f"{key.lower()}.yaml") for key in _CONFIG_KEYS}
    for key, path in paths.items():
        monkeypatch.setenv(key, path)

    result, receipt = _run(tmp_path)

    for key, path in paths.items():
        assert (
            f"config override: {key} resolved from environment "
            f"(source=contract_overlay_env) path={path}"
        ) in result.stderr
    assert receipt["config_overrides"] == [
        {"config_key": key, "source": "contract_overlay_env", "resolved_path": path}
        for key, path in paths.items()
    ]


def test_no_env_overrides_leave_no_provenance(
    tmp_path: Path,
) -> None:
    result, receipt = _run(tmp_path)

    assert "config override:" not in result.stderr
    assert "config_overrides" not in receipt
