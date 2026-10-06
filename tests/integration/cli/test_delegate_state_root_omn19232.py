# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` writes its receipt under the resolved state root (OMN-19232).

The unit module drives the resolver and writer with the runtime stubbed. This
one goes through the real command, the real ``run_receipt_mode`` and the real
``RuntimeLocal`` against the correlated no-op contract, from a working
directory that is neither the environment root nor the home default. Before
the fix the command ignored ``ONEX_STATE_DIR`` and wrote ``runs/<run_id>/``
under that working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_caller import CALLER_LANE_ENV_VARS

pytestmark = pytest.mark.integration

_CONTRACT = (
    "---\n"
    "name: correlated_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.handler_correlated_noop\n"
    "  class: HandlerCorrelatedNoop\n"
    "  input_model: tests.fixtures.handler_correlated_noop"
    ".ModelCorrelatedNoopRequest\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.handler_correlated_noop"
    ":HandlerCorrelatedNoop\n"
)

_CALLER_ENV_CLEARED: dict[str, str | None] = dict.fromkeys(
    (*CALLER_LANE_ENV_VARS, "CLAUDE_CODE_SESSION_ID", "ONEX_LANE_REGISTRY_ROOT")
)


@pytest.fixture(autouse=True)
def _stand_in_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.delenv("ONEX_STATE_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract_path = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract_path.parent.mkdir()
    contract_path.write_text(_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()


def _invoke(tmp_path: Path, *extra: str) -> tuple[Result, Path]:
    cwd = tmp_path / "unrelated_cwd"
    cwd.mkdir()
    runner = CliRunner(env=_CALLER_ENV_CLEARED)
    with runner.isolated_filesystem(temp_dir=cwd) as work:
        result = runner.invoke(
            delegate_command,
            [
                "Reply with exactly the word READY",
                "--json",
                "--task-type",
                "summarization",
                "--bus",
                "inmemory",
                "--locus",
                "in-process",
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
                *extra,
            ],
            catch_exceptions=False,
        )
    return result, Path(work)


def _only_run(root: Path) -> Path:
    runs = list((root / "runs").iterdir())
    assert len(runs) == 1, f"expected exactly one run under {root}, got {runs}"
    return runs[0]


def test_env_root_receives_the_receipt_and_the_cwd_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = tmp_path / "env_root"
    monkeypatch.setenv("ONEX_STATE_DIR", str(env_root))

    result, work = _invoke(tmp_path)

    assert result.exit_code == 0, result.stderr
    run_dir = _only_run(env_root)
    assert (run_dir / "receipt.json").is_file()
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["state_root"] == str(env_root.resolve())
    assert f"state_root={env_root.resolve()}" in result.stderr
    assert not any(work.rglob("receipt.json"))
    assert not (work / ".onex_state").exists()


def test_unset_env_lands_under_the_home_default(tmp_path: Path) -> None:
    result, work = _invoke(tmp_path)

    assert result.exit_code == 0, result.stderr
    home_root = (tmp_path / "home" / ".onex_state").resolve()
    assert (_only_run(home_root) / "receipt.json").is_file()
    assert not any(work.rglob("receipt.json"))


def test_explicit_flag_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = tmp_path / "env_root"
    flag_root = tmp_path / "flag_root"
    monkeypatch.setenv("ONEX_STATE_DIR", str(env_root))

    result, _work = _invoke(tmp_path, "--state-root", str(flag_root))

    assert result.exit_code == 0, result.stderr
    assert (_only_run(flag_root) / "receipt.json").is_file()
    assert not (env_root / "runs").exists()
