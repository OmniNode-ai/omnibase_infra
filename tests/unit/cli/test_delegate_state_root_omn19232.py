# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` anchors its state root; it never follows the working directory.

OMN-19232. ``--state-root`` defaulted to the cwd-relative string
``.onex_state`` and ignored ``ONEX_STATE_DIR``, so a lane that ran the
command from its worktree wrote ``runs/<run_id>/receipt.json`` under that
worktree. Two lab runs cited in a TERMINAL were unreadable for that reason
(ledger row 2026-09-28T22:21:00Z, "stray root").

Every test drives the real ``delegate_command`` entrypoint from a tmp cwd with
``run_receipt_mode`` replaced by a stand-in that hands the CLI's own receipt
callback a recorded-shape receipt, so the receipt is written by the production
writer under whatever root the CLI resolved.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.receipt_mode import resolve_state_root
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.unit

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)


def _receipt() -> ModelSkillResult[dict[str, object]]:
    correlation_id = uuid.uuid4()
    return ModelSkillResult[dict[str, object]](
        skill_name="delegate",
        node_name="node_delegate_skill_orchestrator",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=correlation_id,
        run_id=uuid.uuid4(),
        exit_code=0,
        duration_ms=12,
        result={
            "correlation_id": str(correlation_id),
            "task_type": "research",
            "model_name": "qwen3.8",
            "provider": "http://local.invalid/v1/chat/completions",
            "response": "answer",
            "attempts": [
                {
                    "tier": "local",
                    "backend_id": "local-coder",
                    "model_id": "qwen3.8",
                    "quality_gate_passed": True,
                    "acceptance_decision": "accept",
                }
            ],
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


@pytest.fixture
def written_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[ModelSkillResult[dict[str, object]]]:
    """Replace the runtime with a stand-in that feeds the CLI's receipt callback."""
    contract_path = tmp_path / "contract.yaml"
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _n: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    sent: list[ModelSkillResult[dict[str, object]]] = []

    def _fake_run_receipt_mode(
        *, receipt_callback: Callable[..., object], **_kwargs: object
    ) -> int:
        receipt = _receipt()
        sent.append(receipt)
        receipt_callback(receipt)
        return 0

    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
    return sent


def _invoke(cwd: Path, *extra: str) -> Result:
    cwd.mkdir(parents=True, exist_ok=True)
    return CliRunner().invoke(
        delegate_command,
        [
            "research the routing architecture",
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--emit-socket",
            str(cwd / "no-daemon.sock"),
            *extra,
        ],
        catch_exceptions=False,
    )


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("ONEX_STATE_DIR", raising=False)
    return fake_home


def test_env_root_wins_over_cwd_when_no_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    written_receipts: list[ModelSkillResult[dict[str, object]]],
) -> None:
    """AC1: ONEX_STATE_DIR set, no --state-root, unrelated cwd."""
    env_root = tmp_path / "env_root"
    cwd = tmp_path / "unrelated_cwd"
    cwd.mkdir()
    monkeypatch.setenv("ONEX_STATE_DIR", str(env_root))
    monkeypatch.chdir(cwd)

    result = _invoke(cwd)

    assert result.exit_code == 0, result.output
    run_id = str(written_receipts[0].run_id)
    assert (env_root / "runs" / run_id / "receipt.json").is_file()
    assert not (cwd / ".onex_state").exists()
    assert not any(cwd.rglob("receipt.json"))


def test_unset_env_defaults_to_home_anchor_from_any_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    written_receipts: list[ModelSkillResult[dict[str, object]]],
) -> None:
    """AC2: no env, no flag: two different cwds land under the same home root."""
    cwd_a = tmp_path / "cwd_a"
    cwd_b = tmp_path / "cwd_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    monkeypatch.chdir(cwd_a)
    assert _invoke(cwd_a).exit_code == 0
    monkeypatch.chdir(cwd_b)
    assert _invoke(cwd_b).exit_code == 0

    home_root = home / ".onex_state"
    for receipt in written_receipts:
        assert (home_root / "runs" / str(receipt.run_id) / "receipt.json").is_file()
    assert len(written_receipts) == 2
    assert not any(cwd_a.rglob("receipt.json"))
    assert not any(cwd_b.rglob("receipt.json"))
    assert not (cwd_a / ".onex_state").exists()
    assert not (cwd_b / ".onex_state").exists()


def test_explicit_flag_beats_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    written_receipts: list[ModelSkillResult[dict[str, object]]],
) -> None:
    """AC3: --state-root wins over ONEX_STATE_DIR (positive control: env root stays empty)."""
    env_root = tmp_path / "env_root"
    flag_root = tmp_path / "flag_root"
    cwd = tmp_path / "unrelated_cwd"
    cwd.mkdir()
    monkeypatch.setenv("ONEX_STATE_DIR", str(env_root))
    monkeypatch.chdir(cwd)

    result = _invoke(cwd, "--state-root", str(flag_root))

    assert result.exit_code == 0, result.output
    run_id = str(written_receipts[0].run_id)
    assert (flag_root / "runs" / run_id / "receipt.json").is_file()
    assert not (env_root / "runs").exists()


def test_run_json_and_artifacts_line_carry_the_absolute_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    written_receipts: list[ModelSkillResult[dict[str, object]]],
) -> None:
    """AC4: a reader finds the receipt without knowing the caller's cwd."""
    env_root = tmp_path / "env_root"
    cwd = tmp_path / "unrelated_cwd"
    cwd.mkdir()
    monkeypatch.setenv("ONEX_STATE_DIR", str(env_root))
    monkeypatch.chdir(cwd)

    result = _invoke(cwd)

    assert result.exit_code == 0, result.output
    run_id = str(written_receipts[0].run_id)
    run_json = json.loads(
        (env_root / "runs" / run_id / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["state_root"] == str(env_root.resolve())
    assert Path(run_json["state_root"]).is_absolute()
    artifacts_line = next(
        line for line in result.stderr.splitlines() if "delegate artifacts" in line
    )
    assert str(env_root.resolve()) in artifacts_line
    assert "state_root=" in artifacts_line


def test_option_help_names_the_resolution_order() -> None:
    """AC5: flag, then ONEX_STATE_DIR, then the home default, in that order."""
    option = next(
        p
        for p in delegate_command.params
        if isinstance(p, click.Option) and p.name == "state_root"
    )
    text = " ".join((option.help or "").split())
    flag_at = text.find("--state-root")
    env_at = text.find("ONEX_STATE_DIR")
    home_at = text.find("~/.onex_state")
    assert -1 not in (env_at, home_at), text
    assert env_at < home_at
    assert "flag" in text.lower()
    assert option.default is None
    assert flag_at == -1 or flag_at < env_at


def test_relative_env_root_is_refused_not_anchored_to_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    written_receipts: list[ModelSkillResult[dict[str, object]]],
) -> None:
    """A relative ONEX_STATE_DIR is the cwd-dependence bug again: fail fast."""
    monkeypatch.setenv("ONEX_STATE_DIR", ".onex_state")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        delegate_command,
        ["hello", "--bus", "inmemory", "--locus", "in-process"],
    )

    assert result.exit_code != 0
    assert "ONEX_STATE_DIR" in result.output
    assert not written_receipts


def test_resolver_refuses_a_root_under_claude_and_an_unresolvable_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """Fail fast where no anchor can be resolved, never fall back to the cwd."""
    monkeypatch.setenv("ONEX_STATE_DIR", str(home / ".claude" / "state"))
    with pytest.raises(ProtocolConfigurationError, match=r"~/\.claude"):
        resolve_state_root(None)

    monkeypatch.delenv("ONEX_STATE_DIR")

    def _no_home() -> Path:
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "home", staticmethod(_no_home))
    with pytest.raises(ProtocolConfigurationError, match="home directory"):
        resolve_state_root(None)
    assert resolve_state_root(tmp_path / "flag") == (tmp_path / "flag").resolve()
