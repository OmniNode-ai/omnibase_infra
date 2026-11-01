# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21604 AC1 -- ``onex delegate`` waits on the budget the authority answers.

A site overlay at ``<root>/delegation_budgets/overlay.yaml`` raises a class's
ceiling to 600 s. The authority omnimarket registers answers ``execution_budget``
through that overlay, so the CLI must size its terminal wait from the answer
and must not clamp to the contract's 240 s or refuse a request the overlay
admits. omnibase_infra's test environment has no omnimarket (layering), so the
stand-in authority reads the overlay file the way the real resolver does; the
assertion is about what the CLI does with the budget it is given. The real
resolver is exercised on a lab runtime (see contracts/OMN-21604.yaml).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import run_delegate
from omnibase_infra.cli.protocol_execution_budget import ProtocolExecutionBudget
from tests.helpers.cli_registry_stand_in import (
    StandInExecutionBudget,
    StandInTaskClassAuthority,
    use_stand_in_registry,
    wiring_authority,
)

pytestmark = pytest.mark.unit

_TASK_CLASS = "document"
_OVERLAY_CEILING_SECONDS = 600
_MARGIN_SECONDS = 60
_REQUESTED_SECONDS = 300

_CORRELATED_NOOP_CONTRACT = (
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


def _authority_through_overlay(overlay_root: Path | None) -> StandInTaskClassAuthority:
    """The contract authority with ``<root>/delegation_budgets/overlay.yaml`` applied."""
    base = wiring_authority(terminal_delivery_margin_seconds=_MARGIN_SECONDS)
    if overlay_root is None:
        return base
    raw = yaml.safe_load(
        (overlay_root / "delegation_budgets" / "overlay.yaml").read_text("utf-8")
    )
    budgets = dict(base.budgets)
    for name, fields in raw["execution_budgets"].items():
        budgets[name] = StandInExecutionBudget(
            **{**budgets[name].model_dump(), **fields}
        )
    return base.model_copy(update={"budgets": budgets})


def _write_overlay(root: Path) -> Path:
    overlay = root / "delegation_budgets" / "overlay.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        yaml.safe_dump(
            {
                "execution_budgets": {
                    _TASK_CLASS: {
                        "task_class_timeout_ceiling_seconds": _OVERLAY_CEILING_SECONDS
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def _run_and_capture_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlay_root: Path | None,
) -> tuple[int, int, int | None]:
    """Run the CLI once; return (exit code, terminal wait, requested seconds)."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    use_stand_in_registry(monkeypatch, _authority_through_overlay(overlay_root))

    seen: list[tuple[int | None, int]] = []
    real = cli_delegate._terminal_wait_seconds

    def _spy(
        *,
        requested_timeout_seconds: int | None,
        execution_budget: ProtocolExecutionBudget,
    ) -> int:
        wait = real(
            requested_timeout_seconds=requested_timeout_seconds,
            execution_budget=execution_budget,
        )
        seen.append((requested_timeout_seconds, wait))
        return wait

    monkeypatch.setattr(cli_delegate, "_terminal_wait_seconds", _spy)
    exit_code = run_delegate(
        prompt="summarize the loop",
        task_type=_TASK_CLASS,
        max_tokens=None,
        state_root=tmp_path / "state",
        timeout=_REQUESTED_SECONDS,
        verbose=False,
        emit_socket=tmp_path / "no-daemon.sock",
    )
    assert len(seen) == 1, "the CLI sized its terminal wait exactly once"
    requested, wait = seen[0]
    return exit_code, wait, requested


def test_overlay_ceiling_of_600_lets_a_300_second_request_wait_300_plus_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_overlay(tmp_path / "overlay-root")
    exit_code, wait, requested = _run_and_capture_wait(tmp_path, monkeypatch, root)

    assert requested is not None
    # The probe reserve (`_timeout_left_after_probe`) trims whole seconds off the
    # request, so the request the wait is sized from sits just under 300.
    assert _REQUESTED_SECONDS - 5 <= requested <= _REQUESTED_SECONDS
    assert requested > 240, "not clamped to the contract's 240 s ceiling"
    assert wait == requested + _MARGIN_SECONDS
    # The CLI dispatched the 300 s request and did not refuse it for the ceiling.
    # (The run's exit code is not asserted: the fixture handler declares a 240 s
    # ceiling of its own and rejects the request after the CLI has dispatched.)
    receipts = list((tmp_path / "state" / "runs").glob("*/receipt.json"))
    assert len(receipts) == 1
    assert "timeout_exceeds_task_class_ceiling" not in receipts[0].read_text("utf-8")
    assert exit_code in (0, 1)


def test_without_the_overlay_the_same_request_is_clamped_to_the_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: the overlay is what moves the wait, not the harness."""
    _exit_code, wait, requested = _run_and_capture_wait(tmp_path, monkeypatch, None)

    assert requested is not None
    assert requested > 240
    assert wait == 240 + _MARGIN_SECONDS
