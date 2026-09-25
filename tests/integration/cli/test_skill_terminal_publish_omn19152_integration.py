# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex skill`` publishes a dispatch's returned result end to end (OMN-19152).

The unit module (``tests/unit/cli/test_skill_terminal_publish_omn19152.py``)
proves ``publish_skill_terminal_event`` in isolation, against receipts it
constructs by hand, and proves the CLI's exit-code invariant with
``run_receipt_mode`` stubbed out entirely. Neither drives a receipt the real
dispatch pipeline produced through the real callback wiring added in
``run_skill_by_name`` (``receipts.append`` -> ``publish_skill_terminal_event``
-> ``resolve_skill_terminal_target``).

This is an end-to-end test of the real command: the real click entry point,
the real declarative skill mapping (a stand-in entry, since the shipped
``dod_verify`` entry dispatches an ``omnimarket`` node this repo does not
depend on by layering), the real ``run_receipt_mode``, the real in-memory
dispatch of a local fixture handler, the real ``resolve_skill_terminal_target``
reading a real (but unreachable) lane declaration, and the real
``publish_skill_terminal_event`` attempting a real Kafka connect against a
closed local port. Only the environment (workspace root, packaged-contract
resolution, the omnimarket drift pre-flight) is redirected -- the same
substitution ``tests/integration/cli/test_delegate_pre_publish_failure_omn19131.py``
and ``test_delegate_timeout_typed_refusal.py`` make for the identical reason,
each citing ``feedback_real_dispatch_path_tests``: a test that stubbed
``run_receipt_mode`` would pass whether or not the callback/publish wiring
this PR adds was ever connected.

AC-2 (fail-soft) is what these tests exercise: the terminal-publish attempt
never has a broker to succeed against in a hermetic run, so the outcome under
test is the FAILED report -- and the assertion that matters is that it never
changes the dispatch's own receipt or exit code.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from pydantic import JsonValue

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_skill
from omnibase_infra.cli.cli_skill import run_skill_by_name
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.enum_skill_arg_type import EnumSkillArgType
from omnibase_infra.cli.model_skill_arg_spec import ModelSkillArgSpec
from omnibase_infra.cli.model_skill_mapping import ModelSkillMapping
from omnibase_infra.cli.model_skill_mapping_registry import ModelSkillMappingRegistry
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
)

pytestmark = pytest.mark.integration

# tests.fixtures.handler_correlated_noop is imported by dotted path (not the
# module's own __name__) below, because the runtime resolves the handler and
# its result model by the import path named in the contract, and that path
# must match the one this test imports the assertions' types through.
_FIXTURE_MODULE = "tests.fixtures.handler_correlated_noop"
_REQUEST_MODEL = f"{_FIXTURE_MODULE}.ModelCorrelatedNoopRequest"
_HANDLER_CLASS = f"{_FIXTURE_MODULE}:HandlerCorrelatedNoop"
_RESULT_MODEL = f"{_FIXTURE_MODULE}.ModelDelegateSkillFixtureTerminal"
_TERMINAL_TOPIC = "onex.evt.proof.terminal-publish-omn19152.v1"

_STAND_IN_CONTRACT = (
    "---\n"
    "name: correlated_noop_terminal_publish_proof\n"
    "node_type: compute\n"
    f"terminal_event: {_TERMINAL_TOPIC}\n"
    f"input_model: {_REQUEST_MODEL}\n"
    "handler:\n"
    f"  module: {_FIXTURE_MODULE}\n"
    "  class: HandlerCorrelatedNoop\n"
    f"  input_model: {_REQUEST_MODEL}\n"
    "handler_routing:\n"
    f"  default_handler: {_HANDLER_CLASS}\n"
)

# Referenced only so ruff/pyflakes see the fixture symbols as used: the real
# consumers are the dotted-path strings above, resolved dynamically by the
# runtime out of the stand-in contract, not by this import.
_ = (HandlerCorrelatedNoop, ModelCorrelatedNoopRequest)


def _closed_port() -> int:
    """A local TCP port nothing is listening on, bound then released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _mapping(*, lane: str | None) -> ModelSkillMapping:
    return ModelSkillMapping(
        skill_name="terminal_publish_proof",
        node_name="correlated_noop_terminal_publish_proof",
        result_model=_RESULT_MODEL,
        event_bus="inmemory",
        publish_terminal_to_lane=lane,
        args=(
            ModelSkillArgSpec(
                name="text",
                payload_field="prompt",
                arg_type=EnumSkillArgType.STRING,
                positional=True,
                required=True,
            ),
        ),
    )


def _workspace_with_declared_lane(root: Path, *, broker: str) -> Path:
    """A real ``<root>/omnimarket/config/ci_bus_lanes.yaml`` declaring ``dev``.

    PLAINTEXT so resolution never needs a stored or ambient SASL identity --
    the subject under test is the publish attempt itself, not credential
    resolution (that is ``test_omn18432_lane_identity_cli.py``'s subject).
    """
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(
        f'lanes:\n  dev:\n    broker: "{broker}"\n    security_protocol: PLAINTEXT\n',
        encoding="utf-8",
    )
    return root


@pytest.fixture
def stand_in_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Resolve the real command onto the stand-in mapping and contract.

    Returns ``(state_root, workspace_root)``.
    """
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.setattr(cli_skill, "check_omnimarket_drift", lambda **_: None)

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_STAND_IN_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_skill, "_resolve_packaged_contract", lambda _node_name: contract_path
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return tmp_path / "state", workspace


def _invoke(
    *,
    mapping: ModelSkillMapping,
    state_root: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Result:
    monkeypatch.setattr(
        cli_skill,
        "load_skill_registry",
        lambda: ModelSkillMappingRegistry(skills=(mapping,)),
    )
    return CliRunner().invoke(
        run_skill_by_name,
        [
            mapping.skill_name,
            "integration proof of OMN-19152",
            "--state-root",
            str(state_root),
            "--omnibase-path",
            str(workspace),
            "--emit-socket",
            str(state_root / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )


class TestARealDispatchIsPublishedFailSoft:
    """The real callback -> publish wiring, against a real unreachable broker."""

    def test_the_dispatch_succeeds_and_reports_a_failed_publish(
        self,
        stand_in_registry: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_root, workspace = stand_in_registry
        _workspace_with_declared_lane(workspace, broker=f"127.0.0.1:{_closed_port()}")
        mapping = _mapping(lane="dev")

        started = time.monotonic()
        result = _invoke(
            mapping=mapping,
            state_root=state_root,
            workspace=workspace,
            monkeypatch=monkeypatch,
        )
        elapsed = time.monotonic() - started

        # AC-2: the dispatch itself is unaffected by the publish attempt.
        assert result.exit_code == 0, result.stderr
        receipt = ModelSkillResult[JsonValue].model_validate(
            json.loads(result.stdout.strip())
        )
        assert receipt.status is EnumSkillResultStatus.SUCCESS
        assert receipt.result_model == _RESULT_MODEL

        # The real publish attempt against a closed port fails soft and
        # reports itself on stderr, bounded well under the CLI's own
        # dispatch timeout rather than hanging the command.
        assert "onex skill: terminal publish failed for lane dev" in result.stderr
        assert "verdict, receipt and exit code unchanged" in result.stderr
        assert elapsed < 30.0, (
            f"a fail-soft publish must not hold the command open ({elapsed:.1f}s)"
        )

    def test_a_mapping_with_no_declared_lane_prints_no_publish_line(
        self,
        stand_in_registry: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Negative control: no ``publish_terminal_to_lane`` -> nothing to fail-soft.

        Without this, the assertion above would pass identically for a
        command that attempts no publish at all, and the FAILED-report
        assertions would not be evidence of anything this PR added.
        """
        state_root, workspace = stand_in_registry
        _workspace_with_declared_lane(workspace, broker=f"127.0.0.1:{_closed_port()}")
        mapping = _mapping(lane=None)

        result = _invoke(
            mapping=mapping,
            state_root=state_root,
            workspace=workspace,
            monkeypatch=monkeypatch,
        )

        assert result.exit_code == 0, result.stderr
        assert "onex skill: terminal publish" not in result.stderr

    def test_an_undeclared_lane_is_also_reported_fail_soft(
        self,
        stand_in_registry: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No lane declaration file at all: ``resolve_skill_terminal_target``
        itself raises, which is the other real (non-network) failure mode the
        fail-soft boundary must also absorb.
        """
        state_root, workspace = stand_in_registry
        # Deliberately no ci_bus_lanes.yaml under workspace.
        mapping = _mapping(lane="dev")

        result = _invoke(
            mapping=mapping,
            state_root=state_root,
            workspace=workspace,
            monkeypatch=monkeypatch,
        )

        assert result.exit_code == 0, result.stderr
        assert "onex skill: terminal publish failed for lane dev" in result.stderr
