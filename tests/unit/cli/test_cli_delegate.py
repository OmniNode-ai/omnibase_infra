# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``onex delegate`` single-command delegation (OMN-13096).

The acceptance probe is STRUCTURAL (no size assertions, plan Phase 2 item 1):

- ``classify_task_type`` maps prompt keywords to the delegate task taxonomy
  (first match wins, research fallback);
- ``run_delegate`` writes its scratch payload under ``<state-root>/tmp/`` with
  a run_id-suffixed name — never ``/tmp`` (``feedback_no_tmp_use_workspace``);
- the payload validates against the delegate node's input model
  (``ModelDelegateSkillRequest``) — prompt, task_type, source, and
  ``max_tokens`` ONLY when an explicit ``--max-tokens`` override is supplied
  (omitted otherwise so the node resolves it per-backend from the routing
  contract, OMN-13161);
- the command dispatches through receipt mode so stdout is exactly ONE
  ``ModelSkillResult`` JSON with zero RuntimeLocal log leakage.

The end-to-end probe against the live delegate node (which requires a vLLM
endpoint) lives in the OCC evidence run, not the unit suite. These unit tests
exercise the REAL CLI wiring against a committed proof contract by pointing
``_resolve_packaged_contract`` at it — the dispatch path, payload write, and
receipt envelope are all real.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
import time
import uuid
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.backends import auto_configure
from omnibase_infra.backends.auto_configure import (
    BUS_TYPE_OVERRIDE_ENV,
    EventBusResolutionAmbiguousError,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    BUS_CHOICES,
    DEFAULT_BUS,
    DEFAULT_TASK_TYPE,
    DELEGATE_SOURCE,
    DELEGATE_SOURCE_CHOICES,
    DelegateTimeoutExceededError,
    _write_local_run_files,
    build_backend_overrides,
    classify_task_type,
    delegate_command,
    resolve_default_bus,
    run_delegate,
)
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from omnibase_infra.cli.omnimarket_drift_guard import (
    DRIFT_OVERRIDE_ENV,
    check_omnimarket_drift,
)
from omnibase_infra.cli.task_class_selection import TaskClassContractError
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.enums.enum_task_type_resolution import EnumTaskTypeResolution
from omnibase_infra.runtime_identity import collect_runtime_identity
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_DELEGATION_REQUEST

pytestmark = pytest.mark.unit

KAFKA_BOOTSTRAP_ARG = "$KAFKA_BOOTSTRAP_SERVERS"
_RECEIPT_ENV_SENTINEL = "OMN15569_DELEGATION_RECEIPT_TEST_SENTINEL"


@pytest.fixture(autouse=True)
def _clear_kafka_bootstrap_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must not depend on ambient ``KAFKA_BOOTSTRAP_SERVERS`` (OMN-14376).

    When ``--bus`` is omitted, ``run_delegate`` now probes
    ``KAFKA_BOOTSTRAP_SERVERS`` to auto-resolve the default bus. Tests that
    don't care about bus selection (payload shape, task classification,
    single-receipt-on-stdout, etc.) must stay deterministic regardless of what
    the developer's shell / ``~/.omnibase/.env`` happens to export — clear the
    var by default here; ``TestBusSelection`` / ``TestResolveDefaultBus`` tests
    that DO want to exercise the configured-broker path set it explicitly.
    """
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)


@pytest.fixture(autouse=True)
def _clear_bus_type_override_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must not depend on ambient ``ONEX_EVENT_BUS_TYPE`` (OMN-17304).

    ``ONEX_EVENT_BUS_TYPE`` holds NO tier in the transport ladder any more —
    it is set-and-ignored (with a warning). Clearing it by default keeps the
    warning-path assertions deterministic; tests that exercise the
    set-and-ignored warning set it explicitly.
    """
    monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)


@pytest.fixture(autouse=True)
def _clear_contracts_dir_pointer_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must not depend on an ambient ``ONEX_CONTRACTS_DIR`` (OMN-17304).

    The bootstrap pointer names WHERE the per-runtime configured authority
    lives. An ambient value in the developer's shell would silently swap the
    authority every delegate-path test resolves against. Tests that exercise
    the configured-authority tier set it explicitly to a tmp contracts dir.
    """
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)


def _write_authority_config(
    tmp_path: Path, *, bus_type: str, profile: str | None = None
) -> Path:
    """Write a minimal per-runtime config and return its contracts dir.

    The returned path is what ``ONEX_CONTRACTS_DIR`` (the bootstrap pointer)
    should be set to — the file lands at the kernel-standard location
    ``<contracts_dir>/runtime/runtime_config.yaml``.
    """
    contracts_dir = tmp_path / "authority-contracts"
    (contracts_dir / "runtime").mkdir(parents=True, exist_ok=True)
    lines = ["event_bus:", f'  type: "{bus_type}"']
    if profile is not None:
        lines.append(f'  profile: "{profile}"')
    (contracts_dir / "runtime" / "runtime_config.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return contracts_dir


def _probe_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin that transport resolution never touches the network (OMN-17304).

    Under the ruled ladder — explicit flag > configured authority > shipped
    tier-0 default — the delegate path ALWAYS has a configured answer (the
    shipped default is itself config), so the broker probe is structurally
    unreachable. A probe call is a regression to transport-by-environmental-
    accident, so it fails the test rather than returning a stub state.
    """

    def _fail(**_kwargs: object) -> ModelProbeResult:
        raise AssertionError(
            "resolve_bus_type probed the network — the delegate path must "
            "resolve from the configured authority / shipped tier-0 default"
        )

    monkeypatch.setattr(auto_configure, "probe_kafka", _fail)


@pytest.fixture(autouse=True)
def _no_omnimarket_drift_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the omnimarket pre-flight drift guard for this file (OMN-13930).

    ``run_delegate`` dispatches ``node_delegate_skill_orchestrator`` -- an
    omnimarket-provided node -- so it now runs the same guard ``onex skill``
    and ``onex node`` run. Without this fixture every CLI-wiring test here
    would pass or fail on the ambient shell's ``$OMNI_HOME`` and whether this
    venv happens to have omnimarket co-installed. The guard's own behavior is
    proven in ``test_omnimarket_drift_guard.py``; the delegate call-site
    wiring is proven in ``test_drift_guard_fires_before_delegate_dispatch``,
    which restores the real guard within its own scope.
    """
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


# A proof contract that runs a deterministic in-process handler — no vLLM, no
# network. It stands in for the delegate node so the CLI wiring (payload write,
# receipt-mode dispatch, single typed result) is exercised end-to-end.
_PROOF_NOOP_CONTRACT = (
    "---\n"
    "name: proof_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.noop-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.handler_proof_noop\n"
    "  class: HandlerProofNoop\n"
    "  input_model: tests.fixtures.handler_proof_noop.ModelProofNoopRequest\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.handler_proof_noop:HandlerProofNoop\n"
)

# Delegate-SHAPED stand-in (OMN-17295). ``ModelProofNoopRequest`` declares no
# ``correlation_id``, so a run against it produces a terminal stamped with an
# id RuntimeLocal minted for itself and no caller can attribute — which the
# OMN-17295 correlation join correctly refuses. The real delegate request model
# DOES declare ``correlation_id`` (and is frozen, so RuntimeLocal's event-driven
# overwrite is refused and the CLI's minted id survives onto the wire). Tests
# whose subject is the SHAPE of the delegate receipt must therefore run against
# a contract that round-trips the correlation id, or they are asserting against
# a stand-in the real path does not resemble.
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


_OMN18305_PROBE_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18305"
    / "task_class_contracts_probe.yaml"
)


class TestClassifyTaskType:
    """OMN-18305: classification is contract-declared, not keyword density.

    The rules themselves (word boundaries, presence-not-frequency, shape
    gating, priority) are pinned in ``test_task_class_selection.py`` against a
    probe contract so they run without omnimarket. What is pinned HERE is the
    CLI-facing behaviour and the two reproductions from the ticket.
    """

    @staticmethod
    def _probe_classes() -> tuple[object, ...]:
        from omnibase_infra.cli.task_class_selection import (
            load_selectable_task_classes,
        )

        return load_selectable_task_classes(_OMN18305_PROBE_CONTRACT)

    def test_the_latest_window_is_not_a_test_task(self) -> None:
        """The four-word reproduction: 'latest' must not match 'test'."""
        assert (
            classify_task_type("the latest window", classes=self._probe_classes())
            != "short_keyword"
        )

    def test_a_genuine_test_request_still_resolves_to_the_test_class(self) -> None:
        assert (
            classify_task_type(
                "write a test for the handler", classes=self._probe_classes()
            )
            == "short_keyword"
        )

    def test_no_keyword_table_survives_in_cli_source(self) -> None:
        """AC2: the hardcoded table is gone, not merely bypassed."""
        source = Path(cli_delegate.__file__).read_text(encoding="utf-8")
        assert "_CLASSIFICATION_RULES" not in source


class TestTaskTypeVocabulary:
    """AC4: the CLI's selectable vocabulary equals the contract's public set."""

    #: The contract's ``gateway_exposure: public`` projection, as of
    #: OMN-18305. THE OTHER HALF OF THIS ASSERTION LIVES IN OMNIMARKET:
    #: ``tests/unit/inference/test_task_class_selection_omn18305.py`` pins the
    #: live contract to this same list. Neither repo can see the other (repo
    #: layering forbids importing omnimarket here, and omnimarket pins this
    #: package from the registry), and omnibase_infra's own test suite refuses
    #: to run at all with omnimarket installed -- the OMN-15620 venv-purity
    #: gate treats a co-installed omnimarket as duplicate node registration.
    #: So a live comparison is not available in either suite; two pinned
    #: halves of one list is. Changing the contract turns omnimarket's half
    #: red, which is the signal to update this one.
    EXPECTED_PUBLIC_CLASSES = (
        "code_generation",
        "code_review",
        "complex_reasoning",
        "document",
        "planning",
        "reasoning",
        "refactor",
        "research",
        "review",
        "summarization",
        "test",
    )

    def test_choices_mirror_matches_the_contracts_public_projection(self) -> None:
        assert sorted(cli_delegate.TASK_TYPE_CHOICES) == sorted(
            self.EXPECTED_PUBLIC_CLASSES
        )

    def test_stand_in_contract_carries_the_same_vocabulary(self) -> None:
        """The test stand-in cannot drift away from the mirror it stands in for."""
        from omnibase_infra.cli.task_class_selection import (
            load_selectable_task_classes,
        )
        from tests.unit.cli.conftest import STAND_IN_TASK_CLASS_CONTRACT

        stand_in = load_selectable_task_classes(STAND_IN_TASK_CLASS_CONTRACT)
        assert sorted(entry.name for entry in stand_in) == sorted(
            cli_delegate.TASK_TYPE_CHOICES
        )

    def test_summarization_and_planning_are_reachable(self) -> None:
        """The two classes an engineering standup belongs to were unreachable."""
        assert "summarization" in cli_delegate.TASK_TYPE_CHOICES
        assert "planning" in cli_delegate.TASK_TYPE_CHOICES

    def test_explicit_task_type_fails_closed_when_the_contract_is_unresolvable(
        self,
    ) -> None:
        """OMN-18342 AC(a)/regression guard.

        An explicit ``--task-type`` must never be decided by ``TASK_TYPE_CHOICES``
        alone. When the contract cannot be resolved (e.g. omnimarket absent), the
        run fails closed naming the resolution failure -- it does NOT silently
        fall back to validating against the mirror, even for a value the mirror
        would accept.
        """
        with patch.object(
            cli_delegate,
            "resolve_task_class_contract_path",
            side_effect=TaskClassContractError("omnimarket absent"),
        ):
            with pytest.raises(TaskClassContractError, match="omnimarket absent"):
                cli_delegate.resolve_task_class(
                    "summarise the ledger", explicit="summarization"
                )

    def test_explicit_task_type_present_in_contract_is_accepted(self) -> None:
        """OMN-18342 AC(b): positive control.

        An explicit value that IS present in the resolved contract is accepted,
        and the resolution is recorded as EXPLICIT provenance.
        """
        from tests.unit.cli.conftest import STAND_IN_TASK_CLASS_CONTRACT

        with patch.object(
            cli_delegate,
            "resolve_task_class_contract_path",
            return_value=STAND_IN_TASK_CLASS_CONTRACT,
        ):
            resolved = cli_delegate.resolve_task_class(
                "summarise the ledger", explicit="summarization"
            )

        assert resolved.task_type == "summarization"
        assert resolved.resolution is EnumTaskTypeResolution.EXPLICIT

    def test_explicit_task_type_absent_from_contract_is_refused_naming_the_contract(
        self, tmp_path: Path
    ) -> None:
        """OMN-18342 AC(a): RED test.

        A value present in the ``TASK_TYPE_CHOICES`` mirror but ABSENT from the
        resolved contract must be refused, and the error names the contract's
        own vocabulary -- never the mirror. This is the falsifier for the
        regression introduced by ``0181d708d``: that commit's early-return
        validated ``explicit`` against ``TASK_TYPE_CHOICES`` without ever
        resolving the contract, so a mirror-only class would have been wrongly
        accepted here.
        """
        assert "planning" in cli_delegate.TASK_TYPE_CHOICES  # present in the mirror

        truncated_contract = tmp_path / "truncated_task_class_contract.yaml"
        truncated_contract.write_text(
            "\n".join(
                [
                    "task_classes:",
                    "  research:",
                    "    gateway_exposure: public",
                    "    selection:",
                    "      priority: 1",
                    "      phrases: []",
                ]
            ),
            encoding="utf-8",
        )

        with patch.object(
            cli_delegate,
            "resolve_task_class_contract_path",
            return_value=truncated_contract,
        ):
            with pytest.raises(
                TaskClassContractError, match="the task-class contract exposes"
            ):
                cli_delegate.resolve_task_class(
                    "plan the migration", explicit="planning"
                )


class TestReceiptEnvironmentIsolation:
    """Receipt-mode delegation must not load a home dotenv file."""

    def test_01_receipt_mode_ignores_controlled_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / "omnibase.env"
        env_file.write_text(
            f"{_RECEIPT_ENV_SENTINEL}=loaded-by-receipt-mode\n",
            encoding="utf-8",
        )
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setenv("OMNIBASE_ENV_FILE", str(env_file))
        monkeypatch.delenv(_RECEIPT_ENV_SENTINEL, raising=False)
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

        run_delegate(
            prompt="implement an HTTP server",
            task_type=None,
            max_tokens=None,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        assert _RECEIPT_ENV_SENTINEL not in os.environ

    def test_02_receipt_environment_did_not_escape_previous_test(self) -> None:
        assert _RECEIPT_ENV_SENTINEL not in os.environ


class TestPayloadScratch:
    def test_payload_written_under_state_root_tmp_not_slash_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        run_delegate(
            prompt="implement an HTTP server",
            task_type=None,
            max_tokens=None,
            state_root=state_root,
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        scratch_dir = state_root / "tmp"
        assert scratch_dir.is_dir(), "scratch dir must be under <state-root>/tmp/"
        payloads = list(scratch_dir.glob("delegate-input-*.json"))
        assert len(payloads) == 1, "exactly one run_id-suffixed scratch payload"
        # No scratch leaked to the system temp dir.
        assert not list(Path(tempfile.gettempdir()).glob("delegate-input-*.json"))
        # With no explicit --max-tokens override, the key is omitted entirely so
        # the delegate node resolves it per-backend from its routing contract
        # (OMN-13161 — no CLI-side default).
        payload = json.loads(payloads[0].read_text(encoding="utf-8"))
        assert "max_tokens" not in payload

    def test_payload_validates_against_delegate_request_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        run_delegate(
            prompt="refactor the loop",
            # The subject here is the PAYLOAD SHAPE, so the class is stated
            # rather than classified -- this test should not also be a test of
            # whichever selection predicate happens to claim this prompt
            # (OMN-18305 moved that decision into the task-class contract).
            task_type="refactor",
            max_tokens=4096,
            state_root=state_root,
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        payload_path = next((state_root / "tmp").glob("delegate-input-*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        # With an EXPLICIT --max-tokens override the payload carries exactly the
        # fields the delegate node's input model (ModelDelegateSkillRequest)
        # requires from a consumer: prompt, task_type, source, correlation_id,
        # max_tokens. omnibase_infra does NOT depend on omnimarket (layering),
        # so the node owns model validation at dispatch; the CLI's contract
        # here is the payload shape. (When no override is supplied, max_tokens
        # is omitted — see test_payload_written_under_state_root_tmp_not_slash_tmp,
        # OMN-13161. ``correlation_id`` is always present — OMN-14397.)
        assert uuid.UUID(str(payload.pop("correlation_id")))
        assert payload == {
            "prompt": "refactor the loop",
            "task_type": "refactor",
            "source": DELEGATE_SOURCE,
            "max_tokens": 4096,
        }

    def test_explicit_task_type_overrides_classification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        # Prompt would classify as code_generation; explicit flag wins.
        run_delegate(
            prompt="write an HTTP server",
            task_type="research",
            max_tokens=None,
            state_root=state_root,
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )
        payload_path = next((state_root / "tmp").glob("delegate-input-*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["task_type"] == "research"


class TestSourceFlag:
    """OMN-15185: ``--source`` threads a registered adapter source into the
    delegation payload's ``source`` field, closed to
    :data:`DELEGATE_SOURCE_CHOICES` (mirroring the wire model's
    ``ModelDelegateSkillRequest.source`` Literal). Omitting the flag must
    preserve pre-OMN-15185 behavior exactly (``DELEGATE_SOURCE``,
    ``"claude-code"``).
    """

    def test_default_omitted_flag_uses_delegate_source_constant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        # No --source / source= override at all -- the regression case: a
        # pre-OMN-15185 caller must see byte-identical payload["source"].
        run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            state_root=state_root,
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )
        payload_path = next((state_root / "tmp").glob("delegate-input-*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["source"] == DELEGATE_SOURCE == "claude-code"

    @pytest.mark.parametrize("source_choice", DELEGATE_SOURCE_CHOICES)
    def test_each_choice_lands_in_payload(
        self,
        source_choice: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            source=source_choice,
            state_root=state_root,
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )
        payload_path = next((state_root / "tmp").glob("delegate-input-*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["source"] == source_choice

    @pytest.mark.parametrize("source_choice", DELEGATE_SOURCE_CHOICES)
    def test_cli_flag_each_choice_reaches_overrides(
        self,
        source_choice: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # End-to-end through the click CLI flag, not just the function call.
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            payload = json.loads(
                Path(str(kwargs["input_path"])).read_text(encoding="utf-8")
            )
            captured["source"] = payload["source"]
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "research the routing architecture",
                "--source",
                source_choice,
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert captured["source"] == source_choice

    def test_cli_flag_omitted_defaults_to_claude_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            payload = json.loads(
                Path(str(kwargs["input_path"])).read_text(encoding="utf-8")
            )
            captured["source"] = payload["source"]
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "research the routing architecture",
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert captured["source"] == "claude-code"

    def test_cli_invalid_source_rejected_by_parser(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "research the routing architecture",
                "--source",
                "not-a-real-source",
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "not-a-real-source" in result.output


class TestSourceFlagDriftGuard:
    """``DELEGATE_SOURCE_CHOICES`` duplicates omnimarket's wire model Literal
    (``ModelDelegateSkillRequest.source``) because omnibase_infra does not
    depend on omnimarket -- repo layering runs compat -> core -> spi -> infra,
    and separately omnimarket depends on omnibase-infra, never the reverse
    (importing omnimarket here would be circular/wrong-direction). This is
    exactly OMN-15175's duplicate-alias failure class: a hand-rolled Literal
    silently fell out of sync with this same wire model after it was widened.

    When omnimarket IS importable in the test env, assert the tuple matches
    the LIVE Literal args exactly. It normally is NOT importable in
    omnibase_infra's own test env (no omnimarket dependency); in that case,
    assert against the documented value list stated in the
    ``DELEGATE_SOURCE_CHOICES`` docstring/comment in ``cli_delegate.py``, so a
    silent edit that changes one without the other still fails this test.
    """

    # Mirrors the value list documented in cli_delegate.py's
    # DELEGATE_SOURCE_CHOICES comment -- update BOTH together.
    _DOCUMENTED_CHOICES = ("claude-code", "codex", "external-client")

    def test_choices_match_wire_model_or_documented_fallback(self) -> None:
        try:
            from omnimarket.models.delegation.wire.model_delegate_skill_request import (
                ModelDelegateSkillRequest,
            )
        except ImportError:
            assert set(DELEGATE_SOURCE_CHOICES) == set(self._DOCUMENTED_CHOICES), (
                "DELEGATE_SOURCE_CHOICES drifted from its own documented "
                "value list (OMN-15175 duplicate-alias failure class) -- "
                "omnimarket is not importable in this test env to check "
                "against the live wire model directly, so verify by hand "
                "against omnimarket's "
                "model_delegate_skill_request.py:ModelDelegateSkillRequest"
                ".source Literal."
            )
            return
        source_field = ModelDelegateSkillRequest.model_fields["source"]
        live_choices = get_args(source_field.annotation)
        assert set(DELEGATE_SOURCE_CHOICES) == set(live_choices), (
            f"DELEGATE_SOURCE_CHOICES {DELEGATE_SOURCE_CHOICES} drifted from "
            f"the live ModelDelegateSkillRequest.source Literal {live_choices}"
        )


class TestSingleReceiptOnStdout:
    def test_stdout_is_exactly_one_validated_skill_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "implement an HTTP server",
                "--state-root",
                str(state_root),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        stripped = result.stdout.strip()
        # Exactly one JSON object — any RuntimeLocal log line would break this.
        parsed = json.loads(stripped)
        assert isinstance(parsed, dict)
        assert "\n" not in stripped, "receipt must be a single JSON line"
        ModelSkillResult.model_validate(parsed)

    def test_no_runtime_info_logs_on_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "research the routing architecture",
                "--state-root",
                str(state_root),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "INFO" not in result.stdout
        assert "RuntimeLocal" not in result.stdout


class TestBusSelection:
    """The CLI targets the live bus BY DEFAULT (OMN-13532 / OMN-14376).

    ``run_delegate`` no longer hardcodes the in-memory bus, and no longer
    requires an explicit ``--bus kafka`` to reach the shared platform
    substrate: when ``--bus`` is omitted, :func:`resolve_default_bus` probes
    ``KAFKA_BOOTSTRAP_SERVERS`` and auto-selects ``kafka`` when it is
    configured and healthy — the SAME bus the rest of the system is
    configured with — falling back to ``inmemory`` (with a clear WARNING
    signal) when it is unset or unhealthy (e.g. the OMN-14380 off-box
    advertised-listener gap), so a stale broker degrades gracefully instead of
    hanging the CLI. An explicit ``--bus`` / ``--kafka-bootstrap`` is never
    second-guessed and flows through ``backend_overrides`` to ``RuntimeLocal``
    unchanged (``feedback_bus_is_the_transport``).
    """

    def test_choices_mirror_runtime_supported_values(self) -> None:
        # The CLI must not advertise a bus the runtime rejects, nor omit one it
        # supports — RuntimeLocal is the source of truth.
        from omnibase_core.runtime.runtime_local import SUPPORTED_EVENT_BUS_VALUES

        assert set(BUS_CHOICES) == set(SUPPORTED_EVENT_BUS_VALUES)
        # The safe fallback floor auto-resolution always lands on when the
        # shared bus is not provably reachable (see TestResolveDefaultBus).
        assert DEFAULT_BUS == "inmemory"

    def test_default_overrides_are_inmemory(self) -> None:
        assert build_backend_overrides(bus="inmemory", kafka_bootstrap=None) == {
            "event_bus": "inmemory"
        }

    def test_kafka_with_bootstrap_threads_broker(self) -> None:
        # The live-bus path: event_bus=kafka + the configured broker bootstrap so
        # RuntimeLocal routes through EventBusKafka.from_bootstrap.
        assert build_backend_overrides(
            bus="kafka", kafka_bootstrap=KAFKA_BOOTSTRAP_ARG
        ) == {"event_bus": "kafka", "kafka_bootstrap": KAFKA_BOOTSTRAP_ARG}

    def test_kafka_without_an_address_is_refused(self) -> None:
        # OMN-16871 flipped this. It used to assert {"event_bus": "kafka"} --
        # an override map with no bootstrap, which left EventBusKafka to read
        # KAFKA_BOOTSTRAP_SERVERS. On the launching host that variable names
        # the governed stability-test lane, so the omission WAS the defect.
        # There is now no argument combination that yields a kafka bus whose
        # address this process did not resolve explicitly.
        with pytest.raises(ValueError, match="requires a broker address"):
            build_backend_overrides(bus="kafka", kafka_bootstrap=None)

    def test_bootstrap_with_inmemory_fails_loud(self) -> None:
        # Passing a broker with the default in-memory bus is a misconfiguration
        # (the command would silently never reach a broker) — fail loud.
        with pytest.raises(ValueError, match="only valid with --bus kafka"):
            build_backend_overrides(bus="inmemory", kafka_bootstrap=KAFKA_BOOTSTRAP_ARG)

    def test_unknown_bus_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="Unsupported bus"):
            build_backend_overrides(bus="redis", kafka_bootstrap=None)

    def test_run_delegate_passes_kafka_overrides_to_receipt_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end wiring: run_delegate must forward the resolved
        # backend_overrides to run_receipt_mode unchanged — no hardcoded bus.
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        exit_code = run_delegate(
            prompt="document the router",
            task_type="document",
            max_tokens=None,
            bus="kafka",
            # OMN-17304: this test's subject is TRANSPORT plumbing, so it pins
            # the locus rather than inheriting the new transport-derived
            # default (kafka -> deployed-lane), which would make it a lane
            # dispatch and drag a live-consumer precondition into an
            # assertion about a dict.
            locus=EnumDelegateLocus.IN_PROCESS,
            kafka_bootstrap=KAFKA_BOOTSTRAP_ARG,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        assert exit_code == 0
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": KAFKA_BOOTSTRAP_ARG,
        }

    def test_run_delegate_defaults_to_inmemory_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OMN-17304: no configured authority -> the shipped tier-0 default
        # runtime config answers (inmemory), with no network probe attempted.
        _probe_must_not_run(monkeypatch)
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            # OMN-17304: the subject is auto-resolution of the TRANSPORT, so
            # the locus is pinned; otherwise resolving to kafka would also
            # resolve to a lane dispatch and require a live consumer.
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        assert captured["backend_overrides"] == {"event_bus": "inmemory"}

    def test_run_delegate_resolves_kafka_from_configured_authority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OMN-17304: the per-runtime configured authority — not a broker
        # probe, not an env var — selects kafka WITHOUT an explicit --bus
        # flag. The CLI's embedded runtime resolves like every other runtime.
        contracts_dir = _write_authority_config(tmp_path, bus_type="kafka")
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))
        _probe_must_not_run(monkeypatch)
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        # OMN-16871 changed the second half of this test, not the first. The
        # configured authority still selects kafka with no --bus flag, which
        # is what OMN-17304 established. What it no longer does is hand that
        # transport an address nobody chose: the run is REFUSED, naming the
        # missing lane selection, instead of letting EventBusKafka read
        # KAFKA_BOOTSTRAP_SERVERS and publish onto whichever lane the shell
        # happened to name.
        with pytest.raises(click.ClickException, match="--lane"):
            run_delegate(
                prompt="research the routing architecture",
                task_type=None,
                max_tokens=None,
                # OMN-17304: the subject is auto-resolution of the TRANSPORT,
                # so the locus is pinned; otherwise resolving to kafka would
                # also resolve to a lane dispatch and require a live consumer.
                locus=EnumDelegateLocus.IN_PROCESS,
                state_root=tmp_path / "state",
                timeout=60,
                verbose=False,
                emit_socket=tmp_path / "no-daemon.sock",
            )

        assert captured == {}

    def test_run_delegate_reachable_broker_does_not_decide_the_transport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OMN-17304: the pre-ruling defect was exactly this shape — a broker
        # that happens to be reachable (KAFKA_BOOTSTRAP_SERVERS exported in
        # the shell) used to flip the transport to kafka via the probe tier.
        # Execution locus is a resolved property, not an environmental
        # accident: with no configured authority the shipped tier-0 default
        # answers inmemory and the broker is never even probed.
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.example:9092")
        _probe_must_not_run(monkeypatch)
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            # OMN-17304: the subject is auto-resolution of the TRANSPORT, so
            # the locus is pinned; otherwise resolving to kafka would also
            # resolve to a lane dispatch and require a live consumer.
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        assert captured["backend_overrides"] == {"event_bus": "inmemory"}

    def test_run_delegate_bootstrap_without_explicit_bus_is_value_error(
        self, tmp_path: Path
    ) -> None:
        # A bare --kafka-bootstrap (no --bus) is never silently absorbed into
        # the auto-resolved default — the caller must say --bus kafka too.
        with pytest.raises(ValueError, match="only valid with --bus kafka"):
            run_delegate(
                prompt="document the router",
                task_type="document",
                max_tokens=None,
                kafka_bootstrap=KAFKA_BOOTSTRAP_ARG,
                state_root=tmp_path / "state",
                timeout=60,
                verbose=False,
                emit_socket=tmp_path / "no-daemon.sock",
            )

    def test_cli_flag_bus_kafka_reaches_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The click flags --bus/--kafka-bootstrap must thread through to
        # backend_overrides exactly as the function-call path does.
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "document the router",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                # OMN-17304: transport test, so the locus is pinned rather
                # than inherited (kafka now implies a lane dispatch).
                "--locus",
                "in-process",
                "--kafka-bootstrap",
                KAFKA_BOOTSTRAP_ARG,
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": KAFKA_BOOTSTRAP_ARG,
        }

    def test_cli_bootstrap_without_kafka_is_usage_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "document the router",
                "--kafka-bootstrap",
                KAFKA_BOOTSTRAP_ARG,
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error:" in result.output
        assert "only valid with --bus kafka" in result.output


class TestCorrelationId:
    """OMN-14397: correlation_id must be fresh per invocation, never reused.

    Two consecutive ``onex delegate`` calls from the same working
    directory/state-root previously returned the SAME ``correlation_id`` (and
    stale response content) on the second call. The CLI now mints and writes
    ``correlation_id`` explicitly per invocation instead of leaving it to an
    implicit downstream default.
    """

    def test_two_runs_get_distinct_correlation_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[str] = []

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            payload = json.loads(
                Path(str(kwargs["input_path"])).read_text(encoding="utf-8")
            )
            captured.append(payload["correlation_id"])
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        # Same working directory / state-root for both runs — the exact
        # OMN-14397 reproduction shape.
        for _ in range(2):
            run_delegate(
                prompt="research the routing architecture",
                task_type=None,
                max_tokens=None,
                state_root=tmp_path / "state",
                timeout=60,
                verbose=False,
                emit_socket=tmp_path / "no-daemon.sock",
            )

        assert len(captured) == 2
        # Each is a real UUID and the two are distinct.
        for raw in captured:
            uuid.UUID(raw)
        assert captured[0] != captured[1]


class TestHardTimeoutBackstop:
    """OMN-14397: ``--timeout`` must abort a hung call, not just RuntimeLocal's
    cooperative ``asyncio.wait_for``.

    ``RuntimeLocal``'s internal timeout only preempts at an ``await`` point; a
    call stuck in synchronous, non-cooperative blocking I/O never yields
    control back, so that timeout silently never fires — the defect that left
    an orphaned process on ``.201`` requiring a manual ``kill``. These tests
    drive a genuinely blocking stub (``time.sleep``, not an
    asyncio-cancelable coroutine) to prove the ``SIGALRM``-based hard backstop
    aborts it anyway.
    """

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM is POSIX-only")
    def test_hard_timeout_aborts_blocking_call(self) -> None:
        started = time.monotonic()
        with pytest.raises(DelegateTimeoutExceededError):
            with cli_delegate._hard_timeout(1):
                # Real blocking sleep, not asyncio-cooperative — proves
                # SIGALRM preempts even non-cooperative blocking I/O.
                time.sleep(5)
        elapsed = time.monotonic() - started
        assert elapsed < 3, f"hard timeout did not abort promptly: {elapsed}s"

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM is POSIX-only")
    def test_hard_timeout_cancels_alarm_on_clean_exit(self) -> None:
        # A call that finishes well inside the window must not leave a
        # dangling SIGALRM armed for later, unrelated code to trip over.
        with cli_delegate._hard_timeout(5):
            pass
        # signal.alarm(0) returns the seconds remaining on any previously
        # scheduled alarm (0 if none is armed).
        assert signal.alarm(0) == 0

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM is POSIX-only")
    def test_run_delegate_aborts_hung_dispatch_despite_receipt_mode_broad_except(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """OMN-14397 round 2: the real ``run_receipt_mode`` wraps the exact
        call that hangs (``RuntimeLocal(...); runtime.run()``) in a broad
        ``except Exception as exc:`` that logs and continues rather than
        re-raising (``receipt_mode.py`` ~509-526). A plain ``time.sleep``
        stub with no surrounding except does not replicate that collaborator
        shape and proves nothing beyond what
        ``test_hard_timeout_aborts_blocking_call`` already proves in
        isolation — this stub reproduces the real try/except-Exception shape
        so the test proves the timeout signal survives it and still reaches
        ``run_delegate``'s own handler (clear stderr message, not just an
        accidental exit code from the swallowed exception).
        """

        def _swallowing_run_receipt_mode(**_kwargs: object) -> int:
            exit_code = 1  # pre-initialized, exactly like receipt_mode.py:505
            try:
                time.sleep(10)  # stands in for the hanging runtime.run() call
                exit_code = 0  # pragma: no cover - never reached within the bound
            except Exception:
                # Mirrors receipt_mode.py's real shape exactly (including the
                # logger.exception call): logs and continues rather than
                # re-raising. A RuntimeError-based timeout signal would die
                # right here, silently.
                logging.getLogger(__name__).exception("receipt_mode: runtime raised")
            return exit_code

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(
            cli_delegate, "run_receipt_mode", _swallowing_run_receipt_mode
        )
        # Shrink the grace window so the test doesn't wait out the full sleep.
        monkeypatch.setattr(cli_delegate, "_HARD_TIMEOUT_GRACE_SECONDS", 1)

        started = time.monotonic()
        exit_code = run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            state_root=tmp_path / "state",
            timeout=1,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )
        elapsed = time.monotonic() - started
        captured = capsys.readouterr()

        assert exit_code == 1
        assert elapsed < 5, f"hung call was not aborted within bound: {elapsed}s"
        # The clear-error contract must fire from run_delegate's own
        # DelegateTimeoutExceededError handler — not an accidental exit code
        # falling out of the stub's own pre-initialized `exit_code = 1` after
        # the exception was silently swallowed by its broad except.
        assert "exceeded hard timeout" in captured.err, (
            f"timeout signal did not survive the broad except — stderr: {captured.err!r}"
        )


class TestResolveDefaultBus:
    """The delegate transport resolves from the configured authority (OMN-17304).

    ``resolve_default_bus`` is the function ``run_delegate`` calls whenever
    ``--bus`` is omitted. Per the OMN-17304 operator ruling it now resolves
    the CLI's EMBEDDED runtime the way every other runtime resolves — from
    that runtime's OWN configuration, through the ONE shared authority
    (``backends/auto_configure.py::resolve_bus_type``), by passing
    ``config_bus=`` from the per-runtime config:

    * ``ONEX_CONTRACTS_DIR`` is a BOOTSTRAP pointer (it names where config
      lives, never what the transport is); the runtime config found there is
      the configured authority.
    * With no pointer (or a pointer to a dir with no runtime config), the
      SHIPPED tier-0 default runtime config answers: in-memory bus, local
      profile. An unconfigured install is still config-resolved — the default
      IS the shipped overlay.
    * ``ONEX_EVENT_BUS_TYPE`` holds NO tier. Set-and-ignored produces a
      warning naming the removal; it never decides the transport.
    * The broker probe is structurally unreachable on this path — a config
      answer always exists, so a reachable broker can no longer flip the
      transport (the pre-ruling environmental accident). Every test here
      installs a probe stub that FAILS the test if the network is touched.
    """

    def test_no_authority_resolves_shipped_tier0_default_without_probing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The golden precondition: no bootstrap pointer, no env override, no
        # broker — the shipped tier-0 default runtime config answers.
        _probe_must_not_run(monkeypatch)

        bus, reason = resolve_default_bus()

        assert bus == "inmemory"
        assert "config.event_bus.type=inmemory" in reason
        assert "tier-0" in reason

    def test_configured_kafka_authority_resolves_kafka_without_probing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contracts_dir = _write_authority_config(tmp_path, bus_type="kafka")
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))
        _probe_must_not_run(monkeypatch)

        bus, reason = resolve_default_bus()

        assert bus == "kafka"
        assert "config.event_bus.type=kafka" in reason
        # Provenance names the actual file, so a receipt/capture reader can
        # tell WHICH authority answered — not merely that one did.
        assert str(contracts_dir) in reason

    def test_configured_local_profile_inmemory_is_first_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 'inmemory' is a first-class CONFIGURED value under the local
        # profile, not only the absent-authority default (operator ruling
        # constraint 2/3).
        contracts_dir = _write_authority_config(
            tmp_path, bus_type="inmemory", profile="local"
        )
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))
        _probe_must_not_run(monkeypatch)

        bus, reason = resolve_default_bus()

        assert bus == "inmemory"
        assert "config.event_bus.type=inmemory" in reason
        assert str(contracts_dir) in reason
        # This is the configured authority speaking, NOT the shipped default.
        assert "tier-0" not in reason

    def test_lane_profile_rejects_inmemory_fails_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The profile axis (ruling constraint 3): lane-profile runtimes still
        # reject the in-memory bus — the validator was NOT weakened to make
        # tier-0 expressible. A lane config declaring inmemory is a
        # misconfiguration and fails loud at load time.
        from omnibase_infra.errors import ProtocolConfigurationError

        contracts_dir = _write_authority_config(
            tmp_path, bus_type="inmemory", profile="lane"
        )
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))
        _probe_must_not_run(monkeypatch)

        with pytest.raises(ProtocolConfigurationError) as excinfo:
            resolve_default_bus()

        assert "lane" in str(excinfo.value)

    def test_pointer_without_config_falls_back_to_shipped_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A pointer to a contracts dir with no runtime config is still
        # config-resolved: the shipped tier-0 default answers (same as the
        # kernel's own absent-file behaviour).
        empty_dir = tmp_path / "empty-contracts"
        empty_dir.mkdir()
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(empty_dir))
        _probe_must_not_run(monkeypatch)

        bus, reason = resolve_default_bus()

        assert bus == "inmemory"
        assert "tier-0" in reason

    def test_env_var_no_longer_resolves_the_transport(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # OMN-17304 (ruling constraint 4): ONEX_EVENT_BUS_TYPE is removed from
        # the ladder ENTIRELY. Set to kafka with no configured authority, the
        # shipped inmemory default still answers — and the set-and-ignored
        # state is warned about, never silently absorbed.
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "kafka")
        _probe_must_not_run(monkeypatch)

        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.backends.auto_configure"
        ):
            bus, reason = resolve_default_bus()

        assert bus == "inmemory"
        assert BUS_TYPE_OVERRIDE_ENV not in reason
        warnings = [
            r.getMessage()
            for r in caplog.records
            if BUS_TYPE_OVERRIDE_ENV in r.getMessage()
        ]
        assert warnings, (
            "a set-and-ignored ONEX_EVENT_BUS_TYPE produced no warning — the "
            "operator has no signal that their export stopped doing anything"
        )
        assert any("ignored" in msg for msg in warnings)

    def test_env_var_does_not_outrank_the_configured_authority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pre-ruling ladder ranked the env var ABOVE config; a shell
        # profile decided every delegation's transport. Now the configured
        # authority wins and the env var is inert.
        contracts_dir = _write_authority_config(tmp_path, bus_type="kafka")
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "inmemory")
        _probe_must_not_run(monkeypatch)

        bus, _reason = resolve_default_bus()

        assert bus == "kafka"

    def test_kafka_bootstrap_argument_does_not_trigger_a_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The bootstrap argument names an ENDPOINT for an already-resolved
        # kafka transport; it is not a resolution input. With no configured
        # authority the shipped default answers inmemory even when a broker
        # address is supplied (run_delegate separately rejects the flag
        # combination at the CLI boundary — this pins the resolver seam).
        _probe_must_not_run(monkeypatch)

        bus, _reason = resolve_default_bus(kafka_bootstrap="broker.example:9092")

        assert bus == "inmemory"


class TestOfflineStandaloneGolden:
    """OMN-17304 AC3 golden test: the offline/standalone flow is UNCHANGED.

    The AC's own bar: with no authority configured and no broker reachable,
    ``onex delegate`` resolves ``inmemory`` and behaves IDENTICALLY to the
    pre-change offline flow — asserted here, not by prose. These tests drive
    the REAL dispatch path (``run_receipt_mode`` -> ``RuntimeLocal`` on the
    in-process bus) against the committed proof contract, with the broker
    probe rigged to FAIL the test if any network resolution is attempted.

    If a change makes this flow probe the network, flip the transport, alter
    the override map, print anything but the single receipt JSON on stdout,
    or exit non-zero, these tests fail — that is the drift alarm.
    """

    def test_golden_offline_delegation_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        _probe_must_not_run(monkeypatch)
        state_root = tmp_path / "state"

        runner = CliRunner()
        with caplog.at_level(logging.INFO, logger="omnibase_infra.cli.cli_delegate"):
            result = runner.invoke(
                delegate_command,
                [
                    "research the routing architecture",
                    "--state-root",
                    str(state_root),
                    "--emit-socket",
                    str(tmp_path / "no-daemon.sock"),
                ],
                catch_exceptions=False,
            )

        # Identical exit + stdout contract: exactly ONE ModelSkillResult JSON
        # line, no RuntimeLocal leakage, no provenance lines on stdout.
        assert result.exit_code == 0, result.output
        stripped = result.stdout.strip()
        parsed = json.loads(stripped)
        assert isinstance(parsed, dict)
        assert "\n" not in stripped, "receipt must be a single JSON line"
        ModelSkillResult.model_validate(parsed)
        # Identical degradation signal: the inmemory warning (stderr/capture
        # only) still names the local-SQLite consequence, so the offline flow
        # is not silently mistaken for shared-substrate evidence.
        warning_messages = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "inmemory" in r.getMessage()
        ]
        assert any("SQLite" in msg for msg in warning_messages), (
            "the offline flow lost its local-SQLite degradation warning: "
            f"{warning_messages}"
        )

    def test_golden_offline_override_map_is_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The wire-level pin: the offline flow hands RuntimeLocal EXACTLY the
        # override map it received before the ruling — one key, no additions.
        _probe_must_not_run(monkeypatch)
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        exit_code = run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        assert exit_code == 0
        assert captured["backend_overrides"] == {"event_bus": "inmemory"}


# ---------------------------------------------------------------------------
# omnimarket drift guard wiring (OMN-13930)
# ---------------------------------------------------------------------------

_DRIFT_FAKE_SHA = "cccccccccccccccccccccccccccccccccccccccc"


def test_drift_guard_fires_before_delegate_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``onex delegate`` runs the same pre-flight guard as ``onex skill``/``onex node``.

    ``DELEGATE_NODE_NAME`` (``node_delegate_skill_orchestrator``) is provided
    by omnimarket, so the delegate CLI has always been exposed to the exact
    stale/absent co-install failure the guard exists to catch -- yet it was
    the one dispatch surface of the three with ZERO guard wiring. A drifted
    venv surfaced there as a bare contract-resolution failure with no pointer
    to the cause or the repair command.

    Fails under the pre-fix ``cli_delegate.py``: with no
    ``check_omnimarket_drift`` attribute on the module, the autouse fixture's
    ``monkeypatch.setattr`` errors out before the test body runs.
    """
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", check_omnimarket_drift)
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
        lambda: None,
    )
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
        lambda omni_home=None: _DRIFT_FAKE_SHA,
    )

    # Any dispatch past the guard is a bug -- prove the guard short-circuits
    # FIRST rather than inferring it from a downstream error string.
    def _must_not_run(**_: object) -> int:
        raise AssertionError("dispatch ran despite a drifted omnimarket install")

    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _must_not_run)

    runner = CliRunner()
    result = runner.invoke(
        cli_delegate.delegate_command,
        [
            "explain the router",
            "--state-root",
            str(tmp_path),
            "--omni-home",
            "/fake/omni_home",
        ],
    )

    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "NOT INSTALLED" in combined
    assert _DRIFT_FAKE_SHA[:12] in combined
    assert "install-node-skill-package.sh --execute" in combined
    # The refusal must carry the escape hatch, not just the diagnosis.
    assert DRIFT_OVERRIDE_ENV in combined


class TestExplicitOverrideProvenance:
    """OMN-17304 AC1 -- ``--bus``/``--kafka-bootstrap`` announce overrides."""

    _CLI_LOGGER = "omnibase_infra.cli.cli_delegate"

    def _dispatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        **bus_kwargs: object,
    ) -> int:
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", lambda **_kwargs: 0)
        return run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            # OMN-17304: this class's subject is TRANSPORT provenance
            # (did --bus/--kafka-bootstrap announce themselves as overrides).
            # The locus is pinned so a kafka case does not additionally become
            # a lane dispatch with a live-consumer precondition; locus
            # provenance has its own tests in test_delegate_locus.py.
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            **bus_kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def _override_records(caplog: pytest.LogCaptureFixture) -> list[str]:
        """OVERRIDE lines about the TRANSPORT flags only.

        OMN-17304 added a second override axis — ``--locus`` announces itself
        the same way — and this class's counter-assertion
        ("an auto-resolved transport is not labelled an override") is only
        meaningful about the flags it is actually testing. Matching the bare
        word OVERRIDE would make that assertion fail on an unrelated locus
        line, i.e. it would stop testing what it claims to test.
        """
        return [
            r.getMessage()
            for r in caplog.records
            if "OVERRIDE" in r.getMessage()
            and ("--bus" in r.getMessage() or "--kafka-bootstrap" in r.getMessage())
        ]

    def test_explicit_bus_logs_itself_as_an_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger=self._CLI_LOGGER):
            assert self._dispatch(tmp_path, monkeypatch, bus="inmemory") == 0

        overrides = self._override_records(caplog)
        assert overrides, (
            "an explicit --bus produced no OVERRIDE provenance line; the "
            "receipt cannot distinguish a resolved transport from a typed one"
        )
        assert any("--bus" in msg and "inmemory" in msg for msg in overrides)

    def test_explicit_kafka_bootstrap_logs_itself_as_an_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger=self._CLI_LOGGER):
            assert (
                self._dispatch(
                    tmp_path,
                    monkeypatch,
                    bus="kafka",
                    kafka_bootstrap=KAFKA_BOOTSTRAP_ARG,
                )
                == 0
            )

        # OMN-16871 changed the VERB, not the requirement. The flag no longer
        # "overrides" an address the run would otherwise have resolved,
        # because the run resolves none: it states the address directly, and
        # says so. What still has to hold is that the provenance names the
        # flag and the value, so a receipt can tell a typed address apart
        # from a declared one.
        records = [
            record.getMessage()
            for record in caplog.records
            if record.name == self._CLI_LOGGER
        ]
        assert any(
            "--kafka-bootstrap" in msg and KAFKA_BOOTSTRAP_ARG in msg for msg in records
        ), (
            "an explicit --kafka-bootstrap produced no provenance line; "
            f"captured records: {records}"
        )

    def test_auto_resolved_bus_is_not_labelled_an_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)

        with caplog.at_level(logging.INFO, logger=self._CLI_LOGGER):
            assert self._dispatch(tmp_path, monkeypatch) == 0

        assert not self._override_records(caplog)
        assert any("inmemory" in r.getMessage() for r in caplog.records), (
            "an auto-resolved run logged no transport provenance at all"
        )

    def test_override_provenance_never_reaches_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert (
            self._dispatch(
                tmp_path,
                monkeypatch,
                bus="kafka",
                kafka_bootstrap=KAFKA_BOOTSTRAP_ARG,
            )
            == 0
        )
        assert capsys.readouterr().out == ""


class TestBusHelpTextTellsTheTruth:
    """OMN-17295 AC4: ``--help`` must not claim an execution mode that does not exist.

    As shipped, ``onex delegate --help`` said ``--bus kafka`` "publishes the
    typed delegate-skill command to the broker so a deployed runtime consumer
    dispatches it." That was false: :func:`build_backend_overrides` — the ONLY
    thing ``--bus`` feeds — returns ``{"event_bus": <bus>}`` (plus an optional
    ``kafka_bootstrap``), and nothing in that map names an executor.

    That remains true of ``--bus``, and this class still pins it. What changed
    with OMN-17304 is that the CLI grew a real remote mode under a DIFFERENT
    flag: ``--locus``. So the truthful help text is no longer "execution is
    always in-process" — the previous revision of this class asserted exactly
    that string and would have kept the help text lying in the other
    direction. It now asserts the pair: ``--bus`` is transport, ``--locus``
    owns where the work runs.
    """

    @staticmethod
    def _help_text() -> str:
        result = CliRunner().invoke(delegate_command, ["--help"])
        assert result.exit_code == 0, result.output
        return result.output

    def test_bus_only_ever_sets_the_event_bus_backend(self) -> None:
        """The structural fact the help text has to match."""
        assert build_backend_overrides(bus="kafka", kafka_bootstrap="broker:19092") == {
            "event_bus": "kafka",
            "kafka_bootstrap": "broker:19092",
        }
        # Nothing in the override map names a remote executor, a deployed
        # consumer, or an execution locality.
        for overrides in (
            build_backend_overrides(bus="kafka", kafka_bootstrap="broker:19092"),
            build_backend_overrides(bus="inmemory", kafka_bootstrap=None),
        ):
            assert set(overrides) <= {"event_bus", "kafka_bootstrap"}

    def test_help_does_not_claim_a_deployed_consumer_dispatches_the_work(
        self,
    ) -> None:
        help_text = " ".join(self._help_text().split()).lower()
        for false_claim in (
            "a deployed runtime consumer dispatches it",
            "so a deployed runtime dispatches it",
            "deployed runtime consumer picks it up",
        ):
            assert false_claim not in help_text, (
                f"--help still claims remote execution: {false_claim!r}"
            )

    def test_help_separates_transport_from_where_the_work_runs(self) -> None:
        """OMN-17304: --bus is transport, --locus is locus, and --help says so."""
        help_text = " ".join(self._help_text().split()).lower()
        assert "transport" in help_text, (
            "--help must state that --bus selects the event transport"
        )
        assert "--locus" in help_text, (
            "--help must offer the flag that actually decides where work runs"
        )
        assert "where the work runs" in help_text
        assert "deployed-lane" in help_text and "in-process" in help_text

    def test_help_states_a_lane_dispatch_refuses_rather_than_running_here(
        self,
    ) -> None:
        """The fail-closed promise is part of the contract with the operator.

        Without it a reader may assume the friendly behaviour — fall back to
        local — which is precisely the defect: a locally-produced receipt read
        as a lane result.
        """
        help_text = " ".join(self._help_text().split()).lower()
        assert "refuses" in help_text


class TestCorrelationReachesTheReceipt:
    """OMN-17295 / OMN-14872: the CLI must TELL receipt-mode which run this is.

    ``run_delegate`` mints the correlation id and writes it into the payload,
    but before this change it never handed that id to ``run_receipt_mode`` —
    so the receipt layer had nothing to join against and fell back to reading
    whatever correlation the workflow content happened to declare. That is why
    a stale terminal envelope could be printed as this run's result.
    """

    def test_run_delegate_threads_its_correlation_id_into_receipt_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, str]] = []

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            payload = json.loads(
                Path(str(kwargs["input_path"])).read_text(encoding="utf-8")
            )
            seen.append(
                (payload["correlation_id"], str(kwargs["expected_correlation_id"]))
            )
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            # OMN-17304: the subject is auto-resolution of the TRANSPORT, so
            # the locus is pinned; otherwise resolving to kafka would also
            # resolve to a lane dispatch and require a live consumer.
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        assert len(seen) == 1
        payload_correlation, receipt_correlation = seen[0]
        uuid.UUID(payload_correlation)
        assert payload_correlation == receipt_correlation, (
            "the id written into the request and the id the receipt joins on "
            "must be the same run identity"
        )

    def test_receipt_correlation_equals_the_payload_correlation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-14872: the receipt announced an id the request never carried.

        Reported shape: the outer receipt's ``correlation_id`` differed from
        the id written into ``<state-root>/tmp/delegate-input-*.json``, so a
        caller correlating its own request against the printed receipt was
        matching on two different identities. Real dispatch path, no mocks.
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        state_root = tmp_path / "state"

        result = CliRunner().invoke(
            delegate_command,
            [
                "research the routing architecture",
                "--state-root",
                str(state_root),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        receipt = json.loads(result.stdout.strip())
        payloads = list((state_root / "tmp").glob("delegate-input-*.json"))
        assert len(payloads) == 1
        request = json.loads(payloads[0].read_text(encoding="utf-8"))

        assert receipt["correlation_id"] == request["correlation_id"], (
            "the receipt must be reported under the id the request carried"
        )


class TestLocalRunArtifacts:
    """OMN-16999: persist only route-attributed delegate receipts."""

    @staticmethod
    def _receipt(*, accepted: bool = True) -> ModelSkillResult[dict[str, object]]:
        correlation_id = uuid.uuid4()
        attempt: dict[str, object] = {
            "tier": "local",
            "backend_id": "local-coder",
            "model_id": "qwen3.8",
            "quality_gate_passed": accepted,
            "acceptance_decision": "accept" if accepted else "climb",
        }
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
                "attempts": [attempt],
            },
            result_model=(
                "omnimarket.models.delegation.wire."
                "model_delegate_skill_response.ModelDelegateSkillCompleted"
            ),
            runtime_identity=collect_runtime_identity(config_source="test"),
        )

    def test_writes_three_files_with_accepted_route(self, tmp_path: Path) -> None:
        receipt = self._receipt()
        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="research the route",
            task_type="research",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == "answer"
        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_data["receipt_id"] == str(receipt.correlation_id)
        assert receipt_data["backend_id"] == "local-coder"
        assert receipt_data["model"] == "qwen3.8"
        assert receipt_data["routing_tier"] == "local"
        assert receipt_data["endpoint"] == "http://local.invalid/v1/chat/completions"
        run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_data == {
            "correlation_id": str(receipt.correlation_id),
            "lane": "local",
            "prompt": "research the route",
            "run_id": str(receipt.run_id),
            "task_type": "research",
            # OMN-18305: a customer can see that a class was chosen for them,
            # and how, without re-reading the prompt.
            "task_type_resolution": "explicit",
        }

    def test_attributes_no_route_without_an_accepted_attempt(
        self, tmp_path: Path
    ) -> None:
        """OMN-18306 amended this: the refusal is about the ROUTE, not the run.

        Before OMN-18306 this raised and wrote nothing, and because the raise
        travelled up through ``run_receipt_mode``'s callback it also erased the
        receipt. The route attribution is still fail-closed -- no backend,
        model, tier or endpoint is written -- but the run is recorded.
        """
        receipt = self._receipt(accepted=False)
        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="research the route",
            task_type="research",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_data["route_attributed"] is False
        for field in ("backend_id", "model", "endpoint", "routing_tier"):
            assert receipt_data.get(field) in (None, "")
        assert receipt_data["attempts"][0]["backend_id"] == "local-coder"

    def test_ignores_non_delegate_typed_receipt(self, tmp_path: Path) -> None:
        receipt = self._receipt().model_copy(
            update={"result_model": "tests.fixtures.ModelProofNoopResult"}
        )
        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="proof",
            task_type="research",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )
        assert not (tmp_path / "runs").exists()

    def test_refuses_to_fabricate_task_type_resolution(self, tmp_path: Path) -> None:
        receipt = self._receipt()
        with pytest.raises(ValueError, match="task_type_resolution is required"):
            _write_local_run_files(
                receipt=receipt,
                state_root=tmp_path,
                prompt="research the route",
                task_type="research",
            )


class TestLocalRunArtifactsOnEscalatedRun:
    """A run that escalated past a failed rung still wrote no artifacts (OMN-16999).

    THE DEFECT. ``receipt_mode`` builds the typed ``ModelSkillResult[JsonValue]``
    receipt only when ``status.is_success_like``; otherwise it wraps the run in a
    ``ModelReceiptRuntimeSummary`` whose ``result_model`` is that summary class.
    ``_write_local_run_files`` keyed its type guard on the string
    ``"ModelDelegateSkill"``, so on every non-success receipt it returned
    silently and wrote nothing.

    That is not a rare path — it is the ordinary one. Measured live on this
    workstation 2026-09-05, before the binding repoint in this same ticket:

        attempt 1  local-heavy-reasoning  model_attribution_mismatch  -> climb
        attempt 2  cloud-gemini-pro       quality_gate_passed=true    -> ACCEPT

    The delegation returned the answer ``"OK"`` from an ACCEPTED attempt, and
    still terminalized ``status=failed`` / ``terminal_failure_cause=provider_error``
    because an earlier attempt had errored. So the customer got an answer, the
    receipt named the route that produced it, and ``.onex_state/runs/`` did not
    exist — which is exactly the B5 gap this ticket was opened for, surviving the
    writer that was supposed to close it.

    The accepted attempt is the contract, not the overall status: an escalation
    that ends in an accepted answer is an answer that must be written down.
    """

    @staticmethod
    def _summary_receipt(
        *, accepted: bool = True, workflow: str | None = None
    ) -> ModelSkillResult[ModelReceiptRuntimeSummary]:
        """The shape ``receipt_mode`` emits for a run that terminalized failed."""
        correlation_id = uuid.uuid4()
        terminal_payload: dict[str, object] = {
            "correlation_id": str(correlation_id),
            "task_type": "research",
            "provider": (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
                "chat/completions"
            ),
            "model_name": "gemini-2.5-flash",
            "response": "OK",
            "error_message": "delegation terminalized as failed: provider_error",
            "attempts": [
                {
                    "tier": "local",
                    "backend_id": "local-heavy-reasoning",
                    "model_id": "Qwen3.6-35B-A3B",
                    "quality_gate_passed": False,
                    "acceptance_decision": "climb",
                },
                {
                    "tier": "cheap_cloud",
                    "backend_id": "cloud-gemini-pro",
                    "model_id": "gemini-2.5-flash",
                    "quality_gate_passed": accepted,
                    "acceptance_decision": "accept" if accepted else "climb",
                },
            ],
        }
        summary = ModelReceiptRuntimeSummary(
            workflow_result="failed",
            exit_code=1,
            workflow=(
                workflow
                if workflow is not None
                else "/site-packages/omnimarket/nodes/"
                "node_delegate_skill_orchestrator/contract.yaml"
            ),
            terminal_payload=terminal_payload,
            handler_result=terminal_payload,
            error="",
            capture_log="",
        )
        return ModelSkillResult[ModelReceiptRuntimeSummary](
            skill_name=cli_delegate.DELEGATE_NODE_NAME,
            node_name=cli_delegate.DELEGATE_NODE_NAME,
            status=EnumSkillResultStatus.FAILED,
            correlation_id=correlation_id,
            run_id=uuid.uuid4(),
            exit_code=1,
            duration_ms=94_687,
            result=summary,
            result_model=(
                "omnibase_infra.cli.model_receipt_runtime_summary."
                "ModelReceiptRuntimeSummary"
            ),
            runtime_identity=collect_runtime_identity(config_source="test"),
        )

    def test_writes_artifacts_for_the_accepted_attempt_of_a_failed_run(
        self, tmp_path: Path
    ) -> None:
        receipt = self._summary_receipt()

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="Reply with exactly: OK",
            task_type="research",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == "OK"

        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        # Route identity is the ACCEPTED attempt's, never the first attempted
        # rung's -- attributing this answer to local-heavy-reasoning would be a
        # lie about which model produced it.
        assert receipt_data["backend_id"] == "cloud-gemini-pro"
        assert receipt_data["model"] == "gemini-2.5-flash"
        assert receipt_data["routing_tier"] == "cheap_cloud"
        assert receipt_data["endpoint"].startswith(
            "https://generativelanguage.googleapis.com/"
        )
        assert receipt_data["receipt_id"] == str(receipt.correlation_id)
        # The run really did terminalize failed; the artifact says so rather
        # than laundering an escalated run into a clean success.
        assert receipt_data["status"] == EnumSkillResultStatus.FAILED.value

        run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_data["prompt"] == "Reply with exactly: OK"
        assert run_data["lane"] == "cheap_cloud"
        assert run_data["correlation_id"] == str(receipt.correlation_id)

    def test_attributes_no_route_after_the_summary_unwrap(self, tmp_path: Path) -> None:
        """Fail-closed ATTRIBUTION survives the unwrap (amended by OMN-18306).

        No accepted rung, so no route is named -- but the run is written down,
        including the rung that was attempted and climbed, which is the only
        way a customer can see which backend refused them.
        """
        receipt = self._summary_receipt(accepted=False)
        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="Reply with exactly: OK",
            task_type="research",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_data["route_attributed"] is False
        assert [attempt["backend_id"] for attempt in receipt_data["attempts"]] == [
            "local-heavy-reasoning",
            "cloud-gemini-pro",
        ]
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == "OK"

    def test_ignores_a_failed_run_of_some_other_node(self, tmp_path: Path) -> None:
        """The unwrap is scoped to the delegate contract, not to any summary.

        ``run_receipt_mode`` is shared with ``onex node``/``onex skill``. A
        failed proof run of an unrelated node must not be mistaken for an
        unattributed delegation and raise.
        """
        _write_local_run_files(
            receipt=self._summary_receipt(
                workflow="/site-packages/omnimarket/nodes/node_gap_compute/contract.yaml"
            ),
            state_root=tmp_path,
            prompt="proof",
            task_type="research",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )
        assert not (tmp_path / "runs").exists()


_OMN18306_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18306"
    / "failed_no_accepted_attempt_receipt.json"
)


def _recorded_failed_receipt() -> ModelSkillResult[ModelReceiptRuntimeSummary]:
    """The verbatim receipt of a real terminally-failed lab delegation.

    Recorded 2026-09-13 against the ``.201`` lab model endpoint; see the
    fixture's README for the run, the redactions, and what it does not cover.
    Nothing about its shape is synthesised — this is what the runtime produced.
    """
    return ModelSkillResult[ModelReceiptRuntimeSummary].model_validate(
        json.loads(_OMN18306_FIXTURE.read_text(encoding="utf-8"))
    )


class TestFailedDelegationIsRendered:
    """OMN-18306: a terminally-failed delegation printed nothing at all.

    THE DEFECT. ``_write_local_run_files`` refuses — correctly — to attribute a
    route for a run with no accepted attempt, and ``run_receipt_mode`` invoked
    that writer as ``receipt_callback`` BEFORE reaching its own
    ``click.echo(receipt.model_dump_json())``. So a correct refusal about route
    ATTRIBUTION erased the ANSWER: stdout zero bytes, one error line, and no way
    for the customer to tell a refused run from a crashed one, which rung failed,
    or whether they were billed.

    Route attribution and receipt rendering must not share a failure domain.
    """

    def test_recorded_failure_has_no_accepted_attempt(self) -> None:
        """Positive control on the fixture itself: it really is the bad case."""
        receipt = _recorded_failed_receipt()
        payload = receipt.result.terminal_payload
        assert isinstance(payload, dict)
        attempts = payload["attempts"]
        assert isinstance(attempts, list) and len(attempts) == 2
        assert all(
            str(attempt["acceptance_decision"]) != "accept" for attempt in attempts
        )
        assert payload["status"] == "failed"
        assert payload["terminal_failure_cause"] == "provider_error"

    def test_writer_records_every_rung_of_a_failed_run(self, tmp_path: Path) -> None:
        """AC2: the three files exist for a failed run and carry the evidence."""
        receipt = _recorded_failed_receipt()

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="summarise the coordination ledger",
            task_type="complex_reasoning",
            task_type_resolution=EnumTaskTypeResolution.CONTRACT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        assert (run_dir / "result.txt").exists()
        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_data["terminal_failure_cause"] == "provider_error"
        assert "503" in receipt_data["failure_reason"]
        assert receipt_data["cost_usd"] == 0.0
        recorded = receipt_data["attempts"]
        assert [attempt["backend_id"] for attempt in recorded] == [
            "local-heavy-reasoning",
            "cloud-gemini-pro",
        ]
        assert recorded[0]["failure_class"] == "context_too_large"
        assert recorded[0]["input_tokens_measured"] == 14149
        assert recorded[0]["input_token_budget"] == 8000
        assert recorded[0]["acceptance_decision"] == "climb"
        assert recorded[1]["failure_class"] == "model_unavailable"
        assert recorded[1]["tier"] == "cheap_cloud"
        run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_data["task_type"] == "complex_reasoning"
        assert run_data["run_id"] == str(receipt.run_id)

    def test_writer_synthesises_no_route_for_a_failed_run(self, tmp_path: Path) -> None:
        """AC3: fail-closed attribution survives — no route is invented."""
        receipt = _recorded_failed_receipt()

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="summarise the coordination ledger",
            task_type="complex_reasoning",
            task_type_resolution=EnumTaskTypeResolution.CONTRACT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_data["route_attributed"] is False
        assert "no accepted routing attempt" in receipt_data["route_unattributed"]
        # The LAST attempted backend is the lie this guard exists to prevent.
        for field in ("backend_id", "model", "endpoint", "routing_tier"):
            assert receipt_data.get(field) in (None, "")
        run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_data["lane"] is None
        assert run_data["route_attributed"] is False

    def test_writer_carries_model_content_when_a_rung_produced_some(
        self, tmp_path: Path
    ) -> None:
        """AC2's content clause, on the same recorded envelope.

        The recorded run produced no content because no rung answered. The gate
        -rejection shape that DOES carry content was not reproducible on demand
        (fixture README), so the mapping is pinned on the recorded envelope with
        a response substituted into its terminal payload — the ONLY field
        changed, and one the runtime demonstrably populates on that path.
        """
        receipt = _recorded_failed_receipt()
        payload = dict(receipt.result.terminal_payload or {})
        payload["response"] = "a partial answer the gate then rejected"
        receipt = receipt.model_copy(
            update={
                "result": receipt.result.model_copy(
                    update={"terminal_payload": payload, "handler_result": payload}
                )
            }
        )

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="summarise the coordination ledger",
            task_type="complex_reasoning",
            task_type_resolution=EnumTaskTypeResolution.CONTRACT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == (
            "a partial answer the gate then rejected"
        )

    def test_render_survives_a_raising_artifact_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC1/AC5: an exception in artifact writing cannot suppress the receipt.

        This is the ordering pin. Reorder the callback back in front of the
        render and stdout goes to zero bytes, which is the defect verbatim.
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: contract_path,
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

        def _refuse(**_kwargs: object) -> None:
            raise ValueError(
                "delegate receipt has no accepted routing attempt; refusing to "
                "write unattributed route artifacts"
            )

        monkeypatch.setattr(cli_delegate, "_write_local_run_files", _refuse)

        runner = CliRunner()
        result = runner.invoke(
            delegate_command,
            [
                "summarise the coordination ledger",
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        stripped = result.stdout.strip()
        assert stripped, "a failed artifact write must not erase the receipt"
        parsed = json.loads(stripped)
        ModelSkillResult.model_validate(parsed)
        # AC4: rendering the receipt is not reporting success.
        assert result.exit_code != 0


_OMN18569_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18569"
    / "dispatched_envelope_carrier_receipt.json"
)


def _recorded_dispatched_receipt() -> ModelSkillResult[ModelReceiptRuntimeSummary]:
    """The verbatim receipt of a real DISPATCHED delegation to the ``.201`` dev lane.

    Recorded 2026-09-17, correlation ``83aa8b6c-8189-49f0-953d-80c8f015ed0a``.
    See the fixture's README for the run and its redactions. Nothing about its
    shape is synthesised — this is the CLI's own stdout.
    """
    return ModelSkillResult[ModelReceiptRuntimeSummary].model_validate(
        json.loads(_OMN18569_FIXTURE.read_text(encoding="utf-8"))
    )


class TestDispatchedDelegationWritesRunFiles:
    """OMN-18569: a dispatched delegation wrote none of the three files, silently.

    THE DEFECT. ``receipt_mode`` takes the ``ModelReceiptRuntimeSummary`` branch
    whenever no handler result exists, and a DISPATCHED run hosts no handlers —
    so every dispatched run lands there, success included. On that path the
    terminal arrives inside its event envelope, so the delegation fields sit at
    ``terminal_payload.payload`` and not at ``terminal_payload``. The unwrap
    looked for an ``attempts`` list directly on ``terminal_payload``, found
    none, returned ``None``, and the writer returned early.

    Measured live on the lane the product is demonstrated from: exit 0, a
    correct answer, quality 1.0 — and no ``result.txt``, ``receipt.json`` or
    ``run.json``, with no line on stderr saying so. That is goal-board row B5
    failing on the dispatched path while reporting success.
    """

    def test_recorded_receipt_carries_its_terminal_inside_an_envelope(self) -> None:
        """Positive control on the fixture: it really is the envelope carrier.

        Without this, a green suite below could mean the fixture happens to be
        the bare shape the old code already handled.
        """
        receipt = _recorded_dispatched_receipt()
        carrier = receipt.result.terminal_payload
        assert isinstance(carrier, dict)
        assert "envelope_id" in carrier, "the recorded carrier is an event envelope"
        assert "attempts" not in carrier, (
            "the delegation fields are NOT at the top level — that is the defect"
        )
        nested = carrier["payload"]
        assert isinstance(nested, dict)
        assert isinstance(nested["attempts"], list) and len(nested["attempts"]) == 1
        # The second key the old unwrap tried is null on this path, so it could
        # not have rescued the miss either.
        assert receipt.result.handler_result is None
        assert receipt.result.handler_locus == "dispatched"
        # Exit 0 with a correct answer: nothing about the RUN was wrong.
        assert receipt.exit_code == 0
        assert receipt.status is EnumSkillResultStatus.SUCCESS
        assert nested["response"] == "2, 3, 5, 7, 11"

    def test_writes_three_files_for_a_dispatched_run(self, tmp_path: Path) -> None:
        """AC1: the three customer artifacts exist. RED before the unwrap fix."""
        receipt = _recorded_dispatched_receipt()

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt=(
                "List the first five prime numbers in ascending order, "
                "separated by commas, and nothing else."
            ),
            task_type="summarization",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == "2, 3, 5, 7, 11"
        assert (run_dir / "receipt.json").exists()
        assert (run_dir / "run.json").exists()

    def test_receipt_names_the_lane_rung_that_answered(self, tmp_path: Path) -> None:
        """AC1: the receipt attributes the run to the rung that actually answered."""
        receipt = _recorded_dispatched_receipt()

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="List the first five prime numbers",
            task_type="summarization",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        receipt_data = json.loads(
            (run_dir / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_data["model"] == "Qwen3.6-35B-A3B"
        assert receipt_data["routing_tier"] == "local"
        assert receipt_data["backend_id"] == "a3428e79-1694-5248-ab00-8e532196a515"
        assert receipt_data["receipt_id"] == str(receipt.correlation_id)
        assert receipt_data["correlation_id"] == (
            "83aa8b6c-8189-49f0-953d-80c8f015ed0a"
        )
        run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_data["lane"] == "local"
        assert run_data["task_type"] == "summarization"

    def test_announces_the_paths_on_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A customer learns where the files are from the run, not from ``ls``."""
        receipt = _recorded_dispatched_receipt()

        _write_local_run_files(
            receipt=receipt,
            state_root=tmp_path,
            prompt="List the first five prime numbers",
            task_type="summarization",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )

        announced = capsys.readouterr().err
        assert "delegate artifacts: " in announced
        for name in ("result.txt", "receipt.json", "run.json"):
            assert str(tmp_path / "runs" / str(receipt.run_id) / name) in announced


class TestUnresolvableTerminalFailsLoudly:
    """OMN-18569: an unresolvable terminal must never be a silent skip.

    The nesting miss was one level and cheap. What let it survive a release was
    the early return: a writer that cannot find its terminal and exits 0 looks,
    from outside, exactly like one that had nothing to write. Both this defect
    and OMN-16999's before it were found by a human noticing an empty directory
    long afterwards.
    """

    @staticmethod
    def _summary_receipt_without_a_terminal(
        *, workflow: str
    ) -> ModelSkillResult[ModelReceiptRuntimeSummary]:
        return ModelSkillResult[ModelReceiptRuntimeSummary](
            skill_name=cli_delegate.DELEGATE_NODE_NAME,
            node_name=cli_delegate.DELEGATE_NODE_NAME,
            status=EnumSkillResultStatus.SUCCESS,
            correlation_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            exit_code=0,
            duration_ms=11,
            result=ModelReceiptRuntimeSummary(
                workflow_result="completed",
                exit_code=0,
                workflow=workflow,
                terminal_payload=None,
                handler_result=None,
            ),
            result_model=(
                "omnibase_infra.cli.model_receipt_runtime_summary."
                "ModelReceiptRuntimeSummary"
            ),
            runtime_identity=collect_runtime_identity(config_source="test"),
        )

    def test_absent_terminal_raises_naming_both_carrier_fields(
        self, tmp_path: Path
    ) -> None:
        """AC3: the refusal says WHICH field was missing, not that something failed."""
        receipt = self._summary_receipt_without_a_terminal(
            workflow=(
                "/site-packages/omnimarket/nodes/"
                "node_delegate_skill_orchestrator/contract.yaml"
            )
        )

        with pytest.raises(DelegateTerminalUnresolvedError) as raised:
            _write_local_run_files(
                receipt=receipt,
                state_root=tmp_path,
                prompt="List the first five prime numbers",
                task_type="summarization",
                task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
            )

        message = str(raised.value)
        assert "terminal_payload: absent" in message
        assert "handler_result: absent" in message
        assert not (tmp_path / "runs").exists()

    def test_terminal_of_an_unrecognised_shape_names_the_absent_field(
        self, tmp_path: Path
    ) -> None:
        """A present-but-wrong terminal is named too, not just an absent one."""
        receipt = self._summary_receipt_without_a_terminal(
            workflow=(
                "/site-packages/omnimarket/nodes/"
                "node_delegate_skill_orchestrator/contract.yaml"
            )
        )
        receipt = receipt.model_copy(
            update={
                "result": receipt.result.model_copy(
                    update={"terminal_payload": {"status": "completed"}}
                )
            }
        )

        with pytest.raises(DelegateTerminalUnresolvedError) as raised:
            _write_local_run_files(
                receipt=receipt,
                state_root=tmp_path,
                prompt="List the first five prime numbers",
                task_type="summarization",
                task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
            )

        message = str(raised.value)
        assert "attempts" in message, "the field that identifies a terminal"
        assert "envelope_id" in message, "and the one that identifies its envelope"

    def test_some_other_nodes_run_is_still_skipped_silently(
        self, tmp_path: Path
    ) -> None:
        """AC5: the new refusal did not widen into every receipt.

        ``run_receipt_mode`` is shared with ``onex node``/``onex skill``. A
        failed proof run of an unrelated node reaches this writer with no
        terminal at all, and must still write nothing WITHOUT raising —
        otherwise this fix converts an unrelated node's failure into a delegate
        error.
        """
        _write_local_run_files(
            receipt=self._summary_receipt_without_a_terminal(
                workflow="/site-packages/omnimarket/nodes/node_gap_compute/contract.yaml"
            ),
            state_root=tmp_path,
            prompt="proof",
            task_type="summarization",
            task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        )
        assert not (tmp_path / "runs").exists()

    def test_the_refusal_exits_non_zero_and_reaches_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3's other half, end to end through the real command.

        The unit tests above prove the writer raises and names the field. This
        proves what the CUSTOMER sees when it does: a non-zero exit and the
        message on stderr, with the receipt still on stdout — ``receipt_mode``
        catches a callback failure rather than letting it erase the answer
        (OMN-18306), and the two guarantees have to hold together.
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

        def _unresolvable(**_kwargs: object) -> None:
            raise DelegateTerminalUnresolvedError(
                "delegate receipt carries no resolvable delegation terminal, so "
                "the customer artifacts cannot be written -- terminal_payload: "
                "absent; handler_result: absent"
            )

        monkeypatch.setattr(cli_delegate, "_write_local_run_files", _unresolvable)

        result = CliRunner().invoke(
            delegate_command,
            [
                "List the first five prime numbers",
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code != 0, "no files written is not a successful run"
        assert "terminal_payload: absent" in result.stderr
        # The refusal is about the FILES; the answer still reaches the customer.
        ModelSkillResult.model_validate(json.loads(result.stdout.strip()))
