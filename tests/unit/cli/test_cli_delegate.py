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

import pytest
from click.testing import CliRunner

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
    build_backend_overrides,
    classify_task_type,
    delegate_command,
    resolve_default_bus,
    run_delegate,
)
from omnibase_infra.cli.omnimarket_drift_guard import (
    DRIFT_OVERRIDE_ENV,
    check_omnimarket_drift,
)
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


class TestClassifyTaskType:
    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            ("write unit tests for verify.py", "test"),
            ("add a pytest for the parser", "test"),
            ("document the routing module", "document"),
            ("write a docstring for this fn", "document"),
            ("refactor the dispatch loop", "refactor"),
            ("simplify the config parsing", "refactor"),
            ("review this PR for correctness", "review"),
            ("audit the auth flow", "review"),
            ("reason through the tradeoffs", "reasoning"),
            ("compare two architectures", "reasoning"),
            ("implement an HTTP server", "code_generation"),
            ("build a CLI scaffold", "code_generation"),
            ("what does a calendar app need", DEFAULT_TASK_TYPE),
        ],
    )
    def test_keyword_mapping(self, prompt: str, expected: str) -> None:
        assert classify_task_type(prompt) == expected

    def test_first_match_wins_test_before_code_generation(self) -> None:
        # "write" maps to code_generation, "test" maps to test; "test" rule is
        # ordered first, so a prompt with both classifies as test.
        assert classify_task_type("write a test for the handler") == "test"

    def test_case_insensitive(self) -> None:
        assert classify_task_type("REFACTOR the LOOP") == "refactor"


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
            task_type=None,
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

    def test_kafka_without_bootstrap_omits_key(self) -> None:
        # No bootstrap => Kafka bus resolves from KAFKA_BOOTSTRAP_SERVERS; the
        # override map must not carry an empty/None bootstrap.
        assert build_backend_overrides(bus="kafka", kafka_bootstrap=None) == {
            "event_bus": "kafka"
        }

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

        run_delegate(
            prompt="research the routing architecture",
            task_type=None,
            max_tokens=None,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
        )

        # No explicit --kafka-bootstrap was passed, so the override map omits
        # it — the Kafka bus resolves its own bootstrap from
        # KAFKA_BOOTSTRAP_SERVERS at RuntimeLocal construction time.
        assert captured["backend_overrides"] == {"event_bus": "kafka"}

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
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            **bus_kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def _override_records(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if "OVERRIDE" in r.getMessage()]

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

        overrides = self._override_records(caplog)
        assert any(
            "--kafka-bootstrap" in msg and KAFKA_BOOTSTRAP_ARG in msg
            for msg in overrides
        ), (
            "an explicit --kafka-bootstrap produced no OVERRIDE provenance "
            f"line; captured override records: {overrides}"
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
    dispatches it." That is false. :func:`build_backend_overrides` — the ONLY
    thing ``--bus`` feeds — returns ``{"event_bus": <bus>}`` (plus an optional
    ``kafka_bootstrap``) and hands it to ``RuntimeLocal``, which executes the
    orchestrator IN-PROCESS out of the local venv on both bus values. The flag
    selects the event TRANSPORT; it never relocates execution.

    A remote-execution mode is explicitly out of scope (operator-reviewed): a
    thin client, if it is ever built, is gateway-mediated. So the fix is the
    help text, not a new mode.
    """

    @staticmethod
    def _help_text() -> str:
        result = CliRunner().invoke(delegate_command, ["--help"])
        assert result.exit_code == 0, result.output
        return result.output

    def test_bus_only_ever_sets_the_event_bus_backend(self) -> None:
        """The structural fact the help text has to match."""
        assert build_backend_overrides(bus="kafka", kafka_bootstrap=None) == {
            "event_bus": "kafka"
        }
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

    def test_help_states_execution_is_in_process_and_bus_is_transport_only(
        self,
    ) -> None:
        help_text = " ".join(self._help_text().split()).lower()
        assert "in-process" in help_text, (
            "--help must state that the orchestrator always runs in-process"
        )
        assert "transport" in help_text, (
            "--help must state that --bus selects the event transport only"
        )
        assert "does not change where the work runs" in help_text


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
