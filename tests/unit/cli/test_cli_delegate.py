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
    """Unit tests must not depend on ambient ``ONEX_EVENT_BUS_TYPE`` (OMN-16678).

    ``ONEX_EVENT_BUS_TYPE`` is now tier 2 of the shared resolution order, so an
    ambient value in the developer's shell would pin the bus and silently make
    the probe-tier tests vacuous. Tests that exercise the override set it
    explicitly.
    """
    monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)


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
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
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
        contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
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
        # No KAFKA_BOOTSTRAP_SERVERS configured -> resolve_default_bus
        # short-circuits to inmemory with no network probe attempted (see
        # TestResolveDefaultBus.test_no_bootstrap_short_circuits_to_inmemory).
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
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

    def test_run_delegate_auto_resolves_kafka_when_broker_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OMN-14376: a configured, healthy broker is selected WITHOUT an
        # explicit --bus kafka flag — delegation reaches the shared bus by
        # default, no flag required.
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.example:9092")
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=EnumProbeState.AUTHORITATIVE,
                reason="stub healthy",
                backend_label="event_bus_kafka",
            ),
        )
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

    def test_run_delegate_falls_back_to_inmemory_when_broker_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OMN-14376/OMN-14380: a configured but unreachable/unhealthy broker
        # (e.g. an off-box caller hitting an advertised-listener gap) must
        # degrade gracefully to inmemory, never hang the CLI on a broken
        # default.
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "unreachable.example:9092")
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=EnumProbeState.DISCOVERED,
                reason="TCP connect to unreachable.example:9092 failed",
                backend_label="event_bus_kafka",
            ),
        )
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
    """Direct unit coverage of the OMN-14376 probe-then-select seam.

    ``resolve_default_bus`` is the function ``run_delegate`` calls whenever
    ``--bus`` is omitted; these tests exercise it in isolation from the rest
    of the CLI wiring. ``resolve_default_bus`` never reads
    ``KAFKA_BOOTSTRAP_SERVERS`` itself — it delegates entirely to
    :func:`omnibase_infra.backends.backend_probe.probe_kafka` (the existing,
    already-approved boundary for that env lookup; ``cli_delegate.py`` is not
    on the ``check-env-reads`` allowlist). Tests that exercise the
    configured-broker path stub ``probe_kafka`` so no real network call is
    made from the unit suite; the "nothing configured" test calls the REAL
    ``probe_kafka`` because its own short-circuit (no env, no override) never
    touches the network either.

    OMN-16678: the stub target moved from ``cli_delegate.probe_kafka`` to
    ``auto_configure.probe_kafka``. ``resolve_default_bus`` is now a thin call
    into the shared ``resolve_bus_type`` authority, so ``cli_delegate`` no
    longer imports the probe at all and patching the old name would be a
    silent no-op. The generic resolution-order and indeterminate-probe pins
    live in ``tests/unit/backends/test_bus_resolution_order.py``; what this
    class pins is that the DELEGATE path is wired to that authority and passes
    the delegation topic through.
    """

    def test_no_bootstrap_short_circuits_to_inmemory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No stub: probe_kafka's own "no bootstrap configured" branch
        # short-circuits before any socket call, so this stays a fast,
        # deterministic unit test against the real function.
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        bus, reason = resolve_default_bus()

        assert bus == "inmemory"
        assert "not set" in reason

    def test_healthy_broker_resolves_kafka(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.example:9092")
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=EnumProbeState.HEALTHY,
                reason="stub healthy",
                backend_label="event_bus_kafka",
            ),
        )

        bus, reason = resolve_default_bus()

        assert bus == "kafka"
        assert reason == "stub healthy"

    def test_authoritative_broker_resolves_kafka(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.example:9092")
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=EnumProbeState.AUTHORITATIVE,
                reason="stub authoritative",
                backend_label="event_bus_kafka",
            ),
        )

        bus, _reason = resolve_default_bus()

        assert bus == "kafka"

    def test_offbox_healthy_broker_passes_delegation_authority_topic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-16529 regression: the off-box-healthy scenario.

        Live-reproduced (OMN-16529): an off-box LAN caller reaching a
        Tailscale/MagicDNS-fronted broker via its plain LAN IP gets
        ``probe_kafka(...) == HEALTHY`` forever (broker-identity string
        match can never pass — the broker advertises its MagicDNS hostname,
        never the caller's dialed IP) even though a live, ``Stable``
        consumer group is genuinely bound to the delegation-request topic
        and ready to serve. ``resolve_default_bus`` must:
          (a) still select ``kafka`` on the (pre-existing) HEALTHY floor, and
          (b) pass the delegation command topic through as
              ``authority_topic`` so ``probe_kafka`` can resolve the
              *correct* determination (AUTHORITATIVE via consumer-group
              liveness) instead of silently accepting the mislabelled
              HEALTHY-forever off-box symptom.

        Fails under the pre-OMN-16529 ``resolve_default_bus``: it called
        ``probe_kafka(bootstrap_servers=...)`` with no topic argument at
        all, so this assertion on the received kwarg would find nothing to
        check the fix against.
        """
        # kafka-fallback-ok: live-reproduced OMN-16529 fixture, not a real fallback
        offbox_broker = "192.168.86.201:19092"  # kafka-fallback-ok
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", offbox_broker)
        seen: dict[str, object] = {}

        def _probe(
            *, bootstrap_servers: str, authority_topic: str | None = None
        ) -> ModelProbeResult:
            seen["bootstrap_servers"] = bootstrap_servers
            seen["authority_topic"] = authority_topic
            # Mirrors the live off-box symptom exactly: HEALTHY, never
            # AUTHORITATIVE, via the broker-mismatch reason text.
            return ModelProbeResult(
                state=EnumProbeState.HEALTHY,
                reason="Kafka reachable with 1621 topics but broker mismatch",
                backend_label="event_bus_kafka",
            )

        monkeypatch.setattr(auto_configure, "probe_kafka", _probe)

        bus, reason = resolve_default_bus()

        assert bus == "kafka"
        assert seen["authority_topic"] == SUFFIX_DELEGATION_REQUEST
        assert "broker mismatch" in reason

    def test_unreachable_broker_falls_back_to_inmemory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DISCOVERED is the determinate negative (TCP connect refused / no
        # broker configured / unparseable address): a conclusive "no usable
        # broker", so degrading to the in-process bus is repeatable. Bootstrap
        # passed explicitly (not via env) so the stub receives it directly —
        # resolve_default_bus forwards whatever it's given straight through
        # without touching the env itself.
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=EnumProbeState.DISCOVERED,
                reason=f"TCP connect failed for {bootstrap_servers}",
                backend_label="event_bus_kafka",
            ),
        )

        bus, reason = resolve_default_bus(kafka_bootstrap="broker.example:9092")

        assert bus == "inmemory"
        assert EnumProbeState.DISCOVERED.name in reason
        assert "broker.example:9092" in reason

    def test_indeterminate_broker_refuses_instead_of_coin_flipping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-16678 regression: REACHABLE must not resolve a transport.

        Pre-fix this returned ``("inmemory", ...)`` while ``select_event_bus``
        returned a Kafka bus for the identical state — and because a plain
        ``AdminClient.list_topics`` timeout against a HEALTHY broker degrades
        to REACHABLE, the same unchanged environment produced kafka on 14 of
        20 calls and inmemory on the other 6 (``knowledge-base#59``). A
        delegation that silently lands on inmemory never reaches the deployed
        orchestrator consumer, so the failure is data loss, not a slow path.
        """
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=EnumProbeState.REACHABLE,
                reason="TCP reachable but topic list failed: Broker: Request timed out",
                backend_label="event_bus_kafka",
            ),
        )

        with pytest.raises(EventBusResolutionAmbiguousError) as excinfo:
            resolve_default_bus(kafka_bootstrap="broker.example:9092")

        assert "REACHABLE" in str(excinfo.value)
        assert BUS_TYPE_OVERRIDE_ENV in str(excinfo.value)

    @pytest.mark.parametrize(
        ("override", "expected"), [("inmemory", "inmemory"), ("kafka", "kafka")]
    )
    def test_env_override_pins_the_delegate_bus(
        self, override: str, expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-16678 regression for defect 1: the override reaches THIS path.

        ``ONEX_EVENT_BUS_TYPE`` was read by ``select_event_bus`` and ignored
        here, so setting it to pin ``onex delegate`` did nothing at all. It is
        now tier 2 of the one shared order and outranks the probe — which is
        stubbed to the OPPOSITE answer here so a regression that reinstates
        probe-first cannot pass by coincidence.
        """
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, override)
        probe_state = (
            EnumProbeState.AUTHORITATIVE
            if expected == "inmemory"
            else EnumProbeState.DISCOVERED
        )
        monkeypatch.setattr(
            auto_configure,
            "probe_kafka",
            lambda *, bootstrap_servers, authority_topic=None: ModelProbeResult(
                state=probe_state,
                reason="probe result that the override must outrank",
                backend_label="event_bus_kafka",
            ),
        )

        bus, reason = resolve_default_bus(kafka_bootstrap="broker.example:9092")

        assert bus == expected
        assert BUS_TYPE_OVERRIDE_ENV in reason

    def test_explicit_kafka_bootstrap_override_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "env-broker.example:9092")
        seen: dict[str, str] = {}

        def _probe(
            *, bootstrap_servers: str, authority_topic: str | None = None
        ) -> ModelProbeResult:
            seen["bootstrap_servers"] = bootstrap_servers
            return ModelProbeResult(
                state=EnumProbeState.HEALTHY,
                reason="stub healthy",
                backend_label="event_bus_kafka",
            )

        monkeypatch.setattr(auto_configure, "probe_kafka", _probe)

        bus, _reason = resolve_default_bus(kafka_bootstrap="override.example:9092")

        assert bus == "kafka"
        assert seen["bootstrap_servers"] == "override.example:9092"


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
    """OMN-17304 AC1 — ``--bus``/``--kafka-bootstrap`` announce themselves as overrides.

    Both flags are tier 1 of the shared resolution authority
    (``auto_configure.resolve_bus_type``): when either is supplied the CLI
    performs no resolution at all. Before this class, that decision was
    *silent* — every provenance line in ``run_delegate`` lived inside the
    ``if bus is None:`` branch, so an explicit ``--bus kafka`` produced no log
    record whatsoever and nothing downstream (capture file, receipt, operator
    reading stderr) could distinguish "the configured authority selected
    kafka" from "a human typed --bus kafka and the authority was never
    consulted".

    That distinction is the whole point of the flags being overrides. A probe
    result and a typed flag are different kinds of evidence, and a lane-probe
    receipt that cannot tell them apart is the same class of instrument defect
    as OMN-17295.
    """

    _CLI_LOGGER = "omnibase_infra.cli.cli_delegate"

    def _dispatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        **bus_kwargs: object,
    ) -> int:
        """Run ``run_delegate`` with the real bus wiring and a stubbed dispatch."""
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
        # An explicit --bus bypasses resolution entirely (tier 1). Say so.
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
        # The broker address is a second, independent override: --bus names the
        # transport, --kafka-bootstrap names the endpoint. Both are typed, so
        # both must be attributable in the capture.
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
        # The counter-assertion that keeps the label meaningful: when no flag
        # is typed the transport IS resolved, so nothing may claim to be an
        # override. Without this, labelling every run "override" would pass
        # the two tests above while carrying no information.
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)

        with caplog.at_level(logging.INFO, logger=self._CLI_LOGGER):
            assert self._dispatch(tmp_path, monkeypatch) == 0

        assert not self._override_records(caplog)
        # ...and the resolution itself is still announced, so the run is not
        # simply silent about where its transport came from.
        assert any("inmemory" in r.getMessage() for r in caplog.records), (
            "an auto-resolved run logged no transport provenance at all"
        )

    def test_override_provenance_never_reaches_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # stdout is reserved for exactly one ModelSkillResult JSON. Provenance
        # is stderr/capture-only, like every other line in run_delegate.
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
