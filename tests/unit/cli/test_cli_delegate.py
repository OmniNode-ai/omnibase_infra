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
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    BUS_CHOICES,
    DEFAULT_BUS,
    DEFAULT_TASK_TYPE,
    DELEGATE_SOURCE,
    build_backend_overrides,
    classify_task_type,
    delegate_command,
    run_delegate,
)

pytestmark = pytest.mark.unit

KAFKA_BOOTSTRAP_ARG = "$KAFKA_BOOTSTRAP_SERVERS"

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
        # requires from a consumer: prompt, task_type, source, max_tokens.
        # omnibase_infra does NOT depend on omnimarket (layering), so the node
        # owns model validation at dispatch; the CLI's contract here is the
        # payload shape. (When no override is supplied, max_tokens is omitted —
        # see test_payload_written_under_state_root_tmp_not_slash_tmp, OMN-13161.)
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
    """The CLI can target the live bus (OMN-13532 / OMN-13408 re-proof).

    ``run_delegate`` no longer hardcodes the in-memory bus: ``--bus`` /
    ``--kafka-bootstrap`` flow through ``backend_overrides`` to ``RuntimeLocal``
    so the typed delegate-skill command can be published to the live broker
    where a deployed consumer dispatches it (``feedback_bus_is_the_transport``).
    """

    def test_choices_mirror_runtime_supported_values(self) -> None:
        # The CLI must not advertise a bus the runtime rejects, nor omit one it
        # supports — RuntimeLocal is the source of truth.
        from omnibase_core.runtime.runtime_local import SUPPORTED_EVENT_BUS_VALUES

        assert set(BUS_CHOICES) == set(SUPPORTED_EVENT_BUS_VALUES)
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
