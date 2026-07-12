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
import signal
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    BUS_CHOICES,
    DEFAULT_BUS,
    DEFAULT_TASK_TYPE,
    DELEGATE_SOURCE,
    DelegateTimeoutExceededError,
    build_backend_overrides,
    classify_task_type,
    delegate_command,
    resolve_default_bus,
    run_delegate,
)

pytestmark = pytest.mark.unit

KAFKA_BOOTSTRAP_ARG = "$KAFKA_BOOTSTRAP_SERVERS"


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
            cli_delegate,
            "probe_kafka",
            lambda *, bootstrap_servers: ModelProbeResult(
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
            cli_delegate,
            "probe_kafka",
            lambda *, bootstrap_servers: ModelProbeResult(
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
            cli_delegate,
            "probe_kafka",
            lambda *, bootstrap_servers: ModelProbeResult(
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
            cli_delegate,
            "probe_kafka",
            lambda *, bootstrap_servers: ModelProbeResult(
                state=EnumProbeState.AUTHORITATIVE,
                reason="stub authoritative",
                backend_label="event_bus_kafka",
            ),
        )

        bus, _reason = resolve_default_bus()

        assert bus == "kafka"

    @pytest.mark.parametrize(
        "state", [EnumProbeState.DISCOVERED, EnumProbeState.REACHABLE]
    )
    def test_unhealthy_broker_falls_back_to_inmemory(
        self, state: EnumProbeState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DISCOVERED (TCP failed) and REACHABLE (TCP ok, but e.g. topic list
        # failed — the OMN-14380 advertised-listener symptom) both must
        # gracefully degrade rather than select a bus that cannot actually
        # carry traffic. Bootstrap passed explicitly (not via env) so the
        # stub receives it directly — resolve_default_bus forwards whatever
        # it's given straight to probe_kafka without touching the env itself.
        monkeypatch.setattr(
            cli_delegate,
            "probe_kafka",
            lambda *, bootstrap_servers: ModelProbeResult(
                state=state,
                reason=f"TCP reachable but topic list failed for {bootstrap_servers}",
                backend_label="event_bus_kafka",
            ),
        )

        bus, reason = resolve_default_bus(kafka_bootstrap="broker.example:9092")

        assert bus == "inmemory"
        assert state.name in reason
        assert "broker.example:9092" in reason

    def test_explicit_kafka_bootstrap_override_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "env-broker.example:9092")
        seen: dict[str, str] = {}

        def _probe(*, bootstrap_servers: str) -> ModelProbeResult:
            seen["bootstrap_servers"] = bootstrap_servers
            return ModelProbeResult(
                state=EnumProbeState.HEALTHY,
                reason="stub healthy",
                backend_label="event_bus_kafka",
            )

        monkeypatch.setattr(cli_delegate, "probe_kafka", _probe)

        bus, _reason = resolve_default_bus(kafka_bootstrap="override.example:9092")

        assert bus == "kafka"
        assert seen["bootstrap_servers"] == "override.example:9092"
