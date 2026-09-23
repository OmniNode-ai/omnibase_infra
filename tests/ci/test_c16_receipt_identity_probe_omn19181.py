# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C16 receipt-identity producer (OMN-19181).

Every test here runs against RECORDED observations under
``tests/fixtures/omn19181/`` -- captured read-only from the .201 dev lane's
gateway rows and the orchestrator's bus terminal, provenance in each file -- and performs no
network or database I/O, so the verdict is falsifiable on a laptop and in CI
rather than only on the lab.

The three captured fixtures are real: a completed canary run with a typed dead
run (PASS), a run that died on a provider 429 with NO class or code, and a run
still reading ``published`` two days after submission. The remaining cases are
single-field mutations of the passing capture, each named for the one thing it
changes, so a red here always points at one clause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c16_receipt_identity_probe as probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn19181"
CAPTURED = (
    "lane_healthy_typed_death.json",
    "lane_dead_run_untyped_cause.json",
    "lane_dead_run_never_terminal.json",
)


def _load(name: str = "lane_healthy_typed_death.json") -> dict[str, Any]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _outcomes(payload: dict[str, Any]) -> dict[str, str]:
    record = probe.grade(probe.observations_from_replay(payload))
    return {r.probe: r.outcome for r in record.results}


@pytest.mark.unit
def test_the_three_probes_are_the_criterions_own_and_all_are_graded() -> None:
    assert probe.PROBES == ("R-DELEG-11", "R-DELEG-12", "R-DELEG-26")
    record = probe.grade(probe.observations_from_replay(_load()))
    assert tuple(r.probe for r in record.results) == probe.PROBES


@pytest.mark.unit
@pytest.mark.parametrize("name", CAPTURED)
def test_each_captured_lane_record_grades_as_its_provenance_says(name: str) -> None:
    payload = _load(name)
    assert _outcomes(payload) == payload["expected"]


@pytest.mark.unit
def test_healthy_lane_data_can_pass_and_exits_zero() -> None:
    """The control: a check that cannot pass is worse than one that cannot fail."""
    record = probe.grade(probe.observations_from_replay(_load()))
    assert record.verdict == "pass"
    assert record.exit_code == probe.EXIT_OK


@pytest.mark.unit
def test_skip_skip_pass_is_red_and_never_a_pass() -> None:
    """AC4's falsifier: two probes whose subject never materialised, one PASS."""
    payload = _load()
    healthy = payload["observations"]["healthy"]
    healthy["receipt"] = None
    healthy["status"] = {"status": "published"}
    healthy["error"] = "budget spent before a terminal status"
    record = probe.grade(probe.observations_from_replay(payload))
    outcomes = {r.probe: r.outcome for r in record.results}
    assert outcomes == {
        "R-DELEG-11": "SKIP",
        "R-DELEG-12": "SKIP",
        "R-DELEG-26": "PASS",
    }
    assert record.verdict == "fail"
    assert record.exit_code == probe.EXIT_FINDINGS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("side", "key", "value"),
    [
        # `route` is presence-checked only: the bus terminal carries a backend
        # UUID, not a route NAME, so there is nothing independent to compare
        # it with. Pinned below so that limit cannot quietly become a claim.
        ("receipt", "provider", "gemini"),
        ("receipt", "terminal_model_used", "claude-opus-4-6"),
    ],
)
def test_a_receipt_naming_a_route_the_bus_terminal_does_not_show_fails(
    side: str, key: str, value: str
) -> None:
    payload = _load()
    payload["observations"]["healthy"][side][key] = value
    result = probe.grade(probe.observations_from_replay(payload)).results[1]
    assert result.probe == "R-DELEG-12"
    assert result.outcome == "FAIL"
    assert any(repr(value) in reason for reason in result.reasons)


@pytest.mark.unit
def test_a_receipt_with_no_route_fails_even_when_provider_and_model_agree() -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["route"] = None
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"


@pytest.mark.unit
def test_two_blank_providers_are_not_an_identity() -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["provider"] = None
    payload["observations"]["healthy_terminal"]["payload"]["provider"] = None
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("provider", "gemini"),
        ("model_name", "gemini-2.5-flash"),
        ("status", "failed"),
    ],
)
def test_each_bus_observation_is_compared_not_just_one(key: str, value: str) -> None:
    payload = _load()
    payload["observations"]["healthy_terminal"]["payload"][key] = value
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"


@pytest.mark.unit
def test_the_accepted_attempt_is_the_one_compared_not_the_first() -> None:
    """A re-route answers from its accepted attempt, never an abandoned one."""
    payload = _load()
    attempts = payload["observations"]["healthy_terminal"]["payload"]["attempts"]
    assert [a["acceptance_decision"] for a in attempts] == ["climb", "accept"]
    attempts[0]["model_id"] = "gemini-2.5-flash"
    assert _outcomes(payload)["R-DELEG-12"] == "PASS"
    attempts[1]["model_id"] = "gemini-2.5-flash"
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"


@pytest.mark.unit
def test_an_absent_terminal_fails_and_an_unreadable_bus_skips() -> None:
    payload = _load()
    payload["observations"]["healthy_terminal"]["payload"] = None
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"
    payload = _load()
    payload["observations"]["healthy_terminal"]["error"] = (
        "consumer start failed: KafkaConnectionError"
    )
    assert _outcomes(payload)["R-DELEG-12"] == "SKIP"


@pytest.mark.unit
@pytest.mark.parametrize("passed", [None, "skipped", 1])
def test_a_rule_without_a_boolean_verdict_on_a_completed_run_fails(passed: Any) -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["rule_evaluations"][0]["passed"] = (
        passed
    )
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
def test_a_completed_run_with_no_evaluated_rule_fails() -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["rule_evaluations"] = []
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
def test_a_failed_blocking_rule_on_a_completed_run_fails() -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["rule_evaluations"][0]["passed"] = (
        False
    )
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
@pytest.mark.parametrize("surface", ["status", "receipt"])
def test_a_dying_run_that_reports_success_fails(surface: str) -> None:
    payload = _load()
    payload["observations"]["dying"][surface]["status"] = "completed"
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
def test_a_dying_submission_refused_at_ingress_skips_and_is_red() -> None:
    payload = _load()
    dying = payload["observations"]["dying"]
    dying.update(
        submit_status=400, status=None, receipt=None, error="submit answered 400"
    )
    record = probe.grade(probe.observations_from_replay(payload))
    assert {r.probe: r.outcome for r in record.results}["R-DELEG-26"] == "SKIP"
    assert record.exit_code == probe.EXIT_FINDINGS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("terminal_failure_class", "provider said no"),
        ("terminal_failure_code", "required_bar_missing"),
    ],
)
def test_a_cause_outside_the_typed_grammar_fails(key: str, value: str) -> None:
    payload = _load()
    payload["observations"]["dying"]["receipt"][key] = value
    payload["observations"]["dying"]["status"][key] = value
    assert _outcomes(payload)["R-DELEG-26"] == "FAIL"


@pytest.mark.unit
def test_status_and_receipt_disagreeing_on_the_cause_fails() -> None:
    payload = _load()
    payload["observations"]["dying"]["status"]["terminal_failure_code"] = None
    assert _outcomes(payload)["R-DELEG-26"] == "FAIL"


@pytest.mark.unit
def test_the_dying_run_differs_from_the_healthy_one_only_in_task_class() -> None:
    diff = {
        key
        for key in probe.HEALTHY_PAYLOAD
        if probe.HEALTHY_PAYLOAD[key] != probe.DYING_PAYLOAD[key]
    }
    assert diff == {"task_type"}
    assert set(probe.HEALTHY_PAYLOAD) == set(probe.DYING_PAYLOAD)


@pytest.mark.unit
def test_the_header_form_is_pinned() -> None:
    assert probe.HEADER_NAME == "x-api-key"


@pytest.mark.unit
def test_the_record_carries_no_credential_tenant_or_endpoint(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(_load()), encoding="utf-8")
    out = tmp_path / "record.json"
    assert probe.main(["--replay", str(replay), "--record", str(out)]) == probe.EXIT_OK
    text = out.read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["criterion"] == "C16"
    assert [p["probe"] for p in record["probes"]] == list(probe.PROBES)
    for forbidden in ("tenant_id", "endpoint_url", "api_key", "prompt"):
        assert forbidden not in text


@pytest.mark.unit
def test_a_missing_secret_is_an_input_failure_and_still_writes_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMN19181_TEST_UNSET", raising=False)
    out = tmp_path / "record.json"
    code = probe.main(
        [
            "--base-url",
            "http://127.0.0.1:9",
            "--credential-env",
            "OMN19181_TEST_UNSET",
            "--terminal-topic",
            "unused-because-the-credential-is-refused-first",
            "--runner-identity",
            "test",
            "--record",
            str(out),
        ]
    )
    assert code == probe.EXIT_INPUT
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "could_not_run"


@pytest.mark.unit
def test_route_is_presence_checked_only_and_says_so() -> None:
    """The honest limit: a different route NAME with matching provider/model passes."""
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["route"] = "cloud-gemini-pro"
    assert _outcomes(payload)["R-DELEG-12"] == "PASS"
    assert "no route NAME to\n                compare it with" in (probe.__doc__ or "")


class _FakeRecord:
    def __init__(self, value: bytes) -> None:
        self.value = value


class _FakeConsumer:
    """Stands in for AIOKafkaConsumer: one partition, a scripted fetch."""

    batches: list[list[bytes]] = []
    end = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._position = 0
        self._batches = list(type(self).batches)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def assignment(self) -> set[str]:
        return {"p0"}

    async def end_offsets(self, parts: list[str]) -> dict[str, int]:
        return {p: type(self).end for p in parts}

    async def beginning_offsets(self, parts: list[str]) -> dict[str, int]:
        return dict.fromkeys(parts, 0)

    def seek(self, _part: str, offset: int) -> None:
        self._position = offset

    async def position(self, _part: str) -> int:
        return self._position

    async def getmany(self, **_kwargs: Any) -> dict[str, list[_FakeRecord]]:
        if not self._batches:
            return {}
        batch = self._batches.pop(0)
        self._position += len(batch)
        return {"p0": [_FakeRecord(v) for v in batch]}


def _scan(
    monkeypatch: pytest.MonkeyPatch, batches: list[list[bytes]], end: int
) -> probe.TerminalObservation:
    import asyncio
    import sys
    import types

    _FakeConsumer.batches = batches
    _FakeConsumer.end = end
    fake = types.ModuleType("aiokafka")
    fake.AIOKafkaConsumer = _FakeConsumer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiokafka", fake)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.invalid:9092")
    return asyncio.run(
        probe._scan_terminal(
            "terminal-topic", "cid-1", wait_seconds=2.0, max_records=100
        )
    )


@pytest.mark.unit
def test_a_scan_that_reaches_the_watermark_without_a_match_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = json.dumps({"payload": {"correlation_id": "cid-2"}}).encode()
    seen = _scan(monkeypatch, [[other, other]], end=2)
    assert seen.error is None and seen.payload is None


@pytest.mark.unit
def test_a_scan_that_runs_out_of_window_is_unobserved_not_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 35787593146: a slow broker must grade SKIP, never FAIL."""
    seen = _scan(monkeypatch, [], end=5)
    assert seen.payload is None
    assert seen.error is not None and "did not reach the end" in seen.error
    payload = _load()
    payload["observations"]["healthy_terminal"] = {"payload": None, "error": seen.error}
    assert _outcomes(payload)["R-DELEG-12"] == "SKIP"


@pytest.mark.unit
def test_a_scan_that_finds_the_terminal_returns_its_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mine = json.dumps({"payload": {"correlation_id": "cid-1", "provider": "local"}})
    seen = _scan(monkeypatch, [[mine.encode()]], end=1)
    assert seen.payload == {"correlation_id": "cid-1", "provider": "local"}
