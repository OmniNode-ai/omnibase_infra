# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The output form of ``onex delegate`` follows stdout (OMN-20124).

Programs in other repos parse the one-JSON-line receipt from a subprocess
(the omniclaude delegate skill, the omnimarket probes, the merge-drain
delegation step, the omni_home workflows). An older installed CLI also rejects
``--json``. So the default off a terminal must stay that JSON line, byte for
byte in shape, and only a terminal gets the human form.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "delegation"
_DISPATCHED = _FIXTURES / "omn18569" / "dispatched_envelope_carrier_receipt.json"
_FAILED = _FIXTURES / "omn18306" / "failed_no_accepted_attempt_receipt.json"


class _Receipt:
    def __init__(self, envelope: dict[str, object]) -> None:
        self.envelope = envelope

    def model_dump(self, *, mode: str) -> dict[str, object]:
        return self.envelope

    def model_dump_json(self) -> str:
        return json.dumps(self.envelope, separators=(",", ":"))


def _install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture: Path, *, tty: bool
) -> None:
    envelope = json.loads(fixture.read_text(encoding="utf-8"))
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: x\n", encoding="utf-8")
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)
    monkeypatch.setattr(cli_delegate, "_stdout_is_tty", lambda: tty)

    def _fake(**kwargs: object) -> int:
        receipt = _Receipt(envelope)
        renderer = kwargs.get("receipt_renderer")
        exit_code = int(envelope.get("exit_code") or 0)
        if renderer is None:
            import click

            click.echo(receipt.model_dump_json())
            return exit_code
        assert callable(renderer)
        ok = renderer(receipt)
        return exit_code if ok else max(exit_code, 1)

    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake)


def _invoke(tmp_path: Path, *flags: str):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(
        delegate_command,
        [
            "say hello",
            *flags,
            "--state-root",
            str(tmp_path / "state"),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )


def _terminal_fields(parsed: dict[str, object]) -> tuple[object, object]:
    """The exact path callers read: result.terminal_payload.payload.{response,model_name}."""
    result = parsed["result"]
    assert isinstance(result, dict)
    tp = result["terminal_payload"]
    assert isinstance(tp, dict)
    payload = tp["payload"]
    assert isinstance(payload, dict)
    return payload["response"], payload["model_name"]


@pytest.mark.parametrize("flags", [(), ("--json",)])
def test_non_tty_and_forced_json_keep_the_golden_json_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    fixture_tty = not flags  # the (), non-tty case; --json is forced on a tty
    _install(monkeypatch, tmp_path, _DISPATCHED, tty=not fixture_tty)
    result = _invoke(tmp_path, *flags)
    assert result.exit_code == 0, result.output
    stripped = result.stdout.strip()
    assert "\n" not in stripped
    parsed = json.loads(stripped)
    assert parsed == json.loads(_DISPATCHED.read_text(encoding="utf-8"))
    for key in ("status", "run_id", "result"):
        assert key in parsed
    response, model_name = _terminal_fields(parsed)
    assert isinstance(response, str) and response
    assert isinstance(model_name, str) and model_name


def test_tty_prints_the_human_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED, tty=True)
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert not result.stdout.lstrip().startswith("{")
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_human_flag_forces_the_human_form_off_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED, tty=False)
    result = _invoke(tmp_path, "--human")
    assert result.exit_code == 0, result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_json_and_human_together_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED, tty=False)
    result = _invoke(tmp_path, "--json", "--human")
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_failure_off_a_terminal_is_the_json_receipt_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _FAILED, tty=False)
    result = _invoke(tmp_path)
    assert result.exit_code != 0
    parsed = json.loads(result.stdout.strip())
    assert parsed["exit_code"] != 0
    assert "error_message" in result.stdout


def test_failure_on_a_terminal_is_one_plain_line_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _FAILED, tty=True)
    result = _invoke(tmp_path)
    assert result.exit_code != 0
    assert result.stdout == ""
    failure_lines = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("onex delegate failed: ")
    ]
    assert len(failure_lines) == 1


def test_the_run_delegate_default_is_the_json_contract() -> None:
    """A programmatic caller of run_delegate keeps the JSON line unless it asks."""
    import inspect

    default = inspect.signature(cli_delegate.run_delegate).parameters["json_output"]
    assert default.default is True


def test_stdout_is_tty_is_false_under_the_test_runner() -> None:
    assert cli_delegate._stdout_is_tty() is False
