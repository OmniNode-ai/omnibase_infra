# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Default output is plain text even when redirected; --json preserves the receipt."""

from __future__ import annotations

import json
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


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture: Path) -> None:
    envelope = json.loads(fixture.read_text(encoding="utf-8"))
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: x\n", encoding="utf-8")
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)

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


@pytest.mark.parametrize("flags", [("--json",)])
def test_forced_json_keeps_the_golden_json_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED)
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


def test_default_prints_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED)
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    expected = _terminal_fields(json.loads(_DISPATCHED.read_text()))[0]
    assert result.stdout == f"{expected}\n"
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_human_flag_forces_the_human_form_off_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED)
    result = _invoke(tmp_path, "--human")
    assert result.exit_code == 0, result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_json_and_human_together_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _DISPATCHED)
    result = _invoke(tmp_path, "--json", "--human")
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_explicit_json_failure_is_the_json_receipt_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _FAILED)
    result = _invoke(tmp_path, "--json")
    assert result.exit_code != 0
    parsed = json.loads(result.stdout.strip())
    assert parsed["exit_code"] != 0
    assert "error_message" in result.stdout


def test_default_failure_is_one_plain_line_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, tmp_path, _FAILED)
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


@pytest.mark.parametrize("flags", [(), ("--human",), ("--json",)])
def test_command_success_uses_real_receipt_mode_and_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    """Exercise diagnostics, renderer and writer through the command, off a TTY."""
    _install_recorded_runtime(monkeypatch, tmp_path, failed=False)
    result = _invoke(tmp_path, *flags)
    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "state" / "runs").iterdir())
    receipt = json.loads((run_dir / "receipt.json").read_text())
    response, _ = _terminal_fields(receipt["receipt"])
    assert (run_dir / "result.txt").read_text().strip() == response
    artifacts = (
        "delegate artifacts: "
        + " ".join(
            str(run_dir / name) for name in ("result.txt", "receipt.json", "run.json")
        )
        + f" state_root={run_dir.parent.parent}"
    )
    if flags == ("--json",):
        assert json.loads(result.stdout) == receipt["receipt"]
        assert artifacts in result.stderr
    else:
        assert result.stdout == f"{response}\n{artifacts}\n"
        assert artifacts not in result.stderr
    for diagnostic in ("task class:", "ticket:", "transport:", "onex-runtime:"):
        assert diagnostic not in result.stdout
    assert "task class:" in result.stderr
    assert "ticket:" in result.stderr
    assert "transport:" in result.stderr


@pytest.mark.parametrize("flags", [(), ("--json",)])
def test_command_failure_prints_cause_reason_and_distinct_error_before_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    _install_recorded_runtime(monkeypatch, tmp_path, failed=True)
    result = _invoke(tmp_path, *flags)
    assert result.exit_code != 0, result.output
    lines = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("onex delegate failed:")
    ]
    assert len(lines) == 1
    assert "no_provider_key" in lines[0]
    assert "no model configured" in lines[0]
    assert "Set PROVIDER_KEY to enable this backend" in lines[0]
    if flags:
        parsed = json.loads(result.stdout)
        assert parsed["exit_code"] != 0
        assert result.output.index(lines[0]) < result.output.index(
            result.stdout.strip()
        )
    else:
        assert result.stdout == ""
    run_dir = next((tmp_path / "state" / "runs").iterdir())
    receipt = json.loads((run_dir / "receipt.json").read_text())
    assert receipt["terminal_failure_cause"] == "no_provider_key"


def _install_recorded_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, failed: bool
) -> None:
    """Stand in only for model execution; keep the receipt mode and file writer real."""
    from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
    from omnibase_infra.cli import receipt_mode

    contract = tmp_path / "node_delegate_skill_orchestrator" / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text("name: node_delegate_skill_orchestrator\n")
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

    class RecordedRuntime:
        exit_code = 1 if failed else 0
        handler_result = None

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def run(self) -> EnumWorkflowResult:
            request = json.loads(Path(str(self.kwargs["input_path"])).read_text())
            correlation_id = request["correlation_id"]
            envelope = json.loads(_DISPATCHED.read_text())
            terminal = envelope["result"]["terminal_payload"]
            terminal["correlation_id"] = correlation_id
            terminal["payload"]["correlation_id"] = correlation_id
            if failed:
                terminal["payload"].update(
                    attempts=[],
                    response="",
                    status="failed",
                    terminal_failure_cause="no_provider_key",
                    terminal_failure_reason="no model configured",
                    error_message="Set PROVIDER_KEY to enable this backend",
                )
            state = Path(str(self.kwargs["state_root"]))
            state.mkdir(parents=True, exist_ok=True)
            (state / "workflow_result.json").write_text(
                json.dumps(
                    {
                        "run_id": str(self.kwargs["run_id"]),
                        "wire_correlation_id": correlation_id,
                        "terminal_payload": terminal,
                    }
                )
            )
            return EnumWorkflowResult.FAILED if failed else EnumWorkflowResult.COMPLETED

    monkeypatch.setattr(receipt_mode, "_runtime_factory", lambda _s: RecordedRuntime)
