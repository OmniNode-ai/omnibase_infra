# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``onex node --output receipt`` (OMN-13094).

Acceptance is STRUCTURAL, not size-based (plan Open Question 3 resolution):

- stdout parses as exactly ONE JSON object validating against
  ``ModelSkillResult`` — zero intermediate context leaks;
- zero RuntimeLocal INFO lines on stdout/stderr;
- the full capture log and handler result are content-addressed in the
  artifact store and hash-verified on retrieval;
- ``artifact.captured`` + ``tool.output.captured`` are emitted via the emit
  daemon socket, or spooled locally when the daemon is unreachable;
- ONEX_ARTIFACT_STORE_ROOT defaults to ``<state_root>/artifacts`` when unset
  so durable capture succeeds without a pre-exported env var (OMN-13537);
- a genuine (non-env) artifact-write failure prints the FULL output instead of
  a receipt (no hidden loss);
- node failure produces ``status=failed`` with the full error output inline.

These tests run the REAL dispatch path: the actual click command through the
actual RuntimeLocal with the in-memory bus and the committed proof-noop
fixture handler (no mocked runtime).
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_core.artifacts.artifact_store import ArtifactStore
from omnibase_core.models.artifacts.model_artifact_ref import ModelArtifactRef
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli.cli_node import run_node_by_name
from omnibase_infra.cli.receipt_mode import CAPTURE_DIR_NAME, SPOOL_DIR_NAME

pytestmark = pytest.mark.unit

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

_BROKEN_CONTRACT = (
    "---\n"
    "name: broken_node\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.broken-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.does_not_exist_module\n"
    "  class: HandlerDoesNotExist\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.does_not_exist_module:HandlerDoesNotExist\n"
)


def _write_fixture_inputs(tmp_path: Path, contract_text: str) -> tuple[Path, Path]:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(contract_text, encoding="utf-8")
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"name": "receipt-proof", "count": 3}), encoding="utf-8"
    )
    return contract_path, input_path


def _invoke_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract_text: str = _PROOF_NOOP_CONTRACT,
    store_root: Path | None | str = "default",
    socket_path: Path | None = None,
) -> tuple[Result, Path]:
    """Run the real CLI in receipt mode against the proof-noop fixture."""
    contract_path, input_path = _write_fixture_inputs(tmp_path, contract_text)
    state_root = tmp_path / "state"

    if store_root == "default":
        store_root = tmp_path / "artifacts"
    if store_root is None:
        monkeypatch.delenv("ONEX_ARTIFACT_STORE_ROOT", raising=False)
    else:
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(store_root))

    resolved_socket = socket_path or (tmp_path / "no-daemon.sock")

    runner = CliRunner()
    result = runner.invoke(
        run_node_by_name,
        [
            "proof_noop",
            "--contract",
            str(contract_path),
            "--input",
            str(input_path),
            "--state-root",
            str(state_root),
            "--output",
            "receipt",
            "--emit-socket",
            str(resolved_socket),
        ],
        catch_exceptions=False,
    )
    return result, state_root


def _parse_single_receipt(stdout: str) -> dict[str, object]:
    """stdout must be exactly one JSON object — anything else is a leak."""
    stripped = stdout.strip()
    parsed: object = json.loads(stripped)  # raises if anything else leaked
    assert isinstance(parsed, dict)
    # Round-trip guard: the single line IS the whole stdout payload.
    assert stripped.startswith("{") and stripped.endswith("}")
    assert "\n" not in stripped, "receipt must be a single JSON line"
    return parsed


class TestReceiptModeSuccess:
    def test_stdout_is_exactly_one_validated_skill_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = _invoke_receipt(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output
        payload = _parse_single_receipt(result.stdout)
        receipt = ModelSkillResult.model_validate(payload)
        assert receipt.node_name == "proof_noop"
        assert receipt.status.is_success_like
        assert receipt.exit_code == 0
        # The FULL handler result travels in the receipt, untruncated.
        assert payload["result"] == {
            "status": "success",
            "echoed_name": "receipt-proof",
            "echoed_count": 3,
        }
        assert str(payload["result_model"]).endswith("ModelProofNoopResult")
        assert receipt.artifact_refs, "capture log must be artifact-backed"

    def test_zero_runtime_log_lines_on_stdout_or_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, state_root = _invoke_receipt(tmp_path, monkeypatch)
        assert result.exit_code == 0
        combined = result.stdout + result.stderr
        assert "RuntimeLocal" not in combined
        assert "INFO" not in combined
        # The runtime stream went to the run_id-suffixed capture file instead.
        captures = list((state_root / CAPTURE_DIR_NAME).glob("proof_noop-*.log"))
        assert len(captures) == 1
        capture_text = captures[0].read_text(encoding="utf-8")
        assert "RuntimeLocal" in capture_text

    def test_artifacts_are_retrievable_and_hash_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, state_root = _invoke_receipt(tmp_path, monkeypatch)
        assert result.exit_code == 0
        payload = _parse_single_receipt(result.stdout)
        refs = payload["artifact_refs"]
        assert isinstance(refs, list) and len(refs) == 2  # capture log + result

        store = ArtifactStore()
        blobs = []
        for raw_ref in refs:
            assert isinstance(raw_ref, dict)
            ref = ModelArtifactRef.model_validate(raw_ref)
            # read_blob re-hashes and raises on mismatch — retrieval IS the
            # hash verification.
            blobs.append(store.read_blob(ref))
            meta = store.read_meta(ref)
            assert meta.source_system == "onex_cli"

        capture_file = next((state_root / CAPTURE_DIR_NAME).glob("proof_noop-*.log"))
        assert capture_file.read_bytes() in blobs
        assert json.dumps(payload["result"]).encode("utf-8") in blobs

    def test_events_spooled_when_daemon_unreachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, state_root = _invoke_receipt(tmp_path, monkeypatch)
        assert result.exit_code == 0
        # Receipt still printed — telemetry failure never re-floods the caller.
        _parse_single_receipt(result.stdout)

        spool_dir = state_root / SPOOL_DIR_NAME
        artifact_events = sorted(spool_dir.glob("artifact-captured-*.json"))
        tool_events = sorted(spool_dir.glob("tool-output-captured-*.json"))
        assert len(artifact_events) == 2
        assert len(tool_events) == 1

        for spool_file in artifact_events:
            record = json.loads(spool_file.read_text(encoding="utf-8"))
            assert record["event_type"] == "artifact.captured"
            for field in (
                "artifact_ref",
                "artifact_hash",
                "artifact_size_bytes",
                "artifact_kind",
                "source_system",
                "correlation_id",
            ):
                assert field in record["payload"], field

        tool_record = json.loads(tool_events[0].read_text(encoding="utf-8"))
        assert tool_record["event_type"] == "tool.output.captured"
        for field in ("tool_name", "suppression_decision", "correlation_id"):
            assert field in tool_record["payload"], field
        assert tool_record["payload"]["suppression_decision"] == "receipt_mode"

    def test_events_emitted_when_daemon_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # UDS paths are capped (~104 chars on macOS); pytest tmp_path under
        # xdist can exceed that, so the socket gets its own short tempdir.
        socket_dir = Path(tempfile.mkdtemp(prefix="omn13094-"))
        socket_path = socket_dir / "emit.sock"
        if len(str(socket_path)) > 100:
            pytest.skip("temp dir too long for a Unix domain socket")

        received: list[dict[str, object]] = []
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(8)
        server.settimeout(10.0)
        stop = threading.Event()

        def serve() -> None:
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except (TimeoutError, OSError):
                    return
                with conn:
                    data = b""
                    while not data.endswith(b"\n"):
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    if data:
                        received.append(json.loads(data.decode("utf-8")))
                    conn.sendall(b'{"status": "queued", "event_id": "test"}\n')

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            result, state_root = _invoke_receipt(
                tmp_path, monkeypatch, socket_path=socket_path
            )
        finally:
            stop.set()
            server.close()
            thread.join(timeout=5.0)
            shutil.rmtree(socket_dir, ignore_errors=True)

        assert result.exit_code == 0
        _parse_single_receipt(result.stdout)
        event_types = sorted(str(r["event_type"]) for r in received)
        assert event_types == [
            "artifact.captured",
            "artifact.captured",
            "tool.output.captured",
        ]
        # Daemon accepted everything — nothing may be spooled.
        spool_dir = state_root / SPOOL_DIR_NAME
        assert not spool_dir.exists() or not list(spool_dir.iterdir())


class TestArtifactStoreRootDefault:
    """OMN-13537: ONEX_ARTIFACT_STORE_ROOT defaults to <state_root>/artifacts.

    Before this fix the unset env var raised KeyError in the store constructor
    and the receipt path failed OPEN — receipts were silently never captured.
    """

    def test_unset_env_captures_under_state_root_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset ONEX_ARTIFACT_STORE_ROOT ⇒ default-resolve, receipt captured."""
        result, state_root = _invoke_receipt(tmp_path, monkeypatch, store_root=None)
        assert result.exit_code == 0, result.output
        # No fail-open: the KeyError line must NOT appear.
        assert "artifact capture failed" not in result.stderr
        assert "no hidden loss" not in result.stderr
        # Exactly one receipt JSON on stdout (capture succeeded).
        payload = _parse_single_receipt(result.stdout)
        receipt = ModelSkillResult.model_validate(payload)
        assert receipt.status.is_success_like
        assert receipt.artifact_refs, "capture must be artifact-backed via default"

        # Artifacts physically landed under the defaulted <state_root>/artifacts.
        default_store_root = state_root / "artifacts"
        assert default_store_root.is_dir()
        store = ArtifactStore()  # reads the env var the CLI just defaulted
        assert store.root == default_store_root.resolve()
        for raw_ref in payload["artifact_refs"]:  # type: ignore[union-attr]
            assert isinstance(raw_ref, dict)
            ref = ModelArtifactRef.model_validate(raw_ref)
            store.read_blob(ref)  # re-hashes; raises on mismatch
            assert store.read_meta(ref).source_system == "onex_cli"

    def test_explicit_env_override_wins_over_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ONEX_ARTIFACT_STORE_ROOT is honored, not the default."""
        explicit_root = tmp_path / "operator-chosen-artifacts"
        result, state_root = _invoke_receipt(
            tmp_path, monkeypatch, store_root=explicit_root
        )
        assert result.exit_code == 0, result.output
        _parse_single_receipt(result.stdout)
        # Artifacts went to the operator path, NOT the state-root default.
        assert explicit_root.is_dir()
        assert list(explicit_root.iterdir()), "explicit store root must hold blobs"
        default_store_root = state_root / "artifacts"
        assert not default_store_root.exists(), "default must not be used on override"


class TestReceiptModeFailureAsymmetry:
    def test_genuine_write_failure_prints_full_output_not_receipt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real (non-env) artifact write failure ⇒ FULL output, no receipt.

        Fail-loud is preserved for genuine write errors: pointing the store
        root at an existing regular FILE makes the store's shard ``mkdir`` raise
        ``NotADirectoryError`` (an ``OSError``), which must surface the full
        output (no hidden loss) rather than a receipt.
        """
        store_file = tmp_path / "store-is-a-file"
        store_file.write_text("not a directory", encoding="utf-8")
        result, _ = _invoke_receipt(tmp_path, monkeypatch, store_root=store_file)
        # The run itself succeeded; capture failed for a real reason.
        assert result.exit_code == 0
        assert "artifact capture failed" in result.stderr
        assert "no hidden loss" in result.stderr
        # Full capture stream passes through instead of a receipt.
        assert "RuntimeLocal" in result.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout.strip())

    def test_node_failure_reports_failed_with_error_inline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = _invoke_receipt(
            tmp_path, monkeypatch, contract_text=_BROKEN_CONTRACT
        )
        assert result.exit_code != 0
        payload = _parse_single_receipt(result.stdout)
        receipt = ModelSkillResult.model_validate(payload)
        assert not receipt.status.is_success_like
        result_body = payload["result"]
        assert isinstance(result_body, dict)
        # Errors are never hidden: the full capture log travels inline.
        assert result_body["capture_log"], "failure must inline the capture log"
        assert str(payload["result_model"]).endswith("ModelReceiptRuntimeSummary")
        # The capture log is ADDITIONALLY artifact-backed.
        assert payload["artifact_refs"]


class TestDefaultModeUnchanged:
    def test_default_mode_does_not_print_receipt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_path, input_path = _write_fixture_inputs(
            tmp_path, _PROOF_NOOP_CONTRACT
        )
        monkeypatch.delenv("ONEX_ARTIFACT_STORE_ROOT", raising=False)
        runner = CliRunner()
        result = runner.invoke(
            run_node_by_name,
            [
                "proof_noop",
                "--contract",
                str(contract_path),
                "--input",
                str(input_path),
                "--state-root",
                str(tmp_path / "state"),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # No single-JSON receipt contract in default mode (behavior preserved).
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout.strip() or "not-json")
