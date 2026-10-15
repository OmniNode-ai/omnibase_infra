# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise HTTP probe -> checks JSON -> emit CLI -> typed post-merge event."""

from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

from omnibase_infra.models.lab_pass import ModelPostMergeLabPassReceiptEvent
from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("response", "identity_readable", "verdict", "error_class"),
    [
        (200, True, "PASS", "none"),
        (503, True, "FAIL", "assertion"),
        (None, True, "FAIL", "transport"),
        (200, False, "FAIL", "none"),
    ],
    ids=["ready", "wrong-answer", "no-answer", "unreadable-identity"],
)
def test_post_merge_receipt_event_from_http_through_emit_cli(
    tmp_path: Path,
    response: int | None,
    identity_readable: bool,
    verdict: str,
    error_class: str,
) -> None:
    class ReadyHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if response is None:
                self.close_connection = True
                return
            self.send_response(response)
            self.end_headers()
            self.wfile.write(b"ready" if response == 200 else b"degraded")

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), ReadyHandler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            check = receipts.check_ready(
                "main_ready", f"http://127.0.0.1:{server.server_port}/ready", 2.0
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)

    checks_path = tmp_path / "checks.json"
    checks_path.write_text(json.dumps([check.to_dict()]), encoding="utf-8")
    recorded = json.loads(
        (_ROOT / "tests/fixtures/omn20772/lane_identity_recorded.json").read_text(
            encoding="utf-8"
        )
    )["identity"]
    identity_path = tmp_path / "identity.json"
    if identity_readable:
        identity_path.write_text(json.dumps(recorded), encoding="utf-8")
    sha = recorded["generation"]["revision"]
    events = []
    for attempt in range(2):
        event_path = tmp_path / f"event-{attempt}.json"
        receipt_path = tmp_path / f"receipt-{attempt}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts/ci/lab_pass_receipt.py"),
                "emit",
                "--sha",
                sha,
                "--lane",
                receipts.EnumLabLane.COMPOSE_DEV.value,
                "--started-at",
                "2026-10-15T10:00:00Z",
                "--finished-at",
                "2026-10-15T10:01:00Z",
                "--checks-json",
                str(checks_path),
                "--identity-json",
                str(identity_path),
                "--post-merge-event-out",
                str(event_path),
                "--out",
                str(receipt_path),
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert completed.returncode == (0 if response == 200 else 1), (
            completed.stdout + completed.stderr
        )
        receipt = receipts.parse_receipt(receipt_path.read_text(encoding="utf-8"))
        event = ModelPostMergeLabPassReceiptEvent.model_validate_json(
            event_path.read_text(encoding="utf-8")
        )
        assert event.receipt_key == receipts.artifact_name(receipt.lane, receipt.sha)
        assert event.verdict == verdict
        assert event.receipt_result == ("PASS" if response == 200 else "FAIL")
        assert event.binding_valid is identity_readable
        assert event.checks[0].error_class.value == error_class
        assert event.checks[0].evidence == check.evidence
        dimensions = [] if response == 200 else ["main_ready"]
        if identity_readable:
            assert (
                event.generation.container_ref == recorded["generation"]["container_id"]
            )
            assert event.identity.pins == recorded["pins"]
        else:
            dimensions.append("binding")
            assert "unreadable" in event.binding_reason
        assert set(event.failing_dimensions) == set(dimensions)
        events.append(event)

    assert events[0].event_id == events[1].event_id
