# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 incident replay -- the lab lane had no record and nothing said so.

THE INCIDENT. ``runtime-rebuild-trigger.yml`` run ``34657547387``, on the merge of
``omnibase_infra#3441`` (squash ``71e3da6c``), reported **success**. Its verify job
was SKIPPED, no lab-pass receipt artifact existed for the sha on any lane, and the
k3s ``onex-lab`` lane was pinned to an image stamped 2026-09-08 running
``omnimarket==0.4.30`` against a ``dev`` at 0.4.61. The lane had not been
re-applied across many merges, because ``k8s/onex-lab/apply_lab_lane.sh`` had no
caller anywhere in the org.

THE REGRESSION CLASS IS ``false_green``, and the false green is the absence of any
read at all: there was no step that asked whether the lab lane had been advanced,
so there was nothing to answer wrongly. A guard whose only test fed it a record it
had itself constructed would pass just as happily while the live surface held
nothing.

THE ARTIFACT IS THE LIVE SURFACE'S OWN ANSWER for that exact sha:
``GET /lab-overlay/71e3da6c…`` against the dev-lane deploy agent on the lab host,
captured 2026-09-12T01:19:40Z, headers and body verbatim. Not a reconstruction --
it is the response the reader would have received on the night of the incident,
and it is ``404``, because no lab-overlay record for that commit has ever existed.

The replay serves those bytes back byte-for-byte over a real socket and drives the
real ``poll`` through its real transport, and requires a FAILING check that names
the sha. A reader that treated "no record" as "nothing to report" would satisfy
the old behaviour and fail here.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn18200"
    / "agent-lab-overlay-71e3da6c.http.captured"
)
INCIDENT_SHA = "71e3da6c189a2509d6247054404af320a7e5698d"
EXPECTED_SHA256 = "b0b6723d5b2cf4270b35a3c25cf70153c6f1b60016d5b1b8ecb05f05b341f009"


def _reader_module() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        import fetch_lab_overlay_record

        return fetch_lab_overlay_record
    finally:
        sys.path.pop(0)


def _parse_capture() -> tuple[int, bytes, str]:
    """Split the captured response into status, body and content type.

    Parsed rather than hand-transcribed so the served bytes are the captured
    bytes; a test that retyped the body would be replaying its own authorship.
    """
    raw = FIXTURE.read_bytes()
    head, _, body = raw.partition(b"\r\n\r\n")
    if not head.endswith(b"\r\n") and b"\n\n" in raw:
        head, _, body = raw.partition(b"\n\n")
    lines = head.decode("utf-8").splitlines()
    status = int(lines[0].split()[1])
    content_type = "text/plain"
    for line in lines[1:]:
        if line.lower().startswith("content-type:"):
            content_type = line.split(":", 1)[1].strip()
    return status, body, content_type


def test_the_captured_artifact_is_unmodified() -> None:
    """R1. Editing a capture after the fact breaks the "this is what happened"
    claim, so it must break the build."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXPECTED_SHA256, (
        f"{FIXTURE} has been modified since capture"
    )


@pytest.fixture
def replay_server() -> Any:
    """Serve the captured response for any path, on a real socket."""
    status, body, content_type = _parse_capture()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_the_real_reader_fails_the_sha_whose_lab_lane_was_never_applied(
    replay_server: str,
) -> None:
    """``guard_verdict_on_artifact: reject``.

    The live agent answered 404 for ``71e3da6c`` because no lab-overlay record for
    that commit has ever existed. The reader must turn that into a failing check
    that names the sha, so the receipt emitted from it is a FAIL -- never a silent
    absence, which is what the trigger's green run already was.
    """
    reader = _reader_module()
    checks = reader.poll(
        base_url=replay_server,
        sha=INCIDENT_SHA,
        wait_seconds=0,
        poll_interval_seconds=1,
        request_timeout_seconds=5.0,
        out=sys.stderr,
    )

    assert len(checks) == 1
    check = checks[0]
    assert check["name"] == "lab_overlay_record"
    assert check["ok"] is False
    assert INCIDENT_SHA in check["evidence"]
    assert "404" in check["evidence"]
    # Evidence required on a failure too: a verdict with nothing behind it is
    # the shape the whole receipt contract refuses.
    assert "never returned a record" in check["evidence"]


def test_the_receipt_built_from_that_check_is_a_FAIL(replay_server: str) -> None:
    """The verdict is DERIVED, so the reader's failing check must be capable of
    producing nothing but a FAIL receipt once it reaches the emitter."""
    reader = _reader_module()
    checks = reader.poll(
        base_url=replay_server,
        sha=INCIDENT_SHA,
        wait_seconds=0,
        poll_interval_seconds=1,
        request_timeout_seconds=5.0,
        out=sys.stderr,
    )

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        from lab_pass_receipt import (
            EnumLabLane,
            EnumLabPassResult,
            ModelLabPassCheck,
            build_receipt,
        )
    finally:
        sys.path.pop(0)

    receipt = build_receipt(
        sha=INCIDENT_SHA,
        lane=EnumLabLane.ONEX_LAB_K3S,
        started_at=_ts("2026-09-11T23:20:26Z"),
        finished_at=_ts("2026-09-11T23:50:26Z"),
        checks=[ModelLabPassCheck(**check) for check in checks],
        agent_command_id=None,
    )
    assert receipt.result is EnumLabPassResult.FAIL
    assert json.loads(receipt.to_json())["sha"] == INCIDENT_SHA


def _ts(value: str) -> Any:
    from datetime import UTC, datetime

    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
