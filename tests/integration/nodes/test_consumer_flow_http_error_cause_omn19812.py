# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read real HTTP refusals through the C28 collector, grader and record."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    HandlerConsumerFlow,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)

pytestmark = pytest.mark.integration


def _stable_containers(
    argv: list[str], **_kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Keep Docker observations fixed while urllib uses an actual socket."""
    if argv[1] == "inspect":
        output = json.dumps(
            [{"Id": argv[-1], "State": {"Status": "running", "StartedAt": "stable"}}]
        )
    elif argv[1] == "logs":
        output = ""
    else:
        assert argv[1] == "exec" and "describe" in argv
        output = "PARTITION HIGH-WATERMARK\n0 10\n"
    return subprocess.CompletedProcess(argv, 0, output, "")


@pytest.mark.parametrize(
    ("body", "suffix"),
    [
        (
            b'{"status":"degraded","error":"projection_read_failed","reason":"source timeout"}',
            "; server said: status=degraded error=projection_read_failed reason=source timeout",
        ),
        (b"upstream\x07 not ready\n", "; server said: upstream not ready"),
        (b"", "HTTP Error 503: Service Unavailable"),
        (b"x" * 2048 + b"outside-read-bound", "; server said: " + "x" * 200),
    ],
    ids=["json-cause", "printable-text", "empty-body", "bounded-body"],
)
def test_http_refusal_reaches_indeterminate_record(
    tmp_path: Path, body: bytes, suffix: str
) -> None:
    paths: list[str] = []

    class RefusingProjection(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), RefusingProjection) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            record = tmp_path / "c28-http-error.json"
            request = ModelConsumerFlowRequest(
                subject_lane="landing-l15988-lab",
                base_url=f"http://127.0.0.1:{server.server_port}",
                settle_seconds=0,
                scratch=tmp_path,
                record=record,
            )
            target = HandlerDockerConsumerFlowTarget(runner=_stable_containers)
            result = asyncio.run(HandlerConsumerFlow(target=target).handle(request))
        finally:
            server.shutdown()
            thread.join(timeout=5)

    receipt = json.loads(record.read_text())
    assert result.outcome == "INDETERMINATE"
    assert not result.as_lab_proof_check_result().passed
    assert len(result.reasons) == 1
    assert "HTTP Error 503: Service Unavailable" in result.reasons[0]
    assert result.reasons[0].endswith(suffix)
    assert receipt["result"] == result.model_dump(mode="json")
    assert receipt["observation"]["read_ok"] is False
    assert receipt["observation"]["read_error"].endswith(suffix)
    assert paths == ["/projection/onex.snapshot.projection.consumer-flow.v1"]
