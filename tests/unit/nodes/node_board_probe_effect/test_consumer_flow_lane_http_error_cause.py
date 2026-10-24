# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A refused projection read names the cause the server put in its body (OMN-19812)."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from typing import IO, Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
    ConsumerFlowInputError,
)

pytestmark = pytest.mark.unit

BASE = "http://projection.test"


def _refusing(code: int, body: bytes | None) -> Any:
    def urlopen(url: str, **_kwargs: Any) -> IO[bytes]:
        fp = io.BytesIO(body) if body is not None else None
        raise urllib.error.HTTPError(url, code, "Service Unavailable", Message(), fp)

    return urlopen


def _message(urlopen: Any) -> str:
    lane = ConsumerFlowLane(docker="fake-docker", base_url=BASE, urlopen=urlopen)
    with pytest.raises(ConsumerFlowInputError) as caught:
        lane.page({})
    return str(caught.value)


def test_degraded_json_body_names_status_and_error() -> None:
    body = json.dumps(
        {"status": "degraded", "error": "projection_read_failed", "topic": "t"}
    ).encode()
    message = _message(_refusing(503, body))
    assert "HTTP Error 503: Service Unavailable" in message
    assert message.endswith(
        "; server said: status=degraded error=projection_read_failed"
    )


def test_reason_field_is_named() -> None:
    body = json.dumps({"status": "degraded", "reason": "exposure disabled"}).encode()
    assert _message(_refusing(503, body)).endswith(
        "; server said: status=degraded reason=exposure disabled"
    )


def test_non_json_body_names_its_printable_text() -> None:
    message = _message(_refusing(503, b"upstream\x07 not ready\n"))
    assert message.endswith("; server said: upstream not ready")


def test_no_body_keeps_the_bare_status_line() -> None:
    message = _message(_refusing(503, None))
    assert message.endswith("HTTPError: HTTP Error 503: Service Unavailable")
    assert "server said" not in message


def test_a_served_page_is_returned_unchanged() -> None:
    page = {"rows": [], "next_cursor": None}

    def urlopen(url: str, **_kwargs: Any) -> IO[bytes]:
        return io.BytesIO(json.dumps(page).encode())

    lane = ConsumerFlowLane(docker="fake-docker", base_url=BASE, urlopen=urlopen)
    assert lane.page({}) == page
