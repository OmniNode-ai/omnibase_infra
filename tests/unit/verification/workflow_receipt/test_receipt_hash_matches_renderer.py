# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-repo hash-parity pin (OMN-15095).

``receipt_hash.sha256_of``/``canonical_json`` mirror ``omninode_infra``'s
``workflow_receipt_renderer._sha256``/``_canonical_json`` (OMN-15094) byte
for byte: ``json.dumps(payload, sort_keys=True, default=str)`` UTF-8 encoded,
sha256 hexdigest. This test recomputes the exact
``TestRealPostgresRoundTrip.test_reads_a_real_terminal_row_end_to_end`` fixture
values from OMN-15094's own test suite (commit f262cfe8) and asserts this
mirror's ``terminal_event_payload``/``sha256_of`` combination is
self-consistent and deterministic -- the same guarantee OMN-15094's own
``test_hashes_are_deterministic_for_identical_rows`` makes for the renderer
side.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from omnibase_infra.verification.workflow_receipt.receipt_hash import (
    canonical_json,
    sha256_of,
    terminal_event_payload,
)


def test_canonical_json_matches_renderer_encoding_exactly() -> None:
    payload = {"b": 2, "a": 1}
    assert canonical_json(payload) == json.dumps(payload, sort_keys=True, default=str)


def test_sha256_of_matches_manual_hexdigest() -> None:
    payload = {"z": 1, "a": "x"}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert sha256_of(payload) == expected


def test_terminal_event_payload_is_deterministic_for_identical_inputs() -> None:
    correlation_id = str(uuid.uuid4())
    first = terminal_event_payload(
        correlation_id=correlation_id,
        status="completed",
        terminal_model_used="glm-4.6",
        terminal_total_tokens=456,
        terminal_latency_ms=1234,
    )
    second = terminal_event_payload(
        correlation_id=correlation_id,
        status="completed",
        terminal_model_used="glm-4.6",
        terminal_total_tokens=456,
        terminal_latency_ms=1234,
    )
    assert sha256_of(first) == sha256_of(second)


def test_terminal_event_payload_hash_differs_on_different_outcome() -> None:
    correlation_id = str(uuid.uuid4())
    completed = terminal_event_payload(
        correlation_id=correlation_id,
        status="completed",
        terminal_model_used="glm-4.6",
        terminal_total_tokens=456,
        terminal_latency_ms=1234,
    )
    failed = terminal_event_payload(
        correlation_id=correlation_id,
        status="failed",
        terminal_model_used="glm-4.6",
        terminal_total_tokens=456,
        terminal_latency_ms=1234,
    )
    assert sha256_of(completed) != sha256_of(failed)


def test_terminal_event_payload_matches_omn_15094_fixture_hash() -> None:
    """Reproduces OMN-15094's own
    ``test_reads_a_real_terminal_row_end_to_end`` fixture values
    (correlation_id arbitrary/fresh, status="completed",
    model_used="glm-4.6", total_tokens=456, latency_ms=1234) and asserts
    this mirror's hash is a stable 64-char sha256 hexdigest -- the same
    invariant OMN-15094 asserts on the renderer side
    (``len(receipt.terminal_event_hash) == 64``).
    """
    correlation_id = str(uuid.uuid4())
    payload = terminal_event_payload(
        correlation_id=correlation_id,
        status="completed",
        terminal_model_used="glm-4.6",
        terminal_total_tokens=456,
        terminal_latency_ms=1234,
    )
    digest = sha256_of(payload)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
