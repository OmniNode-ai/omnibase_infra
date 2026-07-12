# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the PyPI publish retry wrapper (OMN-14468, RT-8b).

The release chain died on 2026-05-27 (v0.37.2) when ``uv publish`` hit a bare
PyPI HTTP 500 with no retry. These tests prove ``publish_with_retry`` now:

* retries a transient 500 and then succeeds (the exact v0.37.2 failure mode),
* fails closed after the retry ceiling on a persistent transient error,
* short-circuits to success on an already-published version (idempotent re-run),
* does NOT retry a permanent (auth) error, and
* applies exponential backoff between transient attempts.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from scripts.ci.publish_with_retry import (
    ALREADY_EXISTS,
    PERMANENT,
    TRANSIENT,
    classify_failure,
    publish_with_retry,
)

pytestmark = pytest.mark.unit


def _scripted_runner(
    results: list[tuple[int, str]],
) -> tuple[Callable[[], tuple[int, str]], list[int]]:
    """Return a runner that yields ``results`` in order, plus a call counter."""
    calls: list[int] = []

    def run() -> tuple[int, str]:
        index = len(calls)
        calls.append(index)
        return results[index]

    return run, calls


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Upload failed with status code 500 Internal Server Error", TRANSIENT),
        ("with status code 502 Bad Gateway", TRANSIENT),
        ("status 503 Service Unavailable", TRANSIENT),
        ("status code 429 Too Many Requests", TRANSIENT),
        ("status code 408 Request Timeout", TRANSIENT),
        ("error sending request: connection reset by peer", TRANSIENT),
        ("TLS handshake timed out", TRANSIENT),
        ("File already exists (https://pypi.org/...)", ALREADY_EXISTS),
        ("400 Bad Request: This filename has already been used", ALREADY_EXISTS),
        ("status code 409 Conflict", ALREADY_EXISTS),
        (
            "status code 403 Forbidden — invalid or non-existent authentication",
            PERMANENT,
        ),
        ("status code 401 Unauthorized", PERMANENT),
    ],
)
def test_classify_failure(output: str, expected: str) -> None:
    assert classify_failure(output) == expected


def test_classify_unknown_defaults_to_transient() -> None:
    # No HTTP status, no network marker -> retry (ceiling bounds the cost).
    assert classify_failure("something unexpected happened") == TRANSIENT


# ---------------------------------------------------------------------------
# publish_with_retry
# ---------------------------------------------------------------------------


def test_retries_transient_500_then_succeeds() -> None:
    """The exact v0.37.2 failure: 500 first, success on retry."""
    run, calls = _scripted_runner(
        [
            (1, "Upload failed with status code 500 Internal Server Error"),
            (0, "ok"),
        ]
    )
    slept: list[float] = []

    rc = publish_with_retry(
        run,
        max_attempts=6,
        base_delay=5.0,
        sleep=slept.append,
        log=lambda _msg: None,
    )

    assert rc == 0
    assert len(calls) == 2  # retried exactly once
    assert slept == [5.0]  # one backoff before the retry


def test_fails_closed_after_ceiling_on_persistent_transient() -> None:
    run, calls = _scripted_runner([(1, "status code 503 Service Unavailable")] * 4)
    slept: list[float] = []

    rc = publish_with_retry(
        run,
        max_attempts=4,
        base_delay=1.0,
        sleep=slept.append,
        log=lambda _msg: None,
    )

    assert rc != 0
    assert len(calls) == 4  # exhausted the ceiling
    assert len(slept) == 3  # slept between each of the 4 attempts, not after last


def test_already_exists_is_idempotent_success_without_retry() -> None:
    run, calls = _scripted_runner(
        [(1, "File already exists (https://pypi.org/project/omnibase-infra/)")]
    )
    slept: list[float] = []

    rc = publish_with_retry(run, sleep=slept.append, log=lambda _msg: None)

    assert rc == 0
    assert len(calls) == 1  # no retry needed
    assert slept == []  # never slept


def test_permanent_error_does_not_retry() -> None:
    run, calls = _scripted_runner([(1, "status code 403 Forbidden")])
    slept: list[float] = []

    rc = publish_with_retry(
        run,
        max_attempts=6,
        sleep=slept.append,
        log=lambda _msg: None,
    )

    assert rc != 0
    assert len(calls) == 1  # bailed immediately, did not burn the ceiling
    assert slept == []


def test_exponential_backoff_is_capped() -> None:
    run, _calls = _scripted_runner([(1, "status code 500")] * 6)
    slept: list[float] = []

    publish_with_retry(
        run,
        max_attempts=6,
        base_delay=5.0,
        max_delay=40.0,
        sleep=slept.append,
        log=lambda _msg: None,
    )

    # 5 sleeps between 6 attempts: 5, 10, 20, 40 (cap), 40 (cap).
    assert slept == [5.0, 10.0, 20.0, 40.0, 40.0]
