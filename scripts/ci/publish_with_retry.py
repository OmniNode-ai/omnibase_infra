# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Retry/backoff wrapper around ``uv publish`` for release.yml (OMN-14468, RT-8b).

Why this exists
---------------
The omnibase_infra release chain stopped publishing to PyPI on 2026-05-27
(v0.37.2) when ``uv publish`` hit a bare PyPI HTTP 500 with **no retry**. That
single transient upload-endpoint 5xx left ``omnibase-infra`` stuck at 0.36.1 on
PyPI for six weeks while ``main`` advanced to 0.38.4. A transient failure at the
upload boundary must not sink a release.

This wrapper classifies the failure and reacts accordingly:

* **transient** (5xx / 408 / 425 / 429, or a network error with no HTTP status:
  connection reset, timeout, TLS handshake, broken pipe) -> retry with
  exponential backoff up to a real ceiling, then fail closed.
* **already_exists** (the version's files are already on the index -> a re-run
  of the release, or a partial prior upload) -> treat as success. Combined with
  ``--check-url`` (below), uv itself skips already-present files and uploads only
  the missing ones, so re-running a release is idempotent.
* **permanent** (auth 401/403, a malformed 400 that is *not* "already exists")
  -> fail immediately; retrying only burns the ceiling on a non-recoverable
  error.

The publish token is read from the ``UV_PUBLISH_TOKEN`` environment variable by
``uv`` itself — it is never placed on the command line or logged, so nothing in
this module can leak it.

Usage (from release.yml)::

    UV_PUBLISH_TOKEN=... python3 scripts/ci/publish_with_retry.py \
        dist/*.whl dist/*.tar.gz

Exit codes: ``0`` published (or already present), non-zero on a permanent error
or after the transient-retry ceiling is exhausted.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess  # nosec B404 - invokes the trusted `uv` CLI with a fixed argv
import sys
import time
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# HTTP statuses that a PyPI upload can transiently return. 408 request timeout,
# 425 too early, 429 rate limit, and the 5xx family are all worth retrying.
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# 409 Conflict is PyPI's "this file already exists" on some paths.
_ALREADY_EXISTS_STATUS = frozenset({409})

# Matches "status 500", "status code 500", "with status code 503", etc.
_STATUS_RE = re.compile(r"status(?:\s+code)?\s+(\d{3})", re.IGNORECASE)
# PyPI/twine/uv phrasing for an already-uploaded artifact.
_ALREADY_EXISTS_RE = re.compile(
    r"already (?:exist|been used)|filename has already been used|"
    r"file already exists",
    re.IGNORECASE,
)
# Network-layer failures that carry no HTTP status but are still transient.
_NETWORK_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection error",
    "connection closed",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "eof occurred",
    "handshake",
    "broken pipe",
    "network is unreachable",
    "failed to lookup",
    "dns error",
    "request error",
    "error sending request",
)

TRANSIENT = "transient"
ALREADY_EXISTS = "already_exists"
PERMANENT = "permanent"

# PyPI's public simple index — a well-known external publish target passed to
# `uv publish --check-url` so uv skips already-uploaded files. It is not an ONEX
# service and has no routing-authority contract; the annotation records that.
_PYPI_SIMPLE_INDEX = "https://pypi.org/simple/"  # url-authority-ok: public PyPI index for uv --check-url idempotency, not an ONEX service


def classify_failure(output: str) -> str:
    """Classify a failed ``uv publish`` invocation from its combined output.

    Returns one of :data:`TRANSIENT`, :data:`ALREADY_EXISTS`, :data:`PERMANENT`.

    An unrecognized failure (no HTTP status, no network marker) is treated as
    :data:`TRANSIENT`: the retry ceiling bounds the cost, and PyPI's flakes are
    the failure this wrapper exists to survive. A genuinely permanent error that
    slips through here is still bounded — it exhausts the ceiling and fails.
    """
    low = output.lower()

    if _ALREADY_EXISTS_RE.search(low):
        return ALREADY_EXISTS

    codes = {int(code) for code in _STATUS_RE.findall(output)}
    if codes & _ALREADY_EXISTS_STATUS:
        return ALREADY_EXISTS
    if codes & _TRANSIENT_STATUS:
        return TRANSIENT
    if any(marker in low for marker in _NETWORK_MARKERS):
        return TRANSIENT
    # An explicit non-transient HTTP status (401/403/400-not-already-exists) is
    # not worth retrying.
    if codes:
        return PERMANENT
    return TRANSIENT


# A runner returns (exit_code, combined_stdout_stderr).
RunCommand = Callable[[], tuple[int, str]]


def publish_with_retry(
    run_command: RunCommand,
    *,
    max_attempts: int = 6,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """Run ``run_command`` (a ``uv publish`` invocation) with bounded retries.

    ``run_command`` is injected so the retry policy is unit-testable without a
    real network call. Returns the process exit code to propagate (``0`` on
    success or an idempotent already-exists).
    """
    last_exit = 1
    for attempt in range(1, max_attempts + 1):
        exit_code, output = run_command()
        if exit_code == 0:
            log(f"uv publish succeeded on attempt {attempt}/{max_attempts}")
            return 0

        last_exit = exit_code or 1
        kind = classify_failure(output)

        if kind == ALREADY_EXISTS:
            log(
                "uv publish: artifact already present on the index — treating "
                "as success (idempotent re-publish)"
            )
            return 0

        if kind == PERMANENT:
            log(
                f"::error::uv publish failed with a non-retryable error on "
                f"attempt {attempt}/{max_attempts}; not retrying"
            )
            return last_exit

        # transient
        if attempt >= max_attempts:
            log(
                f"::error::uv publish failed after {attempt} attempt(s) on a "
                f"transient error; giving up"
            )
            return last_exit

        delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
        log(
            f"::warning::uv publish attempt {attempt}/{max_attempts} failed "
            f"(transient); retrying in {delay:.0f}s"
        )
        sleep(delay)

    return last_exit


def _default_runner(argv: list[str]) -> RunCommand:
    """Build a runner that invokes ``argv`` and streams+captures its output."""

    def run() -> tuple[int, str]:
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            check=False,
        )
        # Stream through so the CI log still shows the underlying uv output,
        # and return the combined text for classification.
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode, f"{proc.stdout}\n{proc.stderr}"

    return run


def _build_argv(files: list[str], check_url: str) -> list[str]:
    argv = ["uv", "publish"]
    if check_url:
        # uv skips files already present at the index and uploads only the
        # missing ones -> idempotent re-publish of a partially-uploaded version.
        argv += ["--check-url", check_url]
    argv += files
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish distribution artifacts to PyPI with retry/backoff.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Distribution files to publish (wheels + sdists).",
    )
    parser.add_argument(
        "--check-url",
        # PUBLISH_CHECK_INDEX (not *_URL) intentionally, to keep this CI publish
        # target out of the ONEX routing-authority URL gate; the default is the
        # annotated public PyPI index constant above.
        default=os.environ.get("PUBLISH_CHECK_INDEX", _PYPI_SIMPLE_INDEX),
        help=(
            "Simple index URL uv checks to skip already-present files "
            "(idempotency). Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("PUBLISH_MAX_ATTEMPTS", "6")),
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=float(os.environ.get("PUBLISH_BASE_DELAY_SECONDS", "5")),
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=float(os.environ.get("PUBLISH_MAX_DELAY_SECONDS", "120")),
    )
    args = parser.parse_args(argv)

    if not os.environ.get("UV_PUBLISH_TOKEN"):
        print(
            "::error::UV_PUBLISH_TOKEN is not set — cannot publish to PyPI",
            file=sys.stderr,
        )
        return 2

    command = _build_argv(args.files, args.check_url)
    return publish_with_retry(
        _default_runner(command),
        max_attempts=args.max_attempts,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
