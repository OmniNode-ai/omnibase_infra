# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for node registration (OMN-6997).

Proves the node dispatch layer is alive by verifying at least one node
registers after the handler init fix (PR #1056). This is the gate that
proves the handler factory/container wiring works at the runtime level.

These tests require the Docker runtime stack to be running (infra-up-runtime).
They are marked @slow because they inspect live container logs.

Related:
    - OMN-6995: Platform Subsystem Verification epic
    - OMN-6996: Runtime health integration test (container boot gate)
    - PR #1056: Handler init fix
"""

from __future__ import annotations

import os
import subprocess

import pytest

# Runtime container name — configurable via env for flexibility.
RUNTIME_CONTAINER: str = os.environ.get("ONEX_RUNTIME_CONTAINER", "omninode-runtime")

# Error patterns that indicate the handler init bug is still present.
# These are read from env (comma-separated) or use known bad patterns.
HANDLER_INIT_ERROR_PATTERNS: list[str] = [
    p.strip()
    for p in os.environ.get(
        "ONEX_HANDLER_INIT_ERROR_PATTERNS",
        "NoneType,config is None,initialize() got an unexpected",
    ).split(",")
    if p.strip()
]


def _get_runtime_logs(since: str = "5m") -> str:
    """Retrieve recent logs from the runtime container.

    Args:
        since: Docker log time window (e.g. '5m', '1h').

    Returns:
        Combined stdout + stderr from the container.
    """
    result = subprocess.run(
        ["docker", "logs", "--since", since, RUNTIME_CONTAINER],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.mark.slow
class TestNodeRegistration:
    """Prove node dispatch layer is alive — at least 1 node registered."""

    def test_at_least_one_node_registered(self) -> None:
        """Runtime must have >0 registered nodes after boot.

        Checks container logs for registration evidence. The runtime logs
        handler registration messages during startup.
        """
        logs = _get_runtime_logs(since="30m")
        assert logs.strip(), (
            f"No logs from container {RUNTIME_CONTAINER!r}. "
            "Is the runtime running? Check with: docker ps -a"
        )

        # Look for registration evidence in runtime logs. The runtime emits
        # several patterns during successful node registration:
        # - "node-registration-initiated" topic events
        # - "Emitting registration events" from the orchestrator handler
        # - "introspection event" from MixinNodeIntrospection
        # - "Dispatch started" / "Dispatch completed" from the dispatch engine
        # - "Intent executed" for postgres.upsert_registration
        registration_patterns = [
            "registration",
            "introspection event",
            "dispatch started",
            "dispatch completed",
            "intent executed",
        ]
        evidence = [
            line
            for line in logs.split("\n")
            if any(pat in line.lower() for pat in registration_patterns)
        ]
        assert len(evidence) > 0, (
            "Zero node/handler registration evidence found in runtime logs. "
            "Handler init bug (PR #1056) may not be fixed. "
            f"Searched {len(logs.split(chr(10)))} log lines from {RUNTIME_CONTAINER!r}."
        )

    def test_no_handler_init_errors(self) -> None:
        """No known handler init error patterns should appear in logs.

        These patterns indicate the handler init bug is still present:
        - 'NoneType' errors from uninitialized config
        - 'config is None' from missing container injection
        - 'initialize() got an unexpected' from signature mismatch
        """
        logs = _get_runtime_logs(since="30m")
        if not logs.strip():
            pytest.skip(
                f"No logs from container {RUNTIME_CONTAINER!r} — "
                "cannot verify absence of errors"
            )

        for pattern in HANDLER_INIT_ERROR_PATTERNS:
            # Search case-sensitively since these are specific error signatures
            matches = [line for line in logs.split("\n") if pattern in line]
            assert len(matches) == 0, (
                f"Found handler init error pattern {pattern!r} in runtime logs "
                f"({len(matches)} occurrences). Handler init bug still present. "
                f"First match: {matches[0][:200]}"
            )

    def test_runtime_health_endpoint_reachable(self) -> None:
        """Runtime health endpoint must be reachable and return healthy status.

        This complements the log-based checks by verifying the HTTP health
        endpoint responds. The port is read from env to avoid hardcoding.
        """
        import http.client
        import json

        runtime_port = int(os.environ.get("RUNTIME_PORT", "8085"))
        runtime_host = os.environ.get("RUNTIME_HOST", "localhost")

        try:
            conn = http.client.HTTPConnection(runtime_host, runtime_port, timeout=10)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read().decode()
            data = json.loads(body)
            assert resp.status == 200, f"Health endpoint returned status {resp.status}"
            # The health endpoint should report healthy
            is_healthy = data.get("healthy", data.get("status") == "healthy")
            assert is_healthy, f"Runtime health endpoint reports unhealthy: {data}"
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            pytest.fail(
                f"Cannot reach runtime health endpoint at "
                f"http://{runtime_host}:{runtime_port}/health: {exc}. "
                "Is the runtime running? Check with: docker ps -a"
            )
        finally:
            conn.close()
