# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host composition root of the runtime canary (OMN-26463).

Binds ``ProtocolRuntimeCanaryProbe`` to the canary lane's own containers through
``docker`` and the canary's own HTTP port: the runtime container named by the
instance's ``lane:`` block and its own Redpanda. Nothing here names a dev
container, a dev broker or a dev credential; the container names are arguments,
read from the routing table row of the canary instance by the caller.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import uuid
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from omnibase_infra.protocols.protocol_runtime_canary_probe import (
    ProtocolRuntimeCanaryProbe,
)

__all__: list[str] = ["DockerCanaryProbe"]

#: Run inside the canary's runtime image: the contracts its boot left dormant.
DORMANT_SNIPPET: Final = (
    "import json;"
    "from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts;"
    "from omnibase_infra.handlers.lazy_start_boot_split import "
    "lazy_start_policy_of_this_lane as p, split_boot_manifest as s;"
    "policy = p();"
    "print(json.dumps([] if policy is None else "
    "[c.name for c in s(discover_contracts(), policy)[1]]))"
)


def _docker(*argv: str, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _get(url: str, timeout: float) -> tuple[int, str]:
    """One GET of the canary's own HTTP port; (0, "") when nothing answers."""
    parts = urlsplit(url)
    connection = http.client.HTTPConnection(
        parts.hostname or "", parts.port or 80, timeout=timeout
    )
    try:
        connection.request("GET", parts.path or "/")
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", "replace")
    except (http.client.HTTPException, OSError):
        return 0, ""
    finally:
        connection.close()


@dataclass(frozen=True)
class DockerCanaryProbe(ProtocolRuntimeCanaryProbe):
    """The canary lane's runtime container, read and driven from the host."""

    runtime_container: str
    redpanda_container: str
    main_url: str
    start_command_topic: str
    start_answer_topic: str
    #: Bounds, read by the caller from the node's contract settings.
    docker_timeout_seconds: float
    http_timeout_seconds: float
    wake_answer_seconds: float

    def ready(self) -> bool:
        status, _ = _get(f"{self.main_url}/ready", self.http_timeout_seconds)
        return status == 200

    def heartbeat_ok(self) -> bool:
        status, body = _get(f"{self.main_url}/health", self.http_timeout_seconds)
        if status != 200:
            return False
        try:
            return str(json.loads(body).get("status", "")).lower() == "healthy"
        except (ValueError, AttributeError):
            return False

    def log_text(self) -> str:
        done = _docker(
            "logs", self.runtime_container, timeout=self.docker_timeout_seconds
        )
        return done.stdout + done.stderr

    def dormant_contracts(self) -> tuple[str, ...]:
        done = _docker(
            "exec",
            self.runtime_container,
            "/app/.venv/bin/python",
            "-c",
            DORMANT_SNIPPET,
            timeout=self.docker_timeout_seconds,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"dormant contract listing exited {done.returncode}: {done.stderr.strip()[-300:]}"
            )
        return tuple(json.loads(done.stdout.strip().splitlines()[-1]))

    def wake(self, contract: str) -> bool:
        correlation_id = str(uuid.uuid4())
        request = json.dumps(
            {
                "operation": "node.start.requested",
                "target_contract": contract,
                "topic": self.start_command_topic,
                "correlation_id": correlation_id,
                "envelope": {"canary": True},
            }
        )
        produced = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.redpanda_container,
                "rpk",
                "topic",
                "produce",
                self.start_command_topic,
            ],
            input=request,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.docker_timeout_seconds,
        )
        if produced.returncode != 0:
            return False
        # rpk consume blocks until its timeout; what it printed by then is the answer log.
        try:
            answers = _docker(
                "exec",
                self.redpanda_container,
                "rpk",
                "topic",
                "consume",
                self.start_answer_topic,
                "--offset",
                "start",
                "--format",
                "%v\\n",
                timeout=self.wake_answer_seconds,
            ).stdout
        except subprocess.TimeoutExpired as expired:
            raw = expired.stdout or b""
            answers = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
        return any(
            correlation_id in line and '"DELIVERED"' in line
            for line in answers.splitlines()
        )
