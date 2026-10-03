# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded Docker/HTTP transport for the C28 collector."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import IO, cast

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    ANALYTICS_DB,
    BOOT_CONTAINERS,
    BROKER_CONTAINER,
    BROKER_INTERNAL_ADDRESS,
    EXPOSURE_TOPIC,
    LIVE_WINDOW_SQL,
    PG_CONTAINER,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
    ConsumerFlowInputError,
)
from omnibase_infra.nodes.node_board_probe_effect.models.typed_dict_consumer_flow import (
    TypedDictConsumerFlowIdentity,
    TypedDictConsumerFlowResponse,
)


class ConsumerFlowLane:
    """One request's Docker socket, projection endpoint and injectable I/O."""

    def __init__(
        self,
        *,
        docker: str,
        base_url: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        urlopen: Callable[
            ..., AbstractContextManager[IO[bytes]]
        ] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.docker = docker
        self.base_url = base_url
        self.http_timeout = 30.0
        self.runner = runner
        self.urlopen = urlopen
        self.sleep = sleep
        self.monotonic = monotonic

    # ---- containers -------------------------------------------------------
    def identity(self) -> dict[str, TypedDictConsumerFlowIdentity]:
        out: dict[str, TypedDictConsumerFlowIdentity] = {}
        for name in BOOT_CONTAINERS:
            raw = self._run([self.docker, "inspect", name], timeout=30)
            try:
                info = json.loads(raw)[0]
            except (ValueError, IndexError) as exc:
                raise ConsumerFlowInputError(
                    f"docker inspect {name}: unreadable"
                ) from exc
            state = info.get("State") or {}
            out[name] = {
                "id": str(info.get("Id", ""))[:12],
                "started_at": state.get("StartedAt"),
                "status": state.get("Status"),
                "health": (state.get("Health") or {}).get("Status"),
                "image": (info.get("Config") or {}).get("Image"),
            }
        return out

    def wait_settled(
        self, settle_seconds: float, poll: float = 15.0
    ) -> dict[str, TypedDictConsumerFlowIdentity]:
        deadline = self.monotonic() + settle_seconds
        while True:
            ident = self.identity()
            unsettled = {
                n: (v["status"], v["health"])
                for n, v in ident.items()
                if v["status"] != "running" or v["health"] not in (None, "healthy")
            }
            if not unsettled:
                return ident
            if self.monotonic() >= deadline:
                raise ConsumerFlowInputError(
                    f"lane containers not running and healthy after {settle_seconds:.0f}s: {unsettled}"
                )
            self.sleep(poll)

    def logs(self, container: str, since: str | None = None) -> list[str]:
        args = [self.docker, "logs"]
        if since:
            args += ["--since", since]
        args.append(container)
        try:
            proc = self.runner(
                args, capture_output=True, text=True, timeout=180, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConsumerFlowInputError(
                f"docker logs {container}: {type(exc).__name__}"
            ) from exc
        if proc.returncode != 0:
            raise ConsumerFlowInputError(
                f"docker logs {container} exited {proc.returncode}"
            )
        # The runtime logs to stderr and stdout both.
        return (proc.stdout + proc.stderr).splitlines()

    # ---- broker (credential stays inside the broker container) -----------
    def rpk(self, *args: str, stdin: str | None = None, timeout: float = 60.0) -> str:
        script = (
            f'rpk "$@" -X brokers="{BROKER_INTERNAL_ADDRESS}" '
            '-X user="$DEV_KAFKA_SASL_USERNAME" '
            '-X pass="$DEV_KAFKA_SASL_PASSWORD" -X sasl.mechanism=SCRAM-SHA-256'
        )
        return self._run(
            [
                self.docker,
                "exec",
                "-i",
                BROKER_CONTAINER,
                "sh",
                "-c",
                script,
                "rpk",
                *args,
            ],
            stdin=stdin,
            timeout=timeout,
        )

    def high_watermark(self, topic: str) -> int:
        return parse_high_watermark(self.rpk("topic", "describe", "-p", topic))

    # ---- database ---------------------------------------------------------
    def live_groups(self) -> list[str]:
        sql = LIVE_WINDOW_SQL
        script = f'psql -U "$POSTGRES_USER" -d {ANALYTICS_DB} -AtX -v ON_ERROR_STOP=1 -c "{sql}"'
        out = self._run(
            [self.docker, "exec", PG_CONTAINER, "sh", "-c", script], timeout=120
        )
        return sorted({line.strip() for line in out.splitlines() if line.strip()})

    # ---- exposure ---------------------------------------------------------
    def page(self, query: dict[str, str]) -> TypedDictConsumerFlowResponse:
        url = f"{self.base_url.rstrip('/')}/projection/{EXPOSURE_TOPIC}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        try:
            with self.urlopen(url, timeout=self.http_timeout) as resp:
                body = json.load(resp)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ConsumerFlowInputError(
                f"GET {url}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(body, dict) or not isinstance(body.get("rows"), list):
            raise ConsumerFlowInputError(f"GET {url}: no rows list in the response")
        return cast("TypedDictConsumerFlowResponse", body)

    def _run(
        self, args: Sequence[str], *, stdin: str | None = None, timeout: float = 120.0
    ) -> str:
        try:
            proc = self.runner(
                list(args),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConsumerFlowInputError(
                f"{args[0]} {args[1:3]} could not run: {type(exc).__name__}"
            ) from exc
        if proc.returncode != 0:
            raise ConsumerFlowInputError(
                f"{' '.join(args[:4])} exited {proc.returncode}: {proc.stderr.strip()[:400]}"
            )
        return proc.stdout


def parse_high_watermark(describe_out: str) -> int:
    """Sum of HIGH-WATERMARK over the partitions table of ``rpk topic describe -p``."""
    col: int | None = None
    total = 0
    seen = False
    for line in describe_out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if "HIGH-WATERMARK" in parts:
            col = parts.index("HIGH-WATERMARK")
            continue
        if col is not None and parts[0].isdigit() and len(parts) > col:
            total += int(parts[col])
            seen = True
    if not seen:
        raise ConsumerFlowInputError(
            f"no partition rows in rpk describe output: {describe_out[:200]!r}"
        )
    return total
