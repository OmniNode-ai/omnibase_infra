# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded Docker/HTTP transport for the C28 collector."""

from __future__ import annotations

import json
import re
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, cast

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    ANALYTICS_DB,
    BROKER_INTERNAL_ADDRESS,
    EXPOSURE_TOPIC,
    LIVE_WINDOW_SQL,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
    ConsumerFlowInputError,
    ConsumerFlowLaneUnsettledError,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_lane import (
    ModelConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.models.typed_dict_consumer_flow import (
    TypedDictConsumerFlowIdentity,
    TypedDictConsumerFlowResponse,
)

_BROKER_SASL_MECHANISM = "SCRAM-SHA-256"
_REASON_CHARS = 300
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def check_broker_login_file(principal: str, env_file: Path) -> None:
    """Refuse the broker login unless its password file is present and usable.

    OMN-19812: the stability broker requires SASL, so the probe's rpk calls run
    as the probe's own login. The workflow reads that login's password from the
    dev secret store into ``env_file`` (a mode-600 ``RPK_PASS=`` line) before the
    node runs; a failing read leaves the file absent and its cause in
    ``<env_file>.reason``. Every path that cannot yield a usable file raises
    ``ConsumerFlowInputError``, which the target grades INDETERMINATE, so a
    missing credential never turns into an unauthenticated rpk call. The
    password is never read into the error text.
    """
    try:
        info = env_file.stat()
    except OSError:
        info = None
    if info is None:
        reason = ""
        try:
            raw = env_file.with_name(env_file.name + ".reason").read_text(
                encoding="utf-8", errors="replace"
            )
            reason = _CONTROL_CHARS.sub(" ", raw).strip()[:_REASON_CHARS]
        except OSError:
            pass
        said = f": {reason}" if reason else ""
        raise ConsumerFlowInputError(
            f"broker login {principal} password file is absent{said}"
        )
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ConsumerFlowInputError(
            f"broker login {principal} password file is not a private regular file"
        )
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        raise ConsumerFlowInputError(
            f"broker login {principal} password file is unreadable: {type(exc).__name__}"
        ) from exc
    if not any(
        line.startswith("RPK_PASS=") and len(line) > len("RPK_PASS=") for line in lines
    ):
        raise ConsumerFlowInputError(
            f"broker login {principal} password file holds no RPK_PASS value"
        )


class ConsumerFlowLane:
    """One request's Docker socket, lane containers, projection endpoint and I/O."""

    def __init__(
        self,
        *,
        docker: str,
        base_url: str,
        containers: ModelConsumerFlowLane,
        broker_principal: str | None = None,
        broker_password_env_file: Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        urlopen: Callable[
            ..., AbstractContextManager[IO[bytes]]
        ] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.docker = docker
        self.base_url = base_url
        self.containers = containers
        self.http_timeout = 30.0
        self.broker_principal = broker_principal
        self.broker_password_env_file = broker_password_env_file
        self.runner = runner
        self.urlopen = urlopen
        self.sleep = sleep
        self.monotonic = monotonic

    # ---- containers -------------------------------------------------------
    def identity(self) -> dict[str, TypedDictConsumerFlowIdentity]:
        out: dict[str, TypedDictConsumerFlowIdentity] = {}
        for name in self.containers.boot_containers:
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
        self,
        settle_seconds: float,
        poll: float = 15.0,
        *,
        boot_grace_seconds: float = 0.0,
    ) -> dict[str, TypedDictConsumerFlowIdentity]:
        # OMN-19812: Docker reports health "starting" only while a container is
        # inside its declared start period, so a lane whose every unsettled
        # container is ('running', 'starting') is booting, not unsettled; it
        # earns boot_grace_seconds past the settle window. Any other state ends
        # the window at settle_seconds as before.
        start = self.monotonic()
        deadline = start + settle_seconds
        boot_deadline = deadline + boot_grace_seconds
        while True:
            ident = self.identity()
            unsettled = {
                n: (v["status"], v["health"])
                for n, v in ident.items()
                if v["status"] != "running" or v["health"] not in (None, "healthy")
            }
            if not unsettled:
                return ident
            now = self.monotonic()
            booting = all(sh == ("running", "starting") for sh in unsettled.values())
            if now >= (boot_deadline if booting else deadline):
                raise ConsumerFlowLaneUnsettledError(
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
        # The credential reaches rpk through its environment, never as a -X flag:
        # the expanded flag would sit in the in-container rpk argv, which every user
        # on the host reads through `ps` (OMN-17427).
        if self.broker_principal is not None:
            # OMN-19812: the broker requires SASL. rpk runs as the probe's own
            # login: the user and mechanism ride `-e`, the password rides
            # `--env-file` (a mode-600 file the workflow wrote from the dev secret
            # store), so it is in no argv and no ambient environment. There is no
            # fallback to the container's own credential or to no credential.
            if self.broker_password_env_file is None:
                raise ConsumerFlowInputError(
                    f"broker login {self.broker_principal} has no password file"
                )
            return self._run(
                [
                    self.docker,
                    "exec",
                    "-i",
                    "-e",
                    f"RPK_USER={self.broker_principal}",
                    "-e",
                    f"RPK_SASL_MECHANISM={_BROKER_SASL_MECHANISM}",
                    "--env-file",
                    str(self.broker_password_env_file),
                    self.containers.broker_container,
                    "sh",
                    "-c",
                    f'rpk "$@" -X brokers="{BROKER_INTERNAL_ADDRESS}"',
                    "rpk",
                    *args,
                ],
                stdin=stdin,
                timeout=timeout,
            )
        # OMN-21485: SASL only where the broker container carries a login; the
        # lane's internal listener may take none, and an empty SCRAM user would be
        # refused there.
        script = (
            'if [ -n "${DEV_KAFKA_SASL_USERNAME:-}" ]; then '
            'export RPK_USER="$DEV_KAFKA_SASL_USERNAME" RPK_PASS="$DEV_KAFKA_SASL_PASSWORD" '
            "RPK_SASL_MECHANISM=SCRAM-SHA-256; fi; "
            f'rpk "$@" -X brokers="{BROKER_INTERNAL_ADDRESS}"'
        )
        return self._run(
            [
                self.docker,
                "exec",
                "-i",
                self.containers.broker_container,
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
        PG_CONTAINER = self.containers.postgres_container  # the graded lane's own
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
        except urllib.error.HTTPError as exc:
            # OMN-19812: the projection API names its refusal in the body (a
            # degraded exposure, a source read that failed); the status line
            # alone left the board reading "503 Service Unavailable" with the
            # server's own cause discarded.
            cause = _http_error_cause(exc)
            said = f"; server said: {cause}" if cause else ""
            raise ConsumerFlowInputError(
                f"GET {url}: {type(exc).__name__}: {exc}{said}"
            ) from exc
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


_HTTP_ERROR_BODY_BYTES = 2048
_HTTP_ERROR_TEXT_CHARS = 200


def _http_error_cause(exc: urllib.error.HTTPError) -> str:
    """The refusal the server named in an error body, or "" when it named none.

    A JSON object body contributes its ``status``, ``error`` and ``reason``
    string fields, in that order; any other body contributes its first
    characters with control characters removed.
    """
    try:
        raw = exc.read(_HTTP_ERROR_BODY_BYTES) if exc.fp is not None else b""
    except (OSError, ValueError):
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        parsed: object = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        named = [
            f"{key}={value}"
            for key in ("status", "error", "reason")
            if isinstance(value := parsed.get(key), str) and value
        ]
        if named:
            return " ".join(named)
    return "".join(ch for ch in text if ch.isprintable())[:_HTTP_ERROR_TEXT_CHARS]
