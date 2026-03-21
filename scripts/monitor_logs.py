#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# monitor_logs.py -- Real-time container log monitoring with Slack alerts
#                    and PostgreSQL error event emission to Kafka (OMN-3407)
#
# Watches OmniNode containers, filters for ERROR/CRITICAL/exception lines,
# and posts rate-limited alerts to Slack. Dynamically picks up containers
# as they start and drops them when they stop.
#
# PostgreSQL error monitoring (OMN-3407):
#   Discovers postgres containers dynamically (by label or name prefix), captures
#   multi-line ERROR blocks, computes a SHA-256 dedup fingerprint, and emits
#   structured ModelDbErrorEvent payloads to Kafka topic TOPIC_DB_ERROR_V1.
#   Dedup keys are stored in Valkey ONLY after a successful Kafka publish.
#
# Usage:
#   python scripts/monitor_logs.py                     # Watch all OmniNode containers
#   python scripts/monitor_logs.py --project omnibase-infra-runtime
#   python scripts/monitor_logs.py --dry-run           # Print alerts, don't post
#   python scripts/monitor_logs.py --cooldown 120      # Override per-container cooldown (seconds)
#
# Required env vars (from ~/.omnibase/.env):
#   SLACK_BOT_TOKEN    -- Slack bot OAuth token (xoxb-...)
#   SLACK_CHANNEL_ID   -- Slack channel ID to post alerts to
#
# Optional env vars:
#   MONITOR_PROJECTS   -- Comma-separated compose project names (default: omnibase-infra-runtime,omnibase-infra)
#   MONITOR_COOLDOWN   -- Per-container alert cooldown in seconds (default: 300)
#   KAFKA_BOOTSTRAP_SERVERS -- Kafka bootstrap servers for DB error events
#   VALKEY_HOST        -- Valkey/Redis host for dedup (default: localhost)
#   VALKEY_PORT        -- Valkey/Redis port for dedup (default: 16379)
#   VALKEY_DB          -- Valkey/Redis database index (default: 0)
#   VALKEY_PASSWORD    -- Valkey/Redis password for authentication (required when auth is enabled)

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path


def _load_omnibase_env() -> None:
    """Load ~/.omnibase/.env into os.environ if SLACK_BOT_TOKEN is not set.

    Handles quoted values and comment lines. Skips lines that would
    overwrite values already present in the environment (shell wins).
    """
    env_path = Path.home() / ".omnibase" / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # Strip quotes from value
            try:
                value = shlex.split(value)[0] if value.strip() else ""
            except ValueError:
                value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_omnibase_env()

# ---------------------------------------------------------------------------
# Persistent cooldown (survives monitor restarts / launchd KeepAlive bounces)
# ---------------------------------------------------------------------------

_COOLDOWN_FILE = Path.home() / ".omnibase" / "monitor-cooldowns.json"
_cooldown_lock = threading.Lock()

# Exponential backoff: 5m → 10m → 20m → 40m → 60m (cap)
_BACKOFF_BASE = 300  # 5 minutes
_BACKOFF_CAP = 3600  # 1 hour max


def _cooldown_read(container: str) -> tuple[float, int]:
    """Return (last_alert_time, alert_count) for container."""
    try:
        with _cooldown_lock:
            data = (
                json.loads(_COOLDOWN_FILE.read_text())
                if _COOLDOWN_FILE.exists()
                else {}
            )
        entry = data.get(container, {})
        return float(entry.get("ts", 0.0)), int(entry.get("n", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0, 0


def _cooldown_write(container: str, ts: float, count: int) -> None:
    try:
        with _cooldown_lock:
            data = (
                json.loads(_COOLDOWN_FILE.read_text())
                if _COOLDOWN_FILE.exists()
                else {}
            )
            data[container] = {"ts": ts, "n": count}
            _COOLDOWN_FILE.write_text(json.dumps(data))
    except OSError:
        pass


def _backoff_seconds(count: int) -> int:
    """Exponential backoff: 5m, 10m, 20m, 40m, 60m (cap)."""
    return int(min(_BACKOFF_BASE * (2**count), _BACKOFF_CAP))


# Resolve docker binary at startup so subprocess calls work without a shell PATH.
_DOCKER = shutil.which("docker") or "/usr/local/bin/docker"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROJECTS = ["omnibase-infra-runtime", "omnibase-infra", "omnimemory"]
DEFAULT_COOLDOWN = 300  # 5 minutes between alerts per container
CONTEXT_LINES = 5  # lines of context captured around each error
MAX_SLACK_CHARS = 3000  # Slack block text limit

# Log lines matching these patterns trigger an alert
ERROR_PATTERN = re.compile(
    r"(\[ERROR\]|\[CRITICAL\]|\bERROR\b|\bCRITICAL\b|\bFATAL\b"
    r"|\bTraceback \(most recent call last\)"
    r"|\bProtocolConfigurationError\b"
    r"|\bRuntimeError\b|\bValueError\b|\bKeyError\b"
    r"|\s+raise \w)"
)

# Lines to ignore even if they match ERROR_PATTERN (common false positives)
IGNORE_PATTERN = re.compile(
    r"(error_count=0|no.errors|0 errors|error_rate=0\.0|health.*ok)"
    r"|(DEBUG.*error|error.*DEBUG)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Warning-pattern alerting (OMN-3607)
# ---------------------------------------------------------------------------
# Known recurring WARNING-level issues. Matching lines trigger a labelled Slack
# alert with an independent cooldown (default 1800s / 30 min).

WARNING_PATTERN = re.compile(
    r"\[WARNING\].*("
    r"terminal-state heartbeat ignored"
    r"|Heartbeat received for non-active node"
    r"|dispatch_handlers:.*nacking message for retry"
    r"|DeadlockDetectedError"
    r"|HandlerConsul.*ConnectionError"
    r")"
)

WARNING_COOLDOWN_SECONDS = int(os.environ.get("MONITOR_WARNING_COOLDOWN", "1800"))

# Backoff for warnings: 30m → 60m → cap at 60m
_WARNING_BACKOFF_BASE = 1800  # 30 minutes
_WARNING_BACKOFF_CAP = 3600  # 1 hour max

_WARNING_ISSUE_LABELS: dict[str, str] = {
    # OMN-4826: terminal-state heartbeat ignored — node received a heartbeat
    # after LIVENESS_EXPIRED or REJECTED. Alert includes node_id and current_state.
    "terminal-state heartbeat ignored": "terminal-state-heartbeat",
    "non-active node": "stale-registration",
    "dispatch_handlers": "kafka-nack",
    "DeadlockDetectedError": "schema-deadlock",
    "HandlerConsul": "consul-unavailable",
}


def _warning_issue_label(line: str) -> str:
    """Return a human-readable label for the warning pattern that matched *line*.

    Iterates over ``_WARNING_ISSUE_LABELS`` and returns the first label whose
    fragment key is found in *line*.  Falls back to ``"unknown-warning"`` if no
    fragment matches.
    """
    for fragment, label in _WARNING_ISSUE_LABELS.items():
        if fragment in line:
            return label
    return "unknown-warning"


def _warning_backoff_seconds(count: int) -> int:
    """Exponential backoff for warnings: 30m, 60m, then capped at 60m."""
    return int(min(_WARNING_BACKOFF_BASE * (2**count), _WARNING_BACKOFF_CAP))


# ---------------------------------------------------------------------------
# PostgreSQL error emitter (OMN-3407)
# ---------------------------------------------------------------------------

# Full topic name for PostgreSQL error events.
_TOPIC_DB_ERROR_V1 = "onex.evt.omnibase-infra.db-error.v1"

# Max lines to accumulate into a single postgres error block.
_PG_BLOCK_MAX_LINES = 20

# Postgres timestamp+pid prefix pattern — marks the start of a new log entry.
# Example: "2026-03-02 12:34:56.789 UTC [1234] "
_PG_LOG_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

# PostgreSQL ERROR line pattern. Matches:
#   "ERROR:  [42883] operator does not exist"
#   "ERROR:  relation \"foo\" does not exist"
_PG_ERROR_LINE = re.compile(
    r"\bERROR:\s+(?:\[(?P<sqlstate>[A-Z0-9]{5})\]\s+)?(?P<message>.+)"
)

# Optional supplementary field patterns
_PG_HINT_LINE = re.compile(r"\bHINT:\s+(?P<hint>.+)")
_PG_DETAIL_LINE = re.compile(r"\bDETAIL:\s+(?P<detail>.+)")
_PG_STATEMENT_LINE = re.compile(r"\bSTATEMENT:\s+(?P<statement>.+)")
_PG_CONTEXT_LINE = re.compile(r"\bCONTEXT:\s+(?P<context>.+)")

# Best-effort table name extraction from error messages and SQL statements.
# Matches: relation "foo", table "foo", into "foo", from "foo"
_PG_TABLE_RE = re.compile(
    r'(?:relation|table|into|from)\s+"(?P<table>[^"]+)"',
    re.IGNORECASE,
)

# SQL string literal pattern for normalization: 'value'
_SQL_STRING_LITERAL = re.compile(r"'[^']*'")


def _normalize_text(text: str) -> str:
    """Strip whitespace, collapse runs, and remove SQL string literals."""
    text = _SQL_STRING_LITERAL.sub("''", text)
    return " ".join(text.split())


def _compute_fingerprint(
    error_code: str | None,
    error_message: str,
    table_name: str | None,
    sql_statement: str | None,
) -> str:
    """Return SHA-256 fingerprint of normalized error fields, truncated to 32 chars."""
    parts = ":".join(
        [
            error_code or "",
            _normalize_text(error_message),
            table_name or "",
            _normalize_text(sql_statement or ""),
        ]
    )
    return hashlib.sha256(parts.encode()).hexdigest()[:32]


def _extract_table_name(error_message: str, sql_statement: str | None) -> str | None:
    """Best-effort extraction of table name from error message or SQL statement."""
    for text in (error_message, sql_statement or ""):
        m = _PG_TABLE_RE.search(text)
        if m:
            return m.group("table")
    return None


def _parse_postgres_error_block(
    lines: list[str],
) -> dict[str, str | None] | None:
    """Parse a list of log lines into postgres error fields.

    Returns a dict with keys: error_code, error_message, hint, detail,
    sql_statement, table_name — or None if no ERROR line is found.
    All values are str | None.
    """
    error_code: str | None = None
    error_message: str | None = None
    hint: str | None = None
    detail: str | None = None
    sql_statement: str | None = None

    for line in lines:
        if error_message is None:
            m = _PG_ERROR_LINE.search(line)
            if m:
                error_code = m.group("sqlstate")
                error_message = m.group("message").strip()
            continue
        # Already have error_message — try supplementary fields
        if hint is None:
            mh = _PG_HINT_LINE.search(line)
            if mh:
                hint = mh.group("hint").strip()
                continue
        if detail is None:
            md = _PG_DETAIL_LINE.search(line)
            if md:
                detail = md.group("detail").strip()
                continue
        if sql_statement is None:
            ms = _PG_STATEMENT_LINE.search(line)
            if ms:
                sql_statement = ms.group("statement").strip()
                continue

    if error_message is None:
        return None

    table_name = _extract_table_name(error_message, sql_statement)
    return {
        "error_code": error_code,
        "error_message": error_message,
        "hint": hint,
        "detail": detail,
        "sql_statement": sql_statement,
        "table_name": table_name,
    }


def _discover_postgres_containers() -> list[str]:
    """Return names of running postgres containers.

    Discovery strategy (in order):
    1. Match by compose service label ``com.docker.compose.service=postgres``
    2. Match by container name exact ``omnibase-infra-postgres``
    3. Match by container name prefix ``omnibase-infra-postgres-``

    Returns a deduplicated, sorted list of container names. Logs a warning
    if no containers are found.
    """
    found: set[str] = set()

    # Strategy 1: compose service label
    result = subprocess.run(
        [
            _DOCKER,
            "ps",
            "--filter",
            "label=com.docker.compose.service=postgres",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for name in result.stdout.strip().splitlines():
        if name.strip():
            found.add(name.strip())

    # Strategy 2 & 3: name exact or prefix match
    result2 = subprocess.run(
        [_DOCKER, "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    for name in result2.stdout.strip().splitlines():
        name = name.strip()
        if name == "omnibase-infra-postgres" or name.startswith(
            "omnibase-infra-postgres-"
        ):
            found.add(name)

    if not found:
        print(
            "[monitor] WARNING: no postgres container found at startup; "
            "postgres error monitoring inactive",
            file=sys.stderr,
        )
    return sorted(found)


class PostgresErrorEmitter:
    """Monitors a postgres container log stream and emits error events to Kafka.

    Deduplication is performed via Valkey (Redis-compatible). The Valkey key
    is set **only after** a successful Kafka publish. If Kafka is unavailable,
    the error is logged and the dedup key is not set — allowing retry on the
    next occurrence.
    """

    def __init__(self, container: str, dry_run: bool) -> None:
        self.container = container
        self.dry_run = dry_run
        # These hold runtime-imported clients whose types are not available at
        # type-check time (optional dependencies confluent-kafka and redis).
        # Using Any avoids false attr-defined errors under mypy --strict.
        from typing import Any

        self._producer: Any = None  # confluent_kafka.Producer
        self._valkey: Any = None  # redis.Redis
        self._init_ok = self._init_clients()

    def _init_clients(self) -> bool:
        """Attempt to initialise Kafka producer and Valkey client.

        Returns True if both clients initialised successfully.  On import
        error (library not installed) or missing env var, logs a warning
        and returns False — the emitter is silently disabled.
        """
        kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not kafka_servers:
            print(
                "[monitor] KAFKA_BOOTSTRAP_SERVERS not set; "
                "postgres Kafka emission disabled",
                file=sys.stderr,
            )
            return False

        try:
            import confluent_kafka

            self._producer = confluent_kafka.Producer(
                {
                    "bootstrap.servers": kafka_servers,
                    "acks": "all",
                    "enable.idempotence": "true",
                    "retries": 5,
                    "request.timeout.ms": 10000,
                }
            )
        except ImportError:
            print(
                "[monitor] confluent-kafka not installed; "
                "postgres Kafka emission disabled",
                file=sys.stderr,
            )
            return False
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Failed to create Kafka producer: {exc}; "
                "postgres Kafka emission disabled",
                file=sys.stderr,
            )
            return False

        try:
            import redis

            valkey_host = os.environ.get("VALKEY_HOST", "localhost")
            valkey_port = int(os.environ.get("VALKEY_PORT", "16379"))
            valkey_db = int(os.environ.get("VALKEY_DB", "0"))
            valkey_password = os.environ.get("VALKEY_PASSWORD")
            self._valkey = redis.Redis(
                host=valkey_host,
                port=valkey_port,
                db=valkey_db,
                password=valkey_password,
                socket_timeout=5,
                decode_responses=True,
            )
        except ImportError:
            print(
                "[monitor] redis library not installed; "
                "postgres Kafka emission disabled",
                file=sys.stderr,
            )
            return False
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Failed to create Valkey client: {exc}; "
                "postgres Kafka emission disabled",
                file=sys.stderr,
            )
            return False

        return True

    def maybe_emit(self, lines: list[str]) -> None:
        """Parse error block from lines and emit to Kafka if not deduplicated."""
        if not self._init_ok:
            return

        parsed = _parse_postgres_error_block(lines)
        if parsed is None:
            return

        error_message = parsed["error_message"]
        if not error_message:
            return

        fingerprint = _compute_fingerprint(
            error_code=parsed["error_code"],
            error_message=error_message,
            table_name=parsed["table_name"],
            sql_statement=parsed["sql_statement"],
        )

        # Dedup check
        dedup_key = f"pg_err_dedup:{fingerprint}"
        try:
            if self._valkey is not None:
                existing = self._valkey.get(dedup_key)
                if existing:
                    return
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Valkey dedup check failed for {self.container}: {exc}; "
                "will emit without dedup",
                file=sys.stderr,
            )

        event_payload = {
            "error_code": parsed["error_code"],
            "error_message": error_message,
            "hint": parsed["hint"],
            "detail": parsed["detail"],
            "sql_statement": parsed["sql_statement"],
            "table_name": parsed["table_name"],
            "fingerprint": fingerprint,
            "first_seen_at": datetime.now(UTC).isoformat(),
            "service": self.container,
        }

        if self.dry_run:
            print(f"[DRY RUN] Would emit postgres error event for {self.container}:")
            print(json.dumps(event_payload, indent=2))
            return

        # Kafka publish
        published = False
        try:
            payload_bytes = json.dumps(event_payload).encode("utf-8")
            if self._producer is not None:
                self._producer.produce(
                    topic=_TOPIC_DB_ERROR_V1,
                    value=payload_bytes,
                    key=fingerprint.encode("utf-8"),
                )
                self._producer.flush(timeout=10)
                published = True
                print(
                    f"[monitor] Emitted postgres error event "
                    f"(fingerprint={fingerprint}) from {self.container}"
                )
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Kafka publish failed for {self.container}: {exc}; "
                "dedup key not set — will retry on next occurrence",
                file=sys.stderr,
            )

        # Set dedup key ONLY after successful publish
        if published:
            try:
                if self._valkey is not None:
                    # TTL: 24 hours — avoid infinite suppression of recurring errors
                    self._valkey.setex(dedup_key, 86400, "1")
            except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
                print(
                    f"[monitor] Failed to set Valkey dedup key for {self.container}: {exc}",
                    file=sys.stderr,
                )


class PostgresErrorTailer(threading.Thread):
    """Tails a postgres container log stream and delegates to PostgresErrorEmitter.

    Runs as a daemon thread alongside the existing ContainerTailer instances.
    Captures multi-line error blocks by detecting ERROR: lines and accumulating
    subsequent lines until a new log entry begins or the block size limit is hit.
    """

    def __init__(
        self,
        container: str,
        dry_run: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"pg-err-{container}", daemon=True)
        self.container = container
        self.dry_run = dry_run
        self.stop_event = stop_event
        self._emitter = PostgresErrorEmitter(container, dry_run)

    def run(self) -> None:
        cmd = [
            _DOCKER,
            "logs",
            "--follow",
            "--since",
            "0s",
            "--timestamps",
            self.container,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Cannot tail postgres container {self.container}: {exc}",
                file=sys.stderr,
            )
            return

        print(f"[monitor] Watching postgres errors: {self.container}")

        in_error_block = False
        error_block: list[str] = []

        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if self.stop_event.is_set():
                    break
                line = line.rstrip()

                # Detect start of a new log entry (timestamp prefix)
                is_new_entry = bool(_PG_LOG_PREFIX.search(line))

                if in_error_block:
                    if is_new_entry or len(error_block) >= _PG_BLOCK_MAX_LINES:
                        # Flush the completed block
                        self._emitter.maybe_emit(error_block)
                        error_block = []
                        in_error_block = False

                # Check if this line starts an ERROR block
                if _PG_ERROR_LINE.search(line):
                    in_error_block = True
                    error_block = [line]
                elif in_error_block:
                    error_block.append(line)
        finally:
            # Flush any remaining block
            if error_block:
                self._emitter.maybe_emit(error_block)
            proc.terminate()
            print(f"[monitor] Stopped watching postgres errors: {self.container}")


# ---------------------------------------------------------------------------
# Runtime error emitter (OMN-5649)
# ---------------------------------------------------------------------------
# Classifies ERROR/CRITICAL/FATAL lines from application containers and emits
# structured events to Kafka topic TOPIC_RUNTIME_ERROR_V1. Every classified
# error is emitted (no dedup at emission layer). Dedup happens at the triage
# layer (NodeRuntimeErrorTriageEffect). Recurrence count is tracked via Valkey.

_TOPIC_RUNTIME_ERROR_V1 = "onex.evt.omnibase-infra.runtime-error.v1"

# Error classification regexes
_RE_SCHEMA_MISMATCH = re.compile(
    r"column.*does not exist|relation.*does not exist|undefined column",
    re.IGNORECASE,
)
_RE_MISSING_TOPIC = re.compile(
    r"MISSING_TOPIC|topic.*not in broker|Required topic.*not in broker",
    re.IGNORECASE,
)
_RE_CONNECTION = re.compile(
    r"ConnectionRefused|ConnectionError|connection.*refused",
    re.IGNORECASE,
)
_RE_TIMEOUT = re.compile(
    r"TimeoutError|timed out|deadline exceeded",
    re.IGNORECASE,
)
_RE_OOM = re.compile(
    r"MemoryError|OOMKilled|Cannot allocate memory",
    re.IGNORECASE,
)
_RE_AUTHENTICATION = re.compile(
    r"AuthenticationError|permission denied|access denied",
    re.IGNORECASE,
)

# Logger family extraction: "... omnibase_infra.event_bus.foo: message"
_RE_LOGGER = re.compile(r"[\[\s]?(\w+(?:\.\w+){1,})\s*[:\]]")

# Exception type extraction: "SomeError: message" or "raise SomeError"
_RE_EXCEPTION_TYPE = re.compile(r"(\w+(?:Error|Exception|Failure))\b")

# Missing topic name extraction
_RE_MISSING_TOPIC_NAME = re.compile(
    r"(?:topic|Required topic)\s+['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

# Missing relation/column name extraction
_RE_MISSING_RELATION = re.compile(
    r'(?:column|relation)\s+"([^"]+)"',
    re.IGNORECASE,
)

# Timestamp extraction from container log lines
# Matches: "2026-03-21 13:22:42" or "2026-03-21T13:22:42"
_RE_LOG_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")

# Severity mapping: category -> base severity
_SEVERITY_MAP: dict[str, str] = {
    "OOM": "CRITICAL",
    "SCHEMA_MISMATCH": "HIGH",
    "MISSING_TOPIC": "HIGH",
    "AUTHENTICATION": "HIGH",
    "CONNECTION": "MEDIUM",
    "TIMEOUT": "MEDIUM",
    "UNKNOWN": "MEDIUM",
}


def _classify_runtime_error(line: str) -> str:
    """Classify a runtime error line into a category string."""
    if _RE_OOM.search(line):
        return "OOM"
    if _RE_SCHEMA_MISMATCH.search(line):
        return "SCHEMA_MISMATCH"
    if _RE_MISSING_TOPIC.search(line):
        return "MISSING_TOPIC"
    if _RE_AUTHENTICATION.search(line):
        return "AUTHENTICATION"
    if _RE_CONNECTION.search(line):
        return "CONNECTION"
    if _RE_TIMEOUT.search(line):
        return "TIMEOUT"
    return "UNKNOWN"


def _compute_runtime_fingerprint(
    container: str, error_category: str, error_message: str
) -> str:
    """Return full 64-char SHA-256 hex fingerprint."""
    parts = f"{container}:{error_category}:{error_message}"
    return hashlib.sha256(parts.encode()).hexdigest()


def _extract_logger_family(line: str) -> str:
    """Extract the logger name from a log line, or return 'unknown'."""
    m = _RE_LOGGER.search(line)
    return m.group(1) if m else "unknown"


def _extract_exception_type(line: str) -> str | None:
    """Extract Python exception class name if present."""
    m = _RE_EXCEPTION_TYPE.search(line)
    return m.group(1) if m else None


def _extract_log_timestamp(line: str) -> str | None:
    """Extract ISO timestamp from log line, or return None."""
    m = _RE_LOG_TIMESTAMP.search(line)
    return m.group(1) if m else None


class RuntimeErrorEmitter:
    """Classifies runtime container errors and emits events to Kafka.

    Every classified error is emitted (no dedup suppression). Recurrence
    count is tracked via Valkey and included in the event payload.
    """

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        from typing import Any

        self._producer: Any = None
        self._valkey: Any = None
        self._init_ok = self._init_clients()
        self._environment = os.environ.get("ONEX_ENVIRONMENT", "local")

    def _init_clients(self) -> bool:
        """Attempt to initialise Kafka producer and Valkey client."""
        kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not kafka_servers:
            print(
                "[monitor] KAFKA_BOOTSTRAP_SERVERS not set; "
                "runtime error Kafka emission disabled",
                file=sys.stderr,
            )
            return False

        try:
            import confluent_kafka

            self._producer = confluent_kafka.Producer(
                {
                    "bootstrap.servers": kafka_servers,
                    "acks": "all",
                    "enable.idempotence": "true",
                    "retries": 5,
                    "request.timeout.ms": 10000,
                }
            )
        except ImportError:
            print(
                "[monitor] confluent-kafka not installed; "
                "runtime error Kafka emission disabled",
                file=sys.stderr,
            )
            return False
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Failed to create Kafka producer for runtime errors: {exc}",
                file=sys.stderr,
            )
            return False

        try:
            import redis

            valkey_host = os.environ.get("VALKEY_HOST", "localhost")
            valkey_port = int(os.environ.get("VALKEY_PORT", "16379"))
            valkey_db = int(os.environ.get("VALKEY_DB", "0"))
            valkey_password = os.environ.get("VALKEY_PASSWORD")
            self._valkey = redis.Redis(
                host=valkey_host,
                port=valkey_port,
                db=valkey_db,
                password=valkey_password,
                socket_timeout=5,
                decode_responses=True,
            )
        except ImportError:
            print(
                "[monitor] redis library not installed; "
                "runtime error recurrence tracking disabled (emission still active)",
                file=sys.stderr,
            )
            # Valkey is optional for emission — just tracking recurrence count
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Failed to create Valkey client for runtime errors: {exc}",
                file=sys.stderr,
            )

        return True

    def _get_recurrence_count(self, fingerprint: str) -> int:
        """Get and increment recurrence count for fingerprint in Valkey."""
        dedup_key = f"runtime_err:{fingerprint}"
        try:
            if self._valkey is not None:
                count = self._valkey.incr(dedup_key)
                # Set TTL on first occurrence (1 hour window)
                if count == 1:
                    self._valkey.expire(dedup_key, 3600)
                return int(count)
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Valkey recurrence tracking failed: {exc}",
                file=sys.stderr,
            )
        return 1

    def maybe_emit(self, container: str, line: str) -> None:
        """Classify and emit a runtime error event to Kafka."""
        if not self._init_ok:
            return

        # Classify
        error_category = _classify_runtime_error(line)
        error_message = line.strip()
        fingerprint = _compute_runtime_fingerprint(
            container, error_category, error_message
        )

        # Extract metadata
        logger_family = _extract_logger_family(line)
        exception_type = _extract_exception_type(line)
        log_timestamp = _extract_log_timestamp(line)

        # Determine log level from the line
        log_level = "ERROR"
        if "CRITICAL" in line:
            log_level = "CRITICAL"
        elif "FATAL" in line:
            log_level = "FATAL"

        # Map severity
        severity = _SEVERITY_MAP.get(error_category, "MEDIUM")

        # Get recurrence count
        recurrence = self._get_recurrence_count(fingerprint)

        # Build timestamps
        now = datetime.now(UTC)
        first_seen = now
        if log_timestamp:
            try:
                first_seen = datetime.fromisoformat(
                    log_timestamp.replace(" ", "T")
                ).replace(tzinfo=UTC)
            except ValueError:
                first_seen = now

        # Generate event_id as UUID5 from fingerprint + detected_at
        import uuid

        event_id = str(
            uuid.uuid5(
                uuid.UUID("12345678-1234-5678-1234-567812345678"),
                f"{fingerprint}:{now.isoformat()}",
            )
        )

        # Category-specific parsed fields
        missing_topic_name = None
        missing_relation_name = None
        if error_category == "MISSING_TOPIC":
            m = _RE_MISSING_TOPIC_NAME.search(line)
            if m:
                missing_topic_name = m.group(1)
        elif error_category == "SCHEMA_MISMATCH":
            m = _RE_MISSING_RELATION.search(line)
            if m:
                missing_relation_name = m.group(1)

        event_payload = {
            "event_id": event_id,
            "container": container,
            "source_service": container,
            "logger_family": logger_family,
            "log_level": log_level,
            "error_category": error_category,
            "severity": severity,
            "error_message": error_message,
            "exception_type": exception_type,
            "stack_trace": None,
            "fingerprint": fingerprint,
            "detected_at": now.isoformat(),
            "first_seen_at": first_seen.isoformat(),
            "environment": self._environment,
            "recurrence_count_at_emit": recurrence,
            "raw_line": line,
            "missing_topic_name": missing_topic_name,
            "missing_relation_name": missing_relation_name,
        }

        if self.dry_run:
            print(f"[DRY RUN] Would emit runtime error event for {container}:")
            print(json.dumps(event_payload, indent=2))
            return

        # Kafka publish
        try:
            payload_bytes = json.dumps(event_payload).encode("utf-8")
            if self._producer is not None:
                self._producer.produce(
                    topic=_TOPIC_RUNTIME_ERROR_V1,
                    value=payload_bytes,
                    key=fingerprint.encode("utf-8"),
                )
                self._producer.flush(timeout=10)
                print(
                    f"[monitor] Emitted runtime error event "
                    f"(fingerprint={fingerprint[:16]}..., category={error_category}) "
                    f"from {container}"
                )
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Kafka publish failed for runtime error from {container}: {exc}",
                file=sys.stderr,
            )


class RuntimeErrorTailer(threading.Thread):
    """Tails application container log streams and classifies runtime errors.

    Runs as a daemon thread alongside the existing ContainerTailer instances.
    Watches for ERROR/CRITICAL/FATAL lines in application (non-postgres)
    containers and delegates to RuntimeErrorEmitter for Kafka emission.
    """

    def __init__(
        self,
        container: str,
        dry_run: bool,
        stop_event: threading.Event,
        emitter: RuntimeErrorEmitter | None = None,
    ) -> None:
        super().__init__(name=f"rt-err-{container}", daemon=True)
        self.container = container
        self.dry_run = dry_run
        self.stop_event = stop_event
        self._emitter = emitter or RuntimeErrorEmitter(dry_run)

    def run(self) -> None:
        cmd = [
            _DOCKER,
            "logs",
            "--follow",
            "--since",
            "0s",
            "--timestamps",
            self.container,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"[monitor] Cannot tail runtime errors for {self.container}: {exc}",
                file=sys.stderr,
            )
            return

        print(f"[monitor] Watching runtime errors: {self.container}")

        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if self.stop_event.is_set():
                    break
                line = line.rstrip()

                # Only classify lines that match the ERROR pattern and
                # are not false positives
                if ERROR_PATTERN.search(line) and not IGNORE_PATTERN.search(line):
                    self._emitter.maybe_emit(self.container, line)
        finally:
            proc.terminate()
            print(f"[monitor] Stopped watching runtime errors: {self.container}")


# ---------------------------------------------------------------------------
# Restart watcher (OMN-3596)
# ---------------------------------------------------------------------------

_RESTART_HWM_FILE = Path(
    os.environ.get(
        "MONITOR_RESTART_HWM_FILE",
        str(Path.home() / ".omnibase" / "monitor-restart-hwm.json"),
    )
)
_restart_hwm_lock = threading.Lock()

# Default restart-count delta that triggers an alert.
_RESTART_THRESHOLD = 3

# Default polling interval in seconds.
_RESTART_POLL_INTERVAL = 60


def _restart_hwm_read(container: str) -> int:
    """Return the last-seen restart count for *container* (0 if unknown or corrupt)."""
    try:
        with _restart_hwm_lock:
            data = (
                json.loads(_RESTART_HWM_FILE.read_text())
                if _RESTART_HWM_FILE.exists()
                else {}
            )
        return int(data.get(container, 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _restart_hwm_write(container: str, count: int) -> None:
    """Persist *count* as the high-water mark for *container*.

    Creates parent directories if needed.  If the file contains corrupt JSON
    it is silently reset to ``{}``.  Uses tmp+rename for atomic writes.
    """
    try:
        with _restart_hwm_lock:
            _RESTART_HWM_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                data: dict[str, int] = (
                    json.loads(_RESTART_HWM_FILE.read_text())
                    if _RESTART_HWM_FILE.exists()
                    else {}
                )
            except (json.JSONDecodeError, ValueError):
                data = {}
            data[container] = count
            tmp = _RESTART_HWM_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.rename(_RESTART_HWM_FILE)
    except OSError:
        pass


def _get_worker_state(container: str) -> dict[str, int | str] | None:
    """Return restart count, status, and exit code for *container* via ``docker inspect``.

    Returns ``None`` if the inspect call fails or returns no data.
    """
    result = subprocess.run(
        [
            _DOCKER,
            "inspect",
            "--format",
            "json",
            container,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        items = json.loads(result.stdout)
        if not items:
            return None
        item = items[0]
        return {
            "restart_count": int(item.get("RestartCount", 0)),
            "status": str(item.get("State", {}).get("Status", "unknown")),
            "exit_code": int(item.get("State", {}).get("ExitCode", -1)),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _restart_containers_from_env() -> list[str] | None:
    """Return an explicit container list from ``MONITOR_RESTART_CONTAINERS`` or ``None``."""
    raw = os.environ.get("MONITOR_RESTART_CONTAINERS")
    if not raw:
        return None
    return sorted(name.strip() for name in raw.split(",") if name.strip())


class RestartWatcher(threading.Thread):
    """Daemon thread that polls ``docker inspect`` for restart-count deltas.

    When the restart-count delta since the last high-water mark reaches
    *threshold*, a Slack alert is posted.  The HWM is persisted to disk so
    it survives process restarts.
    """

    def __init__(
        self,
        containers: list[str],
        bot_token: str,
        channel_id: str,
        dry_run: bool,
        stop_event: threading.Event,
        interval: int = _RESTART_POLL_INTERVAL,
        threshold: int = _RESTART_THRESHOLD,
    ) -> None:
        super().__init__(name="restart-watcher", daemon=True)
        self.containers = containers
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.dry_run = dry_run
        self.stop_event = stop_event
        self.interval = interval
        self.threshold = threshold

    def run(self) -> None:
        print(
            f"[monitor] RestartWatcher started (containers={len(self.containers)}, "
            f"interval={self.interval}s, threshold={self.threshold})"
        )
        while not self.stop_event.is_set():
            self._check()
            self.stop_event.wait(timeout=self.interval)

    def _check(self) -> None:
        for container in self.containers:
            state = _get_worker_state(container)
            if state is None:
                continue
            current = int(state["restart_count"])
            hwm = _restart_hwm_read(container)
            delta = current - hwm

            if delta < 0:
                # Container was recreated — reset HWM without alerting
                _restart_hwm_write(container, current)
                continue

            if delta >= self.threshold:
                self._alert(container, state)
                _restart_hwm_write(container, current)
            elif delta > 0:
                # Below threshold — update HWM silently
                _restart_hwm_write(container, current)

    def _alert(self, container: str, state: dict[str, int | str]) -> None:
        hostname = socket.gethostname()
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [
            f":warning: *Restart loop detected:* `{container}`",
            f"*Status:* {state['status']}  |  *Exit code:* {state['exit_code']}",
            f"*Restart count:* {state['restart_count']}",
            f"*Host:* {hostname}  |  *Time:* {timestamp}",
        ]
        if self.dry_run:
            print(f"[DRY RUN] Would post restart alert for {container}:")
            for line in lines:
                print(f"  {line}")
            return
        post_slack(self.bot_token, self.channel_id, container, lines, self.dry_run)


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

# The mrkdwn code-fence wrapper adds exactly 8 chars: "```\n" (4) + "\n```" (4).
# Cap log text at (MAX_SLACK_CHARS - 8) so the assembled field stays within limit.
_BLOCK_TEXT_LIMIT = MAX_SLACK_CHARS - 8

# Matches ANSI CSI sequences (colors, cursor), OSC sequences (hyperlinks, titles),
# and two-byte Fe escape sequences (e.g. ESC [ ... m, ESC ] ... BEL).
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKHF]|\x1b\].*?\x07|\x1b[@-_]")


def _sanitize_log_text(text: str) -> str:
    """Strip ANSI escape codes and non-printable control chars from log content.

    Newlines are preserved; every other control character is replaced with '?'
    so the structure of the log excerpt remains readable.
    """
    text = _ANSI_ESCAPE.sub("", text)
    return "".join(ch if ch >= " " or ch == "\n" else "?" for ch in text)


def _post_slack_plain_text(
    bot_token: str, channel_id: str, container: str, lines: list[str]
) -> None:
    """Fallback: post a plain-text message (no blocks) when blocks are rejected."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = _sanitize_log_text("\n".join(lines))[:MAX_SLACK_CHARS]
    fallback_payload = {
        "channel": channel_id,
        "text": (
            f":rotating_light: *Container error:* `{container}` — {timestamp}\n"
            f"```\n{text}\n```"
        ),
    }
    try:
        data = json.dumps(fallback_payload).encode("utf-8")
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bot_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            result = json.loads(resp.read())
            if not result.get("ok"):
                print(
                    f"[monitor] Slack fallback error for {container}: {result.get('error')}",
                    file=sys.stderr,
                )
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"[monitor] Failed to post Slack fallback for {container}: {exc}",
            file=sys.stderr,
        )


def post_slack(
    bot_token: str, channel_id: str, container: str, lines: list[str], dry_run: bool
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = _sanitize_log_text("\n".join(lines))[:_BLOCK_TEXT_LIMIT]

    payload = {
        "channel": channel_id,
        "text": f":rotating_light: *Container error:* `{container}` — {timestamp}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":rotating_light: *Container error:* `{container}`\n*Time:* {timestamp}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```\n{text}\n```",
                },
            },
        ],
    }

    if dry_run:
        print(f"[DRY RUN] Would post to Slack for {container}:")
        print(json.dumps(payload, indent=2))
        return

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bot_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            result = json.loads(resp.read())
            if not result.get("ok"):
                error = result.get("error")
                if error == "invalid_blocks":
                    print(
                        f"[monitor] invalid_blocks for {container}, retrying with plain text",
                        file=sys.stderr,
                    )
                    _post_slack_plain_text(bot_token, channel_id, container, lines)
                else:
                    print(
                        f"[monitor] Slack API error for {container}: {error}",
                        file=sys.stderr,
                    )
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"[monitor] Failed to post Slack alert for {container}: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Container log tailer
# ---------------------------------------------------------------------------


class ContainerTailer(threading.Thread):
    def __init__(
        self,
        container: str,
        bot_token: str,
        channel_id: str,
        cooldown: int,
        dry_run: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"tail-{container}", daemon=True)
        self.container = container
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.cooldown = cooldown
        self.dry_run = dry_run
        self.stop_event = stop_event
        self._context: deque[str] = deque(maxlen=CONTEXT_LINES)

    def run(self) -> None:
        cmd = [
            _DOCKER,
            "logs",
            "--follow",
            "--since",
            "0s",
            "--timestamps",
            self.container,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(f"[monitor] Cannot tail {self.container}: {exc}", file=sys.stderr)
            return

        print(f"[monitor] Watching {self.container}")
        error_batch: list[str] = []
        batch_timer: threading.Timer | None = None

        def flush_batch() -> None:
            nonlocal error_batch, batch_timer
            if error_batch:
                self._maybe_alert(list(error_batch))
            error_batch = []
            batch_timer = None

        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if self.stop_event.is_set():
                    break
                line = line.rstrip()
                self._context.append(line)

                if ERROR_PATTERN.search(line) and not IGNORE_PATTERN.search(line):
                    if batch_timer:
                        batch_timer.cancel()
                    # Seed with recent context
                    if not error_batch:
                        error_batch.extend(self._context)
                    else:
                        error_batch.append(line)
                    batch_timer = threading.Timer(2.0, flush_batch)
                    batch_timer.daemon = True
                    batch_timer.start()
                elif WARNING_PATTERN.search(line) and not IGNORE_PATTERN.search(line):
                    label = _warning_issue_label(line)
                    self._maybe_warning_alert(label, list(self._context) + [line])
                elif error_batch:
                    # Accumulate lines after the error trigger
                    error_batch.append(line)
        finally:
            if batch_timer:
                batch_timer.cancel()
                flush_batch()
            proc.terminate()
            print(f"[monitor] Stopped watching {self.container}")

    def _maybe_alert(self, lines: list[str]) -> None:
        now = time.time()
        last, count = _cooldown_read(self.container)
        wait = _backoff_seconds(count)
        if now - last < wait:
            remaining = int(wait - (now - last))
            print(
                f"[monitor] Rate-limited {self.container} (cooldown {remaining}s, alert #{count})"
            )
            return
        _cooldown_write(self.container, now, count + 1)
        post_slack(self.bot_token, self.channel_id, self.container, lines, self.dry_run)

    def _maybe_warning_alert(self, label: str, lines: list[str]) -> None:
        """Post a labelled Slack warning alert with independent cooldown.

        Uses a composite cooldown key ``{container}:warn:{label}`` so that each
        container+warning-type pair is rate-limited independently from error-level
        alerts and from other warning labels.
        """
        cooldown_key = f"{self.container}:warn:{label}"
        now = time.time()
        last, count = _cooldown_read(cooldown_key)
        wait = _warning_backoff_seconds(count)
        if now - last < wait:
            remaining = int(wait - (now - last))
            print(
                f"[monitor] Rate-limited warning {label} on {self.container} "
                f"(cooldown {remaining}s, alert #{count})"
            )
            return
        _cooldown_write(cooldown_key, now, count + 1)
        # Include the warning label in the container identifier for Slack triage
        container_label = f"{self.container} [{label}]"
        post_slack(
            self.bot_token, self.channel_id, container_label, lines, self.dry_run
        )


# ---------------------------------------------------------------------------
# Dynamic container watcher
# ---------------------------------------------------------------------------


class LogMonitor:
    def __init__(
        self,
        projects: list[str],
        bot_token: str,
        channel_id: str,
        cooldown: int,
        dry_run: bool,
    ) -> None:
        self.projects = projects
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.cooldown = cooldown
        self.dry_run = dry_run
        self._tailers: dict[str, ContainerTailer] = {}
        self._pg_tailers: dict[str, PostgresErrorTailer] = {}
        self._rt_tailers: dict[str, RuntimeErrorTailer] = {}
        self._rt_emitter = RuntimeErrorEmitter(dry_run)
        self._lock = threading.Lock()

    def _get_project_containers(self) -> list[str]:
        containers: list[str] = []
        for project in self.projects:
            result = subprocess.run(
                [
                    _DOCKER,
                    "ps",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            for name in result.stdout.strip().splitlines():
                if name:
                    containers.append(name)
        return containers

    def _start_tailer(self, container: str) -> None:
        with self._lock:
            if container in self._tailers and self._tailers[container].is_alive():
                return
            stop_event = threading.Event()
            tailer = ContainerTailer(
                container,
                self.bot_token,
                self.channel_id,
                self.cooldown,
                self.dry_run,
                stop_event,
            )
            tailer.start()
            self._tailers[container] = tailer

    def _stop_tailer(self, container: str) -> None:
        with self._lock:
            tailer = self._tailers.pop(container, None)
        if tailer and tailer.is_alive():
            # ContainerTailer is daemon; process termination handles cleanup
            print(f"[monitor] Container gone: {container}")

    def _start_pg_tailer(self, container: str) -> None:
        """Start a PostgresErrorTailer for the given container if not already running."""
        with self._lock:
            existing = self._pg_tailers.get(container)
            if existing and existing.is_alive():
                return
            stop_event = threading.Event()
            tailer = PostgresErrorTailer(container, self.dry_run, stop_event)
            tailer.start()
            self._pg_tailers[container] = tailer

    def _is_postgres_container(self, container: str) -> bool:
        """Return True if container is a postgres container (not an app container)."""
        return container == "omnibase-infra-postgres" or container.startswith(
            "omnibase-infra-postgres-"
        )

    def _start_rt_tailer(self, container: str) -> None:
        """Start a RuntimeErrorTailer for application containers (not postgres)."""
        if self._is_postgres_container(container):
            return  # Postgres errors are handled by PostgresErrorTailer
        with self._lock:
            existing = self._rt_tailers.get(container)
            if existing and existing.is_alive():
                return
            stop_event = threading.Event()
            tailer = RuntimeErrorTailer(
                container, self.dry_run, stop_event, self._rt_emitter
            )
            tailer.start()
            self._rt_tailers[container] = tailer

    def run(self) -> None:
        # Start tailers for already-running project containers
        for container in self._get_project_containers():
            self._start_tailer(container)
            # Also start runtime error tailers for app containers (OMN-5649)
            self._start_rt_tailer(container)

        # Start postgres error tailers for any running postgres containers
        for pg_container in _discover_postgres_containers():
            self._start_pg_tailer(pg_container)

        # Start restart watcher (OMN-3596)
        _restart_stop = threading.Event()
        explicit_containers = _restart_containers_from_env()
        restart_containers = (
            explicit_containers
            if explicit_containers is not None
            else sorted(set(self._get_project_containers()))
        )
        restart_watcher = RestartWatcher(
            containers=restart_containers,
            bot_token=self.bot_token,
            channel_id=self.channel_id,
            dry_run=self.dry_run,
            stop_event=_restart_stop,
        )
        restart_watcher.start()

        # Watch for container start/die events
        cmd = [
            _DOCKER,
            "events",
            "--filter",
            "type=container",
            "--filter",
            "event=start",
            "--filter",
            "event=die",
            "--format",
            "{{.Actor.Attributes.name}} {{.Action}}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

        print(f"[monitor] Monitoring projects: {', '.join(self.projects)}")
        print("[monitor] Listening for container events (Ctrl+C to stop)...")

        project_containers = set(self._get_project_containers())

        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                container, action = parts[0], parts[1]

                # Only track containers from our projects
                current = set(self._get_project_containers())
                if action == "start" and container in current:
                    self._start_tailer(container)
                    # Also start a postgres error tailer if this is a postgres container
                    if self._is_postgres_container(container):
                        self._start_pg_tailer(container)
                    # Start runtime error tailer for app containers (OMN-5649)
                    self._start_rt_tailer(container)
                elif action == "die" and container in project_containers:
                    self._stop_tailer(container)

                project_containers = current
        except KeyboardInterrupt:
            print("\n[monitor] Shutting down...")
        finally:
            _restart_stop.set()
            proc.terminate()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor OmniNode container logs and post errors to Slack"
    )
    parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        help="Compose project name to watch (repeatable, default: all OmniNode projects)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print alerts without posting to Slack",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=int(os.getenv("MONITOR_COOLDOWN", DEFAULT_COOLDOWN)),
        help=f"Per-container alert cooldown in seconds (default: {DEFAULT_COOLDOWN})",
    )
    args = parser.parse_args()

    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    channel_id = os.getenv("SLACK_CHANNEL_ID", "")
    if (not bot_token or not channel_id) and not args.dry_run:
        print(
            "ERROR: SLACK_BOT_TOKEN and SLACK_CHANNEL_ID must be set. "
            "Run with --dry-run to test without posting.",
            file=sys.stderr,
        )
        sys.exit(1)

    projects_env = os.getenv("MONITOR_PROJECTS", "")
    projects: list[str]
    if args.projects:
        projects = args.projects
    elif projects_env:
        projects = [p.strip() for p in projects_env.split(",") if p.strip()]
    else:
        projects = DEFAULT_PROJECTS

    monitor = LogMonitor(
        projects=projects,
        bot_token=bot_token,
        channel_id=channel_id,
        cooldown=args.cooldown,
        dry_run=args.dry_run,
    )
    monitor.run()


if __name__ == "__main__":
    main()
