#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# monitor_logs.py -- Real-time container log monitoring with Slack alerts
#
# Watches OmniNode containers, filters for ERROR/CRITICAL/exception lines,
# and posts rate-limited alerts to Slack. Dynamically picks up containers
# as they start and drops them when they stop.
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

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path


def _load_omnibase_env() -> None:
    """Load ~/.omnibase/.env into os.environ if SLACK_WEBHOOK_URL is not set.

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
    return min(_BACKOFF_BASE * (2**count), _BACKOFF_CAP)


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
# Slack
# ---------------------------------------------------------------------------


def post_slack(
    bot_token: str, channel_id: str, container: str, lines: list[str], dry_run: bool
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = "\n".join(lines)[:MAX_SLACK_CHARS]

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
                print(
                    f"[monitor] Slack API error for {container}: {result.get('error')}",
                    file=sys.stderr,
                )
    except Exception as exc:
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
        except Exception as exc:
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

    def run(self) -> None:
        # Start tailers for already-running containers
        for container in self._get_project_containers():
            self._start_tailer(container)

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
                elif action == "die" and container in project_containers:
                    self._stop_tailer(container)

                project_containers = current
        except KeyboardInterrupt:
            print("\n[monitor] Shutting down...")
        finally:
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
