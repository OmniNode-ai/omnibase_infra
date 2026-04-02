# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Platform capability probe for tier detection.

Extracted from omniclaude/plugins/onex/hooks/lib/capability_probe.py (OMN-5265).
Pure stdlib -- no ONEX imports. Probes available services and writes tier
information to a capabilities file.

Tiers:
    standalone  - No Kafka or none reachable
    event_bus   - Any Kafka host reachable
    full_onex   - Kafka reachable + intelligence /health -> 200

Function renames from omniclaude original:
    _socket_check()      -> socket_check()
    _kafka_reachable()   -> kafka_reachable()
    _http_check()        -> http_health_check()
    probe_tier()         -> probe_platform_tier()
    write_atomic()       -> write_capabilities_atomic()
    read_capabilities()  -> read_capabilities_cached()
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CAPABILITIES_FILE = Path.home() / ".claude" / ".onex_capabilities"
PROBE_TTL_SECONDS = 300  # 5 minutes

TierName = Literal["standalone", "event_bus", "full_onex"]


# ---------------------------------------------------------------------------
# Low-level probes
# ---------------------------------------------------------------------------


def socket_check(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout.

    Args:
        host: Hostname or IP address.
        port: TCP port number.
        timeout: Connection timeout in seconds.

    Returns:
        True if the connection succeeds, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def kafka_reachable(servers: str, timeout: float = 1.0) -> bool:
    """Return True if any Kafka bootstrap server is reachable.

    Args:
        servers: Comma-separated list of host:port entries.
        timeout: Per-host socket timeout in seconds.

    Returns:
        True if at least one host:port is reachable.
    """
    if not servers or not servers.strip():
        return False
    for entry in servers.split(","):
        entry = entry.strip()
        if not entry:
            continue
        host, _, port_str = entry.partition(":")
        if not host or not port_str:
            continue
        try:
            port = int(port_str)
        except ValueError:
            continue
        if socket_check(host, port, timeout):
            return True
    return False


def http_health_check(url: str, timeout: float = 1.0) -> bool:
    """Return True if HTTP GET to url returns 2xx.

    Args:
        url: Full URL to check.
        timeout: HTTP request timeout in seconds.

    Returns:
        True if the response status is 2xx, False otherwise.
    """
    try:
        req = urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
        return 200 <= int(req.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------


def probe_platform_tier(
    kafka_servers: str = "",
    intelligence_url: str = os.environ.get("INTELLIGENCE_URL", ""),
    kafka_timeout: float = 1.0,
    intel_timeout: float = 1.0,
) -> TierName:
    """Probe available services and return the detected tier.

    Args:
        kafka_servers: Comma-separated Kafka bootstrap servers.
        intelligence_url: Base URL of the intelligence service.
        kafka_timeout: Timeout per Kafka host probe in seconds.
        intel_timeout: Timeout for intelligence /health check in seconds.

    Returns:
        One of "standalone", "event_bus", or "full_onex".
    """
    kafka_ok = kafka_reachable(kafka_servers, timeout=kafka_timeout)

    if not kafka_ok:
        return "standalone"

    intel_health = f"{intelligence_url.rstrip('/')}/health"
    intel_ok = http_health_check(intel_health, timeout=intel_timeout)

    if intel_ok:
        return "full_onex"

    return "event_bus"


# ---------------------------------------------------------------------------
# Atomic file I/O with TTL
# ---------------------------------------------------------------------------


def write_capabilities_atomic(
    data: dict[str, object],
    capabilities_file: Path | None = None,
) -> None:
    """Write capabilities data atomically to the capabilities file.

    Uses a .tmp suffix then rename for POSIX atomicity.

    Args:
        data: Dictionary to serialize as JSON.
        capabilities_file: Path to the capabilities file. Defaults to
            DEFAULT_CAPABILITIES_FILE.
    """
    target = capabilities_file or DEFAULT_CAPABILITIES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.rename(target)


def read_capabilities_cached(
    capabilities_file: Path | None = None,
) -> dict[str, object] | None:
    """Read capabilities from file, returning None if missing or stale.

    Args:
        capabilities_file: Path to the capabilities file. Defaults to
            DEFAULT_CAPABILITIES_FILE.

    Returns:
        Parsed capabilities dict, or None if the file is absent or older
        than PROBE_TTL_SECONDS.
    """
    target = capabilities_file or DEFAULT_CAPABILITIES_FILE
    if not target.exists():
        return None
    try:
        raw = target.read_text(encoding="utf-8")
        data: dict[str, object] = json.loads(raw)
        probed_at_str = data.get("probed_at")
        if not isinstance(probed_at_str, str):
            return None
        probed_at = datetime.fromisoformat(probed_at_str)
        # Normalize to UTC for comparison
        if probed_at.tzinfo is None:
            probed_at = probed_at.replace(tzinfo=UTC)
        now = datetime.now(tz=UTC)
        age = (now - probed_at).total_seconds()
        if age > PROBE_TTL_SECONDS:
            return None  # stale
        return data
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.debug("Failed to read capabilities file: %s", exc)
        return None


def run_probe(
    kafka_servers: str,
    intelligence_url: str,
    capabilities_file: Path | None = None,
) -> TierName:
    """Run a full probe cycle: detect tier, write result, return tier.

    Args:
        kafka_servers: Comma-separated Kafka bootstrap servers (may be empty).
        intelligence_url: Base URL of the intelligence service.
        capabilities_file: Path to write capabilities. Defaults to
            DEFAULT_CAPABILITIES_FILE.

    Returns:
        The detected tier name.
    """
    tier = probe_platform_tier(
        kafka_servers=kafka_servers,
        intelligence_url=intelligence_url,
    )
    now_utc = datetime.now(tz=UTC)
    data: dict[str, object] = {
        "tier": tier,
        "probed_at": now_utc.isoformat(),
        "kafka_servers": kafka_servers,
        "intelligence_url": intelligence_url,
    }
    write_capabilities_atomic(data, capabilities_file=capabilities_file)
    return tier
