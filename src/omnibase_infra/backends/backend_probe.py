# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Backend health probes for onex.backends entry point discovery.

Each backend (Kafka, Postgres) provides a probe function that returns a
ModelProbeResult with a 4-state severity:

    DISCOVERED  — entry point found, no connectivity attempted
    REACHABLE   — TCP connect succeeded (or auth failed after connect)
    HEALTHY     — basic operations succeed (topic list, SELECT 1)
    AUTHORITATIVE — safe to replace the local default for this protocol

Authority doctrine:
    AUTHORITATIVE means the backend is ready to be the *sole* provider of
    its protocol at runtime. For Kafka this means brokers match env config
    and at least topic listing works. For Postgres this means the required
    schema tables exist.
"""

from __future__ import annotations

import logging
import os
import socket

from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult

logger = logging.getLogger(__name__)


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def probe_kafka(
    *,
    bootstrap_servers: str | None = None,
    timeout: float = 2.0,
) -> ModelProbeResult:
    """Probe Kafka/Redpanda backend health.

    Probe stages:
        1. TCP connect to first broker → REACHABLE
        2. Topic list via confluent_kafka AdminClient → HEALTHY
        3. Brokers match env config → AUTHORITATIVE

    Auth failure at any stage results in REACHABLE (not HEALTHY).

    Args:
        bootstrap_servers: Comma-separated broker addresses.
            Defaults to KAFKA_BOOTSTRAP_SERVERS env var.
        timeout: TCP connection timeout in seconds.

    Returns:
        ModelProbeResult with probe state and reason.
    """
    backend_name = "event_bus_kafka"

    if bootstrap_servers is None:
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")

    if not bootstrap_servers:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason="KAFKA_BOOTSTRAP_SERVERS not set",
            backend_label=backend_name,
        )

    # Parse first broker for TCP check
    first_broker = bootstrap_servers.split(",")[0].strip()
    parts = first_broker.rsplit(":", 1)
    if len(parts) != 2:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"Cannot parse broker address: {first_broker}",
            backend_label=backend_name,
        )

    host, port_str = parts
    try:
        port = int(port_str)
    except ValueError:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"Invalid port in broker address: {first_broker}",
            backend_label=backend_name,
        )

    # Stage 1: TCP reachability
    if not _tcp_reachable(host, port, timeout=timeout):
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"TCP connect to {host}:{port} failed",
            backend_label=backend_name,
        )

    # Stage 2: Topic listing via AdminClient
    try:
        from confluent_kafka.admin import AdminClient

        admin = AdminClient(
            {
                "bootstrap.servers": bootstrap_servers,
                "socket.timeout.ms": int(timeout * 1000),
                "request.timeout.ms": int(timeout * 1000),
            }
        )
        cluster_metadata = admin.list_topics(timeout=timeout)
        topic_count = len(cluster_metadata.topics)
    except ImportError:
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="confluent_kafka not installed; TCP reachable but cannot list topics",
            backend_label=backend_name,
        )
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        reason = str(exc)
        # Auth failures are REACHABLE, not HEALTHY
        if "auth" in reason.lower() or "sasl" in reason.lower():
            return ModelProbeResult(
                state=EnumProbeState.REACHABLE,
                reason=f"Auth failure: {reason}",
                backend_label=backend_name,
            )
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason=f"TCP reachable but topic list failed: {reason}",
            backend_label=backend_name,
        )

    # Stage 3: Authority check — brokers returned match env config
    try:
        returned_brokers = {b.host for b in cluster_metadata.brokers.values()}
        configured_hosts = {
            addr.split(",")[0].rsplit(":", 1)[0].strip() for addr in [bootstrap_servers]
        }
        brokers_match = bool(returned_brokers & configured_hosts)
    except Exception:  # noqa: BLE001 — best-effort authority check
        brokers_match = False

    if brokers_match and topic_count >= 0:
        return ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason=f"Kafka healthy with {topic_count} topics, brokers match config",
            backend_label=backend_name,
        )

    return ModelProbeResult(
        state=EnumProbeState.HEALTHY,
        reason=f"Kafka reachable with {topic_count} topics but broker mismatch",
        backend_label=backend_name,
    )


def probe_postgres(
    *,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    dbname: str | None = None,
    timeout: float = 2.0,
    required_tables: tuple[str, ...] = ("snapshots", "projections"),
) -> ModelProbeResult:
    """Probe PostgreSQL backend health.

    Probe stages:
        1. TCP connect → REACHABLE
        2. SELECT 1 via psycopg2 → HEALTHY
        3. Required schema tables exist → AUTHORITATIVE

    Auth failure at any stage results in REACHABLE (not HEALTHY).

    Args:
        host: Postgres host. Defaults to localhost.
        port: Postgres port. Defaults to PGPORT env var or 5436.
        user: Postgres user. Defaults to PGUSER or "postgres".
        password: Postgres password. Defaults to POSTGRES_PASSWORD env var.
        dbname: Database name. Defaults to PGDATABASE or "omnibase_infra".
        timeout: TCP connection timeout in seconds.
        required_tables: Tables that must exist for AUTHORITATIVE state.

    Returns:
        ModelProbeResult with probe state and reason.
    """
    backend_name = "state_postgres"

    effective_host: str = host or os.getenv("PGHOST", "localhost") or "localhost"
    try:
        effective_port = port or int(os.getenv("PGPORT", "5436"))
    except ValueError:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"Invalid PGPORT value: {os.getenv('PGPORT', '')}",
            backend_label=backend_name,
        )
    effective_user = user or os.getenv("PGUSER", "postgres")
    effective_password = password or os.getenv("POSTGRES_PASSWORD", "")
    effective_dbname = dbname or os.getenv("PGDATABASE", "omnibase_infra")

    # Stage 1: TCP reachability
    if not _tcp_reachable(effective_host, effective_port, timeout=timeout):
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"TCP connect to {effective_host}:{effective_port} failed",
            backend_label=backend_name,
        )

    # Stage 2: SELECT 1 via psycopg2
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=effective_host,
            port=effective_port,
            user=effective_user,
            password=effective_password,
            dbname=effective_dbname,
            connect_timeout=int(timeout),
        )
    except ImportError:
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="psycopg2 not installed; TCP reachable but cannot query",
            backend_label=backend_name,
        )
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        reason = str(exc)
        if "auth" in reason.lower() or "password" in reason.lower():
            return ModelProbeResult(
                state=EnumProbeState.REACHABLE,
                reason=f"Auth failure: {reason}",
                backend_label=backend_name,
            )
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason=f"TCP reachable but connect failed: {reason}",
            backend_label=backend_name,
        )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        conn.close()
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason=f"Connected but SELECT 1 failed: {exc}",
            backend_label=backend_name,
        )

    # Stage 3: Schema table check for authority
    if not required_tables:
        conn.close()
        return ModelProbeResult(
            state=EnumProbeState.HEALTHY,
            reason="SELECT 1 succeeded, no required tables specified",
            backend_label=backend_name,
        )

    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(required_tables))
            cur.execute(
                f"SELECT table_name FROM information_schema.tables "  # noqa: S608 — parameterized via %s
                f"WHERE table_schema = 'public' AND table_name IN ({placeholders})",
                required_tables,
            )
            found_tables = {row[0] for row in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        conn.close()
        return ModelProbeResult(
            state=EnumProbeState.HEALTHY,
            reason=f"SELECT 1 succeeded but schema check failed: {exc}",
            backend_label=backend_name,
        )
    finally:
        conn.close()

    missing = set(required_tables) - found_tables
    if missing:
        return ModelProbeResult(
            state=EnumProbeState.HEALTHY,
            reason=f"Missing required tables: {sorted(missing)}",
            backend_label=backend_name,
        )

    return ModelProbeResult(
        state=EnumProbeState.AUTHORITATIVE,
        reason=f"Postgres healthy with all required tables: {sorted(required_tables)}",
        backend_label=backend_name,
    )
