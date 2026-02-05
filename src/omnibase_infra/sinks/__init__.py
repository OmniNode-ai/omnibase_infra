# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ledger sink implementations for runtime traceability.

This module provides sink implementations for the ProtocolLedgerSink interface.
Sinks receive ledger events and handle durability, batching, and delivery.

Sinks:
    - InMemoryLedgerSink: Test-only in-memory buffer (NOT durable)
    - FileSpoolLedgerSink: Durable append-only JSONL with rotation

Usage:
    >>> from omnibase_infra.sinks import InMemoryLedgerSink, FileSpoolLedgerSink
    >>>
    >>> # For unit tests only
    >>> test_sink = InMemoryLedgerSink()
    >>>
    >>> # For production (durable)
    >>> production_sink = FileSpoolLedgerSink(
    ...     spool_dir="/var/log/omninode/ledger",
    ...     max_file_size_bytes=10 * 1024 * 1024,  # 10MB
    ... )

Security:
    Sinks write events as-is. Ensure event models exclude sensitive data
    (raw SQL, credentials, PII) before emission.
"""

from omnibase_infra.sinks.sink_ledger_file_spool import FileSpoolLedgerSink
from omnibase_infra.sinks.sink_ledger_inmemory import InMemoryLedgerSink

__all__ = [
    "FileSpoolLedgerSink",
    "InMemoryLedgerSink",
]
