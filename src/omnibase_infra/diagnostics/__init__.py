# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Bus health diagnostics for ONEX Kafka infrastructure.

Generic, domain-free diagnostic engine. Reusable by any ONEX repository.

Quick Start:
    >>> from omnibase_infra.diagnostics import AuditConfig, run_audit
    >>> config = AuditConfig(
    ...     broker="192.168.86.200:29092",
    ...     expected_topics=["onex.evt.platform.node-registration.v1"],
    ... )
    >>> report = run_audit(config)
    >>> print(report.to_human_readable())

See Also:
    models.py: Frozen dataclasses for audit results.
    bus_audit.py: AuditConfig and run_audit() implementation.
    enum_topic_status.py: EnumTopicStatus enum.
    enum_naming_compliance.py: EnumNamingCompliance enum.
    enum_verdict.py: EnumVerdict enum.
"""

from omnibase_infra.diagnostics.bus_audit import AuditConfig, run_audit
from omnibase_infra.diagnostics.enum_naming_compliance import EnumNamingCompliance
from omnibase_infra.diagnostics.enum_topic_status import EnumTopicStatus
from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.diagnostics.models import (
    AuditReport,
    DomainValidationResult,
    EnvelopeStats,
    TopicHealth,
)

__all__: list[str] = [
    # Config + entry point
    "AuditConfig",
    "run_audit",
    # Enums
    "EnumNamingCompliance",
    "EnumTopicStatus",
    "EnumVerdict",
    # Result models
    "AuditReport",
    "DomainValidationResult",
    "EnvelopeStats",
    "TopicHealth",
]
