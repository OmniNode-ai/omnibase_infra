# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the session lifetime policy (OMN-16022).

``test_handlers.py`` proves the handlers apply these bounds. This file
pins the bounds themselves: the boundary condition of each, and — the
point of the ticket — that the two constants stay distinct. The 900s
revalidation ceiling and the 3600s session ceiling were conflated in the
original ticket text and in two revisions of the design doc; a test is
cheaper than another round of that.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_gateway_session_policy as session_policy,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)


def _session(**overrides: object) -> ModelGatewaySession:
    base: dict[str, object] = {
        "session_id": uuid4(),
        "tenant_id": TENANT_ID,
        "tenant_slug": "acme",
        "principal_id": "t-acme",
        "keycloak_client_id": "gw-tenant-acme",
        "edge_instance_id": "edge-201",
        "status": EnumGatewaySessionStatus.ACTIVE,
        "attached_at": NOW,
        "last_heartbeat_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
    }
    base.update(overrides)
    return ModelGatewaySession(**base)  # type: ignore[arg-type]


class TestIsExpired:
    def test_before_expiry_is_not_expired(self) -> None:
        session = _session(expires_at=NOW + timedelta(seconds=1))
        assert session_policy.is_expired(session, now=NOW) is False

    def test_exactly_at_expiry_is_expired(self) -> None:
        """Inclusive boundary: fail-closed is the cheap direction here."""
        session = _session(expires_at=NOW)
        assert session_policy.is_expired(session, now=NOW) is True

    def test_after_expiry_is_expired(self) -> None:
        session = _session(expires_at=NOW - timedelta(seconds=1))
        assert session_policy.is_expired(session, now=NOW) is True


class TestUnverifiedCeiling:
    def test_freshly_validated_session_is_under_the_ceiling(self) -> None:
        config = ModelGatewayAttachConfig()
        assert (
            session_policy.exceeds_unverified_ceiling(
                _session(last_heartbeat_at=NOW), now=NOW, config=config
            )
            is False
        )

    def test_exactly_at_the_ceiling_is_still_under_it(self) -> None:
        """Strictly-greater comparison: the ceiling is a duration allowed, not forbidden."""
        config = ModelGatewayAttachConfig()
        session = _session(
            last_heartbeat_at=NOW
            - timedelta(seconds=config.max_unverified_session_seconds)
        )
        assert (
            session_policy.exceeds_unverified_ceiling(session, now=NOW, config=config)
            is False
        )

    def test_one_second_past_the_ceiling_breaches(self) -> None:
        config = ModelGatewayAttachConfig()
        session = _session(
            last_heartbeat_at=NOW
            - timedelta(seconds=config.max_unverified_session_seconds + 1)
        )
        assert (
            session_policy.exceeds_unverified_ceiling(session, now=NOW, config=config)
            is True
        )

    def test_ceiling_is_measured_from_last_successful_validation(self) -> None:
        config = ModelGatewayAttachConfig()
        stale = _session(
            last_heartbeat_at=NOW - timedelta(seconds=1200),
            # Deliberately far from expiry: a session can breach the
            # revalidation ceiling long before it breaches its own lifetime.
            expires_at=NOW + timedelta(hours=1),
        )
        assert session_policy.is_expired(stale, now=NOW) is False
        assert (
            session_policy.exceeds_unverified_ceiling(stale, now=NOW, config=config)
            is True
        )


class TestConstantsStayDistinct:
    def test_revalidation_ceiling_is_one_attach_token_lifetime(self) -> None:
        assert ModelGatewayAttachConfig().max_unverified_session_seconds == 900

    def test_revalidation_ceiling_is_not_the_session_ceiling(self) -> None:
        """The bug this ticket exists to prevent a repeat of.

        ``max_session_ttl_seconds`` bounds how long a session may live;
        ``max_unverified_session_seconds`` bounds how long it may live
        un-rechecked. Collapsing them to one number silently makes one of
        the two bounds wrong.
        """
        config = ModelGatewayAttachConfig()
        assert config.max_unverified_session_seconds != config.max_session_ttl_seconds
        assert config.max_unverified_session_seconds < config.max_session_ttl_seconds
