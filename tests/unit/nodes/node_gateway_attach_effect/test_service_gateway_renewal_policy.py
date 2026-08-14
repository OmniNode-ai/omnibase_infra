# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the OMN-15952 renewal-cycle policy.

Every test here was RED before ``service_gateway_renewal_policy`` and
``ModelGatewayRenewalDirective`` existed -- the module and the model are the
subject, so there was nothing to import. What is being pinned is not that
some arithmetic is correct but that the arithmetic cannot produce a
directive telling a client to renew at or after its own expiry, and that no
revision of a session may move ``expires_at``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_renewal_mode import (
    EnumGatewayRenewalMode,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_renewal_directive import (
    ModelGatewayRenewalDirective,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.service_gateway_renewal_policy import (
    ExpiryExtensionError,
    assert_expiry_not_extended,
    build_renewal_directive,
    is_renewal_due,
)

pytestmark = pytest.mark.unit

ATTACHED_AT = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)


def _session(*, ttl_seconds: int) -> ModelGatewaySession:
    return ModelGatewaySession(
        session_id=uuid4(),
        tenant_id=uuid4(),
        tenant_slug="acme",
        principal_id="t-acme",
        keycloak_client_id="ga-acme-edge-1",
        edge_instance_id="edge-1",
        status=EnumGatewaySessionStatus.ACTIVE,
        attached_at=ATTACHED_AT,
        last_heartbeat_at=ATTACHED_AT,
        expires_at=ATTACHED_AT + timedelta(seconds=ttl_seconds),
    )


# --------------------------------------------------------------------------- #
# The directive's shape
# --------------------------------------------------------------------------- #


def test_directive_declares_re_attach_not_in_place_renewal() -> None:
    """The mechanism is named on the wire, so no client has to assume it."""
    config = ModelGatewayAttachConfig()
    directive = build_renewal_directive(_session(ttl_seconds=900), config=config)

    assert directive.mode is EnumGatewayRenewalMode.RE_ATTACH
    # There is no second mode to fall back to. If a future change adds one,
    # this assertion is where the contract review starts.
    assert [m.value for m in EnumGatewayRenewalMode] == ["RE_ATTACH"]


def test_directive_window_precedes_expiry_by_the_configured_margin() -> None:
    config = ModelGatewayAttachConfig()
    session = _session(ttl_seconds=900)

    directive = build_renewal_directive(session, config=config)

    assert directive.session_expires_at == session.expires_at
    assert directive.renew_at == session.expires_at - timedelta(
        seconds=config.renewal_margin_seconds
    )
    assert directive.renew_not_before == directive.renew_at - timedelta(
        seconds=config.renewal_jitter_seconds
    )
    assert directive.margin_seconds == config.renewal_margin_seconds
    assert directive.jitter_seconds == config.renewal_jitter_seconds


def test_directive_ordering_invariant_holds_across_token_lifetimes() -> None:
    """renew_not_before <= renew_at < expires_at, for every plausible TTL.

    Including the ones where the margin swallows the whole session: a token
    presented with less life than the configured margin must still yield a
    constructible directive, not a 5xx.
    """
    config = ModelGatewayAttachConfig()
    for ttl in (1, 30, 119, 120, 121, 300, 900, 3600):
        session = _session(ttl_seconds=ttl)
        directive = build_renewal_directive(session, config=config)
        assert directive.renew_not_before <= directive.renew_at, ttl
        assert directive.renew_at < directive.session_expires_at, ttl
        # Never advises a renewal before the session existed.
        assert directive.renew_not_before >= session.attached_at, ttl


def test_directive_for_a_session_shorter_than_the_margin_says_renew_now() -> None:
    """The floor is 'renew immediately', not 'renew in the past'."""
    config = ModelGatewayAttachConfig()
    session = _session(ttl_seconds=30)

    directive = build_renewal_directive(session, config=config)

    assert directive.renew_not_before == session.attached_at
    assert directive.renew_at == session.attached_at
    assert is_renewal_due(session, now=session.attached_at, config=config) is True


def test_directive_rejects_a_renew_at_that_is_not_before_expiry() -> None:
    """Constructed directly -- the model, not the builder, is the guard.

    A builder-only check would be bypassed by any second construction path,
    including deserialization of a payload off the wire.
    """
    expires_at = ATTACHED_AT + timedelta(seconds=900)
    with pytest.raises(ValidationError, match="strictly before session_expires_at"):
        ModelGatewayRenewalDirective(
            mode=EnumGatewayRenewalMode.RE_ATTACH,
            session_expires_at=expires_at,
            renew_not_before=expires_at,
            renew_at=expires_at,
            margin_seconds=120,
            jitter_seconds=30,
        )


def test_directive_rejects_an_inverted_jitter_window() -> None:
    expires_at = ATTACHED_AT + timedelta(seconds=900)
    with pytest.raises(ValidationError, match="renew_not_before must not be later"):
        ModelGatewayRenewalDirective(
            mode=EnumGatewayRenewalMode.RE_ATTACH,
            session_expires_at=expires_at,
            renew_not_before=expires_at - timedelta(seconds=60),
            renew_at=expires_at - timedelta(seconds=120),
            margin_seconds=120,
            jitter_seconds=30,
        )


# --------------------------------------------------------------------------- #
# is_renewal_due
# --------------------------------------------------------------------------- #


def test_renewal_is_not_due_before_the_window_opens() -> None:
    config = ModelGatewayAttachConfig()
    session = _session(ttl_seconds=900)
    directive = build_renewal_directive(session, config=config)

    just_before = directive.renew_not_before - timedelta(seconds=1)
    assert is_renewal_due(session, now=just_before, config=config) is False


def test_renewal_is_due_at_the_instant_the_window_opens() -> None:
    """Inclusive boundary -- an early re-grant costs a token, a late one costs
    the session."""
    config = ModelGatewayAttachConfig()
    session = _session(ttl_seconds=900)
    directive = build_renewal_directive(session, config=config)

    assert (
        is_renewal_due(session, now=directive.renew_not_before, config=config) is True
    )


def test_renewal_stays_due_past_expiry() -> None:
    """Expiry does not retire the obligation; it means the runtime is late."""
    config = ModelGatewayAttachConfig()
    session = _session(ttl_seconds=900)

    assert (
        is_renewal_due(
            session, now=session.expires_at + timedelta(seconds=600), config=config
        )
        is True
    )


def test_jitter_can_be_disabled_without_collapsing_the_window() -> None:
    config = ModelGatewayAttachConfig(renewal_jitter_seconds=0)
    session = _session(ttl_seconds=900)

    directive = build_renewal_directive(session, config=config)

    assert directive.renew_not_before == directive.renew_at
    assert directive.renew_at < directive.session_expires_at


# --------------------------------------------------------------------------- #
# The immutability of expires_at
# --------------------------------------------------------------------------- #


def test_assert_expiry_not_extended_accepts_an_ordinary_heartbeat_revision() -> None:
    session = _session(ttl_seconds=900)
    refreshed = session.model_copy(
        update={
            "last_heartbeat_at": ATTACHED_AT + timedelta(seconds=15),
            "status": EnumGatewaySessionStatus.ACTIVE,
        }
    )

    assert_expiry_not_extended(session, refreshed)


def test_assert_expiry_not_extended_rejects_a_lengthened_ceiling() -> None:
    session = _session(ttl_seconds=900)
    extended = session.model_copy(
        update={"expires_at": session.expires_at + timedelta(seconds=900)}
    )

    with pytest.raises(ExpiryExtensionError, match="immutable"):
        assert_expiry_not_extended(session, extended)


def test_assert_expiry_not_extended_rejects_a_shortened_ceiling_too() -> None:
    """Shortening is not a safe subset -- it is still a second writer."""
    session = _session(ttl_seconds=900)
    shortened = session.model_copy(
        update={"expires_at": session.expires_at - timedelta(seconds=60)}
    )

    with pytest.raises(ExpiryExtensionError):
        assert_expiry_not_extended(session, shortened)
