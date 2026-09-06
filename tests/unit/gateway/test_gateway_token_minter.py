# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""machine grant + attach-token exchange + audience gates + skew (OMN-16687).

Two hops, two audience assertions, and they mirror two DIFFERENT server-side
rules:

* the exchange INPUT rule (``gateway_auth.validate_exchange_input_claims``):
  ``aud`` minus the role-resolved audiences must CONTAIN ``redpanda-events``
  and name nothing outside ``{"redpanda-events", "onex-api"}``, and a token
  already carrying ``gateway-attach`` is refused outright. Two set assertions,
  not one equality -- see OMN-16946, and OMN-15922 for the equality mirror
  that refused the live credential the server had just accepted;
* the attach rule (``gateway_auth.py::_assert_exact_audience``): exact SET
  equality against ``{"gateway-attach"}``, list- or string-valued ``aud``
  normalised first.

Mirroring them client-side does not make the client a security authority --
it cannot, it verifies no signature. It makes the wrong-credential case fail
at the hop that can still explain which credential was held and which was
needed, instead of as an opaque 401 several calls later.

The pre-OMN-16687 shape asserted ``{"gateway-attach"}`` on the DIRECT grant,
which no tenant-holdable credential can ever satisfy: the P0B provisioner
splits the two audiences across two clients and the attach client's secret
never leaves onex-api. That assertion could only ever fail, which is why the
gateway_attach check could not go green on onex-dev.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.gateway.client.gateway_token_minter import (
    EXCHANGE_INPUT_PERMITTED_AUDIENCES,
    EXCHANGE_INPUT_REQUIRED_AUDIENCES,
    GATEWAY_ATTACH_AUDIENCES,
    GATEWAY_TOKEN_EXCHANGE_PATH,
    ROLE_RESOLVED_AUDIENCES,
    GatewayTokenMinter,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)
from tests.unit.gateway.conftest import (
    CLIENT_ID,
    CLIENT_SECRET,
    GATEWAY_BASE_URL,
    TOKEN_ENDPOINT,
    FakeGatewayTransport,
)

# asyncio_mode=auto in this repo's pytest config runs the async tests here;
# an explicit asyncio mark would misfire on the sync tests in the same module.
pytestmark = pytest.mark.unit


def _credential() -> ModelGatewayCredential:
    return ModelGatewayCredential(
        tenant_slug="acme",
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        token_endpoint=TOKEN_ENDPOINT,
        base_url=GATEWAY_BASE_URL,
        edge_instance_id="test-edge",
    )


def _minter(transport: FakeGatewayTransport) -> GatewayTokenMinter:
    return GatewayTokenMinter(transport=transport, credential=_credential())


def test_the_contract_audience_set_is_exactly_gateway_attach() -> None:
    assert frozenset({"gateway-attach"}) == GATEWAY_ATTACH_AUDIENCES


def test_the_exchange_input_audience_sets_mirror_the_servers_two_sets() -> None:
    """Byte-for-byte the server's own constants (OMN-16946).

    ``gateway_auth.EXCHANGE_INPUT_REQUIRED_AUDIENCES`` /
    ``EXCHANGE_INPUT_PERMITTED_AUDIENCES`` in omninode_infra
    ``docker/onex-api/gateway_auth.py``. Two sets, not one: the input rule is
    REQUIRED-is-a-subset plus effective-is-a-subset-of-PERMITTED, never
    equality. Pinning both here is what stops this client drifting stricter
    than the server again.
    """
    assert frozenset({"redpanda-events"}) == EXCHANGE_INPUT_REQUIRED_AUDIENCES
    assert (
        frozenset({"redpanda-events", "onex-api"}) == EXCHANGE_INPUT_PERMITTED_AUDIENCES
    )
    assert EXCHANGE_INPUT_REQUIRED_AUDIENCES <= EXCHANGE_INPUT_PERMITTED_AUDIENCES
    assert frozenset({"account"}) == ROLE_RESOLVED_AUDIENCES
    assert not (EXCHANGE_INPUT_PERMITTED_AUDIENCES & GATEWAY_ATTACH_AUDIENCES)


async def test_a_machine_grant_is_exchanged_for_an_attach_token(
    fake_transport: FakeGatewayTransport,
) -> None:
    """The whole OMN-16687 fix in one assertion pair.

    The credential grants ``redpanda-events`` -- the only audience a tenant
    can hold -- and what comes back carries ``gateway-attach``, because the
    token was minted by the exchange rather than by the realm directly.
    """
    token = await _minter(fake_transport).token_for(now=fake_transport.now)

    assert fake_transport.audiences == ["redpanda-events"]
    assert token.audiences == GATEWAY_ATTACH_AUDIENCES
    assert token.expires_at == fake_transport.now + timedelta(seconds=900)

    # The token handed back is the EXCHANGED one, never the machine token the
    # realm issued -- presenting the latter to the gateway is a 401.
    assert token.access_token.get_secret_value() == fake_transport.exchanged_tokens[-1]
    assert token.access_token.get_secret_value() not in fake_transport.issued_tokens

    url, form = fake_transport.form_requests[-1]
    assert url == TOKEN_ENDPOINT
    assert form["grant_type"] == "client_credentials"
    assert "refresh_token" not in form


async def test_the_exchange_presents_the_machine_token_and_names_no_tenant(
    fake_transport: FakeGatewayTransport,
) -> None:
    """The tenant comes from verified claims, so the request carries none."""
    await _minter(fake_transport).token_for(now=fake_transport.now)

    url, body, headers = fake_transport.json_requests[-1]
    assert url == f"{GATEWAY_BASE_URL}{GATEWAY_TOKEN_EXCHANGE_PATH}"
    assert headers["Authorization"] == f"Bearer {fake_transport.issued_tokens[-1]}"
    assert json.loads(body) == {}
    assert CLIENT_SECRET not in body


async def test_a_base_url_with_a_trailing_slash_does_not_produce_a_doubled_path(
    fake_transport: FakeGatewayTransport,
) -> None:
    credential = _credential().model_copy(update={"base_url": GATEWAY_BASE_URL + "/"})
    minter = GatewayTokenMinter(transport=fake_transport, credential=credential)

    await minter.token_for(now=fake_transport.now)

    url, _, _ = fake_transport.json_requests[-1]
    assert url == f"{GATEWAY_BASE_URL}{GATEWAY_TOKEN_EXCHANGE_PATH}"


async def test_an_attach_audience_credential_is_refused_as_exchange_input(
    fake_transport: FakeGatewayTransport,
) -> None:
    """The ga-* secret is not the credential to store, even if one had it.

    The exchange refuses its own output, so a client holding an attach-audience
    credential must be told that at the grant rather than discover it as a bare
    401 from the exchange.
    """
    fake_transport.audiences = ["gateway-attach"]

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    message = str(caught.value)
    assert "gateway-attach" in message
    assert "rotate" in message
    # Refused locally -- the exchange was never called with it.
    assert fake_transport.exchange_count == 0
    assert fake_transport.json_requests == []


async def test_the_p0b_dual_audience_credential_is_accepted_because_the_server_accepts_it(
    fake_transport: FakeGatewayTransport,
) -> None:
    """The live P0B credential, and the whole of OMN-15922 DoD 2.

    The provisioner stamps BOTH ``redpanda-events`` and ``onex-api`` on a
    tenant's one durable ``t-*`` machine client, because that credential is
    presented at two resource servers. The exchange accepts exactly that
    (``gateway_auth.validate_exchange_input_claims``: REQUIRED is a subset of
    the effective set, and the effective set is a subset of PERMITTED). This
    client asserted bare equality with ``{"redpanda-events"}`` instead, so it
    refused locally -- with ONEX_CORE_163_AUTHENTICATION_ERROR -- a credential
    the gateway had accepted with HTTP 200 minutes earlier. A client mirror
    stricter than the rule it mirrors is a client that cannot mint for any
    tenant the server would honour.
    """
    fake_transport.audiences = ["redpanda-events", "onex-api", "account"]

    token = await _minter(fake_transport).token_for(now=fake_transport.now)

    assert token.audiences == GATEWAY_ATTACH_AUDIENCES
    assert fake_transport.exchange_count == 1


async def test_an_audience_outside_the_permitted_set_is_still_refused(
    fake_transport: FakeGatewayTransport,
) -> None:
    """PERMITTED is a closed allowlist, not membership.

    Widening to accept the server's real rule must not widen past it: an
    IdP-side mapper adding an audience this platform does not mint is the
    case ``EXCHANGE_INPUT_PERMITTED_AUDIENCES`` exists to fail closed on, and
    the server fails closed on it at ``effective - PERMITTED``.
    """
    fake_transport.audiences = ["redpanda-events", "some-other-resource"]

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    message = str(caught.value)
    assert "some-other-resource" in message
    assert fake_transport.exchange_count == 0


async def test_a_token_carrying_only_role_resolved_audiences_is_refused(
    fake_transport: FakeGatewayTransport,
) -> None:
    """The effective set can be EMPTY, and that is a refusal, not a pass.

    ``account`` is discounted before the comparison, so a token carrying
    nothing else reduces to an empty effective set. Empty is a distinct
    branch from the ``onex-api``-only case above: it is the one input for
    which ``EXCHANGE_INPUT_REQUIRED_AUDIENCES - effective`` is the whole
    REQUIRED set and the error path renders ``sorted(effective)`` as an
    empty list. A subset assertion written as ``REQUIRED <= effective``
    reads as satisfiable-by-vacuum to anyone skimming it, so the vacuum case
    is pinned rather than assumed. The server refuses it at the same line.
    """
    fake_transport.audiences = ["account"]

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    message = str(caught.value)
    assert "[]" in message
    assert "redpanda-events" in message
    assert fake_transport.exchange_count == 0


async def test_a_credential_without_the_broker_audience_is_refused(
    fake_transport: FakeGatewayTransport,
) -> None:
    """REQUIRED is a subset assertion: an ``onex-api``-only token is not a
    broker-class P0B credential, and the server says so at
    ``REQUIRED - effective``."""
    fake_transport.audiences = ["onex-api"]

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    assert "redpanda-events" in str(caught.value)
    assert fake_transport.exchange_count == 0


async def test_the_role_resolved_audience_is_discounted_the_way_the_server_discounts_it(
    fake_transport: FakeGatewayTransport,
) -> None:
    """Keycloak adds 'account' on its own; rejecting it would reject reality."""
    fake_transport.audiences = ["redpanda-events", "account"]

    token = await _minter(fake_transport).token_for(now=fake_transport.now)

    assert token.audiences == GATEWAY_ATTACH_AUDIENCES


async def test_an_exchange_that_mints_the_wrong_audience_is_refused(
    fake_transport: FakeGatewayTransport,
) -> None:
    """The set-equality assertion, now on a token that can actually vary.

    Pre-OMN-16687 this assertion sat on the direct grant, where no real
    credential could satisfy it. Here it guards a real server-side defect: an
    exchange minting against a drifted client.
    """
    fake_transport.exchange_audiences = ["gateway-attach", "redpanda-events"]

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    message = str(caught.value)
    assert "server-side defect" in message
    assert "gateway-attach" in message


async def test_a_refused_exchange_names_the_two_things_that_cause_it(
    fake_transport: FakeGatewayTransport,
) -> None:
    fake_transport.exchange_status = 503

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    message = str(caught.value)
    assert "503" in message
    assert GATEWAY_TOKEN_EXCHANGE_PATH in message
    assert CLIENT_SECRET not in message


async def test_a_string_valued_aud_claim_is_accepted_the_same_as_a_single_element_list(
    fake_transport: FakeGatewayTransport,
) -> None:
    """RFC 7519 4.1.3 permits both spellings; only the SET is observable."""
    single = await _minter(fake_transport).token_for(now=fake_transport.now)

    listed = FakeGatewayTransport(
        token_endpoint=TOKEN_ENDPOINT,
        gateway_base_url=GATEWAY_BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        exchange_audiences=["gateway-attach", "gateway-attach"],
    )
    listed_token = await _minter(listed).token_for(now=listed.now)

    assert single.audiences == listed_token.audiences == GATEWAY_ATTACH_AUDIENCES


async def test_a_rejected_credential_fails_closed_and_never_echoes_the_secret(
    fake_transport: FakeGatewayTransport,
) -> None:
    fake_transport.token_endpoint_status = 401

    with pytest.raises(ModelOnexError) as caught:
        await _minter(fake_transport).token_for(now=fake_transport.now)

    assert CLIENT_SECRET not in str(caught.value)
    assert CLIENT_SECRET not in repr(caught.value)


async def test_a_cached_token_is_reused_until_it_enters_the_skew_window(
    fake_transport: FakeGatewayTransport,
) -> None:
    minter = _minter(fake_transport)
    start = fake_transport.now

    await minter.token_for(now=start)
    await minter.token_for(now=start + timedelta(seconds=600))
    assert len(fake_transport.form_requests) == 1
    # The exchange is inside the cached path too: a cache that saved the grant
    # but re-exchanged every call would still be one round trip per call.
    assert fake_transport.exchange_count == 1

    # exp - skew: one mint per token lifetime, not one per call.
    inside_skew = start + timedelta(seconds=900) - timedelta(seconds=30)
    fake_transport.now = inside_skew
    await minter.token_for(now=inside_skew)
    assert len(fake_transport.form_requests) == 2
    assert fake_transport.exchange_count == 2


async def test_an_expired_cached_token_is_never_handed_back(
    fake_transport: FakeGatewayTransport,
) -> None:
    minter = _minter(fake_transport)
    start = fake_transport.now
    first = await minter.token_for(now=start)

    after_expiry = start + timedelta(seconds=1200)
    fake_transport.now = after_expiry
    second = await minter.token_for(now=after_expiry)

    assert (
        second.access_token.get_secret_value() != first.access_token.get_secret_value()
    )
    assert second.expires_at > after_expiry


async def test_the_grant_never_asks_for_a_refresh_token(
    fake_transport: FakeGatewayTransport,
) -> None:
    """client_credentials issues no refresh_token; renewal is re-grant."""
    await _minter(fake_transport).token_for(now=fake_transport.now)

    _, form = fake_transport.form_requests[-1]
    assert form["grant_type"] == "client_credentials"
    assert all("refresh" not in key for key in form)


async def test_a_malformed_token_response_refuses_rather_than_minting_nothing_useful(
    fake_transport: FakeGatewayTransport,
) -> None:
    class _Broken(FakeGatewayTransport):
        async def post_form(self, url: str, *, form: object, headers: object) -> object:  # type: ignore[override]
            from tests.unit.gateway.conftest import FakeHttpResponse

            return FakeHttpResponse(200, '{"token_type":"Bearer"}')

    broken = _Broken(
        token_endpoint=TOKEN_ENDPOINT,
        gateway_base_url=GATEWAY_BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

    with pytest.raises(ModelOnexError) as caught:
        await _minter(broken).token_for(now=datetime.now(UTC))

    assert "access_token" in str(caught.value)
