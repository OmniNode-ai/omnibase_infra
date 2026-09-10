# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Link 5's source is a variable NAME, never a connection string (OMN-16964).

The link-5 leg landed in #3072 with a ``ledger_source`` field taking a RAW
DSN. That is the exact exposure OMN-18060 removed from the sibling projection
leg two weeks later: ``onex skill`` builds the node payload from CLI flags, so
a DSN passed that way lands simultaneously in argv (world-readable through
``/proc/<pid>/cmdline``), in the dispatch step's own echoed log line, and in
the event log as part of the serialised request. Three durable copies of a
credential from one flag.

There is no compatibility shim for the old field. Nothing sets it: no CLI row
in ``skill_mapping.yaml`` ever carried it, so no caller can be broken by the
replacement, and keeping a raw-DSN field alive "just in case" would keep the
exposure alive with it.

No network, no database, no bus.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.models.enum_ledger_replay_status import (
    EnumLedgerReplayStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)

PROBE_URL = "http://127.0.0.1:8085"  # onex-allow-internal-ip test fixture
CORRELATION = "11111111-2222-3333-4444-555555555555"


@pytest.mark.unit
def test_the_raw_dsn_field_is_gone() -> None:
    """``ledger_source`` no longer exists; the field is a NAME now."""
    assert "ledger_source" not in ModelChainCanaryRequest.model_fields
    assert "ledger_source_env" in ModelChainCanaryRequest.model_fields


@pytest.mark.unit
def test_a_dsn_where_a_name_belongs_is_refused_at_the_model_boundary() -> None:
    """A connection string passed as the NAME is refused, value never echoed."""
    offending = "postgresql://someone:somevalue@example.invalid:5432/somedb"

    with pytest.raises(ValueError) as caught:
        ModelChainCanaryRequest(
            correlation_id=CORRELATION, probe_url=PROBE_URL, ledger_source_env=offending
        )

    message = str(caught.value)
    assert "ledger_source_env" in message
    # The refusal must not reproduce the credential it is refusing.
    assert "somevalue" not in message
    assert offending not in message


@pytest.mark.unit
def test_an_ordinary_variable_name_is_accepted() -> None:
    request = ModelChainCanaryRequest(
        correlation_id=CORRELATION,
        probe_url=PROBE_URL,
        ledger_source_env="CHAIN_CANARY_PROJECTION_DSN",
    )
    assert request.ledger_source_env == "CHAIN_CANARY_PROJECTION_DSN"


@pytest.mark.unit
def test_a_credential_free_url_is_not_mistaken_for_a_dsn() -> None:
    """The OMN-18060 #3350 regression: the probe's own address is not a DSN."""
    request = ModelChainCanaryRequest(
        correlation_id=CORRELATION, probe_url=PROBE_URL, ledger_source_env=""
    )
    assert request.ledger_source_env == ""


@pytest.mark.unit
def test_an_unset_variable_makes_no_claim_about_link_five() -> None:
    """Declared but unset resolves to NOT_CONFIGURED, naming the variable."""
    from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
        HandlerChainCanary,
    )

    async def _never_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("the replay must not run without a resolved source")

    handler = HandlerChainCanary(
        ledger_replay=_never_called,  # type: ignore[arg-type]
        ledger_dsn_lookup=lambda _name: "",
    )
    request = ModelChainCanaryRequest(
        correlation_id=CORRELATION,
        probe_url=PROBE_URL,
        ledger_source_env="A_VARIABLE_NOBODY_SET",
    )

    import asyncio

    status, detail = asyncio.run(
        handler._replay_ledger(request, "some-correlation", 1.0)
    )

    assert status is EnumLedgerReplayStatus.SKIPPED_NOT_CONFIGURED
    assert "A_VARIABLE_NOBODY_SET" in detail


@pytest.mark.unit
def test_a_dsn_on_any_flag_refuses_the_leg_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DSN that arrived on some OTHER flag is the same disclosure."""
    from omnibase_infra.nodes.node_chain_canary_effect.handlers import (
        handler_chain_canary as module,
    )

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "onex",
            "skill",
            "chain_canary",
            "--some-other-flag",
            "postgresql://someone:somevalue@example.invalid:5432/somedb",
        ],
    )

    async def _never_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("the replay must not run after a refusal")

    handler = module.HandlerChainCanary(
        ledger_replay=_never_called,  # type: ignore[arg-type]
        ledger_dsn_lookup=lambda _name: "postgresql://resolved",
    )
    request = ModelChainCanaryRequest(
        correlation_id=CORRELATION,
        probe_url=PROBE_URL,
        ledger_source_env="CHAIN_CANARY_PROJECTION_DSN",
    )

    import asyncio

    status, detail = asyncio.run(
        handler._replay_ledger(request, "some-correlation", 1.0)
    )

    assert status is EnumLedgerReplayStatus.REFUSED
    assert "--some-other-flag" in detail
    assert "somevalue" not in detail


@pytest.mark.unit
def test_a_refused_leg_is_not_a_passing_link_five() -> None:
    """REFUSED renders as a non-passing link 5, never as green."""
    from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
        _link_five,
    )
    from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
        EnumChainLinkStatus,
    )

    status, detail = _link_five(EnumLedgerReplayStatus.REFUSED, "some detail")

    assert status is not EnumChainLinkStatus.PASS
    assert detail != ""


@pytest.mark.unit
def test_expected_hops_arrive_from_a_scalar_cli_flag_as_a_tuple() -> None:
    """The workflow passes the declared hops as one comma-separated flag.

    ``skill_mapping.yaml`` arg types are scalar, so without the split this
    would arrive as a single string matching no hop and link 5 would report
    CHAIN_INCOMPLETE against a chain that was in fact complete.
    """
    request = ModelChainCanaryRequest(
        correlation_id=CORRELATION,
        probe_url=PROBE_URL,
        expected_ledger_hops=(
            "onex.cmd.omnimarket.delegate-skill.v1, "
            "onex.evt.omnimarket.delegate-skill-completed.v1"
        ),
    )

    assert request.expected_ledger_hops == (
        "onex.cmd.omnimarket.delegate-skill.v1",
        "onex.evt.omnimarket.delegate-skill-completed.v1",
    )
