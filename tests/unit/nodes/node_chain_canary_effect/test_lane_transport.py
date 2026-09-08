# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The canary's broker comes from the lane declaration, never a host literal.

OMN-17926. ``chain-canary.yml`` defaulted both broker legs to a
Docker-Desktop host alias. From 2026-09-07T17:44Z that took 25 consecutive
runs red with ``quarantine_probe_failed`` — every one of which had already
fired its delegation successfully, so a live chain was reported unreadable.

The literal was wrong in two independent ways, and both are pinned below:

1. **It named a host nothing else can resolve.** The alias works only through
   one runner's ``extra_hosts: host-gateway`` compose mapping. Two runners
   reading the same string reach different machines, or nothing.
2. **It carried no transport.** OMN-18012 Phase B enabled SASL/SCRAM-SHA-256
   on that listener at ~16:40Z on 2026-09-07 — the hour between the last
   passing readback and the first failure. An address literal cannot say that
   a listener started requiring SASL, so the client kept opening plaintext and
   was disconnected mid-handshake.

The lane declaration carries both halves together, which is why the fix reads
it rather than correcting the address. Credentials are NOT in it and never
reach a command line: the overlay declares the protocol and mechanism, and
``KAFKA_SASL_USERNAME`` / ``KAFKA_SASL_PASSWORD`` arrive as environment.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
    DOCKER_DESKTOP_HOST_ALIASES,
    ModelLaneTransport,
    host_aliases_in,
    lane_transport_env,
    load_lane_transport,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)

pytestmark = pytest.mark.unit

#: The exact address the workflow shipped, kept in one place so no test in
#: this file has to spell it twice.
_OLD_LITERAL = "host.docker.internal:19092"

#: Shaped after the live omnimarket config/ci_bus_lanes.yaml `dev` lane as it
#: stands after OMN-18012 Phase B: a real host, SASL over PLAINTEXT, no TLS.
_DECLARED_OVERLAY = """
default: inmemory

lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
  stability:
    broker: inmemory
"""


def _overlay(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _request(**overrides: object) -> ModelChainCanaryRequest:
    payload: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": "http://host.docker.internal:8085",
        "task_type": "test",
    }
    payload.update(overrides)
    return ModelChainCanaryRequest(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Resolution from the declaration
# ---------------------------------------------------------------------------


def test_the_bootstrap_resolves_from_the_lane_declaration(tmp_path: Path) -> None:
    transport = load_lane_transport(_overlay(tmp_path, _DECLARED_OVERLAY), "dev")
    assert transport.bootstrap_servers == "broker.example.internal:19092"
    assert transport.security_protocol == "SASL_PLAINTEXT"
    assert transport.sasl_mechanism == "SCRAM-SHA-256"


def test_the_resolved_address_is_never_a_host_alias(tmp_path: Path) -> None:
    transport = load_lane_transport(_overlay(tmp_path, _DECLARED_OVERLAY), "dev")
    assert host_aliases_in(transport.bootstrap_servers) == ()


def test_a_declaration_that_names_a_host_alias_is_refused(tmp_path: Path) -> None:
    """Positive control: the defect, expressed as a declaration, still fails."""
    overlay = _overlay(
        tmp_path,
        f"""
lanes:
  dev:
    broker: "{_OLD_LITERAL}"
    security_protocol: PLAINTEXT
""",
    )
    with pytest.raises(ValueError, match="Docker-Desktop alias"):
        load_lane_transport(overlay, "dev")


def test_a_lane_with_no_declared_protocol_is_refused(tmp_path: Path) -> None:
    """An address with no transport is the half-declaration that caused this."""
    overlay = _overlay(
        tmp_path,
        """
lanes:
  dev:
    broker: "broker.example.internal:19092"
""",
    )
    with pytest.raises(ValueError, match="security_protocol"):
        load_lane_transport(overlay, "dev")


def test_a_sasl_protocol_without_a_mechanism_is_refused(tmp_path: Path) -> None:
    overlay = _overlay(
        tmp_path,
        """
lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: SASL_PLAINTEXT
""",
    )
    with pytest.raises(ValueError, match="sasl_mechanism"):
        load_lane_transport(overlay, "dev")


def test_a_mechanism_beside_a_plaintext_protocol_is_a_contradiction(
    tmp_path: Path,
) -> None:
    overlay = _overlay(
        tmp_path,
        """
lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
""",
    )
    with pytest.raises(ValueError, match="contradiction"):
        load_lane_transport(overlay, "dev")


def test_an_inmemory_lane_is_refused_rather_than_probed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="in-memory"):
        load_lane_transport(_overlay(tmp_path, _DECLARED_OVERLAY), "stability")


def test_an_undeclared_lane_names_the_lanes_that_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not declared"):
        load_lane_transport(_overlay(tmp_path, _DECLARED_OVERLAY), "judge")


def test_a_missing_overlay_is_refused_not_defaulted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_lane_transport(tmp_path / "absent.yaml", "dev")


# ---------------------------------------------------------------------------
# Credentials never leave the environment
# ---------------------------------------------------------------------------


def test_the_exported_env_carries_the_transport_and_no_credential() -> None:
    env = lane_transport_env(
        ModelLaneTransport(
            lane="dev",
            bootstrap_servers="broker.example.internal:19092",
            security_protocol="SASL_PLAINTEXT",
            sasl_mechanism="SCRAM-SHA-256",
        )
    )
    assert env["KAFKA_BOOTSTRAP_SERVERS"] == "broker.example.internal:19092"
    assert env["KAFKA_SECURITY_PROTOCOL"] == "SASL_PLAINTEXT"
    assert env["KAFKA_SASL_MECHANISM"] == "SCRAM-SHA-256"
    assert "KAFKA_SASL_USERNAME" not in env
    assert "KAFKA_SASL_PASSWORD" not in env


def test_a_plaintext_lane_exports_no_mechanism() -> None:
    env = lane_transport_env(
        ModelLaneTransport(
            lane="dev",
            bootstrap_servers="broker.example.internal:19092",
            security_protocol="PLAINTEXT",
        )
    )
    assert "KAFKA_SASL_MECHANISM" not in env


# ---------------------------------------------------------------------------
# The request model refuses the literal, so it cannot come back as a default
# ---------------------------------------------------------------------------


def test_the_shipped_literal_is_refused_by_the_quarantine_leg() -> None:
    with pytest.raises(ValidationError, match="Docker-Desktop host alias"):
        _request(quarantine_bootstrap_servers=_OLD_LITERAL)


def test_the_shipped_literal_is_refused_by_the_terminal_leg() -> None:
    with pytest.raises(ValidationError, match="Docker-Desktop host alias"):
        _request(terminal_bootstrap_servers=_OLD_LITERAL)


@pytest.mark.parametrize("alias", sorted(DOCKER_DESKTOP_HOST_ALIASES))
def test_every_known_docker_alias_is_refused(alias: str) -> None:
    with pytest.raises(ValidationError):
        _request(quarantine_bootstrap_servers=f"{alias}:19092")


def test_one_bad_entry_in_a_broker_list_is_still_refused() -> None:
    """A list is only as resolvable as its worst member."""
    with pytest.raises(ValidationError):
        _request(
            quarantine_bootstrap_servers=f"broker.example.internal:19092,{_OLD_LITERAL}"
        )


def test_a_declared_address_is_accepted() -> None:
    request = _request(
        quarantine_bootstrap_servers="broker.example.internal:19092",
        terminal_bootstrap_servers="broker.example.internal:19092",
    )
    assert request.quarantine_bootstrap_servers == "broker.example.internal:19092"


def test_an_empty_broker_stays_the_not_configured_sentinel() -> None:
    """Empty still means SKIPPED_NOT_CONFIGURED — the refusal changes nothing here."""
    request = _request(quarantine_bootstrap_servers="")
    assert request.quarantine_bootstrap_servers == ""


def test_the_ingress_probe_url_is_deliberately_not_covered() -> None:
    """The HTTP leg through the host gateway works and stays working.

    Widening the refusal to ``probe_url`` would break the one leg that
    succeeded on all 25 failing runs, to make a point about a different leg.
    """
    request = _request(probe_url="http://host.docker.internal:8085")
    assert request.probe_url == "http://host.docker.internal:8085"


def test_the_sasl_protocols_stay_a_subset_of_the_valid_ones() -> None:
    """The two sets are spelled out separately; they must not drift apart.

    ``_VALID_SECURITY_PROTOCOLS`` is written as a literal rather than composed
    from ``_SASL_PROTOCOLS`` with ``|``, because the repo's union-usage
    validator counts a ``BitOr`` in an annotated assignment as a type union.
    That is a fine trade as long as something notices when one list gains a
    protocol the other does not.
    """
    from omnibase_infra.nodes.node_chain_canary_effect import lane_transport

    assert lane_transport._SASL_PROTOCOLS <= lane_transport._VALID_SECURITY_PROTOCOLS
