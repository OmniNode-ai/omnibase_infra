# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18432 AC4/AC5: which credential answers for a selected lane, and when.

Three surfaces reach this code and they disagree about where a credential
lives, so the resolution order is the whole design and is pinned here rather
than left to the reader:

1. A developer machine holds the sanctioned by-reference client store and no
   SASL environment at all. The store answers.
2. The lab host and the CI runners hold the SASL environment and no client
   store. The environment answers, byte-for-byte as it does today -- this is
   the half that keeps every existing caller unchanged, and it is why the
   environment is a fallback rather than something removed.
3. A machine holding neither, pointed at a lane that DECLARES a SASL protocol,
   is refused before anything connects. The refusal is the point: the
   alternative is an anonymous connect against an auth-required listener, which
   surfaces minutes later as a handshake error naming nothing an operator can
   act on.

The store wins over the environment where both exist. That ordering is
deliberate: the environment on the lab host is shared ambient state that
hundreds of concurrent lanes inherit, and a per-machine identity the operator
stored explicitly is a stronger statement of intent than a variable somebody
exported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.cli.delegate_lane_credentials import (
    DelegateLaneCredentialError,
    resolve_lane_client_transport_for,
)
from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential

pytestmark = pytest.mark.unit

_DECLARED_IN = Path("omnimarket/config/ci_bus_lanes.yaml")


def _sasl_lane() -> ModelDelegateLaneSelection:
    return ModelDelegateLaneSelection(
        lane="dev",
        bootstrap_servers="broker.invalid:19092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        declared_in=_DECLARED_IN,
    )


def _plaintext_lane() -> ModelDelegateLaneSelection:
    return ModelDelegateLaneSelection(
        lane="lab",
        bootstrap_servers="plain.invalid:19092",
        security_protocol="PLAINTEXT",
        declared_in=_DECLARED_IN,
    )


def test_no_lane_selected_binds_nothing(tmp_path: Path) -> None:
    """An explicit --kafka-bootstrap states an address and claims nothing more."""
    assert (
        resolve_lane_client_transport_for(
            lane_target=None, onex_home=tmp_path, environ={}
        )
        is None
    )


def test_the_client_store_answers_for_a_sasl_lane(tmp_path: Path) -> None:
    StoreLaneCredential(onex_home=tmp_path).save(
        lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value"
    )

    transport = resolve_lane_client_transport_for(
        lane_target=_sasl_lane(), onex_home=tmp_path, environ={}
    )

    assert transport is not None
    assert transport.sasl_username == "dev-cli-host"
    assert transport.sasl_password is not None
    assert transport.sasl_password.get_secret_value() == "s3kr3t-value"
    assert transport.security_protocol == "SASL_PLAINTEXT"
    assert transport.sasl_mechanism == "SCRAM-SHA-256"


def test_the_client_store_wins_over_an_ambient_environment(tmp_path: Path) -> None:
    StoreLaneCredential(onex_home=tmp_path).save(
        lane="dev", sasl_username="stored-name", sasl_password="stored-value"
    )

    transport = resolve_lane_client_transport_for(
        lane_target=_sasl_lane(),
        onex_home=tmp_path,
        environ={
            "KAFKA_SASL_USERNAME": "ambient-name",
            "KAFKA_SASL_PASSWORD": "ambient-value",
        },
    )

    assert transport is not None
    assert transport.sasl_username == "stored-name"


def test_the_environment_answers_where_there_is_no_client_store(
    tmp_path: Path,
) -> None:
    """AC5: the lab host and CI keep the behaviour they have today."""
    transport = resolve_lane_client_transport_for(
        lane_target=_sasl_lane(),
        onex_home=tmp_path / "absent",
        environ={
            "KAFKA_SASL_USERNAME": "ambient-name",
            "KAFKA_SASL_PASSWORD": "ambient-value",
        },
    )

    assert transport is None


def test_a_sasl_lane_with_no_credential_anywhere_is_refused(tmp_path: Path) -> None:
    """AC4: refuse before connecting, never open an anonymous session."""
    with pytest.raises(DelegateLaneCredentialError) as caught:
        resolve_lane_client_transport_for(
            lane_target=_sasl_lane(), onex_home=tmp_path / "absent", environ={}
        )

    message = str(caught.value)
    assert "dev" in message
    assert "ci_bus_lanes.yaml" in message
    assert "lane-login" in message


def test_a_half_set_environment_is_refused_rather_than_half_used(
    tmp_path: Path,
) -> None:
    """A username with no password is not a credential; it is a broken machine."""
    with pytest.raises(DelegateLaneCredentialError):
        resolve_lane_client_transport_for(
            lane_target=_sasl_lane(),
            onex_home=tmp_path / "absent",
            environ={"KAFKA_SASL_USERNAME": "ambient-name"},
        )


def test_a_plaintext_lane_binds_its_declared_protocol_and_no_credential(
    tmp_path: Path,
) -> None:
    """AC1 covers non-SASL lanes too: the declaration still states the transport."""
    transport = resolve_lane_client_transport_for(
        lane_target=_plaintext_lane(), onex_home=tmp_path, environ={}
    )

    assert transport is not None
    assert transport.security_protocol == "PLAINTEXT"
    assert transport.sasl_username is None
    assert transport.as_client_config_overrides() == {"security_protocol": "PLAINTEXT"}


def test_a_plaintext_lane_needs_no_store_and_is_never_refused(tmp_path: Path) -> None:
    transport = resolve_lane_client_transport_for(
        lane_target=_plaintext_lane(), onex_home=tmp_path / "absent", environ={}
    )

    assert transport is not None


def test_a_store_that_refuses_for_a_real_defect_is_not_masked_by_the_environment(
    tmp_path: Path,
) -> None:
    """A mis-permissioned credential file is a defect to fix, not to route around.

    Falling through to the environment here would mean a machine whose stored
    credential is unreadable silently authenticates as somebody else -- the
    exact substitution the by-reference store exists to make impossible.
    """
    store = StoreLaneCredential(onex_home=tmp_path)
    store.save(lane="dev", sasl_username="stored-name", sasl_password="stored-value")
    store.credentials_path.chmod(0o644)

    with pytest.raises(DelegateLaneCredentialError) as caught:
        resolve_lane_client_transport_for(
            lane_target=_sasl_lane(),
            onex_home=tmp_path,
            environ={
                "KAFKA_SASL_USERNAME": "ambient-name",
                "KAFKA_SASL_PASSWORD": "ambient-value",
            },
        )

    assert "0644" in str(caught.value)
