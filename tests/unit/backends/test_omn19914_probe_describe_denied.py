# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19914: a group the probe may not describe no longer refuses the dispatch.

Measured on the .201 dev lane 2026-09-28 13:39Z-14:03Z (runs 1b0c5215,
664486d8, d065cd77 under ``.onex_state/runs``): a pre-PR slot shared the dev
broker, its runtime bound ``prepr1.*`` consumer groups whose ids end in the
delegate command topic's scope suffix, and the Mac CLI principal -- granted
CLUSTER DESCRIBE, so it LISTS every group, but GROUP DESCRIBE only on
``local.`` -- could not look one of them up. The coordinator lookup raised
``GroupAuthorizationFailedError`` and every lab delegation failed closed while
the dev lane's own orchestrator group was ``Stable`` throughout.

The two properties pinned here:

* once one candidate is proven ``Stable``, a hidden candidate cannot change
  the answer, so it is skipped and the dispatch proceeds;
* when nothing could be proven, the refusal is typed and names the group and
  the grant it lacked, in words the receipt sanitizer does not redact.

All names below are synthetic.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from aiokafka.errors import GroupAuthorizationFailedError

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupDescribeDeniedError,
    ConsumerGroupLivenessUnknownError,
    live_consumer_groups,
)
from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli import delegate_locus
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusAclRefusedError,
    DelegateLocusRefusedError,
    resolve_delegate_locus,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_BROKER = "lane-broker.invalid:19092"
_PRINCIPAL = "dev-cli-synthetic-host"
_OWN_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    f".__i.runtime-effects.__t.{_TOPIC}"
)
_SLOT_GROUP = (
    "prepr1.omnibase_infra.node_ledger_projection_compute.consume.1.6.0"
    f".__i.runtime-main.__t.{_TOPIC}"
)


class _Response:
    def __init__(self, groups: list[tuple[Any, ...]]) -> None:
        self.groups = groups


class _Admin:
    """aiokafka admin fake: lists the groups, and refuses to look some up."""

    listed: list[str] = []
    denied_raise: set[str] = set()
    denied_code: set[str] = set()
    states: dict[str, str] = {}
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _Admin.kwargs = kwargs

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def describe_cluster(self) -> dict[str, Any]:
        return {"brokers": [{"node_id": 1}]}

    async def _send_request(
        self, request: Any, node_id: int | None = None
    ) -> SimpleNamespace:
        struct = request.prepare({16: (0, 4)})
        assert struct.states_filter == ["Stable"]
        return SimpleNamespace(
            error_code=0,
            groups=[
                (group_id, "consumer", state, {})
                for group_id in self.listed
                if (state := self.states.get(group_id, "Stable")) == "Stable"
            ],
        )

    async def list_consumer_groups(self) -> list[tuple[str, str]]:
        return [(group, "consumer") for group in _Admin.listed]

    async def describe_consumer_groups(self, group_ids: list[str]) -> list[_Response]:
        (group_id,) = group_ids
        if group_id in _Admin.denied_raise:
            # What aiokafka's find_coordinator raises for error code 30.
            raise GroupAuthorizationFailedError(
                f"Unable to get coordinator id for {group_id}"
            )
        code = 30 if group_id in _Admin.denied_code else 0
        state = _Admin.states.get(group_id, "Stable")
        return [_Response([(code, group_id, state, "consumer", "", [])])]


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> type[_Admin]:
    import aiokafka.admin

    for name in (
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _PRINCIPAL)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "synthetic-not-real")
    _Admin.listed = [_OWN_GROUP, _SLOT_GROUP]
    _Admin.denied_raise = set()
    _Admin.denied_code = set()
    _Admin.states = {}
    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _Admin)
    return _Admin


@pytest.mark.unit
def test_a_hidden_foreign_group_does_not_refuse_a_proven_live_lane(
    admin: type[_Admin],
) -> None:
    """The measured shape: own group Stable, slot group hidden -> proceed."""
    admin.denied_raise = {_SLOT_GROUP}

    assert live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER) == (
        _OWN_GROUP,
    )


@pytest.mark.unit
def test_a_denial_reported_in_the_response_is_treated_the_same(
    admin: type[_Admin],
) -> None:
    admin.denied_code = {_SLOT_GROUP}

    assert live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER) == (
        _OWN_GROUP,
    )


@pytest.mark.unit
def test_nothing_proven_and_something_hidden_is_a_typed_refusal(
    admin: type[_Admin],
) -> None:
    """A hidden group may be the only consumer, so this stays UNKNOWN."""
    admin.denied_raise = {_SLOT_GROUP}
    admin.states = {_OWN_GROUP: "Empty"}

    with pytest.raises(ConsumerGroupDescribeDeniedError) as caught:
        live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER)

    exc = caught.value
    # Still UNKNOWN for every existing fail-closed caller.
    assert isinstance(exc, ConsumerGroupLivenessUnknownError)
    assert exc.group_ids == (_SLOT_GROUP,)
    assert exc.principal == _PRINCIPAL
    message = str(exc)
    assert f"ALLOW User:{_PRINCIPAL} DESCRIBE on GROUP '{_SLOT_GROUP}'" in message
    assert "synthetic-not-real" not in message
    # The receipt writer sanitizes this text; the grant must survive it.
    assert sanitize_error_string(message, max_length=4000) == message


@pytest.mark.unit
def test_every_candidate_hidden_is_a_typed_refusal(admin: type[_Admin]) -> None:
    admin.denied_raise = {_OWN_GROUP, _SLOT_GROUP}

    with pytest.raises(ConsumerGroupDescribeDeniedError) as caught:
        live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER)

    assert caught.value.group_ids == tuple(sorted((_OWN_GROUP, _SLOT_GROUP)))


@pytest.mark.unit
def test_any_other_per_group_error_is_still_unknown(
    admin: type[_Admin], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the access denial is tolerated; a coordinator error is not."""

    class _Erroring(_Admin):
        async def describe_consumer_groups(
            self, group_ids: list[str]
        ) -> list[_Response]:
            (group_id,) = group_ids
            code = 16 if group_id == _SLOT_GROUP else 0
            return [_Response([(code, group_id, "Stable", "consumer", "", [])])]

    import aiokafka.admin

    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _Erroring)
    with pytest.raises(ConsumerGroupLivenessUnknownError) as caught:
        live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER)
    assert not isinstance(caught.value, ConsumerGroupDescribeDeniedError)


@pytest.mark.unit
def test_the_locus_gate_refusal_is_typed_and_names_the_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "name": "node_delegate_skill_orchestrator",
                "event_bus": {"subscribe_topics": [_TOPIC]},
            }
        ),
        encoding="utf-8",
    )

    def _deny(
        *, owner: ModelConsumerGroupOwner | None = None, **_: object
    ) -> tuple[str, ...]:
        raise ConsumerGroupDescribeDeniedError(
            group_ids=(_SLOT_GROUP,), bootstrap_servers=_BROKER, principal=_PRINCIPAL
        )

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _deny)
    with pytest.raises(DelegateLocusAclRefusedError) as caught:
        resolve_delegate_locus(
            requested=EnumDelegateLocus.DEPLOYED_LANE,
            bus="kafka",
            kafka_bootstrap=_BROKER,
            contract_path=contract,
            shared_bus_value="kafka",
        )

    exc = caught.value
    assert isinstance(exc, DelegateLocusRefusedError)
    assert exc.group_ids == (_SLOT_GROUP,)
    assert f"DESCRIBE on GROUP '{_SLOT_GROUP}'" in str(exc)
    assert sanitize_error_string(str(exc), max_length=4000) == str(exc)
