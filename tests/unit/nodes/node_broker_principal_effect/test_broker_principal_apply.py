# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20822: the broker principal effect applies a plan in the mint plan's fixed order.

The order (plan D9): list the broker's users (the Admin API reachability check, so
an unreachable Admin API stops the run before any mutation), read the principal's
ACLs and refuse any ACL outside the plan, create the planned ACLs, read them back
and compare as a set, and only then set the password. A failure reports which
mutations completed (``acls_applied``, ``password_set``). The password comes from
the environment variable the request names and appears in no result, log record or
exception.

These tests drive the apply handler through fake Admin API and ACL clients that
record every call in one ordered journal, so the order itself is what is asserted.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import pytest
from pydantic import SecretStr

from omnibase_infra.errors.error_broker_principal import BrokerPrincipalError
from omnibase_infra.handlers.handler_broker_principal_apply import (
    HandlerBrokerPrincipalApply,
)
from omnibase_infra.models.developer_principal import (
    ModelBrokerAclBinding,
    ModelBrokerPrincipalApplyRequest,
    ModelDeveloperPrincipalGrant,
    ModelDeveloperPrincipalPlan,
)

pytestmark = pytest.mark.unit

ADMIN_URL = "http://redpanda:9644"
PASSWORD_ENV = "ONEX_MINT_PASSWORD"
SENTINEL = "sentinel-pw-7f3a9c1e5b2d"
PRINCIPAL = "dev-dev-person-mac"

GRANTS = (
    ModelDeveloperPrincipalGrant(
        resource="topic",
        name="onex.cmd.omnimarket.delegate-skill-requested.v1",
        pattern="literal",
        operations=("describe", "write"),
    ),
    ModelDeveloperPrincipalGrant(
        resource="group",
        name="local.omnibase_core.runtime_local_",
        pattern="prefixed",
        operations=("describe", "read"),
    ),
)


def _planned() -> frozenset[ModelBrokerAclBinding]:
    return frozenset(
        ModelBrokerAclBinding(
            resource=g.resource, name=g.name, pattern=g.pattern, operation=op
        )
        for g in GRANTS
        for op in g.operations
    )


def _plan(decision: str = "create") -> ModelDeveloperPrincipalPlan:
    return ModelDeveloperPrincipalPlan(
        decision=decision,
        principal=PRINCIPAL,
        lane="local",
        grants=GRANTS,
        declaration_digest="a" * 64,
    )


def _request(decision: str = "create") -> ModelBrokerPrincipalApplyRequest:
    return ModelBrokerPrincipalApplyRequest(
        plan=_plan(decision), admin_url=ADMIN_URL, password_env=PASSWORD_ENV
    )


class _Broker:
    """One fake broker: an Admin API side and an ACL side sharing a call journal."""

    def __init__(
        self,
        *,
        users: Iterable[str] = (),
        acls: Iterable[ModelBrokerAclBinding] = (),
        list_users_error: Exception | None = None,
        set_password_error: Exception | None = None,
        create_fails_on: ModelBrokerAclBinding | None = None,
        readback_drops: ModelBrokerAclBinding | None = None,
    ) -> None:
        self.journal: list[str] = []
        self.users = set(users)
        self.acls = set(acls)
        self.passwords: dict[str, str] = {}
        self._list_users_error = list_users_error
        self._set_password_error = set_password_error
        self._create_fails_on = create_fails_on
        self._readback_drops = readback_drops
        self._acl_reads = 0

    # Admin API side
    def list_users(self) -> tuple[str, ...]:
        self.journal.append("list_users")
        if self._list_users_error is not None:
            raise self._list_users_error
        return tuple(sorted(self.users))

    def set_password(self, name: str, password: SecretStr, *, exists: bool) -> None:
        self.journal.append(f"set_password:{'update' if exists else 'create'}")
        if self._set_password_error is not None:
            raise self._set_password_error
        self.users.add(name)
        self.passwords[name] = password.get_secret_value()

    # ACL side
    def list_acls(self, principal: str) -> frozenset[ModelBrokerAclBinding]:
        self._acl_reads += 1
        self.journal.append(f"list_acls:{self._acl_reads}")
        held = set(self.acls)
        if self._readback_drops is not None and self._acl_reads > 1:
            held.discard(self._readback_drops)
        return frozenset(held)

    def create_acls(
        self, principal: str, bindings: Iterable[ModelBrokerAclBinding]
    ) -> None:
        wanted = sorted(bindings, key=lambda b: (b.resource, b.name, b.operation))
        self.journal.append(f"create_acls:{len(wanted)}")
        for binding in wanted:
            if binding == self._create_fails_on:
                raise RuntimeError(f"broker refused {binding.name}")
            self.acls.add(binding)


def _handler(
    broker: _Broker, environ: Mapping[str, str] | None = None
) -> HandlerBrokerPrincipalApply:
    return HandlerBrokerPrincipalApply(
        admin=broker,
        acls=broker,
        environ={PASSWORD_ENV: SENTINEL} if environ is None else environ,
    )


# --- AC1: the password is set only after every planned ACL is read back --------


def test_grants_complete_before_password_on_a_new_principal() -> None:
    broker = _Broker()
    result = _handler(broker).handle(_request())
    assert broker.journal == [
        "list_users",
        "list_acls:1",
        "create_acls:4",
        "list_acls:2",
        "set_password:create",
    ]
    assert result.action == "create"
    assert (result.acls_applied, result.password_set) == (True, True)
    assert broker.acls == set(_planned())
    assert broker.passwords == {PRINCIPAL: SENTINEL}


def test_grants_complete_before_password_no_password_after_a_partial_failure() -> None:
    failing = sorted(_planned(), key=lambda b: (b.resource, b.name, b.operation))[1]
    broker = _Broker(create_fails_on=failing)
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(broker).handle(_request())
    assert not any(step.startswith("set_password") for step in broker.journal)
    assert (raised.value.acls_applied, raised.value.password_set) == (False, False)
    assert raised.value.operation == "broker.principal.apply"


def test_grants_complete_before_password_no_password_when_the_readback_differs() -> (
    None
):
    dropped = next(iter(_planned()))
    broker = _Broker(readback_drops=dropped)
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(broker).handle(_request())
    assert broker.journal[-1] == "list_acls:2"
    assert not any(step.startswith("set_password") for step in broker.journal)
    assert (raised.value.acls_applied, raised.value.password_set) == (False, False)
    assert dropped.name in str(raised.value)


# --- AC2 (unit level): a rerun rotates, drift is refused ------------------------


def test_rerun_rotates_and_leaves_the_acl_set_identical() -> None:
    broker = _Broker(users=(PRINCIPAL,), acls=_planned())
    result = _handler(broker).handle(_request("rotate"))
    assert result.action == "rotate"
    assert broker.acls == set(_planned())
    assert "create_acls:0" in broker.journal or not any(
        step.startswith("create_acls") for step in broker.journal
    )
    assert broker.journal[-1] == "set_password:update"


def test_drift_an_acl_outside_the_plan_is_refused_and_named() -> None:
    extra = ModelBrokerAclBinding(
        resource="topic",
        name="onex.evt.private.v1",
        pattern="literal",
        operation="read",
    )
    broker = _Broker(users=(PRINCIPAL,), acls={*_planned(), extra})
    result = _handler(broker).handle(_request("rotate"))
    assert result.action == "refused"
    assert extra in result.extra_acls
    assert extra.name in result.refusal
    assert broker.journal == ["list_users", "list_acls:1"]
    assert broker.passwords == {}
    assert extra in broker.acls


def test_a_refused_plan_is_not_applied() -> None:
    broker = _Broker()
    request = ModelBrokerPrincipalApplyRequest(
        plan=ModelDeveloperPrincipalPlan(
            decision="refused", refusal_kind="device_cap", refusal="cap reached"
        ),
        admin_url=ADMIN_URL,
        password_env=PASSWORD_ENV,
    )
    with pytest.raises(BrokerPrincipalError):
        _handler(broker).handle(request)
    assert broker.journal == []


# --- AC3: the password appears in no result, log record or exception -----------


def _assert_no_sentinel(text: str) -> None:
    assert SENTINEL not in text


def test_no_secret_in_the_result_or_logs_on_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    result = _handler(_Broker()).handle(_request())
    _assert_no_sentinel(result.model_dump_json())
    _assert_no_sentinel(repr(result))
    for record in caplog.records:
        _assert_no_sentinel(record.getMessage())


@pytest.mark.parametrize(
    "broker_kwargs",
    [
        {"set_password_error": RuntimeError(f"admin said no to {SENTINEL}")},
        {"set_password_error": RuntimeError("admin 500")},
        {"list_users_error": ConnectionError("refused")},
    ],
    ids=["error_text_echoes_the_password", "admin_500", "admin_down"],
)
def test_no_secret_in_the_exception_on_failure_paths(
    caplog: pytest.LogCaptureFixture, broker_kwargs: dict[str, Exception]
) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(_Broker(**broker_kwargs)).handle(_request())
    _assert_no_sentinel(str(raised.value))
    _assert_no_sentinel(repr(raised.value))
    _assert_no_sentinel(repr(raised.value.__dict__))
    for record in caplog.records:
        _assert_no_sentinel(record.getMessage())


def test_no_secret_the_password_variable_must_be_set_before_any_mutation() -> None:
    broker = _Broker()
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(broker, environ={}).handle(_request())
    assert PASSWORD_ENV in str(raised.value)
    assert not any(
        step.startswith(("create_acls", "set_password")) for step in broker.journal
    )


# --- AC5: the Admin API down, before and after the ACLs exist -------------------


def test_admin_down_at_list_names_the_url_and_creates_no_acl() -> None:
    broker = _Broker(list_users_error=ConnectionError("connection refused"))
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(broker).handle(_request())
    assert broker.journal == ["list_users"]
    assert broker.acls == set()
    assert ADMIN_URL in str(raised.value)
    assert raised.value.operation == "broker.principal.apply"
    assert (raised.value.acls_applied, raised.value.password_set) == (False, False)


def test_admin_down_after_the_acls_reports_acls_applied_and_no_password() -> None:
    broker = _Broker(set_password_error=ConnectionError("connection reset"))
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(broker).handle(_request())
    assert broker.acls == set(_planned())
    assert (raised.value.acls_applied, raised.value.password_set) == (True, False)
    assert ADMIN_URL in str(raised.value)


def test_admin_down_on_rotation_keeps_the_previous_password() -> None:
    broker = _Broker(
        users=(PRINCIPAL,),
        acls=_planned(),
        set_password_error=ConnectionError("connection reset"),
    )
    broker.passwords[PRINCIPAL] = "previous-password"
    with pytest.raises(BrokerPrincipalError) as raised:
        _handler(broker).handle(_request("rotate"))
    assert broker.passwords == {PRINCIPAL: "previous-password"}
    assert (raised.value.acls_applied, raised.value.password_set) == (True, False)
