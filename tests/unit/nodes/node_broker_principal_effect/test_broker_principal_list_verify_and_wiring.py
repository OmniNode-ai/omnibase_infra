# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20822: list and verify, and the node's wiring.

``broker.principal.list`` is the Admin API reachability check and returns the
principals that carry the declaration's prefix. ``broker.principal.verify``
reads the principal's ACLs back against the plan, then authenticates as the
principal and lists the topics it can see. The node declares the ``local-dev``
profile (``cli`` is not a registered runtime profile), so neither ``main`` nor
``effects`` attaches it, and one pre-commit entry runs this directory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr, ValidationError

from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
    HandlerBrokerPrincipalList,
)
from omnibase_infra.handlers.handler_broker_principal_verify import (
    HandlerBrokerPrincipalVerify,
)
from omnibase_infra.handlers.handler_developer_principal_plan import principal_name
from omnibase_infra.models.developer_principal import (
    ModelBrokerAclBinding,
    ModelBrokerPrincipalListRequest,
    ModelBrokerPrincipalVerifyRequest,
    ModelDeveloperPrincipalGrant,
    ModelDeveloperPrincipalPlan,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
NODE_DIR = (
    REPO_ROOT / "src" / "omnibase_infra" / "nodes" / "node_broker_principal_effect"
)
ADMIN_URL = "http://redpanda:9644"
PASSWORD_ENV = "ONEX_MINT_PASSWORD"
SENTINEL = "sentinel-pw-0d9e8f7a6b5c"
PRINCIPAL = "dev-dev-person-mac"
GRANTED = "onex.cmd.omnimarket.delegate-skill-requested.v1"

GRANTS = (
    ModelDeveloperPrincipalGrant(
        resource="topic", name=GRANTED, pattern="literal", operations=("describe",)
    ),
)
PLANNED = frozenset(
    {
        ModelBrokerAclBinding(
            resource="topic", name=GRANTED, pattern="literal", operation="describe"
        )
    }
)


def _plan() -> ModelDeveloperPrincipalPlan:
    return ModelDeveloperPrincipalPlan(
        decision="rotate",
        principal=PRINCIPAL,
        lane="local",
        grants=GRANTS,
        declaration_digest="b" * 64,
    )


class _Admin:
    def __init__(
        self, users: Iterable[str] = (), error: Exception | None = None
    ) -> None:
        self._users = tuple(users)
        self._error = error

    def list_users(self) -> tuple[str, ...]:
        if self._error is not None:
            raise self._error
        return self._users

    def set_password(self, name: str, password: SecretStr, *, exists: bool) -> None:
        raise AssertionError("list and verify never set a password")


class _Acls:
    def __init__(self, held: Iterable[ModelBrokerAclBinding]) -> None:
        self._held = frozenset(held)

    def list_acls(self, principal: str) -> frozenset[ModelBrokerAclBinding]:
        return self._held

    def create_acls(
        self, principal: str, bindings: Iterable[ModelBrokerAclBinding]
    ) -> None:
        raise AssertionError("verify never creates an ACL")


class _Session:
    def __init__(
        self, topics: Iterable[str] = (GRANTED,), error: Exception | None = None
    ) -> None:
        self._topics = tuple(topics)
        self._error = error
        self.logins: list[tuple[str, str]] = []

    def list_topics_as(self, principal: str, password: SecretStr) -> tuple[str, ...]:
        self.logins.append((principal, password.get_secret_value()))
        if self._error is not None:
            raise self._error
        return self._topics


# --- list ----------------------------------------------------------------------


def test_list_returns_only_principals_with_the_prefix() -> None:
    """The declarations' own prefix, ``dev-``, already ends in the separator."""
    handler = HandlerBrokerPrincipalList(
        admin=_Admin(
            users=("superuser", "dev-a-mac", "dev-b-air", "devx", "prod-a-mac")
        )
    )
    result = handler.handle(
        ModelBrokerPrincipalListRequest(admin_url=ADMIN_URL, prefix="dev-")
    )
    assert result.principals == ("dev-a-mac", "dev-b-air")


@pytest.mark.parametrize("prefix", ["dev-", "dev", "lab-dev-"])
def test_list_finds_every_name_the_plan_node_gives_under_that_prefix(
    prefix: str,
) -> None:
    """Names come from the plan node's ``principal_name``: ``<prefix><login>-<device>``."""
    named = principal_name(prefix, "person@example.com", "mac")
    other = principal_name("ops-", "person@example.com", "mac")
    handler = HandlerBrokerPrincipalList(admin=_Admin(users=(named, other)))
    result = handler.handle(
        ModelBrokerPrincipalListRequest(admin_url=ADMIN_URL, prefix=prefix)
    )
    assert result.principals == (named,)


@pytest.mark.parametrize("prefix", ["", "Dev-", "1dev", "dev_", "d" * 17])
def test_list_refuses_a_prefix_the_declaration_rule_refuses(prefix: str) -> None:
    with pytest.raises(ValidationError):
        ModelBrokerPrincipalListRequest(admin_url=ADMIN_URL, prefix=prefix)


def test_list_admin_down_is_a_typed_error_naming_the_url_and_operation() -> None:
    handler = HandlerBrokerPrincipalList(admin=_Admin(error=ConnectionError("refused")))
    with pytest.raises(BrokerPrincipalError) as raised:
        handler.handle(
            ModelBrokerPrincipalListRequest(admin_url=ADMIN_URL, prefix="dev")
        )
    assert ADMIN_URL in str(raised.value)
    assert raised.value.operation == "broker.principal.list"


# --- verify --------------------------------------------------------------------


def _verify(acls: _Acls, session: _Session) -> HandlerBrokerPrincipalVerify:
    return HandlerBrokerPrincipalVerify(
        acls=acls, session=session, environ={PASSWORD_ENV: SENTINEL}
    )


def _verify_request() -> ModelBrokerPrincipalVerifyRequest:
    return ModelBrokerPrincipalVerifyRequest(
        plan=_plan(), admin_url=ADMIN_URL, password_env=PASSWORD_ENV
    )


def test_verify_passes_when_the_acls_match_and_the_principal_authenticates() -> None:
    session = _Session()
    result = _verify(_Acls(PLANNED), session).handle(_verify_request())
    assert result.verified is True
    assert result.visible_topics == (GRANTED,)
    assert session.logins == [(PRINCIPAL, SENTINEL)]
    assert SENTINEL not in result.model_dump_json()


def test_verify_fails_when_the_acl_set_differs_from_the_plan() -> None:
    session = _Session()
    result = _verify(_Acls(()), session).handle(_verify_request())
    assert result.verified is False
    assert GRANTED in result.reason
    assert session.logins == [], "no login is tried when the ACLs already disagree"


def test_verify_a_refused_login_is_a_typed_error_without_the_secret() -> None:
    session = _Session(error=PermissionError(f"SASL authentication failed: {SENTINEL}"))
    with pytest.raises(BrokerPrincipalError) as raised:
        _verify(_Acls(PLANNED), session).handle(_verify_request())
    assert raised.value.operation == "broker.principal.verify"
    chain: list[BaseException] = []
    link: BaseException | None = raised.value
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__context__ or link.__cause__
    for exc in chain:
        assert SENTINEL not in str(exc), type(exc).__name__
        assert SENTINEL not in repr(vars(exc)), type(exc).__name__


# --- AC6: no main or effects runtime owns the node; one pre-commit entry ----------


@pytest.mark.parametrize("profile", ["main", "effects"])
def test_profile_ownership_main_and_effects_own_none_of_the_node(profile: str) -> None:
    manifest = discover_contracts_from_paths([NODE_DIR / "contract.yaml"])
    assert [c.name for c in manifest.contracts] == ["node_broker_principal_effect"]
    result = filter_manifest_for_runtime_profile(manifest, profile, environ={})
    assert [c.name for c in result.manifest.contracts] == []


def test_profile_ownership_the_local_dev_profile_owns_the_node() -> None:
    """Positive control: the filter is not refusing every contract."""
    manifest = discover_contracts_from_paths([NODE_DIR / "contract.yaml"])
    result = filter_manifest_for_runtime_profile(manifest, "local-dev", environ={})
    assert [c.name for c in result.manifest.contracts] == [
        "node_broker_principal_effect"
    ]


def test_the_contract_routes_the_three_operations_to_their_handlers() -> None:
    contract = yaml.safe_load((NODE_DIR / "contract.yaml").read_text())
    routes = {
        h["operation"]: h["handler"]["name"]
        for h in contract["handler_routing"]["handlers"]
    }
    assert routes == {
        "broker.principal.list": "HandlerBrokerPrincipalList",
        "broker.principal.apply": "HandlerBrokerPrincipalApply",
        "broker.principal.verify": "HandlerBrokerPrincipalVerify",
    }
    assert contract["node_type"] == "EFFECT_GENERIC"
    assert contract["runtime_profiles"] == ["local-dev"]


def test_pre_commit_entry_runs_the_node_tests() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == "broker-principal-effect"
    ]
    assert len(hooks) == 1, "no single broker-principal-effect pre-commit entry"
    (hook,) = hooks
    assert "tests/unit/nodes/node_broker_principal_effect" in hook["entry"]
    assert hook["pass_filenames"] is False
    files = re.compile(hook["files"])
    for path in (
        "src/omnibase_infra/nodes/node_broker_principal_effect/contract.yaml",
        "src/omnibase_infra/handlers/handler_broker_principal_apply.py",
        "src/omnibase_infra/models/developer_principal/model_broker_acl_binding.py",
    ):
        assert files.search(path), path
