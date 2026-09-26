# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17454: a standalone projection writer resolves one binding per domain.

A standalone writer (omnimarket ``BaseProjectionRunner``) used to open one pool
as one principal, so a writer whose tables span the tenant and internal domains
always had something refused -- at minimum its ``projection_watermarks`` write.
``resolve_standalone_projection_bindings`` gives it what the in-process path
already has: every binding its tables need, each with the principal the
topology declares and a resolved DSN, the watermark included.

Each test pins one failure mode of that resolution against the real shipped
topology (``load_topology_profile``), not a hand-built one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import SecretStr

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.standalone_projection_bindings import (
    PROJECTION_WATERMARK_TABLE,
    StandaloneProjectionBindings,
    resolve_standalone_projection_bindings,
)
from omnibase_infra.topology import load_topology_profile

if TYPE_CHECKING:
    from omnibase_infra.runtime.secret_resolver import SecretResolver

_TENANT_DSN = "postgresql://tenant_projection_writer@db/omnidash_analytics"
_INTERNAL_DSN = "postgresql://omninode_runtime@db/omnidash_analytics"


def _table(name: str, schema: str, access: str) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema=schema,
        migration=f"proof/{name}.sql",
        access=access,
        role=name,
    )


_TENANT_TABLE = _table("delegation_events", "public", "read_write")
_INTERNAL_TABLE = _table("generation_events", "omninode_internal", "read_write")


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_TENANT_DB_URL", _TENANT_DSN)
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _INTERNAL_DSN)


@pytest.mark.unit
@pytest.mark.usefixtures("local_env")
def test_mixed_writer_resolves_one_binding_per_domain_with_the_watermark_internal() -> (
    None
):
    resolved = resolve_standalone_projection_bindings(
        (_TENANT_TABLE, _INTERNAL_TABLE), load_topology_profile("local")
    )

    assert set(resolved.bindings) == {"tenant_projection", "omninode_runtime_service"}
    assert (
        resolved.bindings["tenant_projection"].principal == "tenant_projection_writer"
    )
    assert resolved.bindings["omninode_runtime_service"].principal == "omninode_runtime"
    assert resolved.write_binding_for("delegation_events") == "tenant_projection"
    assert resolved.write_binding_for("generation_events") == "omninode_runtime_service"
    assert resolved.watermark_binding == "omninode_runtime_service"
    assert resolved.bindings["tenant_projection"].dsn.get_secret_value() == _TENANT_DSN
    assert (
        resolved.bindings["omninode_runtime_service"].dsn.get_secret_value()
        == _INTERNAL_DSN
    )


@pytest.mark.unit
@pytest.mark.usefixtures("local_env")
def test_the_watermark_is_added_once_even_when_the_contract_declares_it() -> None:
    resolved = resolve_standalone_projection_bindings(
        (_TENANT_TABLE, PROJECTION_WATERMARK_TABLE), load_topology_profile("local")
    )

    assert resolved.tables.count(PROJECTION_WATERMARK_TABLE.name) == 1
    assert resolved.watermark_binding == "omninode_runtime_service"


@pytest.mark.unit
@pytest.mark.usefixtures("local_env")
def test_a_single_domain_internal_writer_gets_one_binding() -> None:
    resolved = resolve_standalone_projection_bindings(
        (_INTERNAL_TABLE,), load_topology_profile("local")
    )

    assert set(resolved.bindings) == {"omninode_runtime_service"}


@pytest.mark.unit
def test_an_unset_env_carrier_is_refused_naming_the_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _INTERNAL_DSN)
    monkeypatch.delenv("ONEX_TENANT_DB_URL", raising=False)

    with pytest.raises(ValueError, match="ONEX_TENANT_DB_URL") as refused:
        resolve_standalone_projection_bindings(
            (_TENANT_TABLE, _INTERNAL_TABLE), load_topology_profile("local")
        )

    # The binding and its principal are asserted separately: the principal
    # ``tenant_projection_writer`` contains the binding name, so a bare
    # ``tenant_projection`` match would pass a message that dropped the binding.
    assert "binding 'tenant_projection'" in str(refused.value)
    assert "principal 'tenant_projection_writer'" in str(refused.value)
    assert _INTERNAL_DSN not in str(refused.value)


@pytest.mark.unit
def test_a_secret_ref_binding_without_a_resolver_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _INTERNAL_DSN)

    with pytest.raises(
        ValueError, match=re.escape("secret_ref=database.tenant_projection.dsn")
    ):
        resolve_standalone_projection_bindings(
            (_TENANT_TABLE, _INTERNAL_TABLE), load_topology_profile("onex-dev")
        )


class _FakeSecretResolver:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values
        self.asked: list[str] = []

    def get_secret(self, logical_name: str, required: bool = True) -> SecretStr | None:
        self.asked.append(logical_name)
        value = self._values.get(logical_name)
        return None if value is None else SecretStr(value)


@pytest.mark.unit
def test_a_secret_ref_binding_resolves_through_the_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _INTERNAL_DSN)
    resolver = _FakeSecretResolver({"database.tenant_projection.dsn": _TENANT_DSN})

    resolved = resolve_standalone_projection_bindings(
        (_TENANT_TABLE, _INTERNAL_TABLE),
        load_topology_profile("onex-dev"),
        secret_resolver=cast("SecretResolver", resolver),
    )

    assert resolver.asked == ["database.tenant_projection.dsn"]
    assert resolved.bindings["tenant_projection"].dsn.get_secret_value() == _TENANT_DSN


@pytest.mark.unit
@pytest.mark.usefixtures("local_env")
def test_no_dsn_value_appears_in_the_resolved_repr() -> None:
    resolved = resolve_standalone_projection_bindings(
        (_TENANT_TABLE, _INTERNAL_TABLE), load_topology_profile("local")
    )

    rendered = repr(resolved)
    assert _TENANT_DSN not in rendered
    assert _INTERNAL_DSN not in rendered


@pytest.mark.unit
def test_a_watermark_with_no_write_binding_is_refused_naming_the_table() -> None:
    # The resolver always adds the watermark and refuses any table it cannot
    # bind, so only a hand-built value reaches this; the runner must still get
    # a named refusal it can report, not a lookup error from a private map.
    bindings = StandaloneProjectionBindings(
        physical_database="omnidash_analytics",
        bindings={},
        tables=("delegation_events",),
        _write_binding_by_table={"delegation_events": "tenant_projection"},
        _read_binding_by_table={},
    )

    with pytest.raises(ValueError, match="projection_watermarks"):
        _ = bindings.watermark_binding


@pytest.mark.unit
@pytest.mark.usefixtures("local_env")
def test_a_table_the_topology_does_not_grant_is_still_refused() -> None:
    with pytest.raises(ValueError, match="lacks declared write privileges"):
        resolve_standalone_projection_bindings(
            (_table("omn17454_not_granted", "omninode_internal", "write"),),
            load_topology_profile("local"),
        )
