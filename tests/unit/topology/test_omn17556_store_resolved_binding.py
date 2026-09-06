# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ``tenant_projection`` DSN is store-carried on onex-dev (OMN-17556).

No onex-dev pod may hold a ``tenant_projection_writer`` DSN in its environment
(operator ruling, 2026-09-03: no credential env var on ANY pod -- shared
runtime, effects, AND dedicated writer pods). The binding therefore declares a
``secret_ref`` the runtime resolves through ``SecretResolver`` at the *binding
boundary*, and the checked-in topology carries the REF while the store carries
the VALUE.

These tests pin the three things that make that safe, and each one fails for a
DIFFERENT real regression rather than restating the same fact three ways:

1. The onex-dev binding really is store-carried, and carries NO env fallback.
   A regression here is someone "fixing" a boot failure by adding
   ``dsn_env: ONEX_TENANT_DB_URL`` back -- the exact move the ruling forbids.
2. ``_resolve_binding_dsn`` reads the declared carrier and ONLY that carrier.
   A regression here is an ``os.environ`` fallback creeping into the store
   branch, which would let a stale env var silently shadow the store.
3. An unresolvable carrier still fails CLOSED, and the refusal names the
   carrier. A regression here is the store branch degrading to a warning while
   the handler wires with an empty DSN.

The ``local`` and ``onex-prod`` instances deliberately keep ``dsn_env`` and are
asserted to, so this migration cannot silently widen: ``local`` is fed by the
compose lanes (OMN-17562's scope, not this ticket's) and ``onex-prod`` would
need a prod promotion gate this ticket does not open.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseBindingTarget,
    _resolve_binding_dsn,
)
from omnibase_infra.topology import load_environment_topology

pytestmark = pytest.mark.unit

_APPLICATION = "application"
_BINDING = "tenant_projection"
_EXPECTED_SECRET_REF = "database.tenant_projection.dsn"


class _StubResolver:
    """Minimal stand-in for ``SecretResolver.get_secret``.

    Deliberately not a MagicMock: the point of these tests is that exactly one
    logical name is asked for, so the stub records the ask and refuses anything
    it was not primed with, which a permissive mock would silently satisfy.
    """

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values
        self.asked: list[str] = []

    def get_secret(self, logical_name: str, required: bool = True) -> SecretStr | None:
        self.asked.append(logical_name)
        value = self._values.get(logical_name)
        return None if value is None else SecretStr(value)


def test_onex_dev_tenant_projection_binding_is_store_carried() -> None:
    """onex-dev carries the ref, and carries no env fallback beside it."""
    topology = load_environment_topology("onex-dev", None)
    binding = topology.databases[_APPLICATION].bindings[_BINDING]

    assert binding.secret_ref == _EXPECTED_SECRET_REF
    assert binding.dsn_env is None, (
        "the onex-dev tenant_projection binding must carry NO env var beside "
        "its secret_ref. A dsn_env here is a credential materialized into a "
        "pod environment, which the 2026-09-03 operator ruling forbids -- and "
        "with both set, a stale env var would shadow the store silently."
    )
    assert binding.principal == "tenant_projection_writer"


@pytest.mark.parametrize("instance", ["local", "onex-prod"])
def test_unmigrated_instances_keep_their_env_carrier(instance: str) -> None:
    """Only onex-dev moved. The other two are asserted, not assumed."""
    topology = load_environment_topology(instance, None)
    binding = topology.databases[_APPLICATION].bindings[_BINDING]

    assert binding.dsn_env == "ONEX_TENANT_DB_URL"
    assert binding.secret_ref is None


def test_store_carried_binding_resolves_through_the_resolver_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store branch reads the store, and never the environment."""
    binding = ProjectionDatabaseBindingTarget(
        binding_ref=_BINDING,
        database_ref=_APPLICATION,
        physical_database="omnidash_analytics",
        principal="tenant_projection_writer",
        secret_ref=_EXPECTED_SECRET_REF,
    )
    # A DECOY on the legacy env name. If any os.environ fallback survives in
    # the store branch, this value is what leaks through -- and it is the
    # wrong principal, so it would reproduce the OMN-15425 identity mismatch
    # that DLQ'd 100% of tenant-projection input.
    monkeypatch.setenv("ONEX_TENANT_DB_URL", "postgresql://decoy@host/db")
    resolver = _StubResolver({_EXPECTED_SECRET_REF: "postgresql://store@host/db"})

    resolved = _resolve_binding_dsn(binding, resolver)  # type: ignore[arg-type]

    assert resolved == "postgresql://store@host/db"
    assert resolver.asked == [_EXPECTED_SECRET_REF]


def test_env_carried_binding_still_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy carrier is unchanged -- this migration is additive."""
    binding = ProjectionDatabaseBindingTarget(
        binding_ref="omninode_runtime_service",
        database_ref=_APPLICATION,
        physical_database="omnidash_analytics",
        principal="omninode_runtime",
        dsn_env="OMNINODE_INTERNAL_DB_URL",
    )
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", "postgresql://internal@host/db")

    assert _resolve_binding_dsn(binding, None) == "postgresql://internal@host/db"


def test_unresolvable_store_ref_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store that answers nothing yields nothing -- never an env read.

    The empty string is what routes into the caller's ``ValueError``; the
    caller is what refuses. This asserts the resolver does not paper over the
    miss, including when the legacy env var happens to be populated.
    """
    binding = ProjectionDatabaseBindingTarget(
        binding_ref=_BINDING,
        database_ref=_APPLICATION,
        physical_database="omnidash_analytics",
        principal="tenant_projection_writer",
        secret_ref=_EXPECTED_SECRET_REF,
    )
    monkeypatch.setenv("ONEX_TENANT_DB_URL", "postgresql://decoy@host/db")

    assert _resolve_binding_dsn(binding, _StubResolver({})) == ""  # type: ignore[arg-type]
    # No resolver at all is the deployment mistake of wiring this contract into
    # a process that holds no store. Also empty, also refused by the caller.
    assert _resolve_binding_dsn(binding, None) == ""


def test_refusal_names_the_store_carrier_not_a_null_env_var() -> None:
    """The operator-facing error must name the thing that is missing.

    Before OMN-17556 the message interpolated ``binding.dsn_env``. For a
    store-carried binding that is ``None``, so the refusal would have read
    ``tenant_projection:None`` -- naming nothing an operator can act on.
    """
    binding = ProjectionDatabaseBindingTarget(
        binding_ref=_BINDING,
        database_ref=_APPLICATION,
        physical_database="omnidash_analytics",
        principal="tenant_projection_writer",
        secret_ref=_EXPECTED_SECRET_REF,
    )

    assert binding.carrier_description == f"secret_ref={_EXPECTED_SECRET_REF}"
    assert "None" not in binding.carrier_description
