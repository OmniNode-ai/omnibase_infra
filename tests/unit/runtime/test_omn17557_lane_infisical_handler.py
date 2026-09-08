# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A config that declares an Infisical source gets a resolver that can read it (OMN-17557).

OMN-17556 moved the onex-dev ``tenant_projection`` binding from ``dsn_env`` to
``secret_ref: database.tenant_projection.dsn``, resolved through
``SecretResolver`` at the binding boundary. The ref, the rendered lane config
and the pod's bootstrap identity were all correct and live -- verified by
read-only SSM readback of the onex-dev
``omnimarket-tenant-projection-writer`` Deployment on 2026-09-08 -- and the
binding still did not resolve.

The reason was one branch: ``SecretResolver.from_container`` treated an absent
``HandlerInfisical`` in the container's service registry as graceful
degradation and built a resolver without one. NOTHING in either repo ever
registers ``HandlerInfisical`` there -- it is not contract-declared, and the
only other construction site (``RuntimeHostProcess._prefetch_config_from_infisical``)
builds an INLINE handler and shuts it down again -- so that branch was taken on
every runtime process, permanently. Every ``source_type: infisical`` mapping
then answered ``None`` behind a single "Infisical handler not configured"
WARNING, ``_resolve_binding_dsn`` returned ``""``, and all eight tenant-domain
projection contracts refused to wire with "Projection handler requires topology
bindings with configured DSNs: tenant_projection".

These tests pin the four distinct things that make the fix safe. Each fails for
a different real regression:

1. A declared Infisical source really does get a handler, and the logical name
   really resolves through it. The regression is the graceful-degradation
   branch coming back.
2. An incomplete bootstrap identity is a REFUSAL naming the missing variables.
   The regression is that refusal decaying into a warning, which is exactly
   what made the original defect invisible for as long as it lasted.
3. A lane with no Infisical source demands no identity. The regression is a
   lane that never touches the store being made to hold credentials for it.
4. A namespace-only declaration counts too. Reading ``mappings`` and not
   ``namespaces`` is the OMN-16944 defect, and this is a second site where it
   could recur.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    INFISICAL_BOOTSTRAP_VARS,
    build_lane_infisical_handler,
)
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_namespace_rule import (
    ModelSecretNamespaceRule,
)
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec
from omnibase_infra.runtime.secret_resolver import (
    SecretResolver,
    config_declares_infisical_source,
)

pytestmark = pytest.mark.unit

_TENANT_DSN_REF = "database.tenant_projection.dsn"
_TENANT_DSN_PATH = "/dev/onex-runtime/ONEX_TENANT_DB_URL"
_STORE_VALUE = "postgresql://tenant_projection_writer@store/omnidash_analytics"


class _StubInfisicalHandler:
    """Stands in for ``HandlerInfisical`` at the one method the sync path uses.

    Deliberately not a MagicMock: the point is that the resolver asks for one
    flat key inside one declared folder, and a permissive mock would satisfy
    any ask at all -- including the re-rooted one OMN-16984 fixed.
    """

    def __init__(self, values: dict[tuple[str, str | None], str]) -> None:
        self._values = values
        self.asked: list[tuple[str, str | None]] = []

    def get_secret_sync(
        self, secret_name: str, secret_path: str | None = None
    ) -> SecretStr | None:
        self.asked.append((secret_name, secret_path))
        value = self._values.get((secret_name, secret_path))
        return None if value is None else SecretStr(value)


class _EmptyServiceRegistry:
    """A registry that resolves nothing -- the live onex-dev shape.

    Every runtime process reaches ``from_container`` with a registry in exactly
    this state for ``HandlerInfisical``, because no code path registers one.
    """

    async def resolve_service(self, service_type: object) -> object:
        raise LookupError(f"not registered: {service_type!r}")


class _ContainerWithoutInfisical:
    service_registry = _EmptyServiceRegistry()


def _infisical_mapping_config() -> ModelSecretResolverConfig:
    return ModelSecretResolverConfig(
        mappings=[
            ModelSecretMapping(
                logical_name=_TENANT_DSN_REF,
                source=ModelSecretSourceSpec(
                    source_type="infisical",
                    source_path=_TENANT_DSN_PATH,
                ),
            )
        ],
        enable_convention_fallback=False,
    )


def _env_only_config() -> ModelSecretResolverConfig:
    return ModelSecretResolverConfig(
        mappings=[
            ModelSecretMapping(
                logical_name="gateway.attach.keycloak.issuer",
                source=ModelSecretSourceSpec(
                    source_type="env",
                    source_path="KEYCLOAK_ISSUER",
                ),
            )
        ],
        enable_convention_fallback=False,
    )


def _namespace_only_config() -> ModelSecretResolverConfig:
    return ModelSecretResolverConfig(
        mappings=[],
        namespaces=[
            ModelSecretNamespaceRule(
                namespace="tenant_inference_credentials",
                ref_pattern=r"^cred_[A-Za-z0-9._:-]+_[A-Za-z0-9_-]+_[0-9a-f]{32}$",
                source_type="infisical",
                source_path_template="/tenant-inference-credentials/{ref}",
            )
        ],
        enable_convention_fallback=False,
    )


def _set_complete_bootstrap_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set every bootstrap variable to a NON-secret placeholder.

    Values are inert placeholders; the construction seam is replaced in the
    tests that use them, so nothing here ever reaches a real Infisical server.
    """
    placeholders = {
        "INFISICAL_ADDR": "http://infisical.invalid:8080",
        "INFISICAL_CLIENT_ID": "test-client-id",
        "INFISICAL_CLIENT_SECRET": "test-client-secret",  # pragma: allowlist secret
        "INFISICAL_PROJECT_ID": "00000000-0000-0000-0000-000000000000",
        "INFISICAL_ENVIRONMENT_SLUG": "dev",
    }
    assert set(placeholders) == set(INFISICAL_BOOTSTRAP_VARS)
    for name, value in placeholders.items():
        monkeypatch.setenv(name, value)


@pytest.mark.asyncio
async def test_declared_infisical_source_gets_a_handler_and_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store-carried logical name resolves through the lane-built handler.

    Before OMN-17557 this returned ``None``: the container registers no
    handler, so the resolver was built without one and the mapping was dead.
    """
    _set_complete_bootstrap_identity(monkeypatch)
    stub = _StubInfisicalHandler(
        {("ONEX_TENANT_DB_URL", "/dev/onex-runtime"): _STORE_VALUE}
    )

    async def _factory() -> object:
        return stub

    resolver = await SecretResolver.from_container(
        _ContainerWithoutInfisical(),  # type: ignore[arg-type]
        _infisical_mapping_config(),
        infisical_handler_factory=_factory,  # type: ignore[arg-type]
    )
    resolved = resolver.get_secret(_TENANT_DSN_REF, required=False)

    assert resolved is not None, (
        "the binding boundary must resolve a declared Infisical mapping; "
        "None here is the OMN-17557 defect"
    )
    assert resolved.get_secret_value() == _STORE_VALUE
    # The folder declared by the mapping is carried through per read -- not
    # re-rooted to the handler default (OMN-16984).
    assert stub.asked == [("ONEX_TENANT_DB_URL", "/dev/onex-runtime")]


@pytest.mark.asyncio
async def test_incomplete_bootstrap_identity_refuses_and_names_the_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared store source with no identity to read it is a refusal.

    The refusal names the missing VARIABLES and never a value. Degrading this
    to a warning reproduces the original defect exactly: a resolver that looks
    built and answers nothing.
    """
    _set_complete_bootstrap_identity(monkeypatch)
    monkeypatch.delenv("INFISICAL_CLIENT_SECRET")
    monkeypatch.setenv("INFISICAL_PROJECT_ID", "   ")

    with pytest.raises(ProtocolConfigurationError) as excinfo:
        await SecretResolver.from_container(
            _ContainerWithoutInfisical(),  # type: ignore[arg-type]
            _infisical_mapping_config(),
            infisical_handler_factory=lambda: build_lane_infisical_handler(None),
        )

    message = str(excinfo.value)
    assert "INFISICAL_CLIENT_SECRET" in message
    assert "INFISICAL_PROJECT_ID" in message
    # A variable that IS set must not be reported missing.
    assert (
        "INFISICAL_ADDR'"
        not in message.split("Missing or blank:")[1].split(
            "Declared bootstrap variables"
        )[0]
    )


@pytest.mark.asyncio
async def test_env_only_config_demands_no_infisical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane that declares no store source needs no store credential.

    The negative control for the test above: without it, "refuses when the
    identity is missing" and "refuses always" are the same observation.
    """
    for name in INFISICAL_BOOTSTRAP_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEYCLOAK_ISSUER", "https://keycloak.invalid/realms/onex")

    resolver = await SecretResolver.from_container(
        _ContainerWithoutInfisical(),  # type: ignore[arg-type]
        _env_only_config(),
        # Supplied on purpose: the factory must NOT be called for a config that
        # declares no store source, so an incomplete identity is not a refusal.
        infisical_handler_factory=lambda: build_lane_infisical_handler(None),
    )
    resolved = resolver.get_secret("gateway.attach.keycloak.issuer", required=False)

    assert resolved is not None
    assert resolved.get_secret_value() == "https://keycloak.invalid/realms/onex"


def test_namespace_only_declaration_counts_as_an_infisical_source() -> None:
    """Reading ``mappings`` and not ``namespaces`` is the OMN-16944 defect.

    A BYOK ref is minted per request, so it can never appear in ``mappings``;
    a lane whose only store source is a namespace rule must still get a
    handler.
    """
    assert config_declares_infisical_source(_namespace_only_config()) is True
    assert config_declares_infisical_source(_infisical_mapping_config()) is True
    assert config_declares_infisical_source(_env_only_config()) is False


def test_file_backed_namespace_is_not_an_infisical_declaration() -> None:
    """A file-backed namespace must not make a lane start requiring Infisical."""
    config = ModelSecretResolverConfig(
        mappings=[],
        namespaces=[
            ModelSecretNamespaceRule(
                namespace="file_backed",
                ref_pattern=r"^ref_[a-z0-9]+$",
                source_type="file",
                source_path_template="/run/secrets/{ref}",
            )
        ],
        enable_convention_fallback=False,
    )
    assert config_declares_infisical_source(config) is False


@pytest.mark.asyncio
async def test_binding_boundary_factory_serves_the_tenant_projection_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The PRODUCTION path, end to end, from rendered config to resolved DSN.

    ``build_topology_secret_resolver`` is what the projection arm actually
    calls, and the four tests above would all still pass if it simply never
    supplied the factory. This drives the real function against a rendered
    config file shaped like the one read back live from the onex-dev
    ``omnimarket-tenant-projection-writer`` pod
    (``/app/data/delegation/secret_resolver.yaml``), and then feeds the result
    to ``_resolve_binding_dsn`` -- the exact call whose empty return produced
    "Projection handler requires topology bindings with configured DSNs:
    tenant_projection" on all eight contracts.
    """
    _set_complete_bootstrap_identity(monkeypatch)
    rendered = tmp_path / "secret_resolver.yaml"
    rendered.write_text(
        "enable_convention_fallback: false\n"
        "mappings:\n"
        f"  - logical_name: {_TENANT_DSN_REF}\n"
        "    source:\n"
        "      source_type: infisical\n"
        f"      source_path: {_TENANT_DSN_PATH}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SECRET_RESOLVER_CONFIG_PATH", str(rendered))

    stub = _StubInfisicalHandler(
        {("ONEX_TENANT_DB_URL", "/dev/onex-runtime"): _STORE_VALUE}
    )

    async def _seam(container: object | None = None) -> object:
        return stub

    monkeypatch.setattr(handler_wiring, "build_lane_infisical_handler", _seam)

    resolver = await handler_wiring.build_topology_secret_resolver(
        _ContainerWithoutInfisical()
    )
    assert resolver is not None

    binding = handler_wiring.ProjectionDatabaseBindingTarget(
        binding_ref="tenant_projection",
        database_ref="application",
        physical_database="omnidash_analytics",
        principal="tenant_projection_writer",
        secret_ref=_TENANT_DSN_REF,
    )
    # A DECOY on the legacy env name: an os.environ fallback creeping back into
    # the store branch would leak the WRONG principal here.
    monkeypatch.setenv("ONEX_TENANT_DB_URL", "postgresql://decoy@host/db")

    assert handler_wiring._resolve_binding_dsn(binding, resolver) == _STORE_VALUE


@pytest.mark.asyncio
async def test_binding_boundary_needs_no_identity_without_a_store_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control for the test above, at the same production entry point.

    A rendered config whose sources are all env-carried must not make the
    binding boundary demand an Infisical identity it never uses.
    """
    for name in INFISICAL_BOOTSTRAP_VARS:
        monkeypatch.delenv(name, raising=False)
    rendered = tmp_path / "secret_resolver.yaml"
    rendered.write_text(
        "enable_convention_fallback: false\n"
        "mappings:\n"
        "  - logical_name: gateway.attach.keycloak.issuer\n"
        "    source:\n"
        "      source_type: env\n"
        "      source_path: KEYCLOAK_ISSUER\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SECRET_RESOLVER_CONFIG_PATH", str(rendered))
    monkeypatch.setenv("KEYCLOAK_ISSUER", "https://keycloak.invalid/realms/onex")

    resolver = await handler_wiring.build_topology_secret_resolver(
        _ContainerWithoutInfisical()
    )

    assert resolver is not None
    resolved = resolver.get_secret("gateway.attach.keycloak.issuer", required=False)
    assert resolved is not None
    assert resolved.get_secret_value() == "https://keycloak.invalid/realms/onex"
