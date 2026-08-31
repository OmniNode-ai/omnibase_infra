# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16944: rule-based (not per-credential) secret sources for minted refs.

The lane secret map is an exact-match dict. A BYOK credential ref is minted at
request time as ``cred_{tenant}_{provider}_{uuid4hex}``, so it can never be
pre-declared as a ``mappings`` entry -- the resolver returns ``None`` and the
effect boundary fails closed 100% of the time.

These tests pin the fix: a lane declares a **namespace rule** (an anchored ref
pattern -> a store-backed source template) once, and every credential minted
after that lane was deployed resolves through it with no manifest edit and no
redeploy. The fail-closed posture is preserved and strengthened:

* a ref matching a declared namespace resolves ONLY through that namespace's
  source -- convention fallback is structurally unreachable for it, whether or
  not ``enable_convention_fallback`` is on;
* a namespace may never declare ``source_type="env"``: a namespace-matched name
  is tenant-influenced input, and an env-shaped source would let it name a
  platform (house) variable;
* an undeclared minted-shape ref still resolves to ``None`` (fail closed).
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_namespace_rule import (
    ModelSecretNamespaceRule,
)
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec
from omnibase_infra.runtime.secret_resolver import SecretResolver

pytestmark = pytest.mark.unit

# The shape omnimarket's ``credential_publisher.mint_api_key_ref`` emits.
TENANT_CREDENTIAL_REF_PATTERN = r"^cred_[A-Za-z0-9._:-]+_[A-Za-z0-9_-]+_[0-9a-f]{32}$"

_MINTED_REF = "cred_acme-corp_openrouter_" + "0" * 32


def _tenant_namespace(
    *,
    source_type: str = "infisical",
    source_path_template: str = "{ref}",
) -> ModelSecretNamespaceRule:
    return ModelSecretNamespaceRule(
        namespace="tenant_inference_credentials",
        ref_pattern=TENANT_CREDENTIAL_REF_PATTERN,
        source_type=source_type,  # type: ignore[arg-type]
        source_path_template=source_path_template,
    )


class TestNamespaceRuleResolution:
    """AC1 -- the rule, not the entry, is what the lane declares."""

    def test_undeclared_minted_ref_has_no_source_and_fails_closed(self) -> None:
        """RED baseline: today's lane shape. No namespace -> no source spec."""
        config = ModelSecretResolverConfig(enable_convention_fallback=False)
        resolver = SecretResolver(config=config)
        assert resolver.get_source_info(_MINTED_REF) is None
        assert resolver.get_secret(_MINTED_REF, required=False) is None

    def test_declared_namespace_resolves_a_ref_minted_after_deploy(self) -> None:
        """One declared rule serves refs the lane has never seen before."""
        config = ModelSecretResolverConfig(
            namespaces=[_tenant_namespace()],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        for ref in (
            _MINTED_REF,
            "cred_tenant-b_gemini_" + "f" * 32,
            "cred_tenant-c_openai_" + "a1b2c3d4" * 4,
        ):
            info = resolver.get_source_info(ref)
            assert info is not None, ref
            assert info.source_type == "infisical"

    def test_source_path_template_is_interpolated_with_the_ref(self) -> None:
        config = ModelSecretResolverConfig(
            namespaces=[
                _tenant_namespace(source_path_template="/tenant-credentials/{ref}")
            ],
        )
        resolver = SecretResolver(config=config)
        spec = resolver.resolve_namespace_source(_MINTED_REF)
        assert spec == ModelSecretSourceSpec(
            source_type="infisical",
            source_path=f"/tenant-credentials/{_MINTED_REF}",
        )

    def test_non_matching_name_is_not_claimed_by_the_namespace(self) -> None:
        config = ModelSecretResolverConfig(namespaces=[_tenant_namespace()])
        resolver = SecretResolver(config=config)
        assert resolver.resolve_namespace_source("llm.openrouter.api_key") is None
        # An un-anchored substring match must not claim it either.
        assert resolver.resolve_namespace_source(f"prefix_{_MINTED_REF}") is None

    def test_explicit_mapping_still_wins_over_a_namespace(self) -> None:
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name=_MINTED_REF,
                    source=ModelSecretSourceSpec(
                        source_type="file", source_path="/run/secrets/pinned"
                    ),
                )
            ],
            namespaces=[_tenant_namespace()],
        )
        resolver = SecretResolver(config=config)
        info = resolver.get_source_info(_MINTED_REF)
        assert info is not None
        assert info.source_type == "file"

    def test_namespace_names_are_listed_for_introspection_without_values(self) -> None:
        config = ModelSecretResolverConfig(namespaces=[_tenant_namespace()])
        resolver = SecretResolver(config=config)
        assert resolver.list_configured_namespaces() == ["tenant_inference_credentials"]


class TestNamespaceStaysFailClosed:
    """AC2 -- a namespace can never become a route to a house key."""

    def test_namespace_may_not_declare_an_env_source(self) -> None:
        with pytest.raises(ValidationError):
            _tenant_namespace(source_type="env")

    def test_matched_ref_never_falls_through_to_convention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Convention ON + a planted env var must NOT satisfy a namespaced ref.

        Without the namespace, ``enable_convention_fallback`` maps the ref to
        ``CRED_ACME-CORP_OPENROUTER_000...`` -- a name a tenant partly controls.
        With the namespace declared, that path is unreachable: the only source
        is the declared store, which has no value here, so resolution is None.
        """
        env_name = re.sub(r"[^A-Z0-9]", "_", _MINTED_REF.upper())
        monkeypatch.setenv(env_name, "house-key-value")

        config = ModelSecretResolverConfig(
            namespaces=[_tenant_namespace()],
            enable_convention_fallback=True,
        )
        resolver = SecretResolver(config=config)
        # No Infisical handler wired -> the declared source yields nothing.
        assert resolver.get_secret(_MINTED_REF, required=False) is None

    def test_a_namespace_may_not_shadow_a_declared_required_secret(self) -> None:
        with pytest.raises(ValidationError):
            ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="llm.openrouter.api_key",
                        source=ModelSecretSourceSpec(
                            source_type="infisical", source_path="OPENROUTER_API_KEY"
                        ),
                    )
                ],
                required_secrets=["llm.openrouter.api_key"],
                namespaces=[
                    ModelSecretNamespaceRule(
                        namespace="too-broad",
                        ref_pattern=r"^llm\..*$",
                        source_type="infisical",
                        source_path_template="{ref}",
                    )
                ],
            )

    def test_a_namespace_may_not_shadow_a_bootstrap_secret(self) -> None:
        with pytest.raises(ValidationError):
            ModelSecretResolverConfig(
                bootstrap_secrets=["infisical.client_secret"],
                namespaces=[
                    ModelSecretNamespaceRule(
                        namespace="too-broad",
                        ref_pattern=r"^infisical\..*$",
                        source_type="infisical",
                        source_path_template="{ref}",
                    )
                ],
            )


class TestNamespaceRuleValidation:
    """The rule itself must be unambiguous at config-construction time."""

    def test_ref_pattern_must_be_anchored(self) -> None:
        with pytest.raises(ValidationError):
            ModelSecretNamespaceRule(
                namespace="unanchored",
                ref_pattern=r"cred_.*",
                source_type="infisical",
                source_path_template="{ref}",
            )

    def test_ref_pattern_must_compile(self) -> None:
        with pytest.raises(ValidationError):
            ModelSecretNamespaceRule(
                namespace="broken",
                ref_pattern=r"^cred_[(*$",
                source_type="infisical",
                source_path_template="{ref}",
            )

    def test_ref_pattern_must_not_match_everything(self) -> None:
        """A catch-all namespace would silently claim every logical name."""
        with pytest.raises(ValidationError):
            ModelSecretNamespaceRule(
                namespace="catch-all",
                ref_pattern=r"^.*$",
                source_type="infisical",
                source_path_template="{ref}",
            )

    def test_source_path_template_must_reference_the_ref(self) -> None:
        with pytest.raises(ValidationError):
            ModelSecretNamespaceRule(
                namespace="static",
                ref_pattern=TENANT_CREDENTIAL_REF_PATTERN,
                source_type="infisical",
                source_path_template="/a/fixed/path",
            )

    def test_namespace_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            ModelSecretResolverConfig(
                namespaces=[_tenant_namespace(), _tenant_namespace()]
            )
