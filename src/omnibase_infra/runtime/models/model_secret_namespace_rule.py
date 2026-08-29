# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rule-based secret source for *minted* logical names (OMN-16944).

``mappings`` declares one entry per known logical name. That is the right shape
for a platform secret, whose name is fixed at design time. It is the wrong shape
for a name minted at request time -- a BYOK credential ref
(``cred_{tenant}_{provider}_{uuid4hex}``) contains a uuid4, so it cannot be
pre-declared, and adding an entry per issued credential is a manual step plus a
lane redeploy per customer key rather than a mechanism.

A ``ModelSecretNamespaceRule`` declares the *rule* instead: an anchored ref
pattern and the store-backed source template every name matching it resolves
through. One declaration serves every credential registered after the lane was
deployed, with no manifest edit and no redeploy.

Fail-closed by construction:

* ``source_type`` is restricted to store-backed sources (``infisical``,
  ``file``). ``env`` is rejected: a namespace-matched name is partly
  tenant-supplied, so an env-shaped source would let a tenant ref name a
  platform (house) environment variable -- the exact drift OMN-15631 exists to
  prevent.
* the pattern must be anchored and must not match the empty string, so a rule
  can never become a catch-all that silently claims every logical name.
* ``source_path_template`` must reference ``{ref}``; a namespace whose source
  path is constant would map every tenant's ref onto one shared secret.
* a namespace only ever supplies a *source*. When the store holds no value for
  the interpolated path, resolution still returns ``None`` and a
  ``required=True`` caller still raises.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Store-backed only. ``env`` is deliberately absent -- see module docstring.
SecretNamespaceSourceType = Literal["infisical", "file"]

_REF_PLACEHOLDER = "{ref}"


class ModelSecretNamespaceRule(BaseModel):
    """A pattern-scoped, store-backed source for runtime-minted secret refs.

    Attributes:
        namespace: Stable identifier for this rule. Used in structured logs and
            introspection only -- never a lookup key, never a secret value.
        ref_pattern: Anchored regular expression the logical name must match in
            full for this rule to apply.
        source_type: Where matching refs resolve from. Store-backed only.
        source_path_template: The source path, with ``{ref}`` replaced by the
            matched logical name.
        ttl_seconds: Optional cache TTL override for refs matched by this rule.

    Example:
        >>> rule = ModelSecretNamespaceRule(
        ...     namespace="tenant_inference_credentials",
        ...     ref_pattern=r"^cred_[A-Za-z0-9._:-]+_[A-Za-z0-9_-]+_[0-9a-f]{32}$",
        ...     source_type="infisical",
        ...     source_path_template="{ref}",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    namespace: str = Field(
        ...,
        min_length=1,
        description="Stable identifier for this namespace rule (logs/introspection "
        "only; never a secret value).",
    )
    ref_pattern: str = Field(
        ...,
        min_length=1,
        description="Anchored regular expression ('^...$') that a logical name must "
        "match in full for this rule to supply its source.",
    )
    source_type: SecretNamespaceSourceType = Field(
        ...,
        description="Store-backed source type for matching refs. 'env' is not "
        "permitted: a namespace-matched name is partly tenant-supplied and must "
        "never be able to name a platform environment variable.",
    )
    source_path_template: str = Field(
        ...,
        min_length=1,
        description="Source path for matching refs. Must contain '{ref}', which is "
        "replaced by the matched logical name.",
    )
    ttl_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Optional cache TTL override in seconds for refs matched by "
        "this rule.",
    )

    @field_validator("ref_pattern")
    @classmethod
    def _validate_ref_pattern(cls, value: str) -> str:
        if not (value.startswith("^") and value.endswith("$")):
            raise ValueError(
                "ref_pattern must be fully anchored ('^...$') so a namespace can "
                "only claim names it matches end to end -- an unanchored pattern "
                f"claims every name containing the match: {value!r}"
            )
        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(
                f"ref_pattern is not a valid regular expression: {exc}"
            ) from exc
        if compiled.fullmatch("") is not None:
            raise ValueError(
                "ref_pattern matches the empty string, so it is a catch-all that "
                "would silently claim every logical name on the lane: "
                f"{value!r}"
            )
        return value

    @field_validator("source_path_template")
    @classmethod
    def _validate_source_path_template(cls, value: str) -> str:
        if _REF_PLACEHOLDER not in value:
            raise ValueError(
                "source_path_template must contain '{ref}' -- a constant source "
                "path would resolve every matching ref onto one shared secret: "
                f"{value!r}"
            )
        return value

    def matches(self, logical_name: str) -> bool:
        """Return whether this rule claims ``logical_name``."""
        return re.compile(self.ref_pattern).fullmatch(logical_name) is not None

    def source_path_for(self, logical_name: str) -> str:
        """Return the concrete source path for a matched ``logical_name``."""
        return self.source_path_template.replace(_REF_PLACEHOLDER, logical_name)


__all__: list[str] = ["ModelSecretNamespaceRule", "SecretNamespaceSourceType"]
