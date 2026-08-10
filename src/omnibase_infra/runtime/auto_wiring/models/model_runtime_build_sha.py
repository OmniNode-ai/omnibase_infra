# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime build-identity SHA value with explicit absent-with-reason semantics.

OMN-10856: the auto-wiring manifest served over ``/v1/introspection/manifest``
had no way to bind reported topology to a specific deployed build (no image
SHA, no deployment SHA). A build-identity value that is simply
``str | None = None`` (the pattern used by
``omnibase_core.models.runtime_manifest.model_runtime_manifest.ModelRuntimeManifest.image_digest``)
cannot distinguish "the environment variable was genuinely unset" from any
other reason a caller might pass ``None`` — this model makes that
distinction load-bearing: exactly one of ``value`` / ``absent_reason`` is
non-``None`` at all times, so absence always carries an explanation instead
of silently reading as "unknown".

This module deliberately does NOT read process environment variables
itself — ``scripts/check-env-reads.sh`` restricts new env-var reads to an
approved boundary set (``service_kernel.py``, ``runtime/overlay/``, etc.)
and this is a plain data/models module. Callers resolve the raw string
(typically via the standard library's environment getter in
``service_kernel.py``, which is on that allowlist) and pass it to
:meth:`from_raw`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelRuntimeBuildSha(BaseModel):
    """A single build-identity SHA (image digest or deployment/source revision).

    Fail-fast, never a silent default: :meth:`from_raw` turns an unset (or
    blank) source value into ``value=None`` with a typed ``absent_reason``
    naming exactly why — never a fabricated placeholder like ``"unknown"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    value: str | None = Field(
        default=None,
        description="The resolved SHA/digest value, or None when absent.",
    )
    absent_reason: str | None = Field(
        default=None,
        description=(
            "Reason the value is absent. Must be None when value is present, "
            "and must be set (non-blank) when value is None."
        ),
    )

    @model_validator(mode="after")
    def _validate_presence_xor_reason(self) -> ModelRuntimeBuildSha:
        if self.value is None and not self.absent_reason:
            raise ValueError(
                "ModelRuntimeBuildSha requires a non-blank absent_reason "
                "when value is None — absence must always be explained, "
                "never silently defaulted."
            )
        if self.value is not None and self.absent_reason is not None:
            raise ValueError(
                "ModelRuntimeBuildSha.absent_reason must be None when value "
                "is present — a real value cannot also carry an absence "
                "reason."
            )
        return self

    @property
    def is_present(self) -> bool:
        """True when a real SHA/digest value was resolved."""
        return self.value is not None

    @classmethod
    def present(cls, value: str) -> ModelRuntimeBuildSha:
        """Construct a present value. Raises if ``value`` is blank."""
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "ModelRuntimeBuildSha.present() requires a non-blank value"
            )
        return cls(value=stripped, absent_reason=None)

    @classmethod
    def absent(cls, reason: str) -> ModelRuntimeBuildSha:
        """Construct an explicit absent-with-reason marker."""
        return cls(value=None, absent_reason=reason)

    @classmethod
    def from_raw(cls, raw: str | None, *, source_name: str) -> ModelRuntimeBuildSha:
        """Classify an already-fetched raw source value (fail-fast, no env I/O).

        ``raw`` is typically the environment lookup for ``source_name``,
        resolved by the caller (e.g. ``service_kernel.py``, the approved
        env-read boundary).
        A blank (whitespace-only) value is treated the same as unset.

        Args:
            raw: The raw value, or None if the source had nothing set.
            source_name: Human-readable name of where ``raw`` came from
                (e.g. the env var name), used only in the absent reason.
        """
        if raw is None or raw.strip() == "":
            return cls.absent(f"{source_name} is not set")
        return cls.present(raw)


__all__: list[str] = ["ModelRuntimeBuildSha"]
