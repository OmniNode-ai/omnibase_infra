# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The caller's Claude Code session model, for the savings baseline (OMN-20557).

Ruling D3 makes a delegation's savings baseline the developer's own session
model when it is known. The /onex:delegate plugin knows it and passes it to
``onex delegate --session-model``. This model checks the value and writes it to
the request ``metadata`` map under ``session_model``, the key omnimarket's
delegate-skill handler resolves the baseline from. ``metadata`` is accepted by
every released request consumer, so an older consumer ignores the key rather
than refusing the request.

A malformed value is a usage error, never dropped: a caller that believes its
savings were stated against its own model, and were not, is the defect being
closed. A model id the pricing manifest does not price is not malformed; the
consumer reports that run as ``BASELINE_UNRESOLVED`` with no saving.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "SESSION_MODEL_METADATA_KEY",
    "ModelDelegateSessionModel",
]

SESSION_MODEL_METADATA_KEY = "session_model"

#: A model id as Claude Code names one (``claude-opus-5-5``, ``claude-opus-4-6[1m]``,
#: ``anthropic/claude-sonnet-5-5``): no whitespace, at most 128 characters.
_SESSION_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@\[\]+-]{0,127}$")


class ModelDelegateSessionModel(BaseModel):
    """One checked session model id."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model: str = Field(..., description="Session model id")

    @field_validator("model")
    @classmethod
    def _is_a_model_id(cls, value: str) -> str:
        """One check for every path in: ``fullmatch``, so no trailing newline."""
        if _SESSION_MODEL_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{value!r} is not a model id")
        return value

    @classmethod
    def from_flag(cls, value: str | None) -> ModelDelegateSessionModel | None:
        """The checked ``--session-model`` value; None when the flag is absent."""
        if value is None:
            return None
        try:
            return cls(model=value)
        except ValueError as exc:
            raise ValueError(
                f"--session-model {value!r} is not a model id: expected a letter "
                "or digit, then letters, digits or . _ : / @ [ ] + -, at most "
                "128 characters and no whitespace, as Claude Code names its model."
            ) from exc

    def as_metadata(self) -> dict[str, str]:
        """The request ``metadata`` entry the consumer reads."""
        return {SESSION_MODEL_METADATA_KEY: self.model}
