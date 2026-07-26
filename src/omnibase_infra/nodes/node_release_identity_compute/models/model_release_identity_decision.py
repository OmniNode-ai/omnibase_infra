# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Release-identity decision model — output of the fitness gate handler.

Output model for ``node_release_identity_compute``. Carries everything the thin
CLI shim needs to reproduce the legacy gate's behavior byte-for-byte: the process
exit code, which stream the message goes to, the (possibly multi-line) message
text, and a machine-readable reason code for downstream/structured consumers.

Exit-code contract (unchanged from the legacy gate, OMN-13412):
    0 — version correctly ahead of the latest published tag, or the diff is exempt
    1 — packaged source changed without bumping past the latest published version
    2 — configuration error (no/malformed project.version)

Ticket: OMN-14471
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelReleaseIdentityDecision(BaseModel):
    """Output model for the release-identity compute handler.

    Attributes:
        exit_code: Process exit code (0, 1, or 2) per the OMN-13412 contract.
        stream: Which standard stream the shim prints ``message`` to. OK results
            go to ``stdout``; FAIL and config-error results go to ``stderr`` —
            preserving the legacy gate's stream routing exactly.
        message: The full message text (may contain embedded newlines for the
            two-line FAIL guidance). The shim prints this once, so a single
            ``\\n``-joined string reproduces the legacy gate's two ``print`` calls.
        reason_code: Machine-readable classification of the decision (e.g.
            ``version_ahead``, ``version_not_ahead``, ``no_published_tag``).

    Warning:
        **Non-standard __bool__ behavior**: Returns ``True`` only when the gate
        passes (``exit_code == 0``). Differs from typical Pydantic behavior where
        any populated model is truthy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    exit_code: Literal[0, 1, 2] = Field(
        ...,
        description="Process exit code: 0 pass/exempt, 1 not-ahead, 2 config error.",
    )
    stream: Literal["stdout", "stderr"] = Field(
        ...,
        description="Standard stream the shim prints `message` to.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Full message text (may be multi-line for FAIL guidance).",
    )
    reason_code: str = Field(
        ...,
        min_length=1,
        description="Machine-readable reason code for the decision.",
    )

    def __bool__(self) -> bool:
        """Truthy only when the gate passes (exit_code == 0).

        Warning:
            **Non-standard __bool__ behavior**: see the class docstring.
        """
        return self.exit_code == 0


__all__: list[str] = ["ModelReleaseIdentityDecision"]
