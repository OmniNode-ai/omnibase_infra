# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegation an ``onex delegate`` run falls back or escalates from (OMN-20606).

A caller that retries failed work (the merge-drain rung chain: the deployed dev
lane, then an in-process run, then GLM) issues a new delegation with its own
correlation id. Measured on the h201 dev lane on 2026-10-05, 93 of 94 failed
delegations that carried a session were answered by such a retry within 120 s,
and the answering ``delegation_events`` row named nothing about the failure it
answered. This model turns three CLI flags into the request ``metadata`` keys
omnimarket's delegate-skill handler, its in-process evidence terminal and the
``delegation_events`` projection read under the same names
(``omnimarket.models.delegation.delegation_lineage``):

* ``parent_correlation_id``: the parent delegation's correlation id;
* ``lineage_kind``: ``fallback`` (another route) or ``escalation`` (a stronger
  model on the same route);
* ``parent_failure_cause``: optional, a short token naming why the parent did
  not answer.

A malformed or partial lineage is a usage error, never dropped: a caller that
believes it linked two runs and silently did not is the defect being closed.
The request ``metadata`` map is accepted by every released request consumer,
so no consumer changes shape.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LINEAGE_KINDS",
    "LINEAGE_KIND_METADATA_KEY",
    "PARENT_CORRELATION_ID_METADATA_KEY",
    "PARENT_FAILURE_CAUSE_METADATA_KEY",
    "ModelDelegateLineage",
]

PARENT_CORRELATION_ID_METADATA_KEY = "parent_correlation_id"
LINEAGE_KIND_METADATA_KEY = "lineage_kind"
PARENT_FAILURE_CAUSE_METADATA_KEY = "parent_failure_cause"

#: The relations a lineage can name; omnimarket's EnumDelegationLineageKind.
LINEAGE_KINDS: tuple[str, ...] = ("fallback", "escalation")

#: A failure cause token; the same pattern omnimarket's projection accepts.
_CAUSE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")


class ModelDelegateLineage(BaseModel):
    """A run's parent delegation, the kind of relation, and why the parent failed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_correlation_id: uuid.UUID = Field(...)
    lineage_kind: Literal["fallback", "escalation"] = Field(...)
    parent_failure_cause: str | None = Field(
        default=None, pattern=_CAUSE_PATTERN.pattern
    )

    @classmethod
    def from_flags(
        cls,
        parent_correlation_id: str | None,
        lineage_kind: str | None,
        parent_failure_cause: str | None,
        *,
        own_correlation_id: uuid.UUID | None = None,
    ) -> ModelDelegateLineage | None:
        """The lineage the three flags name, or None when they name none.

        Raises:
            ValueError: the lineage is partial or malformed. The message names
                the flag at fault, so the CLI reports it as a usage error.
        """
        if parent_correlation_id is None and lineage_kind is None:
            if parent_failure_cause is not None:
                raise ValueError(
                    "--parent-failure-cause needs --parent-correlation-id and "
                    "--lineage-kind: a cause names why a parent failed, so a run "
                    "that names no parent has none."
                )
            return None
        if parent_correlation_id is None or lineage_kind is None:
            raise ValueError(
                "--parent-correlation-id and --lineage-kind go together: a run "
                "that follows another delegation names both the parent and "
                "whether it is a fallback or an escalation."
            )
        try:
            parent = uuid.UUID(parent_correlation_id.strip())
        except ValueError as exc:
            raise ValueError(
                f"--parent-correlation-id {parent_correlation_id!r} is not a UUID: "
                "pass the correlation id of the delegation this run follows."
            ) from exc
        if own_correlation_id is not None and parent == own_correlation_id:
            raise ValueError(
                f"--parent-correlation-id {parent} is this run's own correlation "
                "id: a delegation cannot follow itself."
            )
        kind = lineage_kind.strip()
        if kind == "fallback":
            resolved_kind: Literal["fallback", "escalation"] = "fallback"
        elif kind == "escalation":
            resolved_kind = "escalation"
        else:
            raise ValueError(
                f"--lineage-kind {lineage_kind!r} is not one of "
                f"{', '.join(LINEAGE_KINDS)}."
            )
        cause: str | None = None
        if parent_failure_cause is not None:
            cause = parent_failure_cause.strip()
            if not _CAUSE_PATTERN.fullmatch(cause):
                raise ValueError(
                    f"--parent-failure-cause {parent_failure_cause!r} is not a "
                    "cause token: lowercase, starting with a letter or digit, then "
                    "letters, digits, '_', '.', ':' or '-', at most 64 characters "
                    "(such as provider_quota_exhausted or exit_124)."
                )
        return cls(
            parent_correlation_id=parent,
            lineage_kind=resolved_kind,
            parent_failure_cause=cause,
        )

    def as_metadata(self) -> dict[str, str]:
        """The request metadata keys, as text."""
        metadata = {
            PARENT_CORRELATION_ID_METADATA_KEY: str(self.parent_correlation_id),
            LINEAGE_KIND_METADATA_KEY: self.lineage_kind,
        }
        if self.parent_failure_cause is not None:
            metadata[PARENT_FAILURE_CAUSE_METADATA_KEY] = self.parent_failure_cause
        return metadata
