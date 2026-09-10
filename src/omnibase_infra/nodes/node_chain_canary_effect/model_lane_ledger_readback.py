# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A lane's declared LEDGER readback reference (OMN-16964).

Sibling of ``ModelLaneProjectionReadback`` and deliberately a separate type
rather than a reuse of it. The two carry the same two fields today, and that
is a coincidence of the current lane topology, not a shared contract: link 2
reads ``delegation_workflow_state`` and link 5 reads ``ledger_chain``, and a
single shared type would make it possible to resolve one declaration and hand
it to the other leg without anything noticing.

It lives beside ``lane_transport.py`` rather than in ``models/`` for the same
reason its projection sibling does: ``models/__init__`` imports
``lane_transport``, so a model placed there and imported by ``lane_transport``
would be a genuine import cycle. One file, one model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLaneLedgerReadback(BaseModel):
    """A lane id and the NAME of the variable its ledger DSN arrives in."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    lane: str = Field(description="Lane id this readback was declared under")
    dsn_env: str = Field(
        description="NAME of the environment variable carrying the DSN, never the DSN"
    )


__all__ = ["ModelLaneLedgerReadback"]
