# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to retire one hot-patch ledger row (OMN-17427)."""

from pydantic import BaseModel, ConfigDict


class ModelHotpatchLedgerReconcileRequest(BaseModel):
    """Identify one ledger row and where to verify its retirement.

    Every path is supplied by the caller: the ledger's location is a fact of
    the host that holds it, so it has no default here.

    ``merge_commit`` is needed only for a row that records none; a row that
    records one retires on its own commit. ``deployed_ref`` defaults to the
    HEAD of the row's source clone, the same default the rebuild preflight
    uses for its build ref. ``note`` is appended to the evidence the handler
    writes itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    ledger_path: str
    clones_root: str
    container: str
    file: str
    merge_commit: str = ""
    deployed_ref: str = ""
    note: str = ""
    docker_cmd: str = "docker"
