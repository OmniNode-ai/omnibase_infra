# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node: NodeEvidenceAutocloseSweepEffect — companion-merge -> dod_verify -> governed Done flip.

First slice of OMN-16106: a scheduled sweep that enumerates recently-merged
``onex_change_control`` evidence companions, extracts each companion's
Evidence-Ticket binding, re-runs the EXISTING ``onex skill dod_verify``
verifier exactly as the controller does for governed flips, and either
flips the bound Linear ticket Done (all ACs receipt-proven) or posts the
honest gap as a comment (never flips on any gap). DRY-RUN by default;
``apply=True`` is required to mutate Linear.

Related Tickets:
    - OMN-16106: this ticket (first slice)
    - OMN-13856: dod_verify / DurableEvidenceGate mechanical no-fake-Done guard
"""

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.node import (
    NodeEvidenceAutocloseSweepEffect,
)

__all__ = ["NodeEvidenceAutocloseSweepEffect"]
