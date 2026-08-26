# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Node: NodeSyncRevertWatchdogEffect — detect + re-flip Linear sync-revert.

OMN-16536's own acceptance criteria call for "A scheduled/CI assertion...
that no `GitAutomationState` on any team resolves a `completed`-type
ticket to a non-`completed` state on a bare/non-closing reference" — this
node is that assertion's mechanical layer (fix option 3 in the ticket
body): a durable sweep that detects a silent automation-driven Done-revert
and corrects it, exactly what closeout lanes have been hand-doing on every
occurrence (OMN-15977, OMN-16077, OMN-14498, OMN-15978, OMN-15918,
OMN-15751, OMN-16311, OMN-16544).

Detection uses Linear's own documented automation signature: an
``IssueHistory`` entry with ``actorId`` null and a populated ``botActor``
(Linear's GraphQL schema: "This field may be empty in the case of
integrations or automations" / "Null if the change was made by an
integration, automation, or system process") transitioning a
completed-type state to a non-completed-type state, with no human comment
posted near the transition and no human-driven state change since. This
node is a complementary safety net, NOT a substitute for fixing the
triggering ``GitAutomationState`` config directly (fix options 1/2 in
OMN-16536) — both remain open, separate work.

DRY-RUN by default (``apply=False``); ``apply=True`` is required to
actually flip a ticket's state or post a comment, matching the
``node_evidence_autoclose_sweep_effect`` (OMN-16106) precedent this node
is modeled on.

Related Tickets:
    - OMN-16536: this ticket
    - OMN-15373: original `merge -> Done` automation defect (fixed);
      flagged the `start`-triggered demotion as an unresolved residual
    - OMN-16106: node_evidence_autoclose_sweep_effect, the sibling sweep
      this node's structure/conventions are modeled on
    - OMN-14849: WS1 Done-revert enforcement receiver design (this node's
      "guard is not a mechanism" framing)
"""

from omnibase_infra.nodes.node_sync_revert_watchdog_effect.node import (
    NodeSyncRevertWatchdogEffect,
)

__all__ = ["NodeSyncRevertWatchdogEffect"]
