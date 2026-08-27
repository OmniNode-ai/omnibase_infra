# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Node: NodeChainCanaryEffect — scheduled live delegation-chain probe.

OMN-16773. The 13-class delegation matrix existed only as a recorded manual
recipe. Nothing ran it. An omnimarket contract change on 2026-08-23
(OMN-15631) silently pushed a typed def-B handler onto the runtime's
projection dispatch arm, every delegation began quarantining, and the chain
stayed dead for four days until a human happened to fire the recipe by hand
(OMN-16767).

``delegation-seam-gate.yml`` was green throughout, and correctly so: it
drives the seam over ``InMemoryTransport``, where the wiring is constructed
by the test rather than by the deployed runtime. No in-memory test can see a
deployed wiring arm pick the wrong path.

This node closes that gap the only way it can be closed — by firing a real
delegation at the real deployed ingress on a schedule and failing loudly
when no terminal comes back. It is the phase-1 seed of the observability
work, deliberately one delegation class against one lane. It is not a
framework and should not grow into one here.

Related Tickets:
    - OMN-16773: this ticket
    - OMN-16767: the dead chain this canary would have caught in hours
    - OMN-16769: quarantine-sink depth/projection (aggregate); this node's
      quarantine leg is correlation-SCOPED and does not overlap it
    - OMN-16027: publish_envelope is fail-open — why ok=true with no
      terminal is treated as RED here
    - OMN-15190: dev-lane-liveness, the sibling scheduled probe whose
      "is the lane up" signal this one deliberately does not duplicate
"""

from omnibase_infra.nodes.node_chain_canary_effect.node import NodeChainCanaryEffect

__all__ = ["NodeChainCanaryEffect"]
