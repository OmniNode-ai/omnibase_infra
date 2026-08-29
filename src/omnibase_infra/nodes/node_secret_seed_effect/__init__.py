# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Node: NodeSecretSeedEffect — headless Infisical secret seeding (OMN-16897).

The operator asked for one command that puts a secret into Infisical with no
UI and no interactive login, and asked for it as a node: *"it should be a
node based workflow"*, *"we probably already have the handler written
somewhere."*

The second half was right. ``InfisicalSecretStore.set_secret`` (OMN-10557)
is a complete async upsert over ``AdapterInfisical``'s
``create_secret``/``update_secret`` (OMN-2286), and
``omnimarket.projection.credential_publisher`` has been writing through it
in production for customer BYOK intake. What did not exist was a canonical
front door: the one Infisical EFFECT node in the tree
(``node_setup_infisical_effect``) predates definition-B, takes an envelope
dict, shells out to two standalone scripts, and has no production caller at
all. So the capability was real and simply unreachable headlessly.

This node is the missing wrapper and nothing more. It adds no write path —
it composes the existing one behind a def-B handler and a contract.

What it deliberately refuses to do
----------------------------------
Carry a secret value. Node inputs are serialised onto the bus and into the
event log, so a value on the request would be durably persisted in both.
The request names a local FILE; the handler reads it at execution time and
the values never leave that call. Verification after a write is a NAME
listing — this node has no code path that reads a stored secret value, by
construction (``ProtocolSeedSecretStore`` omits ``get_secret``).

Related Tickets:
    - OMN-16897: this ticket
    - OMN-10557: ``InfisicalSecretStore`` — the reused write path
    - OMN-2286: ``AdapterInfisical`` internal-only policy and its explicit
      bootstrap/admin write carve-out
    - OMN-16451: config control-plane — the read side keeps secrets as
      refs; this is the write-side complement, not a second config path
    - OMN-16316: the customer-BYOK publisher whose value-crosses-once
      discipline this node copies for platform secrets
"""

from omnibase_infra.nodes.node_secret_seed_effect.node import NodeSecretSeedEffect

__all__ = ["NodeSecretSeedEffect"]
