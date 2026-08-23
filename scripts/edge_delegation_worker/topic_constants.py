# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local mirror of the delegation/gateway topic literals this worker needs.

The single approved source for these literals is
``omninode_infra/docker/onex-api/topic_constants.py`` (a different repo, not
importable at runtime from here) and the mirror-topic union declared in
``omnibase_infra/src/omnibase_infra/nodes/node_bus_forwarder_effect/
contract.yaml``. This module re-declares the same literal strings so the
worker's local-bus channel does not need a cross-repo import -- keep it in
sync with both sources by hand; do not add a topic here that is not already
declared in the forwarder's contract mirror-topic union.

This file lives under ``scripts/``, which the ``no-hardcoded-topics``
pre-commit gate excludes (that gate governs literal topic strings inside the
runtime-dispatched node tree, not standalone client scripts). These are
still literal string constants for a reason: an edge worker process has no
contract.yaml of its own to declare them in.
"""

from __future__ import annotations

# Inbound (cloud -> local mirror -> this worker claims from here).
DELEGATION_REQUEST_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
DELEGATION_INFERENCE_REQUEST_TOPIC = (
    "onex.cmd.omnibase-infra.delegation-inference-request.v1"
)

# Outbound (this worker publishes here -> local mirror -> cloud).
INFERENCE_RESPONSE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
DELEGATION_COMPLETED_TOPIC = "onex.evt.omnibase-infra.delegation-completed.v1"
DELEGATION_FAILED_TOPIC = "onex.evt.omnibase-infra.delegation-failed.v1"
OMNIBASE_INFRA_LLM_CALL_COMPLETED_TOPIC = (
    "onex.evt.omnibase-infra.llm-call-completed.v1"
)

INBOUND_TOPICS: tuple[str, ...] = (
    DELEGATION_REQUEST_TOPIC,
    DELEGATION_INFERENCE_REQUEST_TOPIC,
)
OUTBOUND_RESULT_TOPICS: tuple[str, ...] = (
    INFERENCE_RESPONSE_TOPIC,
    DELEGATION_COMPLETED_TOPIC,
)
OUTBOUND_FAILURE_TOPICS: tuple[str, ...] = (DELEGATION_FAILED_TOPIC,)
