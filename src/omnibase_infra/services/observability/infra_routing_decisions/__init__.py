# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infra routing decisions observability service (OMN-8692, OMN-16025).

Kafka -> infra_routing_decisions projection consumer for
onex.evt.omnibase-infra.routing-decision.v1, the delegation routing reducer's
output. OMN-16025 repointed it here from
onex.evt.omnibase-infra.routing-decided.v1, the legacy AdapterModelRouter topic
that no live producer creates.
"""
