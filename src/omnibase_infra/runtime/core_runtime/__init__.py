# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Composition-root wiring for the ONE core runtime (epic OMN-14717, S6, OMN-14758).

This package holds the infra-side composition wiring that binds delegation COMMAND
topics onto ``omnibase_core.runtime.runtime_dispatch.RuntimeDispatch`` over the S3
``KafkaTransport`` (prod) / S2 ``InMemoryTransport`` (local/CI) face. Every surface
here is gated behind the ``ONEX_CORE_RUNTIME_TOPICS`` allowlist (default EMPTY ⇒ zero
behavior change; the legacy kernel owns everything — the single-lever rollback state).

Modules:

* :mod:`routing_map_builder` — contract → ``dict[str, DispatchRoute]`` with the
  single-owner build gate (§a).
* :mod:`dlq_resolver` — the real ``dlq_topic_resolver`` (contract-declared-first,
  else ONEX-canonical derived), plus the provision set (§b).
* :mod:`single_owner` — the boot-time single-owner-per-topic invariants (§c.3).
* :mod:`phantom_alarm` — the belt-and-suspenders phantom-subscription alarm (§d).
* :mod:`composition` — the allowlist parse + ``build_core_runtime`` orchestration (§c).
"""

from __future__ import annotations

__all__: list[str] = []
