# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared bound on concurrently in-flight consumed records (OMN-18852).

Measured on the ``.201`` dev lane 2026-09-19: every LLM inference on the lane,
local *and* cloud, is served by one partition with one consumer-group member,
and ``EventBusKafka._consume_loop`` awaits the handler inline inside the poll
loop. Fifteen consecutive inferences across three providers produced **zero**
overlapping pairs; queue wait grew monotonically 3 s to 445 s across nine
delegations in 26 minutes, and three callers timed out at the CLI's ~306 s
while their answers were produced and published after they had exited.

The serialisation is structural, not a tunable: one ``await`` inside a doubly
nested ``for``. This module supplies the declaration that lets a node opt out
of it.

Shape, as a TOP-LEVEL key in the node's ``contract.yaml``::

    consume_concurrency:
      max_in_flight_records: 4

Two deliberate choices, each of which is the reason this is a loader rather
than a field on a core contract model:

1. **The key is read here, not by** ``ModelEventBusSubcontract`` **in**
   ``omnibase_core``. Adding a field there would be a third-repo layering
   change for a runtime-local concern. The precedent is omnimarket's
   ``handler_execution_budget`` / ``load_handler_execution_budget``, which is
   likewise a top-level contract key read by a loader beside its consumer.
   ``runtime/auto_wiring/discovery.py:_parse_contract`` reads the contract with
   a plain ``yaml.safe_load`` and hand-picks only the keys it knows, so an
   unrecognised top-level key reaches no ``extra="forbid"`` model and cannot
   break parsing.

2. **An absent declaration means 1, and 1 means the inline path.** Every node
   in the fleet omits this key today, so the default must reproduce the
   pre-OMN-18852 behaviour exactly -- not "a semaphore of size one", which is
   a different code path with different failure modes, but the same inline
   ``await``. A malformed declaration is NOT quietly treated as absent: a
   bound that cannot be read is refused, because a declaration that silently
   evaluates to 1 reads downstream exactly like a lane that opted out.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CONSUME_CONCURRENCY_KEY",
    "DEFAULT_MAX_IN_FLIGHT_RECORDS",
    "ModelConsumeConcurrency",
    "load_consume_concurrency",
]

CONSUME_CONCURRENCY_KEY = "consume_concurrency"

# The serial default. Not merely "a small number": it selects the inline
# ``await`` branch of ``_consume_loop``, so an undeclared node is unchanged.
DEFAULT_MAX_IN_FLIGHT_RECORDS = 1

# Upper bound on the declaration. Each in-flight record holds a handler, its
# provider connection and its retry state; an unbounded declaration would
# convert a queueing problem into a resource-exhaustion one, and the consumer
# still has to finish everything it started before ``max.poll.interval.ms``.
MAX_DECLARABLE_IN_FLIGHT_RECORDS = 64


class ModelConsumeConcurrency(BaseModel):
    """How many consumed records one ``(topic, group_id)`` may have in flight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_in_flight_records: int = Field(
        default=DEFAULT_MAX_IN_FLIGHT_RECORDS,
        ge=1,
        le=MAX_DECLARABLE_IN_FLIGHT_RECORDS,
        description=(
            "Maximum records dispatched concurrently from one poll loop. 1 "
            "keeps the inline serial path; greater than 1 dispatches under a "
            "semaphore and gives up global partition ordering for the topic."
        ),
    )

    @property
    def is_serial(self) -> bool:
        """True when this bound selects the unchanged inline path."""
        return self.max_in_flight_records <= DEFAULT_MAX_IN_FLIGHT_RECORDS


def load_consume_concurrency(contract_path: Path) -> ModelConsumeConcurrency:
    """Read ``consume_concurrency`` from a node contract, defaulting to serial.

    A contract path that does not exist on disk resolves to the serial
    default rather than raising. That is not the malformed case: auto-wiring
    reaches this with synthetic and non-disk-sourced contracts whose
    ``contract_path`` points at nothing, and there is no declaration to read
    from a file that is not there. It is the same disposition
    ``load_published_events_map`` takes for the same reason
    (``runtime/event_bus_subcontract_wiring.py:1394``). The consequence,
    stated rather than left implicit: a node whose contract reaches the
    runtime by any route other than this file gets the serial default,
    whatever its YAML says.

    Args:
        contract_path: The node's ``contract.yaml``.

    Returns:
        The declared bound; the serial default when the key is absent or the
        file does not exist.

    Raises:
        ValueError: The file exists and is not a mapping, the key is present
            but is not a mapping, or the declared value fails validation.
            Each of those is a bound the operator believes they set and which
            would not bind.
    """
    if not contract_path.exists():
        return ModelConsumeConcurrency()

    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{contract_path} must contain a mapping")

    if CONSUME_CONCURRENCY_KEY not in raw:
        return ModelConsumeConcurrency()

    declared = raw[CONSUME_CONCURRENCY_KEY]
    if not isinstance(declared, dict):
        raise ValueError(
            f"{contract_path} declares {CONSUME_CONCURRENCY_KEY} as "
            f"{type(declared).__name__}, which is not a mapping; a bound that "
            "cannot be read must not be silently treated as absent"
        )

    return ModelConsumeConcurrency.model_validate(declared)
