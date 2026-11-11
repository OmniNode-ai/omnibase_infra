# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What it means for a transactional producer to be fenced (OMN-21228).

A producer is fenced when a newer producer initialized under the same
``transactional_id``. The broker then refuses the older one's transaction
operations and aborts whatever it had open. Retrying cannot help: the older
producer will keep being refused for as long as it lives.

This is a classifier plus the error it maps to, not an ONEX handler: it defines
no ``handler_type`` or ``handler_category``, so the module name does not carry
the ``handler_`` prefix (the same arrangement as ``kafka_permanent_failure``).

``aiokafka`` is imported lazily so that importing this module, which the
in-memory bus does, does not pay for the client library.
"""

from __future__ import annotations

from functools import lru_cache

from omnibase_infra.errors import InfraUnavailableError


class InfraProducerFencedError(InfraUnavailableError):
    """A newer producer holds this producer's ``transactional_id``.

    Permanent for the producer that raised it: open a new producer to continue.
    """


@lru_cache(maxsize=1)
def _fencing_types() -> tuple[type[BaseException], ...]:
    from aiokafka.errors import (
        InvalidProducerEpoch,
        ProducerFenced,
        TransactionCoordinatorFenced,
    )

    return (ProducerFenced, InvalidProducerEpoch, TransactionCoordinatorFenced)


def is_producer_fenced(exc: BaseException) -> bool:
    """True when ``exc`` or anything in its cause/context chain is a fence.

    Type and chain, never text: aborting a fenced transaction can raise a
    second error whose ``__context__`` is the fence.
    """
    types = _fencing_types()
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, types):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
