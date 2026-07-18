# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Kernel glue for the S6 core runtime (epic OMN-14717, OMN-14758).

Keeps ``service_kernel.bootstrap()`` edits tiny: the kernel parses the allowlist, threads
it into the legacy subscribe path (the strict no-op when empty), and — ONLY when the
allowlist is non-empty — calls :func:`build_and_start_core_runtime` to construct the S3
Kafka transport, build ``RuntimeDispatch`` (§c.1-c.3), and start the supervised loop
(§c.5). Everything here is dormant unless ``ONEX_CORE_RUNTIME_TOPICS`` is set, so the
default boot path is unchanged.

Live scope (R-4/R-5): the live kernel build targets the KAFKA bus (the canary lane is
stability-test on the real broker). The in-memory parity path (§c.1) is exercised by the
S6 seam test, which constructs ``InMemoryTransport`` directly. Setting the allowlist on a
non-Kafka lane fails closed with a clear error rather than silently no-op-ing.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import cast

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.core_runtime.composition import (
    CoreRuntimeHandle,
    CoreTransport,
    build_core_runtime,
)
from omnibase_infra.runtime.core_runtime.routing_map_builder import (
    DefBTarget,
    HandlerResolver,
)
from omnibase_infra.utils import compute_consumer_group_id

logger = logging.getLogger(__name__)

__all__ = ["build_and_start_core_runtime", "build_kernel_handler_resolver"]

# Dedicated consumer group for the ONE core-runtime loop (single group member ⇒ all
# partitions assigned). Distinct from the per-node legacy groups so offsets are
# independent (rollback safety, R-4).
_CORE_RUNTIME_GROUP = "onex.core-runtime.delegation"


def build_kernel_handler_resolver(container: ModelONEXContainer) -> HandlerResolver:
    """Return a resolver that imports + instantiates a contract handler class.

    R-7 (parity): a handler whose ``__init__`` accepts a container is constructed WITH the
    shared kernel container so it wires the same dependencies as the legacy path; a
    no-arg handler is constructed directly. This is best-effort shared-construction; full
    dispatch-engine instance reuse is verified at the canary (steps 8-9, operator-gated).
    """

    def _resolve(ref: ModelHandlerRef) -> DefBTarget:
        module = importlib.import_module(ref.module)
        try:
            cls = getattr(module, ref.name)
        except AttributeError as exc:
            raise ModelOnexError(
                message=(
                    f"S6 kernel glue: handler {ref.name!r} not found in module "
                    f"{ref.module!r}."
                ),
                error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
            ) from exc
        if not (isinstance(cls, type) and hasattr(cls, "handle")):
            raise ModelOnexError(
                message=(
                    f"S6 kernel glue: {ref.module}.{ref.name} is not a def-B handler "
                    "class exposing handle(request) -> response."
                ),
                error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
            )
        # Constructor signature EXCLUDING self (signature(cls) resolves __init__).
        params: Mapping[str, inspect.Parameter]
        try:
            params = inspect.signature(cls).parameters
        except (TypeError, ValueError):
            params = {}
        # __init__(self, container) style vs no-arg style.
        wants_container = "container" in params or len(params) >= 1
        instance: object
        if wants_container:
            try:
                instance = cls(container)
            except TypeError:
                instance = cls()
        else:
            instance = cls()
        return cast("DefBTarget", instance)

    return _resolve


def build_and_start_core_runtime(
    *,
    core_runtime_topics: frozenset[str],
    contracts: Sequence[ModelDiscoveredContract],
    legacy_subscribed_topics: frozenset[str],
    use_kafka: bool,
    kafka_bootstrap_servers: str | None,
    environment: str,
    container: ModelONEXContainer,
) -> CoreRuntimeHandle:
    """Construct the Kafka transport + ``RuntimeDispatch`` and start the loop (§c.1-c.5).

    Precondition: ``core_runtime_topics`` non-empty (the kernel short-circuits on empty).
    """
    if not core_runtime_topics:
        raise ValueError("build_and_start_core_runtime requires a non-empty allowlist")
    if not use_kafka or not kafka_bootstrap_servers:
        raise ModelOnexError(
            message=(
                "S6 core runtime: ONEX_CORE_RUNTIME_TOPICS is set but the kernel bus is "
                "not Kafka. The live core-runtime loop targets the Kafka broker (the "
                "in-memory parity path is exercised by the S6 seam test). Unset the "
                "allowlist on non-Kafka lanes."
            ),
            error_code=EnumCoreErrorCode.INVALID_STATE,
        )

    # Sole group member ⇒ assigned all partitions of its topics. auto_offset_reset=latest
    # for the command spine (a stale queued command must not re-execute — R-4, operator
    # confirm at canary).
    transport = KafkaTransport.from_bootstrap(
        kafka_bootstrap_servers,
        group=_CORE_RUNTIME_GROUP,
        topics=sorted(core_runtime_topics),
        auto_offset_reset="latest",
    )
    # compute_consumer_group_id is referenced to keep group-id derivation consistent with
    # the legacy path's identity model for observability parity.
    _identity = ModelNodeIdentity(
        env=environment,
        service="omnibase_infra",
        node_name="core-runtime",
        version="0.0.0",
    )
    logger.info(
        "S6 core runtime: consumer group=%s (legacy-parity id=%s) topics=%s",
        _CORE_RUNTIME_GROUP,
        compute_consumer_group_id(_identity, EnumConsumerGroupPurpose.CONSUME),
        sorted(core_runtime_topics),
    )

    handle = build_core_runtime(
        core_runtime_topics=core_runtime_topics,
        contracts=contracts,
        # KafkaTransport structurally satisfies CoreTransport (consumer + producer face).
        transport=cast("CoreTransport", transport),
        handler_resolver=build_kernel_handler_resolver(container),
        legacy_subscribed_topics=legacy_subscribed_topics,
    )
    handle.start()
    return handle
