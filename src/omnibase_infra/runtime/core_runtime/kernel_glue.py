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
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast
from uuid import UUID

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.protocols.protocol_topic_provisioner import (
    ProtocolTopicProvisioner,
)
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

if TYPE_CHECKING:
    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

logger = logging.getLogger(__name__)

__all__ = ["build_and_start_core_runtime", "build_kernel_handler_resolver"]

# Dedicated consumer group for the ONE core-runtime loop (single group member ⇒ all
# partitions assigned). Distinct from the per-node legacy groups so offsets are
# independent (rollback safety, R-4).
_CORE_RUNTIME_GROUP = "onex.core-runtime.delegation"


def _param_wants_container(param: inspect.Parameter) -> bool:
    """True when ``param`` is the ONEX container-injection parameter.

    Matched by the canonical name ``container`` or by a ``ModelONEXContainer`` annotation
    (string annotations are compared by substring because ``from __future__ import
    annotations`` defers them to strings).
    """
    if param.name == "container":
        return True
    annotation = param.annotation
    if annotation is ModelONEXContainer:
        return True
    return isinstance(annotation, str) and "ModelONEXContainer" in annotation


def _construct_handler(cls: type, container: ModelONEXContainer) -> object:
    """Construct ``cls`` by EXPLICIT constructor binding — never a positional guess (R-7).

    Deterministic, fail-closed:

    * no required constructor parameter → ``cls()``;
    * exactly one required parameter that is the ONEX container (by name or annotation)
      → bound by keyword (or positionally when positional-only);
    * anything else → :class:`ModelOnexError`. A ``TypeError`` is NOT swallowed and a
      differently-named required argument is NOT silently fed the container.
    """
    try:
        parameters = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        parameters = []
    required = [
        p
        for p in parameters
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if not required:
        return cls()
    if len(required) == 1 and _param_wants_container(required[0]):
        param = required[0]
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return cls(container)
        return cls(**{param.name: container})
    raise ModelOnexError(
        message=(
            f"S6 kernel glue: handler {cls.__module__}.{cls.__qualname__} has an "
            f"unsupported constructor signature (required params "
            f"{[p.name for p in required]}). The core-runtime resolver only constructs a "
            "no-arg handler or one taking a single ONEX container parameter. Register "
            "the handler instance in the shared container service registry for R-7 "
            "instance reuse, or give it a container-only constructor — refusing to guess "
            "constructor arguments."
        ),
        error_code=EnumCoreErrorCode.INVALID_STATE,
    )


def build_kernel_handler_resolver(container: ModelONEXContainer) -> HandlerResolver:
    """Return a resolver that reuses shared handler instances, else builds fail-closed.

    R-7 (parity): a handler already registered as an INSTANCE in the shared container's
    service registry — i.e. the same object the legacy dispatch path wired — is REUSED
    verbatim so a moved node runs with identical dependency wiring. When no shared
    instance exists, the handler class is constructed by EXPLICIT, fail-closed
    constructor binding (see :func:`_construct_handler`), never a positional guess and
    never a swallowed ``TypeError``. Full dispatch-engine instance reuse for every
    handler is verified at the canary (steps 8-9, operator-gated).
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
        # R-7 shared-instance reuse: prefer the instance the legacy wiring already
        # constructed and registered in the shared container (sync resolution — the
        # RuntimeDispatch routing map is built synchronously). A miss raises and falls
        # through to deterministic construction.
        shared: object | None
        try:
            shared = container.get_service_sync(cls)
        except Exception:  # noqa: BLE001 — registry miss must not break resolution
            shared = None
        if shared is not None:
            logger.info(
                "S6 kernel glue: reusing shared %s.%s instance from the container "
                "(R-7 parity).",
                ref.module,
                ref.name,
            )
            return cast("DefBTarget", shared)
        return cast("DefBTarget", _construct_handler(cls, container))

    return _resolve


async def _provision_dlq_topics(
    dlq_topics: frozenset[str],
    *,
    provisioner: ProtocolTopicProvisioner | None,
    correlation_id: UUID | None,
) -> None:
    """Ensure every resolved DLQ topic exists BEFORE the dispatch loop starts (R-6).

    Fail-closed: the dispatch loop must not start with an unprovisioned dead-letter
    target (the first poison message would silently lose its DLQ send). An absent
    provisioner or a failed creation raises rather than starting a lossy loop.
    """
    if not dlq_topics:
        return
    ordered = sorted(dlq_topics)
    if provisioner is None:
        raise ModelOnexError(
            message=(
                f"S6 core runtime: DLQ topics {ordered} must be provisioned before the "
                "dispatch loop starts, but no topic provisioner is available. Refusing "
                "to start the loop with unprovisioned dead-letter targets (R-6)."
            ),
            error_code=EnumCoreErrorCode.INVALID_STATE,
        )
    for topic in ordered:
        created = await provisioner.ensure_topic_exists(
            topic, correlation_id=correlation_id
        )
        if not created:
            raise ModelOnexError(
                message=(
                    f"S6 core runtime: failed to provision DLQ topic {topic!r} before "
                    "loop start. Refusing to start the dispatch loop with an "
                    "unprovisioned dead-letter target (R-6)."
                ),
                error_code=EnumCoreErrorCode.INVALID_STATE,
            )
    logger.info(
        "S6 core runtime: provisioned %d DLQ topic(s) before loop start: %s",
        len(ordered),
        ordered,
    )


async def build_and_start_core_runtime(
    *,
    core_runtime_topics: frozenset[str],
    contracts: Sequence[ModelDiscoveredContract],
    legacy_subscribed_topics: frozenset[str],
    use_kafka: bool,
    kafka_bootstrap_servers: str | None,
    environment: str,
    container: ModelONEXContainer,
    provisioner: ProtocolTopicProvisioner | None = None,
    runtime: RuntimeHostProcess | None = None,
    correlation_id: UUID | None = None,
) -> CoreRuntimeHandle:
    """Construct the Kafka transport + ``RuntimeDispatch`` and start the loop (§c.1-c.5).

    Precondition: ``core_runtime_topics`` non-empty (the kernel short-circuits on empty).

    Ordering (R-6): resolved DLQ topics are provisioned via ``provisioner`` BEFORE the
    dispatch loop starts, so the first dead-letter send always targets an existing topic.
    When ``runtime`` is supplied, the handle's readiness snapshot (loop health + phantom
    alarm, §c.5/§d) is registered as a supplemental readiness probe so a crashed loop or a
    phantom subscription flips ``/ready`` FAIL.
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

    async def _lag_zero_topics() -> frozenset[str]:
        """LAG-zero readback for the phantom alarm (§d): topics whose consumer position
        has caught up to the broker high-water mark."""
        return await transport.caught_up_topics(core_runtime_topics)

    handle = build_core_runtime(
        core_runtime_topics=core_runtime_topics,
        contracts=contracts,
        # KafkaTransport structurally satisfies CoreTransport (consumer + producer face).
        transport=cast("CoreTransport", transport),
        handler_resolver=build_kernel_handler_resolver(container),
        legacy_subscribed_topics=legacy_subscribed_topics,
        lag_zero_provider=_lag_zero_topics,
    )
    # R-6: DLQ targets must exist before the first dead-letter send — provision BEFORE
    # the loop starts.
    await _provision_dlq_topics(
        handle.dlq_provision_topics,
        provisioner=provisioner,
        correlation_id=correlation_id,
    )
    handle.start()
    # §c.5/§d: fold loop-health + phantom-alarm into the live runtime readiness surface.
    if runtime is not None:
        runtime.register_readiness_probe("core_runtime_loop", handle.readiness_snapshot)
    return handle
