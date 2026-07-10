# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Stamp a verified tenant_id into an inbound envelope's payload."""

from __future__ import annotations

from uuid import UUID

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory


class HandlerStampTenantId:
    """Overwrite payload["tenant_id"] with a config-bound, verified value.

    Pure transform: never reads a tenant_id from the payload as authoritative,
    only ever overwrites with the slug the caller supplies (which
    ServiceTenantIngress derives from the tenant-prefixed subscription topic,
    never from message content).
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        envelope: object,
    ) -> ModelHandlerOutput[ModelEventEnvelope[dict[str, object]]]:
        """Dispatch entrypoint for contract-driven invocation (test/validation path).

        Production traffic runs through ServiceTenantIngress, which calls
        ``stamp`` directly with the tenant slug it derived from the wire
        topic -- this entrypoint has no topic context, so it is invoked with
        the tenant slug already resolved by the caller via
        ``envelope`` metadata rather than deriving one itself.
        """
        typed_envelope, tenant_slug, envelope_id, correlation_id = (
            _coerce_dispatch_input(envelope)
        )
        stamped = self.stamp(typed_envelope, tenant_slug)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=envelope_id,
            correlation_id=correlation_id,
            handler_id=type(self).__name__,
            result=stamped,
        )

    def stamp(
        self,
        envelope: ModelEventEnvelope[dict[str, object]],
        tenant_slug: str,
    ) -> ModelEventEnvelope[dict[str, object]]:
        """Return a copy of ``envelope`` with payload["tenant_id"] overwritten."""
        stamped_payload = {**envelope.payload, "tenant_id": tenant_slug}
        return envelope.model_copy(update={"payload": stamped_payload})


def _coerce_dispatch_input(
    envelope: object,
) -> tuple[ModelEventEnvelope[dict[str, object]], str, UUID, UUID]:
    typed_envelope = (
        envelope
        if isinstance(envelope, ModelEventEnvelope)
        else ModelEventEnvelope[dict[str, object]].model_validate(envelope)
    )
    tenant_slug = str(typed_envelope.get_metadata_value("tenant_slug", ""))
    if not tenant_slug:
        raise ValueError(
            "HandlerStampTenantId.handle() requires metadata.tags['tenant_slug'] "
            "-- production traffic runs through ServiceTenantIngress, which "
            "supplies the slug directly via stamp(); this entrypoint exists "
            "for contract-driven test/validation invocation only."
        )
    return (
        typed_envelope,
        tenant_slug,
        typed_envelope.envelope_id,
        typed_envelope.correlation_id or typed_envelope.envelope_id,
    )


__all__ = ["HandlerStampTenantId"]
