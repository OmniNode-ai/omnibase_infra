# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Bootstrap Handler Descriptor Model with Required handler_class.

This module provides ModelBootstrapHandlerDescriptor, a specialized handler
descriptor for bootstrap handlers that REQUIRES the handler_class field to be set.

Bootstrap handlers are hardcoded handlers that must always specify their
implementation class for dynamic import. Unlike contract-discovered handlers
where handler_class may be optional (inferred from convention), bootstrap
handlers have no contract file to derive the class from.

Part of OMN-1087: Implement HandlerBootstrapSource descriptor-based validation.

See Also:
    - ModelHandlerDescriptor: Base descriptor with optional handler_class
    - HandlerBootstrapSource: Source that uses this specialized descriptor
    - BootstrapEffectDefinition: TypedDict for bootstrap handler definitions

.. versionadded:: 0.6.4
    Created as part of OMN-1087 bootstrap handler validation.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from omnibase_infra.models.handlers.model_handler_descriptor import (
    ModelHandlerDescriptor,
)


class ModelBootstrapHandlerDescriptor(ModelHandlerDescriptor):
    """Handler descriptor for bootstrap handlers with required handler_class.

    This specialized descriptor extends ModelHandlerDescriptor to enforce that
    handler_class is always set. Bootstrap handlers are hardcoded and must
    specify their implementation class since there is no contract file to
    derive the class from.

    The key difference from ModelHandlerDescriptor:
    - handler_class: Required (str) instead of optional (str | None)

    All other fields maintain the same constraints as the parent class.

    Attributes:
        handler_id: Unique identifier for the handler (e.g., "proto.consul").
        name: Human-readable name for the handler.
        version: Semantic version (ModelSemVer). Accepts string, dict, or ModelSemVer.
        handler_kind: Handler kind (compute, effect, reducer, orchestrator).
        input_model: Fully qualified input model class path.
        output_model: Fully qualified output model class path.
        description: Optional description of the handler.
        handler_class: REQUIRED fully qualified Python class path for dynamic import.
        contract_path: Path to the source contract file (typically None for bootstrap).

    Example:
        Create a bootstrap handler descriptor:

        >>> descriptor = ModelBootstrapHandlerDescriptor(
        ...     handler_id="proto.consul",
        ...     name="Consul Handler",
        ...     version="1.0.0",
        ...     handler_kind="effect",
        ...     input_model="omnibase_infra.models.types.JsonDict",
        ...     output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
        ...     handler_class="omnibase_infra.handlers.handler_consul.HandlerConsul",
        ... )
        >>> descriptor.handler_class
        'omnibase_infra.handlers.handler_consul.HandlerConsul'

        Missing handler_class raises ValidationError:

        >>> from pydantic import ValidationError
        >>> try:
        ...     ModelBootstrapHandlerDescriptor(
        ...         handler_id="proto.consul",
        ...         name="Consul Handler",
        ...         version="1.0.0",
        ...         handler_kind="effect",
        ...         input_model="omnibase_infra.models.types.JsonDict",
        ...         output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
        ...         # handler_class omitted - will fail
        ...     )
        ... except ValidationError as e:
        ...     print("Validation failed as expected")
        Validation failed as expected

    Raises:
        ValidationError: If handler_class is not provided or is None.

    .. versionadded:: 0.6.4
        Created as part of OMN-1087 bootstrap handler validation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    # Override handler_class to be required (no default, not optional)
    # The Field() definition must include the pattern constraint from parent
    handler_class: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description=(
            "REQUIRED: Fully qualified Python class path for dynamic handler import. "
            "Bootstrap handlers must always specify this field since they have no "
            "contract file to derive the class from. "
            "Example: 'omnibase_infra.handlers.handler_consul.HandlerConsul'"
        ),
    )

    def to_base_descriptor(self) -> ModelHandlerDescriptor:
        """Convert to base ModelHandlerDescriptor for API compatibility.

        This method allows bootstrap descriptors to be used where the base
        ModelHandlerDescriptor type is expected, while maintaining the
        validation benefits of the bootstrap-specific model.

        Implementation Notes:
            Uses ``model_dump()`` without ``exclude_unset=True`` because:

            1. **Field parity**: This child class has NO extra fields beyond the
               parent. The only difference is ``handler_class`` type constraint
               (required ``str`` vs optional ``str | None``).

            2. **Type compatibility**: A ``str`` value from child is valid where
               parent expects ``str | None``.

            3. **Complete copy**: ``model_dump()`` ensures all fields are copied,
               including those set to their default values.

            Using ``exclude_unset=True`` would risk excluding fields that have
            defaults but were explicitly set to those defaults during construction.

            If future versions add child-specific fields not in parent, this
            method MUST be updated to use ``exclude={'new_field'}`` or refactored.

        Returns:
            ModelHandlerDescriptor instance with all fields copied.

        Example:
            >>> bootstrap_desc = ModelBootstrapHandlerDescriptor(
            ...     handler_id="proto.consul",
            ...     name="Consul Handler",
            ...     version="1.0.0",
            ...     handler_kind="effect",
            ...     input_model="omnibase_infra.models.types.JsonDict",
            ...     output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
            ...     handler_class="omnibase_infra.handlers.handler_consul.HandlerConsul",
            ... )
            >>> base_desc = bootstrap_desc.to_base_descriptor()
            >>> isinstance(base_desc, ModelHandlerDescriptor)
            True
        """
        return ModelHandlerDescriptor(**self.model_dump())


__all__ = ["ModelBootstrapHandlerDescriptor"]
