"""
Entity Registry for Type-Safe Database Operations.

Maps entity types to Pydantic models and database tables, providing
strong typing throughout the database adapter pipeline.

ONEX v2.0 Compliance:
- Contract-driven type resolution
- Strong typing with Pydantic v2
- Entity-to-table mapping
- Comprehensive validation

See Also:
    /docs/STRONGLY_TYPED_DATABASE_PATTERN.md for pattern documentation
"""

from typing import Any
from typing import Any as TypingAny
from typing import ClassVar, Union

from pydantic import BaseModel

from .entities.model_bridge_state import ModelBridgeState
from .entities.model_fsm_transition import ModelFSMTransition
from .entities.model_metadata_stamp import ModelMetadataStamp
from .entities.model_node_health_metrics import ModelNodeHealthMetrics
from .entities.model_node_heartbeat import ModelNodeHeartbeat
from .entities.model_workflow_execution import ModelWorkflowExecution
from .entities.model_workflow_step import ModelWorkflowStep
from .enum_entity_type import EnumEntityType


class EntityRegistry:
    """
    Registry mapping entity types to Pydantic models and database tables.

    Provides type-safe entity resolution and validation throughout the system.
    All database operations use this registry to ensure type safety and
    proper table name resolution.

    Features:
        - Entity type to Pydantic model mapping
        - Entity type to database table mapping
        - Entity validation with Pydantic
        - Entity serialization for database insertion
        - Extensible for new entity types

    Usage Example:
        # Get model for validation
        >>> model = EntityRegistry.get_model(EnumEntityType.WORKFLOW_EXECUTION)
        >>> entity = model(**entity_data)  # Validates data

        # Get table name for query building
        >>> table = EntityRegistry.get_table_name(EnumEntityType.WORKFLOW_EXECUTION)
        >>> query = f"SELECT * FROM {table}"

        # Serialize entity for database
        >>> entity_dict = EntityRegistry.serialize_entity(entity)
    """

    # Entity type -> Pydantic model mapping
    _ENTITY_MODELS: ClassVar[dict[EnumEntityType, type[BaseModel]]] = {
        EnumEntityType.WORKFLOW_EXECUTION: ModelWorkflowExecution,
        EnumEntityType.WORKFLOW_STEP: ModelWorkflowStep,
        EnumEntityType.METADATA_STAMP: ModelMetadataStamp,
        EnumEntityType.FSM_TRANSITION: ModelFSMTransition,
        EnumEntityType.BRIDGE_STATE: ModelBridgeState,
        EnumEntityType.NODE_HEARTBEAT: ModelNodeHeartbeat,
        EnumEntityType.NODE_HEALTH_METRICS: ModelNodeHealthMetrics,
    }

    # Entity type -> database table mapping
    _ENTITY_TABLES: ClassVar[dict[EnumEntityType, str]] = {
        EnumEntityType.WORKFLOW_EXECUTION: "workflow_executions",
        EnumEntityType.WORKFLOW_STEP: "workflow_steps",
        EnumEntityType.METADATA_STAMP: "metadata_stamps",
        EnumEntityType.FSM_TRANSITION: "fsm_transitions",
        EnumEntityType.BRIDGE_STATE: "bridge_states",
        EnumEntityType.NODE_HEARTBEAT: "node_registrations",
        EnumEntityType.NODE_HEALTH_METRICS: "node_registrations",  # Virtual entity, same table
    }

    @classmethod
    def get_model(cls, entity_type: EnumEntityType) -> type[BaseModel]:
        """
        Get Pydantic model for entity type.

        Args:
            entity_type: Entity type enum value

        Returns:
            Pydantic model class for the entity type

        Raises:
            ValueError: If no model registered for entity type

        Example:
            >>> model = EntityRegistry.get_model(EnumEntityType.WORKFLOW_EXECUTION)
            >>> entity = model(workflow_id="wf-123", status="processing", ...)
        """
        model = cls._ENTITY_MODELS.get(entity_type)
        if not model:
            raise ValueError(f"No model registered for entity type: {entity_type}")
        return model

    @classmethod
    def get_table_name(cls, entity_type: EnumEntityType) -> str:
        """
        Get database table name for entity type.

        Args:
            entity_type: Entity type enum value

        Returns:
            Database table name as string

        Raises:
            ValueError: If no table registered for entity type

        Example:
            >>> table = EntityRegistry.get_table_name(EnumEntityType.WORKFLOW_EXECUTION)
            >>> print(table)
            'workflow_executions'
        """
        table = cls._ENTITY_TABLES.get(entity_type)
        if not table:
            raise ValueError(f"No table registered for entity type: {entity_type}")
        return table

    @classmethod
    def validate_entity(
        cls, entity_type: EnumEntityType, entity_data: dict[str, Any]
    ) -> BaseModel:
        """
        Validate entity data against Pydantic model.

        Args:
            entity_type: Entity type to validate
            entity_data: Raw entity data dictionary

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If entity data is invalid
            ValueError: If entity type not registered

        Example:
            >>> entity = EntityRegistry.validate_entity(
            ...     EnumEntityType.WORKFLOW_EXECUTION,
            ...     {"workflow_id": "wf-123", "status": "processing", ...}
            ... )
            >>> print(entity.workflow_id)
            'wf-123'
        """
        model = cls.get_model(entity_type)
        return model(**entity_data)

    @classmethod
    def serialize_entity(cls, entity: BaseModel) -> dict[str, Any]:
        """
        Serialize Pydantic model to dictionary for database insertion.

        Excludes None values and auto-managed fields (id, created_at, updated_at)
        to allow database to handle defaults and auto-generation.

        Converts only JSONB fields to JSON strings using schema-aware field metadata.
        Other field types remain as native Python types for proper asyncpg binding.

        Args:
            entity: Pydantic model instance

        Returns:
            Dictionary ready for database insertion with JSONB fields serialized

        Example:
            >>> entity = ModelWorkflowExecution(workflow_id="wf-123", ...)
            >>> entity_dict = EntityRegistry.serialize_entity(entity)
            >>> # entity_dict excludes None, id, created_at, updated_at
            >>> # Only JSONB fields are converted to JSON strings
        """
        import json

        entity_dict = entity.model_dump(
            exclude_none=True, exclude={"id", "created_at", "updated_at"}
        )

        # Schema-aware JSONB serialization using field metadata
        for field_name, field_info in entity.model_fields.items():
            if field_name in entity_dict and cls._is_jsonb_field(field_info):
                value = entity_dict[field_name]
                if isinstance(value, dict):
                    entity_dict[field_name] = json.dumps(value)

        return entity_dict

    @classmethod
    def _is_jsonb_field(cls, field_info) -> bool:
        """
        Determine if a field should be treated as JSONB based on field metadata.

        Checks explicit json_schema_extra metadata first for reliable detection,
        then falls back to complex type checking for backwards compatibility.

        Args:
            field_info: Pydantic field info object

        Returns:
            True if field should be serialized as JSONB, False otherwise
        """
        # Primary check: Explicit json_schema_extra metadata
        if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            if isinstance(field_info.json_schema_extra, dict):
                db_type = field_info.json_schema_extra.get("db_type")
                if db_type == "jsonb":
                    return True

        # Fallback: Check if field has dict[str, Any] type annotation
        field_type = field_info.annotation

        # Handle Optional types
        if hasattr(field_type, "__origin__"):
            origin = field_type.__origin__
            if origin is Union:
                # Check if any of the Union types is dict[str, Any]
                for arg in field_type.__args__:
                    if cls._is_dict_str_any_type(arg):
                        return True
                return False  # Not a Union containing dict[str, Any]

        # Direct dict[str, Any] check
        return cls._is_dict_str_any_type(field_type)

    @classmethod
    def _is_dict_str_any_type(cls, field_type) -> bool:
        """
        Check if type annotation is dict[str, Any] or compatible.

        Args:
            field_type: Type annotation to check

        Returns:
            True if type is dict[str, Any] or compatible
        """
        try:
            # Check if it's a dict type
            if not (
                hasattr(field_type, "__origin__") and field_type.__origin__ is dict
            ):
                return False

            # Check if args are [str, Any] or compatible
            type_args = getattr(field_type, "__args__", ())
            if len(type_args) != 2:
                return False

            key_type, value_type = type_args

            # Key type should be str (or string-like)
            if key_type not in (str,):
                return False

            # Value type should be Any or compatible
            # Handle both typing.Any and the Any object
            if value_type is Any or value_type is TypingAny:
                return True

            # Handle typing.Any by checking its name
            if hasattr(value_type, "__name__") and value_type.__name__ == "Any":
                return True

            # Handle object type (which is compatible with Any)
            return value_type is object

        except (AttributeError, TypeError):
            return False

    @classmethod
    def deserialize_row(
        cls, entity_type: EnumEntityType, row: dict[str, Any]
    ) -> BaseModel:
        """
        Deserialize database row to Pydantic model.

        Automatically converts JSONB fields from JSON strings to dicts
        for proper Pydantic validation.

        Args:
            entity_type: Entity type to deserialize to
            row: Database row as dictionary

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If row data doesn't match model schema
            ValueError: If entity type not registered

        Example:
            >>> # After database query
            >>> row = {"id": 1, "workflow_id": "wf-123", ...}
            >>> entity = EntityRegistry.deserialize_row(
            ...     EnumEntityType.WORKFLOW_EXECUTION,
            ...     row
            ... )
            >>> print(type(entity))
            <class 'ModelWorkflowExecution'>
        """
        import json

        model = cls.get_model(entity_type)

        # Convert JSONB fields from JSON strings to dicts
        row_copy = dict(row)
        for field_name, field_info in model.model_fields.items():
            if field_name in row_copy and cls._is_jsonb_field(field_info):
                value = row_copy[field_name]
                if isinstance(value, str):
                    try:
                        row_copy[field_name] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # If JSON parsing fails, leave as-is and let Pydantic handle validation error
                        pass

        return model(**row_copy)

    @classmethod
    def get_primary_key_field(cls, entity_type: EnumEntityType) -> str:
        """
        Get primary key field name for entity type.

        All entities use 'id' as the primary key field by convention.

        Args:
            entity_type: Entity type

        Returns:
            Primary key field name (always 'id')

        Example:
            >>> pk_field = EntityRegistry.get_primary_key_field(
            ...     EnumEntityType.WORKFLOW_EXECUTION
            ... )
            >>> print(pk_field)
            'id'
        """
        # All entities use 'id' as primary key by convention
        return "id"

    @classmethod
    def get_entity_fields(cls, entity_type: EnumEntityType) -> list[str]:
        """
        Get list of field names for entity type.

        Useful for query building and validation.

        Args:
            entity_type: Entity type

        Returns:
            List of field names from the Pydantic model

        Example:
            >>> fields = EntityRegistry.get_entity_fields(
            ...     EnumEntityType.WORKFLOW_EXECUTION
            ... )
            >>> print(fields)
            ['id', 'workflow_id', 'correlation_id', 'status', ...]
        """
        model = cls.get_model(entity_type)
        return list(model.model_fields.keys())

    @classmethod
    def is_registered(cls, entity_type: EnumEntityType) -> bool:
        """
        Check if entity type is registered.

        Args:
            entity_type: Entity type to check

        Returns:
            True if entity type has registered model and table

        Example:
            >>> if EntityRegistry.is_registered(EnumEntityType.WORKFLOW_EXECUTION):
            ...     # Proceed with entity operations
            ...     pass
        """
        return entity_type in cls._ENTITY_MODELS and entity_type in cls._ENTITY_TABLES
