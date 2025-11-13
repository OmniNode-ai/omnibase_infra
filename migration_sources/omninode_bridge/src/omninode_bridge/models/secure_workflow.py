"""Secure workflow models with comprehensive validation."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from ..security.validation import InputSanitizer, get_security_validator


class SecureTaskType(str, Enum):
    """Validated task types for workflows."""

    AI_TASK = "ai_task"
    AI_GENERATION = "ai_generation"
    CODE_ANALYSIS = "code_analysis"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    FILE_OPERATION = "file_operation"
    WEBHOOK_TRIGGER = "webhook_trigger"
    CONDITIONAL = "conditional"
    PARALLEL_GROUP = "parallel_group"
    WAIT_DELAY = "wait_delay"
    HUMAN_APPROVAL = "human_approval"


class SecureComplexityLevel(str, Enum):
    """Validated complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class SecureTaskConfig(BaseModel):
    """Secure task configuration with validation."""

    complexity: SecureComplexityLevel = SecureComplexityLevel.MODERATE
    ai_task_type: str | None = Field(None, min_length=1, max_length=100)
    timeout: int = Field(default=300, ge=1, le=3600)  # 1 second to 1 hour
    retry_limit: int = Field(default=2, ge=0, le=10)
    max_latency_ms: float | None = Field(None, ge=1, le=300000)  # 5 minutes max

    # Additional config fields with size limits
    additional_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ai_task_type")
    @classmethod
    def validate_ai_task_type(cls, v):
        """Validate AI task type."""
        if v is None:
            return v

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 100)

        # Check for malicious patterns
        validator.validate_input_safety(v, "ai_task_type")

        return v

    @field_validator("additional_config")
    @classmethod
    def validate_additional_config(cls, v):
        """Validate additional configuration."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=5, max_keys=50)

        # Check for malicious patterns
        validator.validate_input_safety(v, "additional_config")

        return v


class SecureValidationContract(BaseModel):
    """Secure validation contract for task outputs."""

    type: str = Field(..., min_length=1, max_length=50)
    fields: list[str] = Field(default_factory=list, max_length=20)
    min_length: int | None = Field(None, ge=0, le=100000)
    max_length: int | None = Field(None, ge=1, le=1000000)
    required_keywords: list[str] = Field(default_factory=list, max_length=10)
    forbidden_patterns: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        """Validate validation type."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 50)

        # Check for malicious patterns
        validator.validate_input_safety(v, "validation_type")

        return v

    @field_validator("fields", "required_keywords", "forbidden_patterns")
    @classmethod
    def validate_string_lists(cls, v):
        """Validate lists of strings."""
        if not isinstance(v, list):
            raise ValueError("Must be a list")

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        validated_list = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("All items must be strings")

            # Sanitize each item
            item = sanitizer.sanitize_string(item, 200)

            # Check for malicious patterns
            validator.validate_input_safety(item, "string_list_item")

            validated_list.append(item)

        return validated_list


class SecureDefinitionOfDone(BaseModel):
    """Secure definition of done criteria."""

    min_content_length: int = Field(default=10, ge=0, le=100000)
    max_content_length: int = Field(default=10000, ge=1, le=1000000)
    required_keywords: list[str] = Field(default_factory=list, max_length=10)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    completion_criteria: dict[str, Any] = Field(default_factory=dict)

    @field_validator("required_keywords")
    @classmethod
    def validate_required_keywords(cls, v):
        """Validate required keywords."""
        if not isinstance(v, list):
            raise ValueError("Must be a list")

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        validated_list = []
        for keyword in v:
            if not isinstance(keyword, str):
                raise ValueError("All keywords must be strings")

            # Sanitize keyword
            keyword = sanitizer.sanitize_string(keyword, 100)

            # Check for malicious patterns
            validator.validate_input_safety(keyword, "keyword")

            validated_list.append(keyword)

        return validated_list

    @field_validator("completion_criteria")
    @classmethod
    def validate_completion_criteria(cls, v):
        """Validate completion criteria."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=3, max_keys=20)

        # Check for malicious patterns
        validator.validate_input_safety(v, "completion_criteria")

        return v

    @field_validator("max_content_length")
    @classmethod
    def validate_max_length(cls, v, values):
        """Ensure max length is greater than min length."""
        min_length = values.get("min_content_length", 0)
        if v <= min_length:
            raise ValueError(
                "max_content_length must be greater than min_content_length",
            )
        return v


class SecureModelFallbackConfig(BaseModel):
    """Secure model fallback configuration."""

    strategy: str = Field(..., min_length=1, max_length=50)
    fallback_models: list[str] = Field(default_factory=list, max_length=10)
    max_attempts: int = Field(default=3, ge=1, le=10)
    timeout_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        """Validate fallback strategy."""
        allowed_strategies = [
            "next_tier",
            "same_tier",
            "fastest",
            "most_reliable",
            "highest_quality",
            "lowest_cost",
            "custom",
        ]

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 50)

        # Check for malicious patterns
        validator.validate_input_safety(v, "strategy")

        # Validate against allowed strategies
        if v not in allowed_strategies:
            raise ValueError(
                f"Strategy must be one of: {', '.join(allowed_strategies)}",
            )

        return v

    @field_validator("fallback_models")
    @classmethod
    def validate_fallback_models(cls, v):
        """Validate fallback models list."""
        if not isinstance(v, list):
            raise ValueError("Must be a list")

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        validated_models = []
        for model in v:
            if not isinstance(model, str):
                raise ValueError("All models must be strings")

            # Sanitize model name
            model = sanitizer.sanitize_string(model, 200)

            # Check for malicious patterns
            validator.validate_input_safety(model, "model_name")

            validated_models.append(model)

        return validated_models


class SecureWorkflowTask(BaseModel):
    """Secure workflow task with comprehensive validation."""

    task_id: str = Field(..., min_length=1, max_length=100)
    task_type: SecureTaskType = SecureTaskType.AI_TASK
    prompt: str = Field(..., min_length=1, max_length=10000)
    config: SecureTaskConfig = Field(default_factory=SecureTaskConfig)

    # Optional validation and completion criteria
    validation_contract: SecureValidationContract | None = None
    definition_of_done: SecureDefinitionOfDone | None = None
    model_fallback_config: SecureModelFallbackConfig | None = None

    # Dependencies and metadata
    dependencies: list[str] = Field(default_factory=list, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v):
        """Validate task ID."""
        sanitizer = InputSanitizer()
        return sanitizer.validate_identifier(v, 100)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        """Validate task prompt."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 10000)

        # For prompts, we're more lenient but still check for obvious attacks
        try:
            validator.validate_input_safety(v, "prompt")
        except (ValueError, TypeError, AttributeError):
            # Log suspicious content but don't block (prompts may contain code examples, etc.)
            pass

        return v

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, v):
        """Validate task dependencies."""
        if not isinstance(v, list):
            raise ValueError("Dependencies must be a list")

        sanitizer = InputSanitizer()

        validated_deps = []
        for dep in v:
            if not isinstance(dep, str):
                raise ValueError("All dependencies must be strings")

            # Validate dependency as identifier
            dep = sanitizer.validate_identifier(dep, 100)
            validated_deps.append(dep)

        return validated_deps

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate task metadata."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=5, max_keys=30)

        # Check for malicious patterns
        validator.validate_input_safety(v, "task_metadata")

        return v


class SecureWorkflowDefinition(BaseModel):
    """Secure workflow definition with comprehensive validation."""

    workflow_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=100,
    )
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=2000)
    tasks: list[SecureWorkflowTask] = Field(..., min_length=1, max_length=50)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)

    # Execution configuration
    global_timeout_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
    )  # 1 second to 24 hours
    max_parallel_tasks: int = Field(default=5, ge=1, le=20)
    priority: int = Field(default=1, ge=1, le=10)

    # Metadata and tags
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v):
        """Validate workflow ID."""
        sanitizer = InputSanitizer()
        return sanitizer.validate_identifier(v, 100)

    @field_validator("name", "description")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 2000)

        # Check for malicious patterns
        validator.validate_input_safety(v, "text")

        return v

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v):
        """Validate tasks and check for circular dependencies."""
        if not isinstance(v, list):
            raise ValueError("Tasks must be a list")

        # Check for duplicate task IDs
        task_ids = [task.task_id for task in v]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found")

        return v

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, v, values):
        """Validate dependencies structure and check for circular references."""
        if not isinstance(v, dict):
            raise ValueError("Dependencies must be a dictionary")

        sanitizer = InputSanitizer()

        # Get task IDs from tasks if available
        task_ids = set()
        if values.get("tasks"):
            task_ids = {task.task_id for task in values["tasks"]}

        validated_deps = {}
        for task_id, deps in v.items():
            # Validate task_id
            task_id = sanitizer.validate_identifier(task_id, 100)

            # Validate dependencies list
            if not isinstance(deps, list):
                raise ValueError(f"Dependencies for {task_id} must be a list")

            validated_task_deps = []
            for dep in deps:
                if not isinstance(dep, str):
                    raise ValueError(f"Dependency must be a string: {dep}")

                dep = sanitizer.validate_identifier(dep, 100)

                # Check if dependency task exists
                if task_ids and dep not in task_ids:
                    raise ValueError(
                        f"Dependency {dep} for task {task_id} not found in tasks",
                    )

                validated_task_deps.append(dep)

            validated_deps[task_id] = validated_task_deps

        # Check for circular dependencies
        def has_circular_dependency(task_id, visited, path):
            if task_id in path:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            path.add(task_id)

            for dep in validated_deps.get(task_id, []):
                if has_circular_dependency(dep, visited, path):
                    return True

            path.remove(task_id)
            return False

        visited = set()
        for task_id in validated_deps:
            if has_circular_dependency(task_id, visited, set()):
                raise ValueError(
                    f"Circular dependency detected involving task {task_id}",
                )

        return validated_deps

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate workflow metadata."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=5, max_keys=50)

        # Check for malicious patterns
        validator.validate_input_safety(v, "workflow_metadata")

        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate workflow tags."""
        if not isinstance(v, list):
            raise ValueError("Tags must be a list")

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        validated_tags = []
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("All tags must be strings")

            # Sanitize tag
            tag = sanitizer.sanitize_string(tag, 50)

            # Check for malicious patterns
            validator.validate_input_safety(tag, "tag")

            validated_tags.append(tag)

        return validated_tags


class SecureWorkflowExecutionRequest(BaseModel):
    """Secure workflow execution request."""

    workflow_definition: SecureWorkflowDefinition
    input_data: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)
    scheduled_at: datetime | None = None
    max_execution_time: int | None = Field(None, ge=1, le=86400)  # Max 24 hours

    @field_validator("input_data")
    @classmethod
    def validate_input_data(cls, v):
        """Validate input data."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=10, max_keys=200)

        # Check for malicious patterns
        validator.validate_input_safety(v, "input_data")

        return v


class SecureWorkflowExecutionEvent(BaseModel):
    """Secure workflow execution event for Kafka."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_source: str = Field(default="workflow_cli", min_length=1, max_length=100)
    workflow_definition: SecureWorkflowDefinition
    input_data: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)
    reply_to: str | None = Field(None, min_length=1, max_length=200)

    @field_validator("request_source", "reply_to")
    @classmethod
    def validate_string_fields(cls, v):
        """Validate string fields."""
        if v is None:
            return v

        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 200)

        # Check for malicious patterns
        validator.validate_input_safety(v, "identifier")

        return v

    @field_validator("input_data")
    @classmethod
    def validate_input_data(cls, v):
        """Validate input data."""
        sanitizer = InputSanitizer()
        validator = get_security_validator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=10, max_keys=200)

        # Check for malicious patterns
        validator.validate_input_safety(v, "input_data")

        return v
