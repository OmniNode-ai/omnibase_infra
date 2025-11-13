"""Input validation and sanitization for security."""

import re
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator


class SecurityPatterns:
    """Security patterns for detecting malicious input with pre-compiled regex."""

    def __init__(self):
        """Initialize with pre-compiled regex patterns for performance."""
        # Pre-compile SQL injection patterns
        self.SQL_INJECTION_COMPILED = [
            re.compile(
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                re.IGNORECASE,
            ),
            re.compile(r"(--|\#|\/\*|\*\/)", re.IGNORECASE),
            re.compile(r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(
                r"(\b(OR|AND)\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
                re.IGNORECASE,
            ),
        ]

        # Pre-compile XSS patterns
        self.XSS_PATTERNS_COMPILED = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"onload=", re.IGNORECASE),
            re.compile(r"onerror=", re.IGNORECASE),
            re.compile(r"onclick=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
        ]

        # Pre-compile command injection patterns
        self.COMMAND_INJECTION_COMPILED = [
            re.compile(r"[;&|`]"),
            re.compile(r"\$\([^)]*\)"),
            re.compile(r"`[^`]*`"),
            re.compile(r"\|\s*(cat|ls|pwd|whoami|id|uname)", re.IGNORECASE),
        ]

        # Pre-compile path traversal patterns
        self.PATH_TRAVERSAL_COMPILED = [
            re.compile(r"\.\.\/"),
            re.compile(r"\.\.\\"),
            re.compile(r"%2e%2e%2f", re.IGNORECASE),
            re.compile(r"%2e%2e\\", re.IGNORECASE),
            re.compile(r"\/etc\/passwd", re.IGNORECASE),
            re.compile(r"\/proc\/", re.IGNORECASE),
        ]

        # Pre-compile LDAP injection patterns
        self.LDAP_INJECTION_COMPILED = [
            re.compile(r"\*\)"),
            re.compile(r"\(\|"),
            re.compile(r"\(&"),
            re.compile(r"\(\!"),
        ]

        # Pattern mapping for error reporting
        self.pattern_names = {
            "sql_injection": self.SQL_INJECTION_COMPILED,
            "xss": self.XSS_PATTERNS_COMPILED,
            "command_injection": self.COMMAND_INJECTION_COMPILED,
            "path_traversal": self.PATH_TRAVERSAL_COMPILED,
            "ldap_injection": self.LDAP_INJECTION_COMPILED,
        }


class InputSanitizer:
    """Input sanitization utilities."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input with enhanced null safety."""
        # Enhanced null safety checks
        if value is None:
            raise ValueError("Value cannot be None")

        if not isinstance(value, str):
            raise ValueError("Value must be a string")

        # Additional empty string validation
        if not value:
            return value  # Return empty string as-is after validation

        # Limit length
        if len(value) > max_length:
            raise ValueError(f"String too long. Maximum length: {max_length}")

        # Remove null bytes
        value = value.replace("\x00", "")

        # Remove control characters except newline and tab
        value = "".join(char for char in value if ord(char) >= 32 or char in "\n\t")

        # Strip leading/trailing whitespace
        value = value.strip()

        return value

    @staticmethod
    def sanitize_json(
        value: dict[str, Any],
        max_depth: int = 10,
        max_keys: int = 100,
    ) -> dict[str, Any]:
        """Sanitize JSON input with enhanced null safety."""
        # Enhanced null safety checks
        if value is None:
            raise ValueError("Value cannot be None")

        if not isinstance(value, dict):
            raise ValueError("Value must be a dictionary")

        # Additional empty dict validation
        if not value:
            return value  # Return empty dict as-is after validation

        def count_keys(obj, depth=0):
            if depth > max_depth:
                raise ValueError(f"JSON too deeply nested. Maximum depth: {max_depth}")

            if isinstance(obj, dict):
                count = len(obj)
                for v in obj.values():
                    if v is not None:  # Skip None values in counting
                        count += count_keys(v, depth + 1)
                return count
            elif isinstance(obj, list):
                count = 0
                for item in obj:
                    if item is not None:  # Skip None items in counting
                        count += count_keys(item, depth + 1)
                return count
            return 0

        total_keys = count_keys(value)
        if total_keys > max_keys:
            raise ValueError(f"Too many keys in JSON. Maximum: {max_keys}")

        return value

    @staticmethod
    def validate_uuid(value: str) -> UUID:
        """Validate and return UUID with enhanced null safety."""
        # Enhanced null safety checks
        if value is None:
            raise ValueError("UUID value cannot be None")

        if not isinstance(value, str):
            raise ValueError("UUID value must be a string")

        if not value.strip():
            raise ValueError("UUID value cannot be empty")

        try:
            return UUID(value)
        except ValueError:
            raise ValueError("Invalid UUID format")

    @staticmethod
    def validate_identifier(value: str, max_length: int = 100) -> str:
        """Validate identifier (alphanumeric + underscore + hyphen) with enhanced null safety."""
        # Enhanced null safety checks
        if value is None:
            raise ValueError("Identifier cannot be None")

        if not isinstance(value, str):
            raise ValueError("Identifier must be a string")

        if not value.strip():
            raise ValueError("Identifier cannot be empty")

        if len(value) > max_length:
            raise ValueError(f"Identifier too long. Maximum length: {max_length}")

        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValueError("Identifier contains invalid characters")

        return value

    @staticmethod
    def validate_sql_identifier(value: str, max_length: int = 63) -> str:
        """Validate SQL identifier (table name, column name, schema name) to prevent SQL injection.

        SQL identifiers in PostgreSQL must:
        - Start with a letter or underscore
        - Contain only letters, numbers, and underscores
        - Be 63 characters or less (PostgreSQL limit)
        - Not be a reserved keyword

        Args:
            value: The identifier to validate
            max_length: Maximum allowed length (default: 63 for PostgreSQL)

        Returns:
            The validated identifier

        Raises:
            ValueError: If identifier is invalid or potentially malicious
        """
        # Enhanced null safety checks
        if value is None:
            raise ValueError("SQL identifier cannot be None")

        if not isinstance(value, str):
            raise ValueError("SQL identifier must be a string")

        if not value.strip():
            raise ValueError("SQL identifier cannot be empty")

        if len(value) > max_length:
            raise ValueError(f"SQL identifier too long. Maximum length: {max_length}")

        # SQL identifier validation - must start with letter or underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
            raise ValueError(
                "SQL identifier contains invalid characters or invalid start character"
            )

        # Check for SQL keywords that could be used in injection
        sql_keywords = {
            # Core SQL commands
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "UNION",
            "JOIN",
            "WHERE",
            "HAVING",
            "GROUP",
            "ORDER",
            "LIMIT",
            "OFFSET",
            "OR",
            "AND",
            "NOT",
            "NULL",
            "TRUE",
            "FALSE",
            "INTO",
            "VALUES",
            "SET",
            "FROM",
            "TABLE",
            "VIEW",
            "INDEX",
            "CASCADE",
            "RESTRICT",
            "RESET",
            "SHOW",
            "DESCRIBE",
            "EXPLAIN",
            "ANALYZE",
            "VACUUM",
            "REINDEX",
            "CLUSTER",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "RETURNING",
            "WITH",
            "RECURSIVE",
            "ILIKE",
            "SIMILAR",
            "REGEX",
            "WINDOW",
            "OVER",
            "PARTITION",
            "RANGE",
            "ROWS",
            "PRECEDING",
            "FOLLOWING",
            "CURRENT",
            "ROW",
            "EXCLUDE",
            "INCLUDE",
            "TIES",
            # Data type keywords
            "INTEGER",
            "VARCHAR",
            "TEXT",
            "BOOLEAN",
            "DATE",
            "TIMESTAMP",
            "UUID",
            "JSON",
            "JSONB",
            "ARRAY",
            "SERIAL",
            "BIGINT",
            "SMALLINT",
            "DECIMAL",
            "NUMERIC",
            "REAL",
            "DOUBLE",
            "PRECISION",
            "INTERVAL",
            # Constraint keywords
            "PRIMARY",
            "FOREIGN",
            "KEY",
            "REFERENCES",
            "UNIQUE",
            "CHECK",
            "CONSTRAINT",
            "DEFAULT",
            "AUTO_INCREMENT",
        }

        if value.upper() in sql_keywords:
            raise ValueError(f"SQL identifier cannot be a reserved keyword: {value}")

        return value


class SecurityValidator:
    """Security-focused input validator."""

    def __init__(self):
        """Initialize security validator."""
        self.patterns = SecurityPatterns()

    def check_for_malicious_patterns(
        self,
        value: str,
        input_type: str = "general",
    ) -> list[str]:
        """Check for malicious patterns in input using pre-compiled regex patterns."""
        # Enhanced null safety checks
        if value is None:
            return []

        if not isinstance(value, str):
            return []

        # Additional safety check for empty strings
        if not value.strip():
            return []

        detected_patterns = []

        # Check SQL injection using pre-compiled patterns
        # Skip SQL keyword check if this looks like a parameterized query (contains $1, $2, etc.)
        has_parameterized_placeholders = re.search(r"\$\d+", value)

        for pattern in self.patterns.SQL_INJECTION_COMPILED:
            # If this is the SQL keyword pattern and we have parameterized placeholders, skip it
            if (
                pattern.pattern
                == r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)"
                and has_parameterized_placeholders
            ):
                continue
            if pattern.search(value):
                detected_patterns.append(f"sql_injection:{pattern.pattern}")

        # Check XSS using pre-compiled patterns
        for pattern in self.patterns.XSS_PATTERNS_COMPILED:
            if pattern.search(value):
                detected_patterns.append(f"xss:{pattern.pattern}")

        # Check command injection using pre-compiled patterns
        for pattern in self.patterns.COMMAND_INJECTION_COMPILED:
            if pattern.search(value):
                detected_patterns.append(f"command_injection:{pattern.pattern}")

        # Check path traversal using pre-compiled patterns
        for pattern in self.patterns.PATH_TRAVERSAL_COMPILED:
            if pattern.search(value):
                detected_patterns.append(f"path_traversal:{pattern.pattern}")

        # Check LDAP injection using pre-compiled patterns
        for pattern in self.patterns.LDAP_INJECTION_COMPILED:
            if pattern.search(value):
                detected_patterns.append(f"ldap_injection:{pattern.pattern}")

        return detected_patterns

    def validate_input_safety(self, value: Any, input_type: str = "general") -> bool:
        """Validate that input is safe from known attack patterns."""
        if isinstance(value, str):
            malicious_patterns = self.check_for_malicious_patterns(value, input_type)
            if malicious_patterns:
                # Log the specific patterns for security monitoring without exposing them to client
                import logging

                security_logger = logging.getLogger("omninode_bridge.security")
                security_logger.warning(
                    f"Malicious input detected in {input_type}: patterns={malicious_patterns[:3]}..."  # Only log first 3 patterns
                )

                raise HTTPException(
                    status_code=400,
                    detail="Invalid input detected. Please check your request and try again.",
                )
        elif isinstance(value, dict):
            # Recursively validate dict values
            for key, val in value.items():
                self.validate_input_safety(val, f"{input_type}.{key}")
        elif isinstance(value, list):
            # Validate list items
            for i, item in enumerate(value):
                self.validate_input_safety(item, f"{input_type}[{i}]")

        return True


# Enhanced Pydantic models with security validation


class SecureWorkflowDefinition(BaseModel):
    """Secure workflow definition with comprehensive validation."""

    workflow_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=2000)
    tasks: list[dict[str, Any]] = Field(..., min_length=1, max_length=50)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    global_timeout_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
    )  # 1 second to 24 hours
    max_parallel_tasks: int = Field(default=5, ge=1, le=20)
    metadata: dict[str, Any] = Field(default_factory=dict)

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
        validator = SecurityValidator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 2000)

        # Check for malicious patterns
        validator.validate_input_safety(v, "text")

        return v

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v):
        """Validate tasks array."""
        if not isinstance(v, list):
            raise ValueError("Tasks must be a list")

        sanitizer = InputSanitizer()
        validator = SecurityValidator()

        for i, task in enumerate(v):
            if not isinstance(task, dict):
                raise ValueError(f"Task {i} must be a dictionary")

            # Validate task structure
            required_fields = ["task_id", "task_type"]
            for field in required_fields:
                if field not in task:
                    raise ValueError(f"Task {i} missing required field: {field}")

            # Sanitize task_id
            task_id = task.get("task_id", "")
            task["task_id"] = sanitizer.validate_identifier(task_id, 100)

            # Validate entire task for safety
            validator.validate_input_safety(task, f"task[{i}]")

        return v

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, v):
        """Validate dependencies."""
        if not isinstance(v, dict):
            raise ValueError("Dependencies must be a dictionary")

        sanitizer = InputSanitizer()

        for task_id, deps in v.items():
            # Validate task_id
            sanitizer.validate_identifier(task_id, 100)

            # Validate dependencies list
            if not isinstance(deps, list):
                raise ValueError(f"Dependencies for {task_id} must be a list")

            for dep in deps:
                if not isinstance(dep, str):
                    raise ValueError(f"Dependency must be a string: {dep}")
                sanitizer.validate_identifier(dep, 100)

        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata."""
        sanitizer = InputSanitizer()
        validator = SecurityValidator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=5, max_keys=50)

        # Check for malicious patterns
        validator.validate_input_safety(v, "metadata")

        return v


class SecureHookPayload(BaseModel):
    """Secure hook payload with validation."""

    action: str = Field(..., min_length=1, max_length=100)
    resource: str = Field(..., min_length=1, max_length=100)
    resource_id: str = Field(..., min_length=1, max_length=255)
    data: dict[str, Any] = Field(default_factory=dict)
    previous_state: dict[str, Any] | None = None
    current_state: dict[str, Any] | None = None

    @field_validator("action", "resource", "resource_id")
    @classmethod
    def validate_identifiers(cls, v):
        """Validate identifier fields."""
        sanitizer = InputSanitizer()
        validator = SecurityValidator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 255)

        # Check for malicious patterns
        validator.validate_input_safety(v, "identifier")

        return v

    @field_validator("data", "previous_state", "current_state")
    @classmethod
    def validate_json_fields(cls, v):
        """Validate JSON fields."""
        if v is None:
            return v

        sanitizer = InputSanitizer()
        validator = SecurityValidator()

        # Sanitize JSON structure
        v = sanitizer.sanitize_json(v, max_depth=10, max_keys=200)

        # Check for malicious patterns
        validator.validate_input_safety(v, "json_data")

        return v


class SecureTaskRequest(BaseModel):
    """Secure task request with validation."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    task_type: str = Field(..., min_length=1, max_length=100)
    context_size: int | None = Field(None, ge=1, le=1000000)
    complexity: str = Field(default="moderate", min_length=1, max_length=50)
    max_latency_ms: float | None = Field(None, ge=1, le=300000)  # 5 minutes max
    preferred_model: str | None = Field(None, min_length=1, max_length=200)
    track_metrics: bool = Field(default=True)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        """Validate prompt field."""
        sanitizer = InputSanitizer()
        validator = SecurityValidator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 10000)

        # Check for malicious patterns (but be more lenient for prompts)
        try:
            validator.validate_input_safety(v, "prompt")
        except HTTPException:
            # For prompts, we might be more lenient with certain patterns
            # but still log suspicious content
            pass

        return v

    @field_validator("task_type", "complexity", "preferred_model")
    @classmethod
    def validate_string_fields(cls, v):
        """Validate string fields."""
        if v is None:
            return v

        sanitizer = InputSanitizer()
        validator = SecurityValidator()

        # Sanitize
        v = sanitizer.sanitize_string(v, 200)

        # Check for malicious patterns
        validator.validate_input_safety(v, "identifier")

        return v


# Global validator instance
security_validator = SecurityValidator()


def get_security_validator() -> SecurityValidator:
    """Get the global security validator instance."""
    return security_validator
