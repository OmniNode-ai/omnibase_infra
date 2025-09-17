"""PostgreSQL query parameter model."""


from pydantic import BaseModel, Field


class ModelPostgresQueryParameter(BaseModel):
    """Strongly typed PostgreSQL query parameter."""

    value_string: str | None = Field(default=None, description="String parameter value")
    value_integer: int | None = Field(default=None, description="Integer parameter value")
    value_float: float | None = Field(default=None, description="Float parameter value")
    value_boolean: bool | None = Field(default=None, description="Boolean parameter value")
    value_null: bool | None = Field(default=None, description="Null parameter value flag")
    parameter_type: str = Field(description="Parameter type (string, integer, float, boolean, null)")
    parameter_index: int = Field(description="Parameter position in query (0-based)")

    def get_value(self) -> object | None:
        """Get the actual parameter value based on type."""
        if self.parameter_type == "string":
            return self.value_string
        if self.parameter_type == "integer":
            return self.value_integer
        if self.parameter_type == "float":
            return self.value_float
        if self.parameter_type == "boolean":
            return self.value_boolean
        if self.parameter_type == "null":
            return None
        return None

    @classmethod
    def from_value(cls, value: object, index: int) -> "ModelPostgresQueryParameter":
        """Create parameter from raw value using protocol-based duck typing (ONEX compliance)."""
        if value is None:
            return cls(parameter_type="null", parameter_index=index, value_null=True)
        if hasattr(value, "encode") and hasattr(value, "strip") and hasattr(value, "split"):  # String-like protocol
            return cls(parameter_type="string", parameter_index=index, value_string=str(value))
        if hasattr(value, "__add__") and hasattr(value, "__mod__") and not hasattr(value, "split") and not hasattr(value, "__truediv__"):  # Integer-like protocol
            return cls(parameter_type="integer", parameter_index=index, value_integer=int(value))
        if hasattr(value, "__add__") and hasattr(value, "__truediv__") and hasattr(value, "is_integer"):  # Float-like protocol
            return cls(parameter_type="float", parameter_index=index, value_float=float(value))
        if hasattr(value, "__bool__") and hasattr(value, "__invert__") and not hasattr(value, "__add__"):  # Boolean-like protocol
            return cls(parameter_type="boolean", parameter_index=index, value_boolean=bool(value))
        # Convert unknown types to string (fallback pattern)
        return cls(parameter_type="string", parameter_index=index, value_string=str(value))


class ModelPostgresQueryParameters(BaseModel):
    """Collection of PostgreSQL query parameters."""

    parameters: list[ModelPostgresQueryParameter] = Field(
        default_factory=list,
        description="List of strongly typed query parameters",
    )
    parameter_count: int = Field(default=0, description="Total number of parameters")

    def add_parameter(self, value: object) -> None:
        """Add a parameter from raw value."""
        param = ModelPostgresQueryParameter.from_value(value, len(self.parameters))
        self.parameters.append(param)
        self.parameter_count = len(self.parameters)

    def get_values(self) -> list[object]:
        """Get list of parameter values for SQL execution."""
        return [param.get_value() for param in self.parameters]

    @classmethod
    def from_list(cls, values: list[object]) -> "ModelPostgresQueryParameters":
        """Create parameters from list of values."""
        instance = cls()
        for value in values:
            instance.add_parameter(value)
        return instance
