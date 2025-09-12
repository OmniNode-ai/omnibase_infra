"""PostgreSQL query parameter model."""

from typing import Optional, List
from pydantic import BaseModel, Field


class ModelPostgresQueryParameter(BaseModel):
    """Strongly typed PostgreSQL query parameter."""
    
    value_string: Optional[str] = Field(default=None, description="String parameter value")
    value_integer: Optional[int] = Field(default=None, description="Integer parameter value")
    value_float: Optional[float] = Field(default=None, description="Float parameter value")
    value_boolean: Optional[bool] = Field(default=None, description="Boolean parameter value")
    value_null: Optional[bool] = Field(default=None, description="Null parameter value flag")
    parameter_type: str = Field(description="Parameter type (string, integer, float, boolean, null)")
    parameter_index: int = Field(description="Parameter position in query (0-based)")
    
    def get_value(self) -> Optional[object]:
        """Get the actual parameter value based on type."""
        if self.parameter_type == "string":
            return self.value_string
        elif self.parameter_type == "integer":
            return self.value_integer
        elif self.parameter_type == "float":
            return self.value_float
        elif self.parameter_type == "boolean":
            return self.value_boolean
        elif self.parameter_type == "null":
            return None
        return None
    
    @classmethod
    def from_value(cls, value: object, index: int) -> "ModelPostgresQueryParameter":
        """Create parameter from raw value."""
        if value is None:
            return cls(parameter_type="null", parameter_index=index, value_null=True)
        elif isinstance(value, str):
            return cls(parameter_type="string", parameter_index=index, value_string=value)
        elif isinstance(value, int):
            return cls(parameter_type="integer", parameter_index=index, value_integer=value)
        elif isinstance(value, float):
            return cls(parameter_type="float", parameter_index=index, value_float=value)
        elif isinstance(value, bool):
            return cls(parameter_type="boolean", parameter_index=index, value_boolean=value)
        else:
            # Convert unknown types to string
            return cls(parameter_type="string", parameter_index=index, value_string=str(value))


class ModelPostgresQueryParameters(BaseModel):
    """Collection of PostgreSQL query parameters."""
    
    parameters: List[ModelPostgresQueryParameter] = Field(
        default_factory=list,
        description="List of strongly typed query parameters"
    )
    parameter_count: int = Field(default=0, description="Total number of parameters")
    
    def add_parameter(self, value: object) -> None:
        """Add a parameter from raw value."""
        param = ModelPostgresQueryParameter.from_value(value, len(self.parameters))
        self.parameters.append(param)
        self.parameter_count = len(self.parameters)
    
    def get_values(self) -> List[object]:
        """Get list of parameter values for SQL execution."""
        return [param.get_value() for param in self.parameters]
    
    @classmethod
    def from_list(cls, values: List[object]) -> "ModelPostgresQueryParameters":
        """Create parameters from list of values."""
        instance = cls()
        for value in values:
            instance.add_parameter(value)
        return instance