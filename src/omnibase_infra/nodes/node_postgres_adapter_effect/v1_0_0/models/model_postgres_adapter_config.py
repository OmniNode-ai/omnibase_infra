"""PostgreSQL Adapter Configuration Model."""

import os
from typing import Optional
from pydantic import BaseModel, Field


class ModelPostgresAdapterConfig(BaseModel):
    """
    Configuration model for PostgreSQL adapter validation limits and security settings.
    
    Supports environment-based configuration for different deployment environments.
    """
    
    # Query validation limits
    max_query_size: int = Field(
        default=50000,
        description="Maximum query size in characters (50KB default)",
        ge=1000,  # At least 1KB
        le=1000000,  # At most 1MB
    )
    
    max_parameter_count: int = Field(
        default=100,
        description="Maximum number of parameters per query",
        ge=1,
        le=1000,
    )
    
    max_parameter_size: int = Field(
        default=10000,
        description="Maximum size per parameter in characters (10KB default)",
        ge=100,  # At least 100 bytes
        le=100000,  # At most 100KB
    )
    
    max_timeout_seconds: int = Field(
        default=300,
        description="Maximum query timeout in seconds (5 minutes default)",
        ge=1,
        le=3600,  # Maximum 1 hour
    )
    
    max_complexity_score: int = Field(
        default=20,
        description="Maximum query complexity score threshold",
        ge=5,  # Minimum complexity limit
        le=100,  # Maximum complexity limit
    )
    
    # Performance settings
    enable_query_complexity_validation: bool = Field(
        default=True,
        description="Whether to enable query complexity validation",
    )
    
    enable_sql_injection_detection: bool = Field(
        default=True,
        description="Whether to enable SQL injection pattern detection",
    )
    
    enable_error_sanitization: bool = Field(
        default=True,
        description="Whether to enable error message sanitization",
    )
    
    # Environment-specific settings
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)",
    )
    
    @classmethod
    def from_environment(cls) -> "ModelPostgresAdapterConfig":
        """
        Create configuration from environment variables.
        
        Environment variable mapping:
        - POSTGRES_ADAPTER_MAX_QUERY_SIZE
        - POSTGRES_ADAPTER_MAX_PARAMETER_COUNT
        - POSTGRES_ADAPTER_MAX_PARAMETER_SIZE
        - POSTGRES_ADAPTER_MAX_TIMEOUT_SECONDS
        - POSTGRES_ADAPTER_MAX_COMPLEXITY_SCORE
        - POSTGRES_ADAPTER_ENABLE_COMPLEXITY_VALIDATION
        - POSTGRES_ADAPTER_ENABLE_INJECTION_DETECTION
        - POSTGRES_ADAPTER_ENABLE_ERROR_SANITIZATION
        - POSTGRES_ADAPTER_ENVIRONMENT
        
        Returns:
            Configured ModelPostgresAdapterConfig instance
        """
        return cls(
            max_query_size=int(os.getenv("POSTGRES_ADAPTER_MAX_QUERY_SIZE", "50000")),
            max_parameter_count=int(os.getenv("POSTGRES_ADAPTER_MAX_PARAMETER_COUNT", "100")),
            max_parameter_size=int(os.getenv("POSTGRES_ADAPTER_MAX_PARAMETER_SIZE", "10000")),
            max_timeout_seconds=int(os.getenv("POSTGRES_ADAPTER_MAX_TIMEOUT_SECONDS", "300")),
            max_complexity_score=int(os.getenv("POSTGRES_ADAPTER_MAX_COMPLEXITY_SCORE", "20")),
            enable_query_complexity_validation=os.getenv("POSTGRES_ADAPTER_ENABLE_COMPLEXITY_VALIDATION", "true").lower() == "true",
            enable_sql_injection_detection=os.getenv("POSTGRES_ADAPTER_ENABLE_INJECTION_DETECTION", "true").lower() == "true",
            enable_error_sanitization=os.getenv("POSTGRES_ADAPTER_ENABLE_ERROR_SANITIZATION", "true").lower() == "true",
            environment=os.getenv("POSTGRES_ADAPTER_ENVIRONMENT", "development"),
        )
    
    @classmethod
    def for_environment(cls, environment: str) -> "ModelPostgresAdapterConfig":
        """
        Create environment-specific configuration with appropriate defaults.
        
        Args:
            environment: Target environment (development, staging, production)
            
        Returns:
            Environment-optimized configuration
        """
        base_config = cls.from_environment()
        base_config.environment = environment
        
        if environment == "production":
            # Production: More restrictive limits
            base_config.max_query_size = min(base_config.max_query_size, 25000)  # 25KB max
            base_config.max_parameter_count = min(base_config.max_parameter_count, 50)
            base_config.max_parameter_size = min(base_config.max_parameter_size, 5000)  # 5KB max
            base_config.max_timeout_seconds = min(base_config.max_timeout_seconds, 180)  # 3 minutes max
            base_config.max_complexity_score = min(base_config.max_complexity_score, 15)
            
        elif environment == "development":
            # Development: More permissive limits for testing
            base_config.max_query_size = 100000  # 100KB max
            base_config.max_parameter_count = 200
            base_config.max_parameter_size = 20000  # 20KB max
            base_config.max_timeout_seconds = 600  # 10 minutes max
            base_config.max_complexity_score = 30
            
        return base_config
    
    def get_complexity_weights(self) -> dict:
        """
        Get complexity scoring weights based on environment.
        
        Returns:
            Dictionary of operation types to complexity weights
        """
        if self.environment == "production":
            # More conservative weights in production
            return {
                "join": 3,
                "subquery": 4, 
                "union": 5,
                "leading_wildcard": 6,
                "regex": 12,
                "expensive_function": 4,
                "order_without_limit": 3,
            }
        else:
            # Standard weights for development/staging
            return {
                "join": 2,
                "subquery": 3,
                "union": 4,
                "leading_wildcard": 5,
                "regex": 10,
                "expensive_function": 3,
                "order_without_limit": 2,
            }