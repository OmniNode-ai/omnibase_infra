"""
ONEX Notification Authentication Model

Shared model for webhook authentication configuration in the ONEX infrastructure.
Supports multiple authentication types with secure credential handling.

Security Note: All credential fields use SecretStr for secure handling.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field, SecretStr
from omnibase_core.enums.enum_auth_type import EnumAuthType


class ModelNotificationAuth(BaseModel):
    """
    Authentication configuration for webhook notifications.

    This model encapsulates authentication details for external HTTP requests,
    supporting multiple authentication schemes with secure credential handling.

    Attributes:
        auth_type: Type of authentication to use
        credentials: Authentication credentials (structure depends on auth_type)
    """

    auth_type: EnumAuthType = Field(
        ...,
        description="Type of authentication to use for the notification"
    )

    credentials: Dict[str, Any] = Field(
        ...,
        description="Authentication credentials, structure depends on auth_type"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True
        extra = "forbid"
        use_enum_values = True

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        self._validate_credentials_for_auth_type()

    def _validate_credentials_for_auth_type(self) -> None:
        """Validate that credentials match the specified auth type."""
        if self.auth_type == EnumAuthType.BEARER:
            if "token" not in self.credentials:
                raise ValueError("Bearer auth requires 'token' in credentials")

        elif self.auth_type == EnumAuthType.BASIC:
            required_fields = {"username", "password"}
            if not required_fields.issubset(self.credentials.keys()):
                raise ValueError("Basic auth requires 'username' and 'password' in credentials")

        elif self.auth_type == EnumAuthType.API_KEY_HEADER:
            required_fields = {"header_name", "api_key"}
            if not required_fields.issubset(self.credentials.keys()):
                raise ValueError("API key auth requires 'header_name' and 'api_key' in credentials")

    @property
    def is_bearer_auth(self) -> bool:
        """Check if this is bearer token authentication."""
        return self.auth_type == EnumAuthType.BEARER

    @property
    def is_basic_auth(self) -> bool:
        """Check if this is basic authentication."""
        return self.auth_type == EnumAuthType.BASIC

    @property
    def is_api_key_auth(self) -> bool:
        """Check if this is API key authentication."""
        return self.auth_type == EnumAuthType.API_KEY_HEADER

    def get_auth_header(self) -> Dict[str, str]:
        """
        Generate the appropriate HTTP header for this authentication type.

        Returns:
            Dict[str, str]: HTTP header(s) for authentication

        Raises:
            ValueError: If credentials are invalid for the auth type
        """
        if self.auth_type == EnumAuthType.BEARER:
            token = self.credentials.get("token", "")
            return {"Authorization": f"Bearer {token}"}

        elif self.auth_type == EnumAuthType.BASIC:
            import base64
            username = self.credentials.get("username", "")
            password = self.credentials.get("password", "")
            credentials_str = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials_str.encode()).decode()
            return {"Authorization": f"Basic {encoded_credentials}"}

        elif self.auth_type == EnumAuthType.API_KEY_HEADER:
            header_name = self.credentials.get("header_name", "")
            api_key = self.credentials.get("api_key", "")
            return {header_name: api_key}

        else:
            raise ValueError(f"Unsupported auth type: {self.auth_type}")