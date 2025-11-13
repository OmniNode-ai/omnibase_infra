"""Pydantic models for common GitHub Actions."""

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionVersion(str, Enum):
    """Common GitHub Action versions."""

    CHECKOUT_V4 = "actions/checkout@v4"
    SETUP_PYTHON_V5 = "actions/setup-python@v5"
    SETUP_NODE_V4 = "actions/setup-node@v4"
    UPLOAD_ARTIFACT_V4 = "actions/upload-artifact@v4"
    DOWNLOAD_ARTIFACT_V4 = "actions/download-artifact@v4"
    CACHE_V4 = "actions/cache@v4"


class PythonVersion(str, Enum):
    """Supported Python versions."""

    PYTHON_3_8 = "3.8"
    PYTHON_3_9 = "3.9"
    PYTHON_3_10 = "3.10"
    PYTHON_3_11 = "3.11"
    PYTHON_3_12 = "3.12"


class GitHubAction(BaseModel):
    """Base model for GitHub Actions."""

    uses: str = Field(..., description="Action reference")
    with_: dict[str, Any] | None = Field(
        None, alias="with", description="Action inputs"
    )
    env: dict[str, str] | None = Field(None, description="Environment variables")
    id: str | None = Field(None, description="Step identifier")
    name: str | None = Field(None, description="Step name")
    if_: str | None = Field(None, alias="if", description="Conditional expression")

    model_config = ConfigDict(populate_by_name=True)


class CheckoutAction(GitHubAction):
    """actions/checkout action configuration."""

    uses: str = Field(
        default=ActionVersion.CHECKOUT_V4, description="Checkout action version"
    )

    def __init__(self, **kwargs):
        # Set default with parameters for checkout
        default_with = {"fetch-depth": 1}
        if "with_" not in kwargs and "with" not in kwargs:
            kwargs["with_"] = default_with
        super().__init__(**kwargs)


class SetupPythonAction(GitHubAction):
    """actions/setup-python action configuration."""

    uses: str = Field(
        default=ActionVersion.SETUP_PYTHON_V5, description="Setup Python action version"
    )

    def __init__(self, python_version: str = PythonVersion.PYTHON_3_12, **kwargs):
        default_with = {"python-version": python_version, "cache": "pip"}
        if "with_" not in kwargs and "with" not in kwargs:
            kwargs["with_"] = default_with
        super().__init__(**kwargs)


class UploadArtifactAction(GitHubAction):
    """actions/upload-artifact action configuration."""

    uses: str = Field(
        default=ActionVersion.UPLOAD_ARTIFACT_V4,
        description="Upload artifact action version",
    )

    def __init__(self, name: str, path: str, retention_days: int = 30, **kwargs):
        default_with = {"name": name, "path": path, "retention-days": retention_days}
        if "with_" not in kwargs and "with" not in kwargs:
            kwargs["with_"] = default_with
        # Pass name to parent class so it gets set as the name attribute
        kwargs["name"] = name
        super().__init__(**kwargs)


class DownloadArtifactAction(GitHubAction):
    """actions/download-artifact action configuration."""

    uses: str = Field(
        default=ActionVersion.DOWNLOAD_ARTIFACT_V4,
        description="Download artifact action version",
    )

    def __init__(self, name: str, path: str | None = None, **kwargs):
        default_with = {"name": name}
        if path:
            default_with["path"] = path
        if "with_" not in kwargs and "with" not in kwargs:
            kwargs["with_"] = default_with
        # Pass name to parent class so it gets set as the name attribute
        kwargs["name"] = name
        super().__init__(**kwargs)


class CacheAction(GitHubAction):
    """actions/cache action configuration."""

    uses: str = Field(
        default=ActionVersion.CACHE_V4, description="Cache action version"
    )

    def __init__(
        self,
        key: str,
        path: Union[str, list[str]],
        restore_keys: list[str] | None = None,
        **kwargs,
    ):
        default_with = {
            "key": key,
            "path": path if isinstance(path, str) else "\n".join(path),
        }
        if restore_keys:
            default_with["restore-keys"] = "\n".join(restore_keys)
        if "with_" not in kwargs and "with" not in kwargs:
            kwargs["with_"] = default_with
        super().__init__(**kwargs)


class ClaudeCodeAction(GitHubAction):
    """anthropics/claude-code-action configuration."""

    uses: str = Field(
        default="anthropics/claude-code-action@v1",
        description="Claude Code action version",
    )

    def __init__(
        self,
        oauth_token: str = "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}",
        github_token: str | None = None,
        prompt: str | None = None,
        claude_args: str | None = None,
        **kwargs,
    ):
        default_with = {"claude_code_oauth_token": oauth_token}
        if github_token:
            default_with["github_token"] = github_token
        if prompt:
            default_with["prompt"] = prompt
        if claude_args:
            default_with["claude_args"] = claude_args

        if "with_" not in kwargs and "with" not in kwargs:
            kwargs["with_"] = default_with
        super().__init__(**kwargs)


class ShellCommand(BaseModel):
    """Shell command step configuration."""

    name: str = Field(..., description="Step name")
    run: str = Field(..., description="Shell command")
    shell: str | None = Field(None, description="Shell type")
    working_directory: str | None = Field(None, alias="working-directory")
    env: dict[str, str] | None = Field(None, description="Environment variables")
    if_: str | None = Field(None, alias="if", description="Conditional expression")
    continue_on_error: bool | None = Field(None, alias="continue-on-error")
    timeout_minutes: int | None = Field(None, alias="timeout-minutes")

    model_config = ConfigDict(populate_by_name=True)


class DockerContainer(BaseModel):
    """Docker container configuration for jobs."""

    image: str = Field(..., description="Container image")
    env: dict[str, str] | None = Field(None, description="Environment variables")
    ports: list[Union[str, int]] | None = Field(None, description="Port mappings")
    volumes: list[str] | None = Field(None, description="Volume mounts")
    options: str | None = Field(None, description="Additional docker options")
    credentials: dict[str, str] | None = Field(None, description="Registry credentials")


class ServiceContainer(BaseModel):
    """Service container configuration."""

    image: str = Field(..., description="Service image")
    env: dict[str, str] | None = Field(None, description="Environment variables")
    ports: list[Union[str, int]] | None = Field(None, description="Port mappings")
    volumes: list[str] | None = Field(None, description="Volume mounts")
    options: str | None = Field(None, description="Docker options")
    credentials: dict[str, str] | None = Field(None, description="Registry credentials")


class ScheduleTrigger(BaseModel):
    """Cron schedule trigger configuration."""

    cron: str = Field(..., description="Cron expression")

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, v):
        """Basic cron validation."""
        parts = v.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts")
        return v
