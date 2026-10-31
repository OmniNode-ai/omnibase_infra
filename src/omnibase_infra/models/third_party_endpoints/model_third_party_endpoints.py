# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared third-party service endpoints (OMN-21623).

The values live in the ``endpoints`` block of
``models/third_party_endpoints/third_party_endpoints.yaml``. The model has no field
defaults, so a block that omits a value fails to load rather than falling back
to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_ENDPOINTS_PATH = Path(__file__).resolve().parent / "third_party_endpoints.yaml"

_GIT_TOKEN_USER = "x-access-token"


class ModelThirdPartyEndpoints(BaseModel):
    """URLs of the third-party services and probes that infra code calls.

    Attributes:
        github_api_base_url: GitHub REST API base.
        github_web_base_url: GitHub web origin (pull request, blob and clone URLs).
        runtime_source_repo_url: Repository the runtime source attestation clones
            when no URL is given.
        linear_graphql_url: Linear GraphQL endpoint.
        linear_oauth_token_url: Linear OAuth client-credentials token endpoint.
        slack_post_message_url: Slack ``chat.postMessage`` endpoint.
        gmail_oauth_token_url: Google OAuth2 token endpoint for the Gmail handler.
        gmail_api_base_url: Gmail REST API base for the authenticated user.
        claude_code_settings_schema_url: JSON schema of the Claude Code settings file.
        loopback_http_base_url: Scheme and host a self-probe uses before
            ``:<port><path>``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    github_api_base_url: str = Field(min_length=1)
    github_web_base_url: str = Field(min_length=1)
    runtime_source_repo_url: str = Field(min_length=1)
    linear_graphql_url: str = Field(min_length=1)
    linear_oauth_token_url: str = Field(min_length=1)
    slack_post_message_url: str = Field(min_length=1)
    gmail_oauth_token_url: str = Field(min_length=1)
    gmail_api_base_url: str = Field(min_length=1)
    claude_code_settings_schema_url: str = Field(min_length=1)
    loopback_http_base_url: str = Field(min_length=1)

    def github_pull_url(self, repo: str, number: int) -> str:
        """Return the web URL of pull request ``number`` in ``repo`` (``owner/name``)."""
        return f"{self.github_web_base_url}/{repo}/pull/{number}"

    def github_blob_url(self, repo: str, sha: str, path: str) -> str:
        """Return the web URL of ``path`` at commit ``sha`` in ``repo``."""
        return f"{self.github_web_base_url}/{repo}/blob/{sha}/{path}"

    def github_pull_api_url(self, repo: str, number: int) -> str:
        """Return the REST URL of pull request ``number`` in ``repo``."""
        return f"{self.github_api_base_url}/repos/{repo}/pulls/{number}"

    def github_clone_url(self, repo: str, token: str = "") -> str:
        """Return the clone URL of ``repo``; a non-empty ``token`` is embedded as userinfo."""
        base = urlsplit(self.github_web_base_url)
        authority = f"{_GIT_TOKEN_USER}:{token}@{base.netloc}" if token else base.netloc
        return f"{base.scheme}://{authority}/{repo}.git"

    def loopback_url(self, port: int, path: str = "") -> str:
        """Return the loopback URL of ``port`` with an optional ``path``."""
        return f"{self.loopback_http_base_url}:{port}{path}"

    @classmethod
    def from_file(cls, endpoints_path: Path) -> ModelThirdPartyEndpoints:
        """Load the endpoints from the ``endpoints`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``endpoints`` mapping.
        """
        with endpoints_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("endpoints") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_third_party_endpoints",
            )
            raise ProtocolConfigurationError(
                f"{endpoints_path.name} has no endpoints mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_third_party_endpoints() -> ModelThirdPartyEndpoints:
    """Return the endpoints declared in ``third_party_endpoints.yaml``."""
    return ModelThirdPartyEndpoints.from_file(_ENDPOINTS_PATH)


__all__: list[str] = ["ModelThirdPartyEndpoints", "load_third_party_endpoints"]
