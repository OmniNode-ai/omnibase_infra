# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Test google-genai dependency install and async client availability.

Verifies that the `google-genai` SDK is installed and that:
  - `from google import genai` succeeds
  - `from google.genai import types` succeeds
  - `genai.Client` exposes an async (`aio`) namespace

The `google-genai` package is consumed by `AdapterLlmProviderGemini`
(OMN-5069) and declared as a primary dependency in `pyproject.toml`.

See: OMN-5065 (Task 1 — add google-genai dependency).
"""

import pytest
from packaging.version import Version


@pytest.mark.unit
class TestGoogleGenaiDependency:
    """Verify google-genai package functions correctly after install."""

    def test_google_genai_imports(self) -> None:
        """Both `genai` and `genai.types` must import without error."""
        from google import genai
        from google.genai import types

        assert genai.__name__ == "google.genai"
        assert types.__name__ == "google.genai.types"

    def test_google_genai_minimum_version(self) -> None:
        """Installed google-genai version must satisfy >=1.0,<2.0."""
        from google import genai

        actual = Version(genai.__version__)
        assert Version("1.0.0") <= actual < Version("2.0.0"), (
            f"google-genai version {actual} is outside the declared "
            f">=1.0,<2.0 range in pyproject.toml."
        )

    def test_google_genai_async_client_available(self) -> None:
        """`genai.Client` must expose the async `aio` namespace.

        The async client surface is required by AdapterLlmProviderGemini
        (OMN-5069), which uses `client.aio.models.generate_content(...)`.
        """
        from google import genai

        assert hasattr(genai, "Client"), (
            "google.genai.Client is missing — the public SDK surface changed."
        )
        assert hasattr(genai.Client, "aio"), (
            "google.genai.Client.aio namespace is missing — the async "
            "client surface required by AdapterLlmProviderGemini is "
            "unavailable."
        )
