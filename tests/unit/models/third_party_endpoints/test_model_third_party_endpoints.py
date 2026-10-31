# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the declared third-party service endpoints (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.third_party_endpoints import (
    ModelThirdPartyEndpoints,
    load_third_party_endpoints,
)
from omnibase_infra.models.third_party_endpoints import (
    model_third_party_endpoints as endpoints_module,
)

pytestmark = pytest.mark.unit

_DECLARED_FILE = Path(endpoints_module.__file__).parent / "third_party_endpoints.yaml"


def _declared_block() -> dict[str, str]:
    document = yaml.safe_load(_DECLARED_FILE.read_text(encoding="utf-8"))
    block = document["endpoints"]
    assert isinstance(block, dict)
    return block


def test_loaded_endpoints_equal_the_declared_file() -> None:
    loaded = load_third_party_endpoints()
    assert loaded.model_dump() == _declared_block()


@pytest.mark.parametrize("omitted", sorted(ModelThirdPartyEndpoints.model_fields))
def test_an_omitted_value_fails_to_load(omitted: str, tmp_path: Path) -> None:
    block = {k: v for k, v in _declared_block().items() if k != omitted}
    path = tmp_path / "endpoints.yaml"
    path.write_text(yaml.safe_dump({"endpoints": block}), encoding="utf-8")
    with pytest.raises(ValidationError, match=omitted):
        ModelThirdPartyEndpoints.from_file(path)


def test_a_file_without_an_endpoints_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.yaml"
    path.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        ModelThirdPartyEndpoints.from_file(path)


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    block = {**_declared_block(), "unexpected": "value"}
    path = tmp_path / "endpoints.yaml"
    path.write_text(yaml.safe_dump({"endpoints": block}), encoding="utf-8")
    with pytest.raises(ValidationError):
        ModelThirdPartyEndpoints.from_file(path)


def test_url_builders_derive_from_the_declared_bases() -> None:
    endpoints = load_third_party_endpoints()
    assert (
        endpoints.github_pull_url("owner/name", 7)
        == f"{endpoints.github_web_base_url}/owner/name/pull/7"
    )
    assert (
        endpoints.github_blob_url("owner/name", "abc123", "dir/file.py")
        == f"{endpoints.github_web_base_url}/owner/name/blob/abc123/dir/file.py"
    )
    assert (
        endpoints.github_pull_api_url("owner/name", 7)
        == f"{endpoints.github_api_base_url}/repos/owner/name/pulls/7"
    )
    assert (
        endpoints.loopback_url(8085, "/health")
        == f"{endpoints.loopback_http_base_url}:8085/health"
    )
    assert endpoints.loopback_url(8085) == f"{endpoints.loopback_http_base_url}:8085"


def test_clone_url_carries_the_token_only_when_given() -> None:
    endpoints = load_third_party_endpoints()
    plain = endpoints.github_clone_url("owner/name")
    assert plain == f"{endpoints.github_web_base_url}/owner/name.git"
    with_token = endpoints.github_clone_url("owner/name", "tok")
    scheme, _, rest = endpoints.github_web_base_url.partition("://")
    assert with_token == f"{scheme}://x-access-token:tok@{rest}/owner/name.git"
