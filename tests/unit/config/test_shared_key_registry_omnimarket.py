# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry tests for OMN-10563: services.omnimarket section + 1.2 schema bump.

Validates:
1. Registry version bumped to 1.2.
2. `services` section exists and parses correctly.
3. `services.omnimarket` declares the keys named in the public-shippable plan
   Task 16 (kafka.KAFKA_GROUP_ID, llm.LLM_*_URL/_MODEL_ID, db.POSTGRES_DATABASE).
4. The new `_services_keys()` loader handles valid + invalid inputs without
   regressing existing loaders (`_load_registry`, `_bootstrap_keys`,
   `_identity_defaults`, `_service_override_required`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "config" / "shared_key_registry.yaml"
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "register-repo.py"


@pytest.fixture(scope="module")
def registry_data() -> dict[str, object]:
    return yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def register_repo_module():  # type: ignore[no-untyped-def]
    """Import scripts/register-repo.py as a module (filename has a hyphen)."""
    spec = importlib.util.spec_from_file_location("register_repo", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["register_repo"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestRegistryVersionBump:
    def test_version_is_1_2(self, registry_data: dict[str, object]) -> None:
        assert registry_data.get("version") == "1.2"

    def test_top_level_sections_present(self, registry_data: dict[str, object]) -> None:
        # All preexisting top-level sections must remain present after the
        # 1.2 schema bump (regression guard).
        for required in (
            "shared",
            "bootstrap_only",
            "identity_defaults",
            "service_override_required",
            "services",
        ):
            assert required in registry_data, f"section {required!r} missing"


class TestServicesOmnimarketSection:
    def test_omnimarket_kafka_includes_group_id(
        self, registry_data: dict[str, object]
    ) -> None:
        services = registry_data["services"]
        assert isinstance(services, dict)
        omnimarket = services["omnimarket"]
        assert "KAFKA_GROUP_ID" in omnimarket["kafka"]

    def test_omnimarket_llm_lists_url_and_model_id_for_coder_and_reasoner(
        self, registry_data: dict[str, object]
    ) -> None:
        omnimarket = registry_data["services"]["omnimarket"]
        llm_keys = set(omnimarket["llm"])
        assert {
            "LLM_CODER_URL",
            "LLM_CODER_MODEL_ID",
            "LLM_REASONER_URL",
            "LLM_REASONER_MODEL_ID",
        }.issubset(llm_keys)

    def test_omnimarket_db_includes_postgres_database(
        self, registry_data: dict[str, object]
    ) -> None:
        omnimarket = registry_data["services"]["omnimarket"]
        assert "POSTGRES_DATABASE" in omnimarket["db"]

    def test_omnimarket_db_includes_connection_keys(
        self, registry_data: dict[str, object]
    ) -> None:
        omnimarket = registry_data["services"]["omnimarket"]
        db_keys = set(omnimarket["db"])
        assert {
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        }.issubset(db_keys)

    def test_omnimarket_infisical_section_present(
        self, registry_data: dict[str, object]
    ) -> None:
        omnimarket = registry_data["services"]["omnimarket"]
        assert "infisical" in omnimarket

    def test_omnimarket_infisical_includes_addr_and_project_id(
        self, registry_data: dict[str, object]
    ) -> None:
        omnimarket = registry_data["services"]["omnimarket"]
        infisical_keys = set(omnimarket["infisical"])
        assert {"INFISICAL_ADDR", "INFISICAL_PROJECT_ID"}.issubset(infisical_keys)


class TestServicesKeysLoader:
    def test_loader_returns_omnimarket_block(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        result = register_repo_module._services_keys()
        assert "omnimarket" in result
        assert "KAFKA_GROUP_ID" in result["omnimarket"]["kafka"]
        assert "POSTGRES_DATABASE" in result["omnimarket"]["db"]
        assert "POSTGRES_PASSWORD" in result["omnimarket"]["db"]
        assert "INFISICAL_ADDR" in result["omnimarket"]["infisical"]
        assert "INFISICAL_PROJECT_ID" in result["omnimarket"]["infisical"]

    def test_loader_returns_empty_dict_when_section_absent(
        self, register_repo_module
    ) -> None:  # type: ignore[no-untyped-def]
        # Older registries without `services:` are still valid; loader returns {}.
        result = register_repo_module._services_keys({"shared": {"/shared/x/": ["A"]}})
        assert result == {}

    def test_loader_rejects_non_mapping_services(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match=r"must be a mapping|to be a mapping"):
            register_repo_module._services_keys({"services": ["omnimarket"]})

    def test_loader_rejects_non_mapping_repo_block(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="to be a mapping"):
            register_repo_module._services_keys(
                {"services": {"omnimarket": ["KAFKA_GROUP_ID"]}}
            )

    def test_loader_rejects_non_list_transport(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="to be a list"):
            register_repo_module._services_keys(
                {"services": {"omnimarket": {"kafka": "KAFKA_GROUP_ID"}}}
            )

    def test_loader_rejects_empty_transport_list(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="empty key list"):
            register_repo_module._services_keys(
                {"services": {"omnimarket": {"kafka": []}}}
            )

    def test_loader_rejects_non_string_keys(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="must be a list of strings"):
            register_repo_module._services_keys(
                {"services": {"omnimarket": {"kafka": [123]}}}
            )


class TestExistingLoadersStillWork:
    """Regression: 1.2 schema bump must not break existing loaders."""

    def test_load_registry_still_returns_shared(self, register_repo_module) -> None:  # type: ignore[no-untyped-def]
        result = register_repo_module._load_registry()
        assert "/shared/db/" in result
        assert "POSTGRES_HOST" in result["/shared/db/"]

    def test_bootstrap_keys_still_includes_postgres_password(
        self, register_repo_module
    ) -> None:  # type: ignore[no-untyped-def]
        result = register_repo_module._bootstrap_keys()
        assert "POSTGRES_PASSWORD" in result

    def test_identity_defaults_still_includes_postgres_database(
        self, register_repo_module
    ) -> None:  # type: ignore[no-untyped-def]
        result = register_repo_module._identity_defaults()
        assert "POSTGRES_DATABASE" in result

    def test_service_override_required_still_includes_kafka_group_id(
        self, register_repo_module
    ) -> None:  # type: ignore[no-untyped-def]
        result = register_repo_module._service_override_required()
        assert "KAFKA_GROUP_ID" in result
