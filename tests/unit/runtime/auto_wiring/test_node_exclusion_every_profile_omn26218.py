# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26232: one exclusion name is safe on every runtime profile the stack runs.

The compose passthrough hands ``ONEX_RUNTIME_NODE_EXCLUSION`` to every runtime
service, so one name reaches the kernel under each ``RUNTIME_PROFILE`` the
catalog declares. The ownership filter refuses a name the manifest it is handed
lacks, so the name is safe everywhere only when that manifest is the whole
discovered one whatever the profile. The kernel hands the filter exactly that
(``discover_contracts`` runs before any profile is applied); this pins the
filter's side: a name any discovered contract carries is accepted under every
profile, including a profile that does not own the contract, and a name none
carries is refused under every profile, loud, before wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_EXCLUSION,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
CATALOG = ROOT / "docker" / "catalog" / "services"


def _catalog_profiles() -> tuple[str, ...]:
    found: set[str] = set()
    for path in sorted(CATALOG.glob("*.yaml")):
        env = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
            "hardcoded_env"
        ) or {}
        if isinstance(env.get("RUNTIME_PROFILE"), str):
            found.add(env["RUNTIME_PROFILE"])
    return tuple(sorted(found))


PROFILES = _catalog_profiles()


def _contract(name: str, profile: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_profiles=(profile,),
    )


def _manifest() -> ModelAutoWiringManifest:
    """One contract owned by each profile the catalog declares."""
    return ModelAutoWiringManifest(
        contracts=tuple(_contract(f"node_of_{p}", p) for p in PROFILES)
    )


def test_the_catalog_declares_the_profiles_the_kernel_services_run() -> None:
    assert {"main", "effects", "workers", "tenant-projection"} <= set(PROFILES)


@pytest.mark.parametrize("profile", PROFILES)
def test_a_discovered_name_is_accepted_under_every_profile(profile: str) -> None:
    manifest = _manifest()
    for owner in PROFILES:
        name = f"node_of_{owner}"
        result = filter_manifest_for_runtime_profile(
            manifest, profile, environ={ENV_RUNTIME_NODE_EXCLUSION: name}
        )
        assert name not in {c.name for c in result.manifest.contracts}
        assert result.manifest.errors == ()
        assert name in result.node_selection_excluded_contracts or owner != profile


@pytest.mark.parametrize("profile", PROFILES)
def test_an_undiscovered_name_is_refused_under_every_profile(profile: str) -> None:
    with pytest.raises(RuntimeHostError, match="node_of_typo"):
        filter_manifest_for_runtime_profile(
            _manifest(), profile, environ={ENV_RUNTIME_NODE_EXCLUSION: "node_of_typo"}
        )


@pytest.mark.parametrize("profile", PROFILES)
def test_an_unset_exclusion_changes_nothing_under_every_profile(profile: str) -> None:
    result = filter_manifest_for_runtime_profile(_manifest(), profile, environ={})
    assert result.node_selection_excluded_contracts == ()
    assert result.manifest.errors == ()
