# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26460: dev's declared node exclusion names only contracts dev can discover.

The recorded incident
---------------------
Dev deploy job ae286985 (``fixtures/dev_job_ae286985/incident_record.json``) failed
at the runtime phase: ``runtime health DEGRADED - discovery_errors: 1 contract(s)
failed to load: compliance_orchestrator``.

``compliance_orchestrator`` loads fine. The row is not a parse or import failure: it
is ``ModelRuntimeNodeSelection.stale_declared_errors`` reporting that the dev
profile's declared exclusion (``contracts/services/runtime_lane_node_exclusion.contract.yaml``)
names a contract "which no discovered contract is". The contract is a node of
``omnibase_core``, and dev's ``ONEX_ACTIVE_RUNTIME_PACKAGES`` is
``omnibase_infra,omnimarket`` (``docker/runtime-policy.env``), so discovery never
returns any ``omnibase_core`` contract on dev and an exclusion of one can never match.
The recorded row is ``fixtures/dev_job_ae286985/discovery_error_row.json``.

These tests run the runtime's own discovery and ownership filter, over the contract
at its real path in the installed ``omnibase_core``, under the dev environment.

Related Tickets:
    - OMN-26460: this regression
    - OMN-26324: the declared exclusion, whose evidence read contract files, not discovery
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.runtime.model_runtime_lane_node_exclusion import (
    LANE_PROFILE_PREFIX,
    RUNTIME_LANE_NODE_EXCLUSION_CONTRACT,
    ModelRuntimeLaneNodeExclusion,
)
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts,
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts_cache_clear,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveryError,
)
from omnibase_infra.utils.util_runtime_packages import (
    ENV_ACTIVE_RUNTIME_PACKAGES,
    normalize_runtime_package_name,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACTS_DIR = REPO_ROOT / "contracts"
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "dev_job_ae286985"
DEV_LANE = f"{LANE_PROFILE_PREFIX}dev"
NAMED_IN_INCIDENT = "compliance_orchestrator"

_FAILED_TO_LOAD = re.compile(
    r"discovery_errors: (?P<count>\d+) contract\(s\) failed to load: (?P<names>.+)$"
)


def _dev_active_packages() -> str:
    """The dev lane's package allowlist, read from the policy file the compose files render."""
    for line in (REPO_ROOT / "docker" / "runtime-policy.env").read_text().splitlines():
        if line.startswith(f"{ENV_ACTIVE_RUNTIME_PACKAGES}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{ENV_ACTIVE_RUNTIME_PACKAGES} is not in runtime-policy.env")


@pytest.fixture
def dev_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    environ = {
        "ONEX_RUNTIME_LANE": DEV_LANE,
        "ONEX_CONTRACTS_DIR": str(CONTRACTS_DIR),
    }
    monkeypatch.setenv(ENV_ACTIVE_RUNTIME_PACKAGES, _dev_active_packages())
    discover_contracts_cache_clear()
    yield environ
    discover_contracts_cache_clear()


def _dev_main_rows(environ: dict[str, str]) -> tuple[ModelDiscoveryError, ...]:
    """The error rows the dev main runtime's health monitor counts."""
    manifest = discover_contracts()
    result = filter_manifest_for_runtime_profile(manifest, "main", environ=environ)
    return result.manifest.errors


def _all_packages_manifest() -> ModelAutoWiringManifest:
    """Discovery with no package allowlist: every contract the image ships."""
    return discover_contracts()


def _dev_excluded() -> tuple[str, ...]:
    declaration = ModelRuntimeLaneNodeExclusion.load(
        ModelRuntimeLaneNodeExclusion.declaration_path(CONTRACTS_DIR)
    )
    return declaration.entry_for(DEV_LANE).excluded_contracts


def test_compliance_orchestrator_has_no_discovery_error_under_dev(
    dev_env: dict[str, str],
) -> None:
    """AC1/AC3: no discovery error row names the contract of the incident."""
    rows = [
        r for r in _dev_main_rows(dev_env) if r.entry_point_name == NAMED_IN_INCIDENT
    ]
    assert not rows, [(r.entry_point_name, r.error) for r in rows]


def test_dev_exclusion_names_no_contract_of_an_inactive_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The class: a declared exclusion of a contract dev can never discover is a stale entry.

    Discovery with no allowlist returns every installed package's contracts. A contract
    whose package is outside dev's allowlist is never discovered on dev, so naming it in
    the dev exclusion makes ``stale_declared_errors`` report it on every boot.
    """
    monkeypatch.delenv(ENV_ACTIVE_RUNTIME_PACKAGES, raising=False)
    discover_contracts_cache_clear()
    try:
        everything = _all_packages_manifest()
    finally:
        discover_contracts_cache_clear()
    active = {
        normalize_runtime_package_name(p) for p in _dev_active_packages().split(",")
    }
    inactive_package_of = {
        c.name: c.package_name
        for c in everything.contracts
        if normalize_runtime_package_name(c.package_name) not in active
    }
    stale = {
        name: inactive_package_of[name]
        for name in _dev_excluded()
        if name in inactive_package_of
    }
    assert not stale, (
        f"dev excludes contracts of packages outside {sorted(active)}: {stale}; "
        "discovery on dev never returns them, so the exclusion is reported as a "
        "discovery error"
    )


def test_incident_row_is_what_the_filter_reports_for_an_undiscovered_exclusion(
    dev_env: dict[str, str], tmp_path: Path
) -> None:
    """Control: a dev exclusion naming an undiscovered contract yields the recorded row.

    A copy of the declaration whose dev exclusion also names the recorded contract
    is read against the same dev discovery. Without this control the tests above could
    pass because the row's text or the filter changed, not because the entry is gone.
    """
    recorded = json.loads((FIXTURES / "discovery_error_row.json").read_text())
    target = tmp_path / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target.parent.mkdir(parents=True)
    data = yaml.safe_load(
        (CONTRACTS_DIR / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT).read_text()
    )
    data["profiles"]["dev"]["excluded_contracts"].append(recorded["entry_point_name"])
    target.write_text(yaml.safe_dump(data), encoding="utf-8")

    rows = _dev_main_rows({**dev_env, "ONEX_CONTRACTS_DIR": str(tmp_path)})
    by_name = {r.entry_point_name: r for r in rows}
    assert by_name[recorded["entry_point_name"]].error == recorded["error"]


def test_incident_record_names_what_the_recorded_row_explains(
    dev_env: dict[str, str],
) -> None:
    """AC2: the contracts the incident named have no row, in the incident's own shape."""
    incident = json.loads((FIXTURES / "incident_record.json").read_text())
    recorded = json.loads((FIXTURES / "discovery_error_row.json").read_text())
    match = _FAILED_TO_LOAD.search(incident["error"])
    assert match is not None, incident["error"]
    named = [n.strip() for n in match.group("names").split(",")]
    assert int(match.group("count")) == len(named)
    assert recorded["entry_point_name"] in named

    rows = _dev_main_rows(dev_env)
    still_reported = [
        r.entry_point_name
        for r in rows
        if r.entry_point_name in named and r.error == recorded["error"]
    ]
    assert not still_reported, still_reported
