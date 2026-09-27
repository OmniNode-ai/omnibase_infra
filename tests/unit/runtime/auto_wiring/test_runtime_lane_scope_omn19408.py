# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19408: a node attaches only on the runtime lanes its contract declares.

The .201 stability-test runtime loaded the lab lane-health projection, whose
scope (OMN-18769 AC6) holds rows for the three lab lanes and nothing else. The
stability-test runtimes publish health with no lane, the fold correctly dropped
every event, the projection wrote zero rows, and the apply-divergence detector
held the lane Docker-unhealthy from about 05:40Z. The detector was right and the
fold was right. The ATTACHMENT was wrong, and nothing could express that:
``runtime_profiles`` is role-scoped (main, effects, workers), and a
stability-test main runtime is ``main``.

``runtime_lanes`` is a contract-declared lane scope, enforced by the same
ownership filter the kernel wiring and the health monitor both read, so the two
cannot disagree about what is attached. Fail-closed: a runtime that owns a
lane-scoped contract by profile but cannot name a registered lane does NOT
attach it, and records a discovery error that names the contract and the
variable -- the health monitor turns that into a DEGRADED ``discovery_errors``
dimension. It is never a silent skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from omnibase_core.constants.constants_runtime_lanes import (
    LAB_RUNTIME_LANES,
    REGISTERED_RUNTIME_LANES,
)
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)
from omnibase_infra.runtime.health import runtime_lane_identity
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_LANE,
    KNOWN_LANES,
    resolve_declared_runtime_lane,
    resolve_runtime_lane,
)

pytestmark = pytest.mark.unit

LAB_PROJECTION = "projection_lab_lane_health"
LAB_SCOPE = ("compose-dev", "onex-lab", "onex-lab-k3s")

_LAB_PROJECTION_CONTRACT = """
name: projection_lab_lane_health
contract_version: {major: 1, minor: 0, patch: 0}
node_type: reducer
runtime_lanes:
  - compose-dev
  - onex-lab
  - onex-lab-k3s
event_bus:
  subscribe_topics:
    - onex.evt.omnibase-infra.runtime-health-check.v1
db_io:
  db_tables:
    - name: lab_lane_health
      database_ref: application
      schema: omninode_internal
      migration: "0000_create_lab_lane_health.sql"
      access: read_write
      role: lane_health
""".strip()


def _write_contract(tmp_path: Path, body: str, directory: str) -> Path:
    contract_dir = tmp_path / directory
    contract_dir.mkdir()
    path = contract_dir / "contract.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _contract(
    name: str,
    *,
    runtime_profiles: tuple[str, ...] = (),
    runtime_lanes: tuple[str, ...] | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_profiles=runtime_profiles,
        runtime_lanes=(
            ModelRuntimeLaneScope(lanes=runtime_lanes)
            if runtime_lanes is not None
            else None
        ),
    )


def _names(result_manifest: ModelAutoWiringManifest) -> list[str]:
    return [contract.name for contract in result_manifest.contracts]


# --------------------------------------------------------------------------
# Discovery reads the field, and refuses a malformed one
# --------------------------------------------------------------------------


def test_discovery_parses_runtime_lanes_into_the_core_scope(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, _LAB_PROJECTION_CONTRACT, "node_lab")

    manifest = discover_contracts_from_paths([path])

    assert manifest.total_errors == 0
    scope = manifest.contracts[0].runtime_lanes
    assert isinstance(scope, ModelRuntimeLaneScope)
    assert scope.lanes == LAB_SCOPE


def test_a_contract_without_runtime_lanes_is_unscoped(tmp_path: Path) -> None:
    body = _LAB_PROJECTION_CONTRACT.replace(
        "runtime_lanes:\n  - compose-dev\n  - onex-lab\n  - onex-lab-k3s\n", ""
    )
    assert "runtime_lanes" not in body
    path = _write_contract(tmp_path, body, "node_unscoped")

    manifest = discover_contracts_from_paths([path])

    assert manifest.total_errors == 0
    assert manifest.contracts[0].runtime_lanes is None


def test_discovery_refuses_an_unregistered_lane_by_name(tmp_path: Path) -> None:
    body = _LAB_PROJECTION_CONTRACT.replace("  - onex-lab-k3s", "  - onex-lab-k3z")
    path = _write_contract(tmp_path, body, "node_typo")

    manifest = discover_contracts_from_paths([path])

    assert manifest.contracts == ()
    assert manifest.total_errors == 1
    assert "onex-lab-k3z" in manifest.errors[0].error


# --------------------------------------------------------------------------
# AC2 of OMN-19408: absent on stability-test, present on compose-dev
# --------------------------------------------------------------------------


def test_ac2_lab_projection_is_not_wired_on_stability_test_and_is_on_compose_dev(
    tmp_path: Path,
) -> None:
    path = _write_contract(tmp_path, _LAB_PROJECTION_CONTRACT, "node_lab")
    manifest = discover_contracts_from_paths([path])

    stability = filter_manifest_for_runtime_profile(
        manifest, "main", environ={ENV_RUNTIME_LANE: "stability-test"}
    )
    dev = filter_manifest_for_runtime_profile(
        manifest, "main", environ={ENV_RUNTIME_LANE: "compose-dev"}
    )

    assert LAB_PROJECTION not in _names(stability.manifest)
    assert stability.lane_excluded_contracts == (LAB_PROJECTION,)
    assert stability.manifest.total_errors == 0, (
        "a DECLARED lane outside the scope is the correct state, not an error"
    )
    assert stability.runtime_lane == "stability-test"

    assert _names(dev.manifest) == [LAB_PROJECTION]
    assert dev.lane_excluded_contracts == ()
    assert dev.manifest.total_errors == 0


def test_every_lab_lane_attaches_the_lab_projection() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(LAB_PROJECTION, runtime_lanes=LAB_SCOPE),)
    )
    for lane in sorted(LAB_RUNTIME_LANES):
        result = filter_manifest_for_runtime_profile(
            manifest, "main", environ={ENV_RUNTIME_LANE: lane}
        )
        assert _names(result.manifest) == [LAB_PROJECTION], lane


@pytest.mark.parametrize(
    "lane", sorted(REGISTERED_RUNTIME_LANES - LAB_RUNTIME_LANES), ids=str
)
def test_no_non_lab_lane_attaches_the_lab_projection(lane: str) -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(LAB_PROJECTION, runtime_lanes=LAB_SCOPE),)
    )
    result = filter_manifest_for_runtime_profile(
        manifest, "main", environ={ENV_RUNTIME_LANE: lane}
    )
    assert result.manifest.contracts == ()
    assert result.lane_excluded_contracts == (LAB_PROJECTION,)
    assert result.manifest.total_errors == 0


# --------------------------------------------------------------------------
# Fail-closed: no lane, or an unknown one, is an error -- never a silent skip
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "environ",
    [{}, {ENV_RUNTIME_LANE: ""}, {ENV_RUNTIME_LANE: "stabilty-test"}],
    ids=["absent", "blank", "unregistered"],
)
def test_an_undeclared_lane_does_not_attach_and_is_recorded_as_an_error(
    environ: dict[str, str],
) -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract(LAB_PROJECTION, runtime_lanes=LAB_SCOPE),
            _contract("unscoped_node"),
        )
    )

    result = filter_manifest_for_runtime_profile(manifest, "main", environ=environ)

    assert _names(result.manifest) == ["unscoped_node"], (
        "fail-closed: a lane-scoped node never attaches on a runtime that "
        "cannot name its lane"
    )
    assert result.manifest.total_errors == 1, (
        "an undeclared lane on a runtime that owns a lane-scoped node must be "
        "an error the health monitor reports, not a silent skip"
    )
    error = result.manifest.errors[0]
    assert error.entry_point_name == LAB_PROJECTION
    assert ENV_RUNTIME_LANE in error.error
    assert LAB_PROJECTION in error.error
    assert result.runtime_lane is None


def test_existing_discovery_errors_are_kept_beside_the_lane_error() -> None:
    from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveryError

    prior = ModelDiscoveryError(entry_point_name="broken", error="parse failed")
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(LAB_PROJECTION, runtime_lanes=LAB_SCOPE),),
        errors=(prior,),
    )

    result = filter_manifest_for_runtime_profile(manifest, "main", environ={})

    assert result.manifest.errors[0] == prior
    assert result.manifest.total_errors == 2


def test_an_unscoped_manifest_needs_no_lane() -> None:
    """The error is about lane-scoped nodes, not about every runtime."""
    manifest = ModelAutoWiringManifest(contracts=(_contract("unscoped_node"),))

    result = filter_manifest_for_runtime_profile(manifest, "main", environ={})

    assert _names(result.manifest) == ["unscoped_node"]
    assert result.manifest.total_errors == 0


def test_a_profile_that_does_not_own_the_node_needs_no_lane() -> None:
    """runtime-effects on the dev lane declares no lane, and never owns this node.

    Profile ownership is decided first. Only a runtime that would otherwise
    attach a lane-scoped node has to be able to name its lane.
    """
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(LAB_PROJECTION, runtime_lanes=LAB_SCOPE),)
    )

    result = filter_manifest_for_runtime_profile(manifest, "effects", environ={})

    assert result.manifest.contracts == ()
    assert result.skipped_contracts == (LAB_PROJECTION,)
    assert result.manifest.total_errors == 0


def test_the_lane_is_read_from_the_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(LAB_PROJECTION, runtime_lanes=LAB_SCOPE),)
    )
    monkeypatch.setenv(ENV_RUNTIME_LANE, "stability-test")

    result = filter_manifest_for_runtime_profile(manifest, "main")

    assert result.manifest.contracts == ()
    assert result.runtime_lane == "stability-test"


# --------------------------------------------------------------------------
# The raw-contract ownership path used by RuntimeHostProcess agrees
# --------------------------------------------------------------------------


def test_raw_contract_ownership_honours_the_lane_scope() -> None:
    raw = {
        "name": LAB_PROJECTION,
        "runtime_lanes": list(LAB_SCOPE),
        "event_bus": {
            "subscribe_topics": ["onex.evt.omnibase-infra.runtime-health-check.v1"]
        },
    }

    assert (
        runtime_profile_owns_contract(
            raw, "main", environ={ENV_RUNTIME_LANE: "compose-dev"}
        )
        is True
    )
    assert (
        runtime_profile_owns_contract(
            raw, "main", environ={ENV_RUNTIME_LANE: "stability-test"}
        )
        is False
    )
    assert runtime_profile_owns_contract(raw, "main", environ={}) is False
    raw_unscoped = {k: v for k, v in raw.items() if k != "runtime_lanes"}
    assert runtime_profile_owns_contract(raw_unscoped, "main", environ={}) is True


# --------------------------------------------------------------------------
# Two resolvers over one variable, and OMN-18769 AC6 unchanged
# --------------------------------------------------------------------------


def test_placement_resolver_accepts_every_registered_lane() -> None:
    for lane in sorted(REGISTERED_RUNTIME_LANES):
        assert resolve_declared_runtime_lane({ENV_RUNTIME_LANE: lane}) == lane
    assert (
        resolve_declared_runtime_lane({ENV_RUNTIME_LANE: " Stability-Test "})
        == "stability-test"
    )


def test_placement_resolver_refuses_absent_and_unregistered_lanes() -> None:
    assert resolve_declared_runtime_lane({}) is None
    assert resolve_declared_runtime_lane({ENV_RUNTIME_LANE: "  "}) is None
    assert resolve_declared_runtime_lane({ENV_RUNTIME_LANE: "prod"}) is None


def test_ac6_unchanged_the_health_event_still_keys_only_a_lab_lane() -> None:
    """Declaring stability-test does NOT make its health keyable.

    OMN-18769 AC6 is that stability-test, judge and the collaborator lane can
    never be keyed onto a lab lane-health row. The health emitter's vocabulary
    stays the lab set, so a stability-test runtime that now names its lane for
    PLACEMENT still publishes its health with no lane.
    """
    assert KNOWN_LANES == LAB_RUNTIME_LANES
    for lane in ("stability-test", "judge", "lakshman"):
        assert resolve_runtime_lane({ENV_RUNTIME_LANE: lane}) is None


def test_a_registered_non_lab_lane_is_not_warned_about_on_every_health_emit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The resolver runs on every health emit for the life of the container.

    An unregistered value is a typo and deserves the warning. A registered
    non-lab lane is the deployment telling the truth; warning about it every
    check interval is the volume that gets a line filtered out.
    """
    runtime_lane_identity._note_non_lab_lane.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                assert (
                    resolve_runtime_lane({ENV_RUNTIME_LANE: "stability-test"}) is None
                )
    finally:
        runtime_lane_identity._note_non_lab_lane.cache_clear()

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
