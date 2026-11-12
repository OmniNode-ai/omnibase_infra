# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23513: the lane census reads policy-folded services as expected absent.

The dev lane's runtime-effects, projection-api and projection-delegation-writer
services are folded into the main runtime on purpose; the runtime policy file
declares it. The manifest loader reads a lane's service of the policy's compose
project, whose name ends with a folded service, as ``kind: profile_gated``
(absent is clean, running is ``profile_gated_present``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load("lane_census_plan", _ROOT / "scripts" / "lane_census_plan.py")
FOLD = PLAN._census_model("model_lane_census_fold_policy")


_LAB_HOST = "lab-01"
_SHIPPED_POLICY = _ROOT / "docker" / "runtime-policy.env"


def _registry() -> Any:
    """A minimal registry mapping the lab host to the lanes under test."""

    class _Registry:
        def lanes_of(self, host_id: str) -> set[str]:
            return {"dev", "other"}

        def hosts_of(self, lane_name: str) -> set[str]:
            return {_LAB_HOST}

        def validate(self, lanes: Any, ambiguous: Any) -> None:
            """The census host-declaration check; a lab registry is always valid."""
            return

        def resolve(self, raw: str) -> str:
            """Resolve a hostname, daemon name or host id to exactly one host id."""
            return _LAB_HOST

    return _Registry()


REGISTRY = _registry()


def _container(
    name: str,
    *,
    lane: str,
    state: str = "running",
    status: str = "Up 3 hours",
    image: str = "omninode-runtime:0.37.0",
) -> dict[str, Any]:
    return {
        "Names": name,
        "State": state,
        "Status": status,
        "Image": image,
        "RestartCount": 0,
        "Labels": f"com.omninode.lane={lane},com.omninode.layer=runtime",
    }


def _kinds(findings: list[dict[str, str]]) -> set[str]:
    return {f["kind"] for f in findings}


def _write_policy(path: Path, *, project: str, profiles: str, foldable: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# generated test policy",
                f"DEV_COMPOSE_PROJECT={project}",
                f"DEV_RUNTIME_MAIN_FOLDED_PROFILES={profiles}",
                f"DEV_RUNTIME_MAIN_FOLDABLE_SERVICES={foldable}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _dev_manifest() -> dict[str, Any]:
    return {
        "default_image_tag_pattern": ".+",
        "lanes": {
            "dev": {
                "compose_project": "omnibase-infra",
                "network": "omnibase-infra-dev-network",
                "services": [
                    {"name": "omnibase-infra-postgres", "kind": "service"},
                    {"name": "omninode-runtime", "kind": "service"},
                    {"name": "omninode-runtime-effects", "kind": "service"},
                    {"name": "omnimarket-projection-api", "kind": "service"},
                    {
                        "name": "omnimarket-projection-delegation-writer",
                        "kind": "service",
                    },
                ],
            },
            "other": {
                "compose_project": "other-project",
                "network": "omnibase-infra-other-network",
                "services": [
                    {"name": "omnibase-infra-postgres", "kind": "service"},
                    {"name": "omninode-runtime", "kind": "service"},
                    {"name": "omninode-runtime-effects", "kind": "service"},
                ],
            },
        },
    }


def _dev_envelope(containers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lane": "dev",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-dev-network"],
        "runtime_tag": None,
    }


def _other_envelope(containers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lane": "other",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-other-network"],
        "runtime_tag": None,
    }


def _folded_policy(tmp_path: Path) -> Path:
    path = tmp_path / "runtime-policy.env"
    _write_policy(
        path,
        project="omnibase-infra",
        profiles="effects,projection-api,projection-writer-delegation",
        foldable=json.dumps(
            {
                "effects": "runtime-effects",
                "projection-api": "projection-api",
                "projection-writer-delegation": "projection-delegation-writer",
            }
        ),
    )
    return path


def _empty_policy(tmp_path: Path) -> Path:
    path = tmp_path / "runtime-policy.env"
    _write_policy(
        path,
        project="omnibase-infra",
        profiles="",
        foldable=json.dumps({}),
    )
    return path


def _plan(
    envelope: dict[str, Any],
    manifest: dict[str, Any],
    policy: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    """The manifest read through the loader with ``policy`` as the runtime policy."""
    manifest_path = tmp_path / "lane-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setenv(FOLD.ENV_RUNTIME_POLICY, str(policy))
    return PLAN.build_plan(envelope, PLAN.load_manifest(manifest_path), REGISTRY)


def test_shipped_policy_golden() -> None:
    """a. the repo's shipped policy folds exactly the three dev services."""
    project, names = FOLD.folded_service_names(_SHIPPED_POLICY)
    assert project == "omnibase-infra"
    expected = {"runtime-effects", "projection-api", "projection-delegation-writer"}
    assert expected <= names, f"missing folded services: {expected - names}"


def test_folded_services_absent_is_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """b. folded services that are absent yield NO container_absent findings."""
    policy = _folded_policy(tmp_path)
    manifest = _dev_manifest()
    containers = [
        _container("omnibase-infra-postgres", lane="dev"),
        _container("omninode-runtime", lane="dev"),
    ]
    plan = _plan(_dev_envelope(containers), manifest, policy, monkeypatch, tmp_path)
    assert "container_absent" not in _kinds(plan["findings"]), plan["findings"]


def test_empty_policy_folded_services_are_absent_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """c. the same inventory with a policy that folds nothing is the control."""
    policy = _empty_policy(tmp_path)
    manifest = _dev_manifest()
    containers = [
        _container("omnibase-infra-postgres", lane="dev"),
        _container("omninode-runtime", lane="dev"),
    ]
    plan = _plan(_dev_envelope(containers), manifest, policy, monkeypatch, tmp_path)
    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    assert {
        "omninode-runtime-effects",
        "omnimarket-projection-api",
        "omnimarket-projection-delegation-writer",
    } <= absent, plan["findings"]


def test_folded_service_running_is_profile_gated_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """d. a running folded container is a profile_gated_present warning."""
    policy = _folded_policy(tmp_path)
    manifest = _dev_manifest()
    containers = [
        _container("omnibase-infra-postgres", lane="dev"),
        _container("omninode-runtime", lane="dev"),
        _container("omninode-runtime-effects", lane="dev"),
    ]
    plan = _plan(_dev_envelope(containers), manifest, policy, monkeypatch, tmp_path)
    present = [
        f
        for f in plan["findings"]
        if f["kind"] == "profile_gated_present"
        and f["container"] == "omninode-runtime-effects"
    ]
    assert present and present[0]["severity"] == "warning", plan["findings"]
    assert "unexpected_container" not in _kinds(plan["findings"]), plan["findings"]


def test_different_compose_project_keeps_absent_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """e. a lane whose compose_project differs keeps container_absent."""
    policy = _folded_policy(tmp_path)
    manifest = _dev_manifest()
    containers = [
        _container("omnibase-infra-postgres", lane="other"),
        _container("omninode-runtime", lane="other"),
    ]
    plan = _plan(_other_envelope(containers), manifest, policy, monkeypatch, tmp_path)
    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    assert "omninode-runtime-effects" in absent, plan["findings"]


def test_malformed_foldable_services_raises(tmp_path: Path) -> None:
    """f. a malformed FOLDABLE_SERVICES value raises LaneCensusConfigError."""
    path = tmp_path / "runtime-policy.env"
    _write_policy(
        path,
        project="omnibase-infra",
        profiles="effects",
        foldable="not-json",
    )
    with pytest.raises(ValueError):
        FOLD.folded_service_names(path)


def test_missing_policy_folds_nothing(tmp_path: Path) -> None:
    """f. a missing policy file folds nothing."""
    project, names = FOLD.folded_service_names(tmp_path / "absent.env")
    assert project == ""
    assert names == frozenset()


def test_foldable_but_not_folded_service_stays_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foldable service whose profile is not in the folded list stays required."""
    path = tmp_path / "runtime-policy.env"
    _write_policy(
        path,
        project="omnibase-infra",
        profiles="effects",
        foldable=json.dumps(
            {"effects": "runtime-effects", "projection-api": "projection-api"}
        ),
    )
    containers = [
        _container("omnibase-infra-postgres", lane="dev"),
        _container("omninode-runtime", lane="dev"),
    ]
    plan = _plan(
        _dev_envelope(containers), _dev_manifest(), path, monkeypatch, tmp_path
    )
    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    assert "omnimarket-projection-api" in absent, plan["findings"]
    assert "omninode-runtime-effects" not in absent, plan["findings"]


def test_folded_profile_missing_from_foldable_raises(tmp_path: Path) -> None:
    """A folded profile the policy cannot map to a service is a config error."""
    path = tmp_path / "runtime-policy.env"
    _write_policy(
        path,
        project="omnibase-infra",
        profiles="effects,nonesuch",
        foldable=json.dumps({"effects": "runtime-effects"}),
    )
    with pytest.raises(ValueError, match="nonesuch"):
        FOLD.folded_service_names(path)


def test_main_path_reads_the_policy_through_the_manifest_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader the collector's main() calls folds; an unfolded copy is untouched."""
    manifest_path = tmp_path / "lane-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_dev_manifest()), encoding="utf-8")
    monkeypatch.setenv(FOLD.ENV_RUNTIME_POLICY, str(_folded_policy(tmp_path)))
    loaded = PLAN.load_manifest(manifest_path)
    kinds = {s["name"]: s["kind"] for s in loaded["lanes"]["dev"]["services"]}
    assert kinds["omninode-runtime-effects"] == "profile_gated"
    assert kinds["omninode-runtime"] == "service"
    assert (
        yaml.safe_load(manifest_path.read_text())["lanes"]["dev"]["services"][2]["kind"]
        == "service"
    )
