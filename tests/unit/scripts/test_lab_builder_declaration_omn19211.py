# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dedicated lab builders are declared once and nothing drifts from it (OMN-19211).

dockerd on the lab host stopped three times in 33 days because the cache index of its
embedded BuildKit was damaged. The fix is two declared builders (lane image
builds, CI image builds) with their own state, a cache ceiling in their daemon
config, and a pinned BuildKit that carries the upstream database-open recovery.
CI cannot reach the host, so these tests prove what it can: the declaration is
well-formed, the ceiling is a ceiling and equals the host GC's, and the three
places that name a builder (the declaration, the CI workflow, the deploy-agent
unit) agree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_DECLARATION = _REPO / "deploy" / "buildkit" / "builders.yaml"
_KEEP_LIST = _REPO / "deploy" / "disk-gc" / "keep-list.yaml"
_WORKFLOW = _REPO / ".github" / "workflows" / "docker-build.yml"
_DEV_UNIT = _REPO / "scripts" / "deploy-agent" / "deploy" / "deploy-agent-dev.service"

# The release that recovers from a database that fails to open
# (moby/buildkit#7137) and writes a freelist at graceful shutdown (#7249).
_FIRST_RECOVERING_BUILDKIT = (0, 34, 0)


def _declaration() -> Any:
    return yaml.safe_load(_DECLARATION.read_text())


def _builders() -> dict[str, Any]:
    return _declaration()["builders"]


def _version(image: str) -> tuple[int, ...]:
    tag = image.rpartition(":")[2]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", tag), f"{image} is not pinned to a release"
    return tuple(int(p) for p in tag.lstrip("v").split("."))


def _daemon_config(spec: dict[str, Any]) -> Any:
    return tomllib.loads((_REPO / str(spec["buildkitd_config"])).read_text())


@pytest.mark.unit
class TestDeclaration:
    def test_a_builder_per_workload_each_with_its_own_name(self) -> None:
        builders = _builders()
        assert {"lane", "ci"} <= set(builders)
        names = [b["name"] for b in builders.values()]
        assert len(names) == len(set(names)), (
            "two workloads share a builder, so one's damaged cache stops the other"
        )

    def test_every_builder_is_a_docker_container_builder(self) -> None:
        """The `docker` driver IS dockerd's embedded BuildKit, the exposure removed."""
        for key, spec in _builders().items():
            assert spec["driver"] == "docker-container", key

    def test_every_builder_pins_a_buildkit_that_recovers_from_a_bad_database(
        self,
    ) -> None:
        for key, spec in _builders().items():
            assert _version(str(spec["image"])) >= _FIRST_RECOVERING_BUILDKIT, key

    def test_the_declared_config_file_exists_and_declares_gc(self) -> None:
        for key, spec in _builders().items():
            worker = _daemon_config(spec)["worker"]["oci"]
            assert worker["gc"] is True, key

    def test_engine_floor_is_declared(self) -> None:
        engine = _declaration()["engine"]
        assert engine["min_docker_ce"]


@pytest.mark.unit
class TestCacheCeiling:
    def _rules(self) -> list[dict[str, Any]]:
        return _daemon_config(_builders()["lane"])["worker"]["oci"]["gcpolicy"]

    def test_catch_all_rule_sets_a_ceiling_not_only_a_floor(self) -> None:
        """`reservedSpace` alone is a floor and prunes toward nothing (0B on the lab host)."""
        catch_all = [r for r in self._rules() if r.get("all") is True]
        assert catch_all, "no catch-all rule; unmatched cache is unbounded"
        for rule in catch_all:
            assert "maxUsedSpace" in rule, rule

    def test_the_deprecated_floor_vocabulary_is_absent(self) -> None:
        raw = (_REPO / "deploy" / "buildkit" / "buildkitd.toml").read_text()
        assert "keepBytes" not in raw and "gckeepstorage" not in raw

    def test_builder_ceiling_equals_the_host_gc_ceiling(self) -> None:
        catch_all = next(r for r in self._rules() if r.get("all") is True)
        keep_list = yaml.safe_load(_KEEP_LIST.read_text())
        assert catch_all["maxUsedSpace"] == keep_list["builder_cache_max_size"], (
            "buildkitd.toml has drifted from keep-list.yaml builder_cache_max_size; "
            "change both in the same PR"
        )

    def test_all_builders_share_one_daemon_config(self) -> None:
        configs = {str(b["buildkitd_config"]) for b in _builders().values()}
        assert len(configs) == 1, "two ceilings for one host disk"


@pytest.mark.unit
class TestConsumersAgreeWithTheDeclaration:
    def test_ci_workflow_builds_on_the_declared_ci_builder(self) -> None:
        spec = _builders()["ci"]
        text = _WORKFLOW.read_text()
        assert f"name: {spec['name']}" in text
        assert f"image={spec['image']}" in text
        for key, value in spec["driver_opts"].items():
            assert f"{key}={value}" in text
        assert f"buildkitd-config: {spec['buildkitd_config']}" in text

    def test_dev_unit_names_a_declared_lane_builder(self) -> None:
        match = re.search(
            r"^Environment=DEPLOY_AGENT_LANE_BUILDER=(\S+)$",
            _DEV_UNIT.read_text(),
            re.MULTILINE,
        )
        assert match is not None, "the dev unit does not opt into a dedicated builder"
        assert match.group(1) in _builders()
