# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A rendered-hash read must enable the profile of every service it names (OMN-18143).

Dev redeploy job da6bc43b failed with ``no such service: cost-estimator-web:
disabled``. OMN-21481 made ``cost-estimator-web`` a member of the agent's named
dev-lane services and gave it its OWN compose profile, ``cost-estimator``,
because the other dev instances layer the overlay and bring up
``--profile runtime``. The compose ``up`` command enables the profile of a
service named on its command line, and the runtime phase relies on that. The
compose ``config --hash <service>`` command does NOT: it resolves the name
against the active profiles only, so the convergence check after a runtime
``up`` (and the dev boot guard's hash read), which name every service of the
scope under ``--profile runtime``, aborted on the one service whose profile is
not ``runtime``.

Reproduced on a lab host, read-only, under a throwaway project name with the
dev-lane file set (compose v5.6.1)::

    --profile runtime  config --hash cost-estimator-web  -> no such service
    --profile cost-estimator config --hash cost-estimator-web -> <hash>
    --profile runtime --profile '*' config --hash cost-estimator-web,omninode-runtime
                                                         -> both hashes
    omninode-runtime's hash is identical with and without the wildcard.

So the fix names no profile by hand: an explicit service list is rendered with
every profile enabled, and the ``["*"]`` form (everything one profile would
start) keeps exactly the requested profile.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from deploy_agent.compose_budget import _read_compose_services
from deploy_agent.events import DEV_LANE_ONLY_RUNTIME_SERVICES, EnumRuntimeLane
from deploy_agent.executor import DeployExecutor

DEV_LANE_OVERLAY = (
    Path(__file__).resolve().parents[4] / "docker" / "docker-compose.dev-lane.yml"
)


def _render(
    services: list[str], *, profile: str | None, tmp_path: Path
) -> dict[str, str]:
    """Call real compose config using this repository's gated service profiles."""
    gated = _profile_gated_named_services()
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "omninode-runtime": {
                        "image": "busybox:latest",
                        "profiles": ["runtime"],
                    },
                    **{
                        name: {"image": "busybox:latest", "profiles": profiles}
                        for name, profiles in gated.items()
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "COMPOSE_PROFILES": ""}
    with (
        patch(
            "deploy_agent.executor._compose_file_args",
            return_value=["-f", str(compose)],
        ),
        patch(
            "deploy_agent.executor.lane_config_for",
            return_value=SimpleNamespace(compose_project="landing-l22913-lab"),
        ),
        patch("deploy_agent.executor._compose_env", return_value=env),
    ):
        return DeployExecutor()._rendered_config_hashes(
            EnumRuntimeLane.DEV, services, profile=profile
        )


def _profile_gated_named_services() -> dict[str, list[str]]:
    """Agent-named dev-lane services whose compose profile excludes ``runtime``."""
    declared = _read_compose_services([str(DEV_LANE_OVERLAY)])

    def effective(profiles: list[str]) -> list[str]:
        # A folded writer's profile is ``${VAR:-runtime}``: compose renders the default.
        return [re.sub(r"^\$\{[^:}]+:-([^}]*)\}$", r"\1", item) for item in profiles]

    return {
        name: effective(declared[name]["profiles"])
        for name in DEV_LANE_ONLY_RUNTIME_SERVICES
        if name in declared
        and declared[name].get("profiles")
        and "runtime" not in effective(declared[name]["profiles"])
    }


def test_the_compose_model_has_a_named_service_outside_the_runtime_profile() -> None:
    """Positive control: the hazard exists in the model, so the tests below bite."""
    assert "cost-estimator-web" in _profile_gated_named_services()


@pytest.mark.parametrize("profile", ["runtime", None])
def test_a_named_profile_gated_service_has_its_profile_enabled(
    tmp_path: Path,
    profile: str | None,
) -> None:
    gated = _profile_gated_named_services()
    names = ["omninode-runtime", *gated]
    hashes = _render(names, profile=profile, tmp_path=tmp_path)
    assert set(hashes) == set(names)
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in hashes.values())


def test_the_everything_in_one_profile_form_keeps_exactly_that_profile(
    tmp_path: Path,
) -> None:
    hashes = _render(["*"], profile="runtime", tmp_path=tmp_path)
    assert set(hashes) == {"omninode-runtime"}


@pytest.mark.parametrize("profile", ["runtime", None])
def test_real_compose_resolves_named_services_and_preserves_runtime_hash(
    tmp_path: Path,
    profile: str | None,
) -> None:
    """Profile activation must preserve the runtime service's rendered hash."""
    gated = _profile_gated_named_services()
    assert "cost-estimator-web" in gated
    runtime_only = _render(["*"], profile="runtime", tmp_path=tmp_path)
    named = _render(["omninode-runtime", *gated], profile=profile, tmp_path=tmp_path)
    assert set(runtime_only) == {"omninode-runtime"}
    assert set(named) == {"omninode-runtime", *gated}
    assert named["omninode-runtime"] == runtime_only["omninode-runtime"]
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in named.values())
