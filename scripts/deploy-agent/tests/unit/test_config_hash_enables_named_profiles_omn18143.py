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

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.compose_budget import _read_compose_services
from deploy_agent.events import DEV_LANE_ONLY_RUNTIME_SERVICES, EnumRuntimeLane
from deploy_agent.executor import DeployExecutor

DEV_LANE_OVERLAY = (
    Path(__file__).resolve().parents[4] / "docker" / "docker-compose.dev-lane.yml"
)
HASH = "a" * 64


def _enabled_profiles(argv: list[str]) -> set[str]:
    return {argv[i + 1] for i, token in enumerate(argv) if token == "--profile"}


def _render(services: list[str], *, profile: str | None) -> list[str]:
    seen: list[list[str]] = []

    def run(
        cmd: list[str], timeout: int, **_: object
    ) -> subprocess.CompletedProcess[str]:
        seen.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 0, "\n".join(f"{s} {HASH}" for s in services), ""
        )

    with (
        patch("deploy_agent.executor._run", side_effect=run),
        patch("deploy_agent.executor._compose_env", return_value={}),
    ):
        DeployExecutor()._rendered_config_hashes(
            EnumRuntimeLane.DEV, services, profile=profile
        )
    assert len(seen) == 1
    return seen[0]


def _profile_gated_named_services() -> dict[str, list[str]]:
    """Agent-named dev-lane services whose compose profile excludes ``runtime``."""
    declared = _read_compose_services([str(DEV_LANE_OVERLAY)])
    return {
        name: declared[name]["profiles"]
        for name in DEV_LANE_ONLY_RUNTIME_SERVICES
        if name in declared
        and declared[name].get("profiles")
        and "runtime" not in declared[name]["profiles"]
    }


def test_the_compose_model_has_a_named_service_outside_the_runtime_profile() -> None:
    """Positive control: the hazard exists in the model, so the tests below bite."""
    assert "cost-estimator-web" in _profile_gated_named_services()


@pytest.mark.parametrize("profile", ["runtime", None])
def test_a_named_profile_gated_service_has_its_profile_enabled(
    profile: str | None,
) -> None:
    gated = _profile_gated_named_services()
    names = ["omninode-runtime", *gated]
    enabled = _enabled_profiles(_render(names, profile=profile))
    for service, profiles in gated.items():
        assert "*" in enabled or set(profiles) <= enabled, (
            f"`compose config --hash {service}` under profiles {sorted(enabled)} "
            f"cannot resolve {service}, whose profiles are {profiles}"
        )


def test_the_everything_in_one_profile_form_keeps_exactly_that_profile() -> None:
    argv = _render(["*"], profile="runtime")
    assert _enabled_profiles(argv) == {"runtime"}
    assert argv[argv.index("--hash") + 1] == "*"
