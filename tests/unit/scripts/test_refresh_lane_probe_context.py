# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Config-as-data guards for the OMN-14958 probe-context fix.

Live failure being eliminated (deploy run 29977968728, inside the
containerized omninode-deploy-runner): refresh_stability_lane.sh's
health-gate + rollback-gate defaulted MANIFEST_URL/HEALTH_URL to
``http://localhost:18085/...``. Inside the runner container (bridge network)
localhost is the container itself, both probes died ``Connection refused``,
the gate reported INFRA_ERROR, and the script printed "the stability-test
lane may be UNHEALTHY right now" -- a FALSE alarm about a lane it never
observed (host-side readback showed the lane healthy and untouched).

Three legs, each guarded here against silent reversion:

1. Probe host parameterization: both refresh scripts derive their default
   gate URLs from ``LANE_PROBE_HOST`` (default localhost) instead of a
   hardcoded localhost; the runner compose sets it to the host-gateway
   alias, with extra_hosts wiring that alias.
2. STOP-wording honesty: an INFRA_ERROR post-rollback gate (probes could not
   RUN) is reported as "unreachable from this probe context", distinct from
   the observed-unhealthy STOP; the "may be UNHEALTHY" claim is reserved for
   a gate that actually ran and failed.
3. Operator-env provisioning: deploy-runtime.sh and both refresh scripts
   honor OMNIBASE_OPERATOR_ENV_FILE, and the runner compose provisions the
   file read-only at a neutral path (never under /home/runner/.omnibase,
   whose state/ subtree must stay runner-writable for receipts).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
STABILITY_SCRIPT = RUNTIME_BUILD / "refresh_stability_lane.sh"
DEV_SCRIPT = RUNTIME_BUILD / "refresh_dev_lane.sh"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"

LANE_PORTS = {STABILITY_SCRIPT: "18085", DEV_SCRIPT: "8085"}


def _deploy_runner_service() -> dict[str, Any]:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    return compose["services"]["omninode-deploy-runner"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Leg 1: probe host parameterization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("script", [STABILITY_SCRIPT, DEV_SCRIPT], ids=lambda p: p.name)
def test_refresh_scripts_derive_gate_urls_from_lane_probe_host(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    port = LANE_PORTS[script]
    assert re.search(
        r'^LANE_PROBE_HOST="\$\{LANE_PROBE_HOST:-localhost\}" # fallback-ok:',
        text,
        re.MULTILINE,
    ), (
        f"{script.name} must default LANE_PROBE_HOST to localhost (host-side "
        f"runs) with the OMN-10741 fallback-ok justification"
    )
    assert (
        f'MANIFEST_URL="http://${{LANE_PROBE_HOST}}:{port}/v1/introspection/manifest"'
        in text
    ), f"{script.name} MANIFEST_URL default must derive from LANE_PROBE_HOST"
    assert f'HEALTH_URL="http://${{LANE_PROBE_HOST}}:{port}/health"' in text, (
        f"{script.name} HEALTH_URL default must derive from LANE_PROBE_HOST"
    )
    # The regression: a hardcoded-localhost default URL.
    hardcoded = [
        line
        for line in text.splitlines()
        if "http://localhost" in line and not line.lstrip().startswith("#")
    ]
    assert not hardcoded, (
        f"{script.name} still hardcodes a localhost probe URL: {hardcoded} -- "
        f"inside the deploy-runner container localhost is the container itself "
        f"(OMN-14958 false 'lane unhealthy' STOP)"
    )


@pytest.mark.unit
def test_compose_deploy_runner_wires_probe_host_and_gateway() -> None:
    svc = _deploy_runner_service()
    env = svc["environment"]
    assert env.get("LANE_PROBE_HOST") == "host.docker.internal", (
        "deploy runner must set LANE_PROBE_HOST=host.docker.internal so the "
        "refresh scripts' gate probes reach the lane ports published on the host"
    )
    extra_hosts = [str(h) for h in svc.get("extra_hosts", [])]
    assert any(
        h.startswith("host.docker.internal:") and h.endswith("host-gateway")
        for h in extra_hosts
    ), (
        f"deploy runner needs extra_hosts host.docker.internal:host-gateway "
        f"(got {extra_hosts}) -- without it LANE_PROBE_HOST does not resolve"
    )


# ---------------------------------------------------------------------------
# Leg 2: STOP-wording honesty (INFRA_ERROR != observed-unhealthy)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("script", [STABILITY_SCRIPT, DEV_SCRIPT], ids=lambda p: p.name)
def test_stop_wording_distinguishes_unreachable_from_unhealthy(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    # A dedicated INFRA_ERROR branch on the post-rollback gate...
    assert re.search(r'elif \[\[ "\$\{GATE2_OVERALL\}" == "INFRA_ERROR" \]\]', text), (
        f"{script.name} must branch on GATE2_OVERALL == INFRA_ERROR: an "
        f"INFRA_ERROR gate never observed the lane, so the STOP message must "
        f"not claim lane damage (OMN-14958 false alarm)"
    )
    # ...that says the truthful thing...
    assert "UNREACHABLE FROM THIS PROBE CONTEXT" in text, (
        f"{script.name} INFRA_ERROR STOP wording must state the lane was "
        f"unreachable from the probe context, not unhealthy"
    )
    assert "LANE_PROBE_HOST" in text
    # ...and the observed-unhealthy claim is scoped to a gate that ran.
    unhealthy_lines = [
        line for line in text.splitlines() if "may be UNHEALTHY right now" in line
    ]
    assert unhealthy_lines, (
        f"{script.name} must keep the observed-unhealthy STOP for a gate that "
        f"actually ran and failed"
    )
    infra_error_idx = text.index('== "INFRA_ERROR" ]]')
    for line in unhealthy_lines:
        assert text.index(line) > infra_error_idx, (
            f"{script.name}: the 'may be UNHEALTHY' claim must live in the "
            f"post-INFRA_ERROR else-branch (observed-unhealthy), never before "
            f"the INFRA_ERROR distinction"
        )


# ---------------------------------------------------------------------------
# Leg 3: operator env provisioning
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "script",
    [DEPLOY_SCRIPT, STABILITY_SCRIPT, DEV_SCRIPT],
    ids=lambda p: p.name,
)
def test_scripts_honor_parameterized_operator_env(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert re.search(
        r'^OMNIBASE_OPERATOR_ENV_FILE="\$\{OMNIBASE_OPERATOR_ENV_FILE:-\$\{HOME\}/\.omnibase/\.env\}"$',
        text,
        re.MULTILINE,
    ), (
        f"{script.name} must honor OMNIBASE_OPERATOR_ENV_FILE with the "
        f"${{HOME}}/.omnibase/.env default (OMN-14958)"
    )
    bare_sources = [
        line
        for line in text.splitlines()
        if re.search(r'^\s*source\s+"\$\{HOME\}/\.omnibase/\.env"', line)
    ]
    assert not bare_sources, (
        f"{script.name} still sources a hardcoded ${{HOME}}/.omnibase/.env: "
        f"{bare_sources}"
    )


@pytest.mark.unit
def test_compose_deploy_runner_provisions_operator_env_readonly() -> None:
    svc = _deploy_runner_service()
    env = svc["environment"]
    assert env.get("OMNIBASE_OPERATOR_ENV_FILE") == "/run/omnibase-operator.env", (
        "deploy runner must point OMNIBASE_OPERATOR_ENV_FILE at the neutral "
        "read-only mount path"
    )
    volumes = [str(v) for v in svc["volumes"]]
    op_binds = [v for v in volumes if v.endswith(":/run/omnibase-operator.env:ro")]
    assert len(op_binds) == 1, (
        f"expected exactly one read-only operator-env bind, got {op_binds}"
    )
    assert op_binds[0].startswith("${DEPLOY_RUNNER_OPERATOR_ENV_FILE:?"), (
        f"operator-env bind must fail-fast interpolate "
        f"DEPLOY_RUNNER_OPERATOR_ENV_FILE (got {op_binds[0]!r}) -- a silent "
        f"default lets docker create a directory at a bogus host path"
    )
    # Never mounted under /home/runner/.omnibase: docker would create
    # root-owned intermediate dirs and break the runner-writable
    # ~/.omnibase/state receipt tree.
    home_omnibase = [v for v in volumes if "/home/runner/.omnibase" in v]
    assert not home_omnibase, (
        f"operator env must not be mounted under /home/runner/.omnibase: "
        f"{home_omnibase}"
    )
