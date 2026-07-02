# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""deploy-runtime.sh verify_deployment must probe the lane-scoped runtime container.

OMN-13826: verify_deployment() resolved the runtime container by the HARDCODED dev
name ``omninode-runtime`` -- an anchored ``docker ps --filter name=^/omninode-runtime$``
plus a ``docker logs omninode-runtime`` on the health-fail path. Each lane overlay
prefixes the runtime ``container_name`` with its lane (dev keeps the bare name):

    omnibase-infra                -> omninode-runtime               (dev, no prefix)
    omnibase-infra-stability-test -> omninode-stability-test-runtime
    omnibase-infra-prod           -> omninode-prod-runtime
    omnibase-infra-judge          -> omninode-judge-runtime

The dev ``omninode-runtime`` container commonly runs ALONGSIDE a non-dev lane on the
same host, so the anchored dev-name filter resolved the wrong (dev) container when
deploying e.g. stability-test and emitted a false "Image label mismatch" warning.

The fix factors the lane->container-name mapping into
``resolve_lane_runtime_container_name`` (mirroring ``resolve_lane_overlay_filename``'s
lane derivation + fail-closed whitelist) and routes verify_deployment's container
probes through it.

These tests assert (behaviorally, by extracting + executing the pure helper -- no
docker required):
  - the helper exists,
  - dev still resolves to the bare ``omninode-runtime`` name,
  - each non-dev lane resolves to its lane-prefixed container name,
  - an unknown non-dev project fails CLOSED,
  - verify_deployment routes its container-name probes through the helper (guards
    against a regression that re-hardcodes the dev name).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """Return the source text of a single top-level bash function ``name()``."""
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


def _run_container_name_resolver(
    compose_project: str,
) -> subprocess.CompletedProcess[str]:
    """Execute the extracted resolver for one compose project.

    Stubs ``log_error`` so the fail-closed path does not depend on the script's
    logging helpers, then echoes the resolved container name.
    """
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            _extract_function("resolve_lane_runtime_container_name"),
            f'resolve_lane_runtime_container_name "{compose_project}"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_defines_lane_runtime_container_name_resolver() -> None:
    text = _script_text()
    assert re.search(
        r"^resolve_lane_runtime_container_name\s*\(\)", text, re.MULTILINE
    ), "deploy-runtime.sh must define resolve_lane_runtime_container_name()"


@pytest.mark.unit
def test_dev_project_resolves_bare_runtime_name() -> None:
    """The bare dev project keeps the un-prefixed ``omninode-runtime`` name."""
    result = _run_container_name_resolver("omnibase-infra")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "omninode-runtime"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("compose_project", "expected"),
    [
        ("omnibase-infra-stability-test", "omninode-stability-test-runtime"),
        ("omnibase-infra-prod", "omninode-prod-runtime"),
        ("omnibase-infra-judge", "omninode-judge-runtime"),
    ],
)
def test_non_dev_project_resolves_lane_prefixed_name(
    compose_project: str, expected: str
) -> None:
    """Each non-dev lane resolves to its lane-prefixed runtime container_name."""
    result = _run_container_name_resolver(compose_project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.unit
def test_unknown_non_dev_project_fails_closed() -> None:
    """An unrecognized non-dev project must abort, not resolve the dev name."""
    result = _run_container_name_resolver("omnibase-infra-mystery-lane")
    assert result.returncode != 0, (
        "unknown non-dev compose project must fail closed, got: "
        f"stdout={result.stdout!r} rc={result.returncode}"
    )
    assert "mystery-lane" in result.stderr or "Unknown lane" in result.stderr


@pytest.mark.unit
def test_verify_deployment_routes_probes_through_resolver() -> None:
    """verify_deployment must not re-hardcode the dev ``omninode-runtime`` name.

    It must derive the container name via resolve_lane_runtime_container_name and
    use that derived name in the ``docker ps --filter name=...`` primary probe and
    the ``docker logs`` health-fail path -- never an anchored literal dev name.
    """
    body = _extract_function("verify_deployment")
    assert "resolve_lane_runtime_container_name" in body, (
        "verify_deployment() must derive the runtime container name via "
        "resolve_lane_runtime_container_name"
    )
    assert "${runtime_container_name}" in body, (
        "verify_deployment() must use the derived ${runtime_container_name}"
    )
    # The pre-fix bug: an anchored literal dev-name filter.
    assert "name=^/omninode-runtime$" not in body, (
        "verify_deployment() must not filter on the hardcoded dev container name "
        "'omninode-runtime' (OMN-13826); use the lane-derived name"
    )
    assert "docker logs omninode-runtime" not in body, (
        "verify_deployment() must not `docker logs` the hardcoded dev container "
        "name (OMN-13826); use the lane-derived name"
    )
