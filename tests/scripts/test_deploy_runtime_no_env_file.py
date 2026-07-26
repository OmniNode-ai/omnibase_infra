# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard: deploy-runtime.sh must not use --env-file.

F65 / OMN-6910: The old setup_env() approach copied ~/.omnibase/.env into a
stale snapshot and then passed --env-file to docker compose. This caused env
var changes to be silently ignored until the next full redeploy.

The fix sources ~/.omnibase/.env at script top and lets docker compose
resolve ${VAR} from the shell environment directly -- no --env-file needed.

These tests ensure the anti-pattern is never reintroduced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"


def _read_script_lines() -> list[str]:
    """Read deploy-runtime.sh, stripping comment-only lines."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


@pytest.mark.unit
def test_no_env_file_flag_in_active_code() -> None:
    """deploy-runtime.sh must not pass --env-file to docker compose."""
    lines = _read_script_lines()
    violations = [(i + 1, line) for i, line in enumerate(lines) if "--env-file" in line]
    assert violations == [], (
        f"Found --env-file in non-comment lines of deploy-runtime.sh: {violations}"
    )


@pytest.mark.unit
def test_no_env_file_args_variable() -> None:
    """deploy-runtime.sh must not declare env_file_args arrays."""
    lines = _read_script_lines()
    violations = [
        (i + 1, line) for i, line in enumerate(lines) if "env_file_args" in line
    ]
    assert violations == [], (
        f"Found env_file_args in non-comment lines of deploy-runtime.sh: {violations}"
    )


@pytest.mark.unit
def test_no_setup_env_function() -> None:
    """deploy-runtime.sh must not define a setup_env() function."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    # Match function definition: setup_env() { (with possible whitespace)
    assert not re.search(r"^setup_env\s*\(\)", text, re.MULTILINE), (
        "setup_env() function definition found in deploy-runtime.sh -- must not exist (F65)"
    )


@pytest.mark.unit
def test_sources_operator_env_at_top() -> None:
    """deploy-runtime.sh must source the operator env early in the script.

    OMN-14958: the path is parameterized via OMNIBASE_OPERATOR_ENV_FILE
    (default ${HOME}/.omnibase/.env) so the containerized deploy runner can
    point at its provisioned read-only mount, and a MISSING file must be a
    NAMED precondition failure -- not bash's bare `source` crash (which
    killed deploy run 29977968728 with no actionable error).
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    # Parameterized default preserved (~/.omnibase/.env stays the default).
    default_match = re.search(
        r'^OMNIBASE_OPERATOR_ENV_FILE="\$\{OMNIBASE_OPERATOR_ENV_FILE:-\$\{HOME\}/\.omnibase/\.env\}"$',
        text,
        re.MULTILINE,
    )
    assert default_match is not None, (
        "deploy-runtime.sh must default OMNIBASE_OPERATOR_ENV_FILE to "
        "${HOME}/.omnibase/.env (OMN-14958 parameterization)"
    )
    # The parameterized file is what gets sourced (never a hardcoded $HOME path).
    match = re.search(
        r'^source\s+"\$\{OMNIBASE_OPERATOR_ENV_FILE\}"$',
        text,
        re.MULTILINE,
    )
    assert match is not None, (
        "deploy-runtime.sh must source ${OMNIBASE_OPERATOR_ENV_FILE} "
        "(OMN-14958: parameterized operator env sourcing)"
    )
    # Ensure the sourcing still happens early (before any deploy logic).
    # OMN-14999: threshold bumped 80 -> 100. OMN-14984's named-error guard
    # blocks (OPERATOR_ENV_MISSING / OPERATOR_ENV_UNREADABLE) pushed the real
    # source line to 82 -- still immediately after set -euo pipefail and the
    # two precondition checks, before any deploy logic, matching this test's
    # actual intent. 100 gives headroom for the next named-guard addition
    # without another line-count chase.
    line_number = text[: match.start()].count("\n") + 1
    assert line_number <= 100, (
        f"source ${{OMNIBASE_OPERATOR_ENV_FILE}} found at line {line_number}, "
        "expected within first 100 lines of script"
    )
    # Named, fail-closed guard for the missing-file case.
    assert "OPERATOR_ENV_MISSING" in text, (
        "deploy-runtime.sh must emit the named OPERATOR_ENV_MISSING error "
        "(exit 64) when the operator env file is absent -- a bare `source` "
        "crash is the OMN-14958 regression"
    )


@pytest.mark.unit
def test_omnibase_env_cannot_override_runtime_health_url() -> None:
    """A stale ~/.omnibase/.env HEALTH_CHECK_URL must not redirect deploy verify."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'OPERATOR_HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-}"' in text
    assert 'export HEALTH_CHECK_URL="${OPERATOR_HEALTH_CHECK_URL}"' in text
    assert "unset HEALTH_CHECK_URL" in text
    assert (
        'readonly HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://${INFRA_HOST:?INFRA_HOST required}:8085/health}"'
        in text
    )


@pytest.mark.unit
def test_compose_project_defaults_to_live_runtime_project() -> None:
    """deploy-runtime.sh must target the canonical live runtime compose project."""
    text = "\n".join(_read_script_lines())

    assert "OMNIBASE_INFRA_COMPOSE_PROJECT:-omnibase-infra" in text
    assert 'local compose_project="omnibase-infra-${COMPOSE_PROFILE}"' not in text


@pytest.mark.unit
def test_verify_deployment_uses_lane_scoped_runtime_container_name() -> None:
    """deploy-runtime.sh must inspect the LANE-SCOPED runtime container by name.

    OMN-13826: verify_deployment previously anchored on the hardcoded dev name
    ``name=^/omninode-runtime$``. Because the dev ``omninode-runtime`` container
    commonly runs alongside a non-dev lane, that anchored filter resolved the wrong
    (dev) container on a non-dev deploy and emitted a false image-label mismatch.
    The name probe must now use the lane-derived ``${runtime_container_name}``
    (per-lane derivation asserted in test_deploy_runtime_verify_container_name.py).

    It must still NOT use the compose-default ``<project>-<service>-1`` naming.
    """
    text = "\n".join(_read_script_lines())

    assert "name=^/${runtime_container_name}$" in text
    assert "name=^/omninode-runtime$" not in text
    assert "${compose_project}-omninode-runtime-1" not in text
    assert '${compose_project}-omninode-runtime" | head -1' not in text
