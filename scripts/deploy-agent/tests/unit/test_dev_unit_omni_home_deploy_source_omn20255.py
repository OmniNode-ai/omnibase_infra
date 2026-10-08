# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The .201 dev unit stages workspace builds from the deploy-source tree (OMN-20255).

A workspace build's RT-1 step (``stage_workspace.sh`` -> ``deploy_source_ref.py``)
runs ``checkout --force``, ``reset --hard`` and ``clean -ffdx`` on every vendored
sibling under ``OMNI_HOME``. The dev unit took ``OMNI_HOME`` from the operator env
store, which names ``/data/omninode/omni_home``: the lanes' canonical tree. Every
dev-lane job then detached omnimarket, omnibase_core and omnibase_compat there
(2026-10-01, jobs 3ef8ac17, 7120389d and cc5545bb, each followed about 13 seconds
later by a detach in that clone's reflog), and the clone guard refused every
in-process delegate on the host while they stayed detached.

The deploy-source root is the tree reconcile-host converges, and the OMN-20154
build/reconcile lock is ``<OMNI_HOME>/.onex-reconcile-host.lock`` on both sides,
so the two must name the same root or the interlock is inert. The unit declares
that root itself and protects the name from the env store.
"""

import re
from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_DEV_UNIT = _DEPLOY_DIR / "deploy-agent-dev.service"
_LAUNCHER = _DEPLOY_DIR / "deploy-agent-launch.sh"
_LOCK = Path(__file__).resolve().parents[2] / "deploy_agent" / "reconcile_host_lock.py"
_RECONCILE_HOST = (
    Path(__file__).resolve().parents[4]
    / "deploy/maintenance/omninode-workspace-reconcile.sh"
)

_DEPLOY_SOURCE_ROOT = "/data/omninode"
_LANES_TREE = "/data/omninode/omni_home"

_ENVIRONMENT_RE = re.compile(r'^Environment="?([A-Za-z_][A-Za-z0-9_]*)=(.*?)"?$')


def _env(unit: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in unit.read_text().splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        match = _ENVIRONMENT_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


@pytest.mark.unit
def test_the_launcher_sources_an_env_store_that_could_override_omni_home() -> None:
    """Positive control: protection is needed only because the store is sourced."""
    launcher = _LAUNCHER.read_text()
    assert '. "$DEPLOY_AGENT_ENV_FILE"' in launcher
    assert "DEPLOY_AGENT_ENV_PROTECTED" in launcher


@pytest.mark.unit
def test_the_build_lock_and_reconcile_host_lock_share_one_root_name() -> None:
    """Both sides derive the lock from OMNI_HOME, so the roots must agree."""
    assert '".onex-reconcile-host.lock"' in _LOCK.read_text()
    assert (
        'exec env OMNI_HOME="$OMNI_HOME" "$RECONCILER"' in _RECONCILE_HOST.read_text()
    )


@pytest.mark.unit
def test_omni_home_deploy_source_is_declared_by_the_dev_unit() -> None:
    env = _env(_DEV_UNIT)
    assert env.get("OMNI_HOME") == _DEPLOY_SOURCE_ROOT, env.get("OMNI_HOME")
    assert env["OMNI_HOME"] != _LANES_TREE


@pytest.mark.unit
def test_omni_home_deploy_source_is_protected_from_the_env_store() -> None:
    protected = set(_env(_DEV_UNIT)["DEPLOY_AGENT_ENV_PROTECTED"].split())
    assert "OMNI_HOME" in protected
