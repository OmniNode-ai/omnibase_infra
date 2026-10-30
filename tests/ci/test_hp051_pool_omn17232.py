# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17232: the HP DL560 Gen8 pool is arch-neutral CI on an opt-in label, with its own slice and git mirror.

Operator RULING 2026-10-29T14:00:26Z: make before break. The pool carries the opt-in label
``omnibase-ci-multiarch`` and never ``omnibase-ci`` (this CPU has no AVX2, so a job that imports an AVX2 wheel must
stay on the ``omnibase-ci`` pools), and its class ``action-multiarch`` is the one no ``omnibase-ci`` spread counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "hp051-ci-runner"
COMPOSE = REPO_ROOT / "docker" / f"docker-compose.runners-{PREFIX}.yml"
POOL_DIR = REPO_ROOT / "docker" / "runners" / "systemd" / "pools" / PREFIX
SLICE = POOL_DIR / "omnirunners.slice"
MIRROR_DAEMON = POOL_DIR / "omninode-git-mirror-daemon.service"
MIRROR_REFRESH = POOL_DIR / "omninode-git-mirror-refresh.service"
MIRROR_TIMER = POOL_DIR / "omninode-git-mirror-refresh.timer"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
OPT_IN_LABEL = "omnibase-ci-multiarch"
# The labels that put a runner in a required-context pool. The opt-in label is the only one this pool carries.
REQUIRED_POOL_LABELS = {"omnibase-ci", "omnibase-verify"}
MIRROR_ROOT = "/data/scratch/git-mirror"


def _compose() -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(COMPOSE.read_text(encoding="utf-8")))


def test_every_runner_carries_the_opt_in_label_and_no_required_pool_label() -> None:
    services = _compose()["services"]
    assert len(services) == 4
    for name, service in services.items():
        labels = set(str(service["environment"]["RUNNER_LABELS"]).split(","))
        assert OPT_IN_LABEL in labels, f"{name} lacks the opt-in label"
        assert not labels & REQUIRED_POOL_LABELS, f"{name} would join a required pool"
        assert "arch-amd64" in labels


def test_the_pool_is_its_own_cgroup_slice_with_declared_limits() -> None:
    services = _compose()["services"]
    assert {s["cgroup_parent"] for s in services.values()} == {"omnirunners.slice"}
    text = SLICE.read_text(encoding="utf-8")
    for key in ("MemoryHigh", "MemoryMax", "MemorySwapMax", "CPUWeight"):
        assert f"{key}=" in text, f"the slice must declare {key}"


def test_the_fleet_declares_the_pool_with_the_multiarch_class_only() -> None:
    document = yaml.safe_load(FLEET.read_text(encoding="utf-8"))
    rows = [row for row in document["hosts"] if row["host"] == "h051"]
    assert len(rows) == 1
    assert rows[0]["runner_name_prefix"] == PREFIX
    assert rows[0]["expected_count"] == 4
    assert rows[0]["arch"] == "amd64"
    assert rows[0]["classes"] == ["action-multiarch"]


def test_the_runner_names_match_the_declared_count() -> None:
    names = sorted(_compose()["services"])
    assert names == [f"{PREFIX}-{n}" for n in range(1, 5)]


def test_the_pool_has_its_own_git_mirror_on_the_data_volume() -> None:
    """The pre-seed reads its default address; the three units serve it from the data volume."""
    daemon = MIRROR_DAEMON.read_text(encoding="utf-8")
    # The address is host configuration: a required file, so the unit refuses to start without it.
    assert "EnvironmentFile=/etc/default/omninode-git-mirror\n" in daemon
    assert "--listen=${OMNI_GIT_MIRROR_LISTEN}" in daemon
    assert "--port=9418" in daemon
    assert f"--base-path={MIRROR_ROOT}" in daemon
    # Read-only exposure: the daemon is never given receive-pack, and listens on no wildcard address.
    assert "--enable=receive-pack" not in daemon
    assert "--listen=0.0.0.0" not in daemon
    refresh = MIRROR_REFRESH.read_text(encoding="utf-8")
    assert f"Environment=OMNI_GIT_MIRROR_ROOT={MIRROR_ROOT}" in refresh
    assert "Type=oneshot" in refresh
    timer = MIRROR_TIMER.read_text(encoding="utf-8")
    assert "Unit=omninode-git-mirror-refresh.service" in timer


def test_no_runner_overrides_the_mirror_address() -> None:
    """The default gateway is right on this host; an override would silently point the pre-seed elsewhere."""
    for name, service in _compose()["services"].items():
        env = service["environment"]
        assert "OMNI_GIT_MIRROR_HOST" not in env, name
        assert "OMNI_GIT_MIRROR_DISABLE" not in env, name
