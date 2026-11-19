# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime carries the product clones its Done guard reads (OMN-25709).

THE RECORDED REFUSAL
--------------------
A Done write through ``onex ticket transition <id> --state Done --bus-lane dev``
reaches ``dod_verify`` inside the dev runtime and is refused: the verdict is
``CONTRACT_MISSING`` with ``product_clones=[]``
(``fixtures/dev_runtime_scope_clones_refusal_omn25709.json``). The verifier finds a
repository's contract only through its canonical clone, ``$OMNI_HOME/<name>`` with a
``.git`` inside, and the runtime image sets ``OMNI_HOME=/app`` with no clone under it.

THE DECLARED SOURCE
-------------------
The dev lane already mounts the host's canonical clones read-only where a runtime
reads them (the omniclaude clone at ``/etc/onex/ticket-guard``, the overlays clone at
``/etc/onex/skill-overlays-store``), each taken from the registry-root layout the
canonical-clone sync keeps current, relative to the ``docker/`` directory. These
checks hold the same declaration for the scope clones: each product repository
mounted read-only at ``/app/<name>``, in the two services that host the ticket node's
Done guard (``runtime-effects`` on an unfolded dev instance, ``omninode-runtime`` on
primary dev where the effects role is folded into main). No credential, no new
operator variable and no host file is involved.

The readers live in omnimarket, which this repository does not install, so these
checks cover the committed declarations. The running container is read on the lab.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / ("dev_runtime_scope_clones_refusal_omn25709.json")
)
DEV_LANE = DOCKER / "docker-compose.dev-lane.yml"
FOLD = DOCKER / "docker-compose.dev-tenant-fold.yml"

#: Where each Done-guard host declares its mounts.
HOSTS: tuple[tuple[Path, str], ...] = (
    (DEV_LANE, "runtime-effects"),
    (FOLD, "omninode-runtime"),
)


def _fixture() -> dict[str, object]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _repositories() -> list[str]:
    repos = _fixture()["governed_repositories"]
    assert isinstance(repos, list) and repos
    return [str(name) for name in repos]


def _omni_home() -> str:
    runtime = _fixture()["runtime"]
    assert isinstance(runtime, dict)
    return str(runtime["omni_home_in_container"])


def _volumes(path: Path, service: str) -> list[str]:
    services = load_compose(path)["services"]
    volumes = services[service].get("volumes", [])
    assert isinstance(volumes, list)
    return [str(v) for v in volumes]


def _clone_mounts(path: Path, service: str) -> dict[str, str]:
    """Container target -> ``source:mode`` for every bind of a host directory directly under OMNI_HOME.

    Named volumes (``effects_data:/app/effects-data``) have no path in their source and are
    not clones.
    """
    mounts: dict[str, str] = {}
    for entry in _volumes(path, service):
        source, _, rest = entry.partition(":")
        target, _, mode = rest.partition(":")
        directly_under_home = target.startswith(f"{_omni_home()}/") and (
            target.count("/") == 2
        )
        if directly_under_home and source.startswith("."):
            mounts[target] = f"{source}:{mode}"
    return mounts


def test_scope_clones_recorded_refusal_names_the_empty_clone_set() -> None:
    """Positive control: the fixture is the refusal the change answers."""
    refusals = _fixture()["refusals"]
    assert isinstance(refusals, list)
    verdicts = [r for r in refusals if r.get("verdict") == "CONTRACT_MISSING"]
    assert len(verdicts) == 1
    assert verdicts[0]["product_clones"] == []
    reasons = {r.get("reason") for r in refusals}
    assert "done_write_refused" in reasons


def test_scope_clones_runtime_image_reads_clones_under_the_fixture_omni_home() -> None:
    dockerfile = (DOCKER / "Dockerfile.runtime").read_text(encoding="utf-8")
    assert re.search(rf"^\s*OMNI_HOME={re.escape(_omni_home())}\s", dockerfile, re.M)


@pytest.mark.parametrize(("path", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_scope_clones_every_product_repository_is_mounted_read_only(
    path: Path, service: str
) -> None:
    mounts = _clone_mounts(path, service)
    expected = {
        f"{_omni_home()}/{name}": f"../../{name}:ro" for name in _repositories()
    }
    assert mounts == expected


@pytest.mark.parametrize(("path", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_scope_clones_mount_sources_are_registry_root_siblings_not_variables(
    path: Path, service: str
) -> None:
    """Derived from the layout the canonical-clone sync keeps current, as the overlays store is."""
    mounts = _clone_mounts(path, service)
    assert mounts
    for source_mode in mounts.values():
        source = source_mode.rsplit(":", 1)[0]
        assert source.startswith("../../")
        assert "$" not in source


def test_scope_clones_the_onex_change_control_clone_is_never_mounted() -> None:
    """The verifier skips the governance repository; mounting it would be a dead weight."""
    assert "onex_change_control" not in _repositories()
    for path, service in HOSTS:
        assert not any("onex_change_control" in e for e in _volumes(path, service))


def test_scope_clones_no_other_dev_lane_service_gains_a_clone_mount() -> None:
    for path, holder in HOSTS:
        services = load_compose(path)["services"]
        holders = sorted(
            name
            for name, service in services.items()
            if isinstance(service, dict) and _clone_mounts(path, name)
        )
        assert holders == [holder]
