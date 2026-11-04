# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The neutral lane census tests read (OMN-20934).

The lab's lane manifest and committed census are private deployment config
(operator RULING 2026-10-10T21:57:38Z). They live in ``config/lab/lane-census``
under the workspace config root, and this repository ships neither, so a test
that exercises the census tools reads the neutral pair under
``tests/fixtures/lane_census`` instead. That directory mirrors a config root:
point ``ONEX_WORKSPACE_CONFIG_ROOT`` at :data:`NEUTRAL_CONFIG_ROOT` and every
census reader resolves the neutral files exactly as it resolves the real ones.

Assertions about which lanes and hosts the lab declares are not made here;
they describe the lab, so they are checked where the lab's manifest lives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

NEUTRAL_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "lane_census"
NEUTRAL_CENSUS_DIR = NEUTRAL_CONFIG_ROOT / "config" / "lab" / "lane-census"
NEUTRAL_MANIFEST = NEUTRAL_CENSUS_DIR / "lane-manifest.yaml"
NEUTRAL_SNAPSHOT = NEUTRAL_CENSUS_DIR / "census-snapshot.json"
#: The neutral registry overlay (OMN-22152): the file ``LANE_CENSUS_REGISTRY``
#: names. It is not under the config root; the registry is the lane census
#: node's overlay data, found through its own variable.
NEUTRAL_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "lane_census"
    / "node_lane_census_effect"
    / "overlay.yaml"
)

#: The required lane on :data:`NEUTRAL_HOSTNAME`, and its network.
NEUTRAL_LANE = "dogfood"
NEUTRAL_LANE_NETWORK = "example-dogfood-network"
#: An optional lane on the same host.
NEUTRAL_OPTIONAL_LANE = "dev"
#: A lane declared for the other host only.
NEUTRAL_OTHER_HOST_LANE = "principal-issuer"
#: The hostname a census on the first host reports, and the id it resolves to.
NEUTRAL_HOSTNAME = "box-a"
NEUTRAL_HOST_ID = "host-a"
#: The second host's hostname.
NEUTRAL_OTHER_HOSTNAME = "box-b"


def use_neutral_lane_census(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every census reader at the neutral set; return the config root."""
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(NEUTRAL_CONFIG_ROOT))
    monkeypatch.setenv("LANE_CENSUS_REGISTRY", str(NEUTRAL_REGISTRY))
    monkeypatch.delenv("LANE_MANIFEST", raising=False)
    return NEUTRAL_CONFIG_ROOT


def generator_consistent_manifest(destination: Path) -> Path:
    """The neutral manifest plus a stub for every lane the table generator maps.

    ``generate_claude_lane_block.py`` keeps its port and boundary maps keyed by
    lane, and the drift checker reports every map entry the manifest does not
    declare. A manifest that checker accepts as clean therefore declares each
    mapped lane. The stubs are read from the generator itself and declare no
    services, so the census checks nothing for them. Writes the manifest to
    ``destination``.
    """
    import sys

    import yaml

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from generate_claude_lane_block import _LANE_BOUNDARY, _LANE_PORT_MAP

    manifest = yaml.safe_load(NEUTRAL_MANIFEST.read_text(encoding="utf-8"))
    for lane in sorted(set(_LANE_PORT_MAP) | set(_LANE_BOUNDARY)):
        manifest["lanes"].setdefault(lane, {"optional": True, "services": []})
    destination.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return destination


def _lab_census_path(name: str) -> Path:
    """Where the lab's own census file is, without reading or requiring it."""
    import sys

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from lane_census_plan import LaneCensusConfigError, lane_census_dir

    try:
        return lane_census_dir() / name
    except LaneCensusConfigError:
        return (
            Path("/ONEX_WORKSPACE_CONFIG_ROOT-is-unset/config/lab/lane-census") / name
        )


#: The lab's own lane manifest and committed census (private config), for the
#: ``lab_overlay`` tests only. Those cross-check this repository's lane compose
#: files and tools against the lab's real declarations, so they need the real
#: files: the marker deselects them from the generic test splits and the Lane
#: Census Staleness job runs them with the overlay fetched. Resolved without
#: being read, so importing a module that names them never fails collection;
#: with no config root the path names the missing variable and reading it fails
#: loud.
LAB_MANIFEST = _lab_census_path("lane-manifest.yaml")
LAB_SNAPSHOT = _lab_census_path("census-snapshot.json")


def lanes_declared_on_host(registry: Any, host_id: str) -> list[str]:
    """Every lane the registry lists for ``host_id``, sorted."""
    return sorted(registry.lanes_of(host_id))


def lab_registry() -> Any:
    """The lab host registry, from the file ``LANE_CENSUS_REGISTRY`` names.

    Private overlay data of the lane census node (OMN-22152), for the
    ``lab_overlay`` tests only. Unset or invalid, reading it fails loud with the
    planner's own not-configured error.
    """
    import sys

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from lane_census_plan import load_registry

    return load_registry()
