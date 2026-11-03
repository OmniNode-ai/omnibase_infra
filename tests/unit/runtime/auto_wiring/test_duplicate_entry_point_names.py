# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No ``onex.nodes`` entry point name may be declared by two installed packages (OMN-22233).

Discovery keys what it loads by contract name and rejects the second package that
ships a name another package already owns, at runtime, after the image is built.
This check reads the installed distributions' own metadata, so it holds in any
environment: the dev runtime image installs both ``omnibase_infra`` and
``omnimarket``, and a name declared by both is a defect there.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from importlib.metadata import Distribution, distributions
from pathlib import Path

import pytest

NODES_GROUP = "onex.nodes"


def entry_point_owners(dists: Iterable[Distribution]) -> dict[str, set[str]]:
    """Map each ``onex.nodes`` entry point name to the distributions declaring it."""
    owners: dict[str, set[str]] = defaultdict(set)
    for dist in dists:
        dist_name = dist.metadata["Name"]
        for ep in dist.entry_points:
            if ep.group == NODES_GROUP:
                owners[ep.name].add(dist_name)
    return owners


def duplicated(owners: dict[str, set[str]]) -> dict[str, set[str]]:
    return {name: dists for name, dists in owners.items() if len(dists) > 1}


def _write_dist(root: Path, name: str, entry_points: tuple[str, ...]) -> None:
    info = root / f"{name.replace('-', '_')}-1.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0.0\n", encoding="utf-8"
    )
    lines = "".join(f"{ep} = {name}.nodes.{ep}\n" for ep in entry_points)
    (info / "entry_points.txt").write_text(
        f"[{NODES_GROUP}]\n{lines}", encoding="utf-8"
    )


@pytest.mark.unit
def test_duplicate_entry_point_names_none_among_installed_packages() -> None:
    assert duplicated(entry_point_owners(distributions())) == {}


@pytest.mark.unit
def test_duplicate_entry_point_names_planted_pair_is_detected(tmp_path: Path) -> None:
    """Positive control: two installed packages naming one entry point are reported."""
    _write_dist(tmp_path, "pkg-one", ("node_shared", "node_only_one"))
    _write_dist(tmp_path, "pkg-two", ("node_shared", "node_only_two"))
    found = duplicated(entry_point_owners(distributions(path=[str(tmp_path)])))
    assert found == {"node_shared": {"pkg-one", "pkg-two"}}


@pytest.mark.unit
def test_duplicate_entry_point_names_disjoint_pair_is_clean(tmp_path: Path) -> None:
    _write_dist(tmp_path, "pkg-one", ("node_one",))
    _write_dist(tmp_path, "pkg-two", ("node_two",))
    assert duplicated(entry_point_owners(distributions(path=[str(tmp_path)]))) == {}
