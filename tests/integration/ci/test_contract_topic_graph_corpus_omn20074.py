# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: the contract graph's corpus no longer includes onex_change_control.

``contract-topic-graph`` resolves its checkout packages from a directory the
workflow populates. The workflow stopped checking the change-control repository
out, so the graph must neither look for it nor fail closed because it is absent.
These tests run the real resolver and the real graph builder over a checkout
root that holds only the packages the workflow still checks out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators import contract_topic_graph as graph

pytestmark = pytest.mark.integration


def _checkout_root(tmp_path: Path, packages: tuple[str, ...]) -> Path:
    for package in packages:
        (tmp_path / package / "src" / package).mkdir(parents=True)
    return tmp_path


def test_resolver_resolves_every_remaining_checkout_package_and_no_change_control(
    tmp_path: Path,
) -> None:
    root = _checkout_root(tmp_path, graph.CHECKOUT_PACKAGES)
    roots = graph.discover_contract_roots(root)
    assert set(graph.CHECKOUT_PACKAGES) <= set(roots)
    assert "onex_change_control" not in roots


def test_resolver_ignores_a_change_control_checkout_if_one_is_present(
    tmp_path: Path,
) -> None:
    root = _checkout_root(tmp_path, (*graph.CHECKOUT_PACKAGES, "onex_change_control"))
    assert "onex_change_control" not in graph.discover_contract_roots(root)


def test_graph_fails_closed_naming_only_the_packages_still_expected(
    tmp_path: Path,
) -> None:
    """Positive control: a missing checkout still refuses, and never names change control."""
    with pytest.raises(RuntimeError) as excinfo:
        graph.build_graph(checkout_root=tmp_path)
    message = str(excinfo.value)
    assert "omnimarket" in message
    assert "onex_change_control" not in message
