# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The GitHub source-control nodes live in omnimarket only (OMN-20915).

Two nodes moved to omnimarket and the dead PR-state, merge-sweep and
model-only directories were deleted. This pins that none of them comes back
in this repository's source tree, its entry points or its GitHub client, and
that what the kernel still needs from the moved webhook node is the secret's
logical name, owned here.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src" / "omnibase_infra"

MOVED_TO_OMNIMARKET = (
    "node_github_pr_poller_effect",
    "node_github_webhook_ingress_effect",
)
DELETED = (
    "node_pr_state_projection_compute",
    "node_pr_state_write_effect",
    "node_merge_sweep_pr_list_effect",
    "node_merge_sweep_classify_compute",
    "node_merge_sweep_workflow_orchestrator",
    "node_merge_sweep_workflow_reducer",
    "node_merge_sweep_auto_merge_effect",
    "node_pr_label_effect",
    "node_pr_state_fetch_effect",
    "node_mergeability_orchestrator",
    "node_mergeability_evaluate_compute",
)


@pytest.mark.parametrize("node", MOVED_TO_OMNIMARKET + DELETED)
def test_the_node_directory_is_gone(node: str) -> None:
    assert not (_SRC / "nodes" / node).exists()


@pytest.mark.parametrize("node", MOVED_TO_OMNIMARKET + DELETED)
def test_the_node_has_no_entry_point(node: str) -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = project["project"]["entry-points"]["onex.nodes"]
    assert node not in entry_points


def test_the_github_adapter_package_is_gone() -> None:
    assert not (_SRC / "adapters" / "github").exists()
    assert importlib.util.find_spec("omnibase_infra.adapters.github") is None


def test_the_merge_sweep_workflow_definition_is_gone() -> None:
    assert not (_SRC / "workflows" / "merge_sweep_workflow.yaml").exists()


def test_no_source_file_imports_a_removed_node_or_the_github_adapter() -> None:
    modules = tuple(
        f"omnibase_infra.nodes.{node}" for node in MOVED_TO_OMNIMARKET + DELETED
    ) + ("omnibase_infra.adapters.github",)
    import_line = re.compile(
        r"^\s*(?:from|import)\s+(?:" + "|".join(map(re.escape, modules)) + r")\b",
        re.MULTILINE,
    )
    offenders = [
        str(path.relative_to(_ROOT))
        for path in _SRC.rglob("*.py")
        if import_line.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_import_scan_sees_an_import_of_a_removed_module() -> None:
    modules = ("omnibase_infra.adapters.github",)
    import_line = re.compile(
        r"^\s*(?:from|import)\s+(?:" + "|".join(map(re.escape, modules)) + r")\b",
        re.MULTILINE,
    )
    assert import_line.search("from omnibase_infra.adapters.github import x\n")
    assert not import_line.search("# mirrors omnibase_infra.adapters.github docs\n")


def test_the_kernel_owns_the_webhook_secret_logical_name() -> None:
    from omnibase_infra.runtime.service_kernel import GITHUB_WEBHOOK_SECRET_REF

    assert GITHUB_WEBHOOK_SECRET_REF == "github.webhook.secret"
