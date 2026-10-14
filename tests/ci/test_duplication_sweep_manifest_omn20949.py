# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20949: the Duplication Sweep reads kafka_boundaries.yaml from omnimarket.

D2 compares omniclaude's topic claims against the Kafka boundary manifest. The
manifest lives beside omnimarket's node_duplication_sweep, so the required
Duplication Sweep job checks out omnimarket at the pinned sha and the script
reads the manifest there; neither names onex_change_control any more.

Falsifier: restore the onex_change_control checkout in duplication-sweep.yml, or
the change-control path in run_duplication_sweep.py, and a test below fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "duplication-sweep.yml"
SCRIPT = REPO_ROOT / "scripts" / "ci" / "run_duplication_sweep.py"
PINS = yaml.safe_load((REPO_ROOT / ".github" / "sibling-pins.yaml").read_text())["pins"]
MANIFEST = "src/omnimarket/nodes/node_duplication_sweep/contracts/kafka_boundaries.yaml"


def _checkout_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for m in re.finditer(r"^\s*repository: OmniNode-ai/(\w+)\s*$", text, re.M):
        blocks[m.group(1)] = text[m.end() : m.end() + 400]
    return blocks


def test_workflow_checks_out_omnimarket_not_change_control() -> None:
    text = WORKFLOW.read_text()
    blocks = _checkout_blocks(text)
    assert "onex_change_control" not in text
    assert "omnimarket" in blocks
    block = blocks["omnimarket"]
    assert f"ref: {PINS['omnimarket']}" in block
    assert "path: sweep-home/omnimarket" in block
    assert "src/omnimarket/nodes/node_duplication_sweep/contracts" in block


def test_script_reads_the_omnimarket_manifest_only() -> None:
    text = SCRIPT.read_text()
    assert "onex_change_control" not in text
    assert MANIFEST in text


def test_pinned_omnimarket_ships_the_manifest_path_the_script_reads() -> None:
    """The workflow's sparse path covers the manifest the script resolves."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("run_duplication_sweep_omn20949", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    relevant = module._RELEVANT_SUBSTRINGS["D2"]
    assert any(MANIFEST in sub for sub in relevant)


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/duplication_sweep_d2_omnimarket_manifest_omn20949.json"
)
def test_d2_replays_the_recorded_omnimarket_manifest_read(
    recorded_response, tmp_path: Path
) -> None:
    """Replay D2 over the manifest omnimarket ships at the pin, with no change-control tree."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("run_duplication_sweep_omn20949_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    response = recorded_response["response"]
    assert response["d2_result"]["status"] == "PASS"
    hooks = tmp_path / "omniclaude" / "src" / "omniclaude" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "topics.py").write_text(
        "".join(
            f'    T{i} = "{topic}"\n'
            for i, topic in enumerate(response["omniclaude_claimed_topics"])
        )
    )
    manifest = tmp_path / "omnimarket" / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(response["manifest"])
    assert not (tmp_path / "onex_change_control").exists()

    assert module.check_d2_topic_conflicts(tmp_path) == response["d2_result"]
