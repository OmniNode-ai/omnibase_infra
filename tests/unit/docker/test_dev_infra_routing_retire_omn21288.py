# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One-runtime plan T14 (OMN-23072, under OMN-21288): the dev stack no longer
runs the infra routing decisions consumer container.

``node_projection_infra_routing_decisions`` (omnimarket) owns the
``infra_routing_decisions`` table and, through the handover its contract declares
(OMN-23065), takes over the retired consumer group. The standalone container
that ran ``omnibase_infra.services.observability.infra_routing_decisions.consumer``
is therefore gone from every dev stack file, every dev deploy set, and the stack-shape baseline. The lane manifest in the overlays repository and the
lane boundary maps that mirror it name the container too; they move in lockstep
with that manifest, not here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

SERVICE = "infra-routing-decisions-consumer"
ROOT = Path(__file__).resolve().parents[3]

# The dev stack: the dev lane, every overlay that derives from it, and the
# pre-PR verify stack. Read by glob so a new dev overlay is covered at once.
DEV_STACK_COMPOSE_FILES = tuple(
    sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "docker").glob("docker-compose.dev-*.yml")
    )
) + ("docker/docker-compose.prepr.yml",)

# Files that name services in code, as bash arrays or Python tuples.
DEPLOY_SET_FILES = (
    "src/omnibase_infra/handlers/handler_runtime_deploy.sh",
    "scripts/runtime_build/refresh_dev_lane.sh",
    "scripts/deploy-agent/deploy_agent/events.py",
)

STACK_SHAPE = ROOT / "config" / "validation" / "stack_shape.yaml"


def _load_compose(path: Path) -> dict[str, object]:
    """Read a compose file, dropping the ``!override`` and ``!reset`` merge tags."""
    text = re.sub(r"!(?:override|reset)\b", "", path.read_text(encoding="utf-8"))
    return yaml.safe_load(text)


def _code_lines(path: Path) -> list[str]:
    """Lines of ``path`` that are not whole-line comments."""
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("relative", DEV_STACK_COMPOSE_FILES)
def test_no_dev_stack_compose_file_declares_the_service(relative: str) -> None:
    compose = _load_compose(ROOT / relative)
    assert SERVICE not in (compose.get("services") or {}), relative


@pytest.mark.parametrize("relative", DEPLOY_SET_FILES)
def test_no_dev_deploy_set_names_the_service(relative: str) -> None:
    offenders = [
        line.strip() for line in _code_lines(ROOT / relative) if SERVICE in line
    ]
    assert offenders == [], relative


def test_stack_shape_baseline_holds_no_entry_for_the_service() -> None:
    declared = yaml.safe_load(STACK_SHAPE.read_text(encoding="utf-8"))
    for section in ("baseline", "command_baseline"):
        names = [entry["service"] for entry in declared.get(section) or []]
        assert SERVICE not in names, section


def test_the_dev_stack_scan_covers_the_dev_lane_and_its_overlays() -> None:
    """Positive control: the glob finds the dev lane and at least one overlay."""
    assert "docker/docker-compose.dev-lane.yml" in DEV_STACK_COMPOSE_FILES
    assert len(DEV_STACK_COMPOSE_FILES) >= 3


def test_the_check_reads_the_service_name_it_refuses() -> None:
    """Positive control: a compose service of this name would be found.

    The scan above reads the ``services`` mapping, so a mapping that holds the
    name must be found by the same read.
    """
    planted = yaml.safe_load(f"services:\n  {SERVICE}:\n    image: x\n")
    assert SERVICE in planted["services"]
    assert re.search(re.escape(SERVICE), f"    {SERVICE}\n")
