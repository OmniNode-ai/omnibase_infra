# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19085: each lane's Redpanda memory reservation has its own dial.

Why this exists. Three lanes -- dev (docker-compose.infra.yml), stability-test
and judge -- all read the same ``REDPANDA_MEMORY``, and the value that actually
takes effect comes from the host-global operator env file that
``deploy-runtime.sh`` resolves via ``OMNIBASE_OPERATOR_ENV_FILE``. There was
therefore no way to change one lane's broker reservation without changing the
others at their next recreate, and nothing in the repository said so. That
coupling is invisible in any single file, which is exactly the kind of thing a
test has to hold, because reading ``infra.yml`` alone does not reveal it.

The dev lane now reads ``DEV_REDPANDA_MEMORY``. These tests pin both halves:
the dev lane is separated, and the separation did not disturb the lanes it was
supposed to leave alone (OMN-19077 AC-1).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose `!override` / `!reset` tags."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that unwraps compose merge/override tags (OMN-13772)."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _memory_arg(compose_path: Path) -> str:
    """Return the token following --memory in that file's redpanda command."""
    raw = compose_path.read_text(encoding="utf-8")
    # _ComposeLoader extends SafeLoader; the extra constructors only unwrap
    # compose merge/override tags.
    compose: dict[str, Any] = yaml.load(raw, Loader=_ComposeLoader)  # noqa: S506
    command = compose["services"]["redpanda"]["command"]
    # Two spellings are in use across these files: "--memory" and its value as
    # separate list entries, or a single "--memory <value>" entry. Handle both
    # rather than asserting one, so this test pins the VALUE and not the style.
    for entry in command:
        text = str(entry)
        if text.startswith("--memory "):
            return text.split(" ", 1)[1]
    index = command.index("--memory")
    return str(command[index + 1])


@pytest.mark.unit
def test_dev_lane_reads_its_own_memory_dial() -> None:
    """The dev lane must not read the shared variable."""
    value = _memory_arg(_DOCKER_DIR / "docker-compose.infra.yml")

    assert value == "${DEV_REDPANDA_MEMORY:-12G}"
    assert "REDPANDA_MEMORY:-" not in value.replace("DEV_REDPANDA_MEMORY:-", "")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("compose_file", "expected"),
    [
        ("docker-compose.stability-test.yml", "${REDPANDA_MEMORY:-8G}"),
        ("docker-compose.judge.yml", "${REDPANDA_MEMORY:-8G}"),
        ("docker-compose.lakshman.yml", "${LAKSHMAN_REDPANDA_MEMORY:-2G}"),
        ("docker-compose.dogfood.yml", "${DOGFOOD_REDPANDA_MEMORY:-2G}"),
        ("docker-compose.prod.yml", "${PROD_REDPANDA_MEMORY:-8G}"),
        ("docker-compose.ci-bus.yml", "${CI_BUS_REDPANDA_MEMORY:-1G}"),
    ],
)
def test_other_lanes_are_undisturbed(compose_file: str, expected: str) -> None:
    """Separating the dev lane must change no other lane's dial or default."""
    assert _memory_arg(_DOCKER_DIR / compose_file) == expected


@pytest.mark.unit
def test_no_two_lanes_share_a_memory_dial() -> None:
    """The coupling this ticket removed must not come back by a new route.

    A future lane added with a copy-pasted ``${REDPANDA_MEMORY:-...}`` would
    silently re-create the problem, so assert the variable names are distinct
    across every lane that declares a broker.
    """
    lane_files = [
        "docker-compose.infra.yml",
        "docker-compose.stability-test.yml",
        "docker-compose.judge.yml",
        "docker-compose.lakshman.yml",
        "docker-compose.dogfood.yml",
        "docker-compose.prod.yml",
        "docker-compose.ci-bus.yml",
    ]
    seen: dict[str, list[str]] = {}
    for name in lane_files:
        value = _memory_arg(_DOCKER_DIR / name)
        variable = value.split(":-", 1)[0].lstrip("${")
        seen.setdefault(variable, []).append(name)

    # stability-test and judge deliberately still share REDPANDA_MEMORY: they
    # were out of scope for OMN-19085, which changed the dev lane only. Pin
    # that as the ONE known sharing so a new one cannot slip in unnoticed.
    shared = {var: files for var, files in seen.items() if len(files) > 1}
    assert shared == {
        "REDPANDA_MEMORY": [
            "docker-compose.stability-test.yml",
            "docker-compose.judge.yml",
        ]
    }
