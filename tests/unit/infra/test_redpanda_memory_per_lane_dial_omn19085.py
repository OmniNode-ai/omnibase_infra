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

OMN-19077 then gave the stability-test lane its own dial the same way,
``STABILITY_TEST_REDPANDA_MEMORY`` defaulting to 12G (operator consent
ROLLING_WORK_LEDGER.md:3220), so ``REDPANDA_MEMORY`` is now read by the judge
lane alone and no two lanes share a dial.
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
        ("docker-compose.stability-test.yml", "${STABILITY_TEST_REDPANDA_MEMORY:-12G}"),
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

    # OMN-19085 left stability-test and judge sharing REDPANDA_MEMORY; OMN-19077
    # gave stability-test its own dial, so there is no known sharing left.
    shared = {var: files for var, files in seen.items() if len(files) > 1}
    assert shared == {}


@pytest.mark.unit
def test_stability_test_fully_overrides_the_base_command() -> None:
    """The dev lane's dial lives in the BASE file, so this override is load-bearing.

    ``docker-compose.infra.yml`` is not the dev lane's alone: the stability-test
    lane composes it too, as ``infra.yml`` + ``stability-test.yml`` (read from
    the live container's ``com.docker.compose.project.config_files`` label on
    2026-09-21). The dev lane's reservation is declared in the base file because
    Compose REPLACES a ``command`` list wholesale rather than merging it, so an
    overlay cannot change one flag without restating all of them -- and a second
    copy of the broker's advertise addresses is a worse hazard than this test.

    What keeps the base-file change off the stability-test lane is that
    ``stability-test.yml`` declares its own ``command`` with Compose's
    ``!override`` tag. If that tag or that command block is ever removed, the
    stability-test broker would silently inherit the dev lane's reservation --
    exactly the cross-lane coupling OMN-19085 exists to remove. Assert it, so
    the removal is a red test rather than a surprise on the next recreate.
    """
    raw = (_DOCKER_DIR / "docker-compose.stability-test.yml").read_text(
        encoding="utf-8"
    )
    compose = yaml.load(raw, Loader=_ComposeLoader)  # noqa: S506
    redpanda = compose["services"]["redpanda"]

    assert "command" in redpanda, (
        "stability-test.yml no longer declares its own redpanda command; it would "
        "now inherit the dev lane's reservation from docker-compose.infra.yml"
    )
    assert "!override" in raw.split("  redpanda:", 1)[1].split("command:", 1)[1][:40], (
        "stability-test.yml's redpanda command lost its !override tag; Compose "
        "would merge rather than replace and the base file's value could leak in"
    )
    assert _memory_arg(_DOCKER_DIR / "docker-compose.stability-test.yml") == (
        "${STABILITY_TEST_REDPANDA_MEMORY:-12G}"
    )


@pytest.mark.unit
def test_dev_dial_needs_no_operator_env_edit() -> None:
    """Applying the change must not require editing an untracked file on the host.

    The value that was live before this ticket came from the host-global operator
    env file, which is in no manifest and no tracked file. The point of carrying
    the dev value as a Compose DEFAULT is that a recreate picks it up with nothing
    edited on the host, so assert the dev lane's dial actually has a default
    rather than being another undeclared variable.
    """
    value = _memory_arg(_DOCKER_DIR / "docker-compose.infra.yml")

    assert ":-" in value, (
        "the dev dial has no default, so it would need a host env edit"
    )
    assert not value.startswith("${DEV_REDPANDA_MEMORY:?"), (
        "the dev dial is declared required; it must carry a default instead"
    )


@pytest.mark.unit
def test_stability_test_dial_is_12g_by_default_and_needs_no_env_edit() -> None:
    """OMN-19077: the stability-test broker drops from the operator env's 24G to 12G.

    The live 24G came from ``REDPANDA_MEMORY`` in the host-global operator env
    file. If this lane still read that variable, or read a new one with no
    default or a ``:?`` guard, the next governed recreate would either keep 24G
    or need an untracked host edit. Assert the lane's own variable, its default,
    and that the default is the dev lane's level.
    """
    value = _memory_arg(_DOCKER_DIR / "docker-compose.stability-test.yml")

    assert value == "${STABILITY_TEST_REDPANDA_MEMORY:-12G}"
    assert not value.startswith("${REDPANDA_MEMORY"), (
        "stability-test reads the shared variable again; the operator env's 24G "
        "would come back at the next recreate"
    )
    dev_default = _memory_arg(_DOCKER_DIR / "docker-compose.infra.yml").split(":-", 1)[
        1
    ]
    assert value.split(":-", 1)[1] == dev_default, (
        "the stability-test default no longer matches the dev lane's level"
    )
