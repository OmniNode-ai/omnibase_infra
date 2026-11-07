# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev boot creates only the topics of the contracts it wires (OMN-22478).

The dev runtime container carried no ``ONEX_BOOT_UNIVERSE_PROVISION``, which the
kernel reads as on, so every boot created every topic any installed contract or
skill manifest declares: 2,463 of the dev broker's 2,857 topics had never been
produced to. The dev compose anchor now sets the switch off.

The chain under test, with no value typed in the test: the dev compose anchor's
own setting -> the environment a container gets from it -> the kernel's own read
of that variable, taken from the kernel source rather than copied here.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = ROOT / "docker"
DEV_COMPOSE = DOCKER_DIR / "docker-compose.infra.yml"
KERNEL = ROOT / "src" / "omnibase_infra" / "runtime" / "service_kernel.py"
SWITCH = "ONEX_BOOT_UNIVERSE_PROVISION"

pytestmark = pytest.mark.unit

_SETTING = re.compile(
    rf"^\s*{SWITCH}:\s*\$\{{(?P<var>[A-Z_]+):-(?P<default>[^}}]*)\}}\s*$",
    re.MULTILINE,
)


def _kernel_universe_warm_enabled(environ: Mapping[str, str]) -> bool:
    """Apply the kernel's own ``_universe_warm_enabled`` expression to ``environ``.

    The expression is read from the kernel source: ``os.environ.get(NAME,
    DEFAULT) != OFF``. A change in its shape fails here, so the test cannot go
    on judging a read the kernel no longer makes.
    """
    for node in ast.walk(ast.parse(KERNEL.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_universe_warm_enabled"
            for target in node.targets
        ):
            continue
        expr = node.value
        assert isinstance(expr, ast.Compare)
        assert isinstance(expr.ops[0], ast.NotEq)
        read = expr.left
        assert isinstance(read, ast.Call)
        name, default = (
            arg.value for arg in read.args if isinstance(arg, ast.Constant)
        )
        off = expr.comparators[0]
        assert isinstance(off, ast.Constant)
        assert name == SWITCH
        return environ.get(name, default) != off.value
    msg = "the kernel no longer assigns _universe_warm_enabled"
    raise AssertionError(msg)


def _container_env(compose_text: str, host_env: Mapping[str, str]) -> dict[str, str]:
    """The switch as a dev runtime container receives it from the anchor."""
    setting = _SETTING.search(compose_text)
    assert setting is not None, (
        f"the dev runtime-env anchor must set {SWITCH} with a default"
    )
    return {SWITCH: host_env.get(setting.group("var"), setting.group("default"))}


def test_dev_boot_does_not_run_the_universe_warm() -> None:
    env = _container_env(DEV_COMPOSE.read_text(encoding="utf-8"), host_env={})

    assert env == {SWITCH: "0"}
    assert _kernel_universe_warm_enabled(env) is False


def test_a_host_can_bring_the_full_warm_back() -> None:
    env = _container_env(
        DEV_COMPOSE.read_text(encoding="utf-8"), host_env={SWITCH: "1"}
    )

    assert _kernel_universe_warm_enabled(env) is True


def test_only_the_dev_compose_changes_the_switch() -> None:
    referencing = sorted(
        path.name
        for path in DOCKER_DIR.glob("docker-compose*.yml")
        if SWITCH in path.read_text(encoding="utf-8")
    )

    assert referencing == ["docker-compose.infra.yml"]


def test_failing_control_a_dev_anchor_that_leaves_the_warm_on_is_caught() -> None:
    """The anchor as it was (no setting) reads the warm as enabled.

    This is the state the lab was in: the proof that the first test can fail.
    """
    without_setting = _SETTING.sub("", DEV_COMPOSE.read_text(encoding="utf-8"))

    assert _SETTING.search(without_setting) is None
    assert _kernel_universe_warm_enabled({}) is True


def test_failing_control_a_default_of_one_in_the_anchor_is_caught() -> None:
    broken = _SETTING.sub(
        lambda m: f"  {SWITCH}: ${{{m.group('var')}:-1}}",
        DEV_COMPOSE.read_text(encoding="utf-8"),
    )

    env = _container_env(broken, host_env={})

    assert _kernel_universe_warm_enabled(env) is True
