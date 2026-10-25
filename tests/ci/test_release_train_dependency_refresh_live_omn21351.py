# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The train's lock refresh replays a real `uv lock --no-build` run (OMN-21351).

The recording is `uv lock --no-build` run by `refresh_locked_dependencies` over
this repository's own `pyproject.toml` and `uv.lock`: exit 0, 195 upgrade targets,
132 third-party packages moved, no first-party package and not the root entry.
The replay rebuilds the lock from the recorded versions and covers the upgrade
targets and version moves of packages present before and after the real run.
Packages the real run added or removed (its "Added"/"Removed" lines) are outside
`lock_version_moves` and are not replayed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("placeholder_routing_overlay")]

_ROOT = Path(__file__).resolve().parents[2]


def _release_train() -> Any:
    name = "release_train_dependency_refresh_replay_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / "ci" / "release_train.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _block(name: str, version: str, source: str) -> str:
    return f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ {source} }}\n'


@pytest.mark.live_contact(
    "tests/ci/fixtures/release_train_dependency_refresh_omn21351.json"
)
def test_release_train_dependency_refresh_replays_the_recorded_uv_run(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    rt = _release_train()
    targets = cast("list[str]", recorded_response["upgrade_packages"])
    first_party = cast("list[str]", recorded_response["first_party_registry_packages"])
    versions = cast("dict[str, str]", recorded_response["versions_before"])
    root = cast("dict[str, str]", recorded_response["root_entry"])
    moves = cast("list[dict[str, str]]", recorded_response["moves"])
    registry = 'registry = "https://pypi.org/simple"'
    before = "version = 1\n" + "".join(
        [
            _block(root["name"], root["version_before"], 'editable = "."'),
            *(_block(name, "1.0", registry) for name in first_party),
            *(_block(name, versions[name], registry) for name in targets),
        ]
    )
    after = before
    for move in moves:
        after = after.replace(
            f'name = "{move["name"]}"\nversion = "{move["from"]}"',
            f'name = "{move["name"]}"\nversion = "{move["to"]}"',
        )
    assert after != before
    (tmp_path / "uv.lock").write_text(before, encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def replay(argv: Sequence[str], cwd: Path) -> Any:
        calls.append((list(argv), cwd))
        (cwd / "uv.lock").write_text(after, encoding="utf-8")
        return type(
            "Completed",
            (),
            {
                "returncode": recorded_response["exit_code"],
                "stderr": recorded_response["stderr"],
            },
        )()

    policy = rt.load_policy(_ROOT / "config" / "release_train_policy.yaml")[
        "omnibase_infra"
    ]
    result = rt.refresh_locked_dependencies(tmp_path, policy, replay)

    assert len(calls) == 1
    argv, cwd = calls[0]
    assert cwd == tmp_path
    recorded_argv = cast("list[str]", recorded_response["argv_head"])
    head = len(recorded_argv)
    assert argv[:head] == recorded_argv
    assert argv[head::2] == ["--upgrade-package"] * len(targets)
    assert argv[head + 1 :: 2] == targets
    stderr = cast("str", recorded_response["stderr"])
    assert len(moves) == sum(
        line.startswith("Updated ") for line in stderr.splitlines()
    )
    assert len(targets) == 195
    assert len(moves) == 132
    assert not set(first_party) & set(targets)
    assert result.status is rt.EnumDependencyRefreshStatus.REFRESHED
    assert [{"name": n, "from": a, "to": b} for n, a, b in result.moves] == moves
    assert root["version_before"] == root["version_after"]
