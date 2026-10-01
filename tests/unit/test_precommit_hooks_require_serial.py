# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Keep filename-consuming OmniNode hooks in one process (OMN-20241).

Without require_serial, pre-commit 4.6.1 partitions filenames into batches of
max(4, ceil(N / cpu_count)) and runs up to cpu_count processes concurrently.
Each validator process pays the full omnibase_core import cost; concurrent
commit lanes multiply that cost. The validators already accept whole file
lists and aggregate exit codes, so require_serial avoids redundant imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pytest
import yaml


class PrecommitHook(TypedDict):
    """Hook fields used by the serial-execution policy."""

    id: str
    pass_filenames: NotRequired[bool]
    language: NotRequired[str]
    require_serial: NotRequired[bool]


class PrecommitRepo(TypedDict):
    """Repository and its consumer hook declarations."""

    repo: str
    hooks: list[PrecommitHook]


class PrecommitConfig(TypedDict):
    """Configuration fields used by the policy checker."""

    repos: list[PrecommitRepo]


def hooks_missing_require_serial(config: PrecommitConfig) -> list[str]:
    """Report filename-consuming local/OmniNode hooks without explicit True."""
    offenders: list[str] = []
    for repo in config["repos"]:
        if repo["repo"] != "local" and not repo["repo"].startswith(
            "https://github.com/OmniNode-ai/"
        ):
            continue
        for hook in repo["hooks"]:
            if hook.get("pass_filenames") is False or hook.get("language") == "fail":
                continue
            if hook.get("require_serial") is not True:
                offenders.append(hook["id"])
    return offenders


@pytest.mark.unit
def test_filename_consuming_omninode_hooks_require_serial() -> None:
    """Enforce the policy for every hook in the repository configuration."""
    config_path = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"
    config = cast(
        "PrecommitConfig", yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    offenders = hooks_missing_require_serial(config)
    assert not offenders, f"Hooks must declare require_serial: true: {offenders}"


@pytest.mark.unit
def test_local_hook_missing_require_serial_is_reported() -> None:
    """A new filename-consuming local hook cannot silently regress."""
    config: PrecommitConfig = {
        "repos": [{"repo": "local", "hooks": [{"id": "new-validator"}]}]
    }
    assert hooks_missing_require_serial(config) == ["new-validator"]


@pytest.mark.unit
def test_hook_without_filenames_is_not_reported() -> None:
    """Hooks that do not receive filenames cannot incur partition fan-out."""
    config: PrecommitConfig = {
        "repos": [
            {
                "repo": "local",
                "hooks": [{"id": "whole-repo-validator", "pass_filenames": False}],
            }
        ]
    }
    assert hooks_missing_require_serial(config) == []


@pytest.mark.unit
def test_third_party_hook_is_not_reported() -> None:
    """Third-party hooks retain their own execution policy."""
    config: PrecommitConfig = {
        "repos": [
            {
                "repo": "https://github.com/pre-commit/pre-commit-hooks",
                "hooks": [{"id": "check-merge-conflict"}],
            }
        ]
    }
    assert hooks_missing_require_serial(config) == []


@pytest.mark.unit
@pytest.mark.parametrize("require_serial", [None, False])
def test_omninode_remote_hook_requires_serial_override(
    require_serial: bool | None,
) -> None:
    """Remote consumer hooks must explicitly opt into serial execution."""
    hook: PrecommitHook = {"id": "remote-validator"}
    if require_serial is not None:
        hook["require_serial"] = require_serial
    config: PrecommitConfig = {
        "repos": [
            {"repo": "https://github.com/OmniNode-ai/omnibase_core", "hooks": [hook]}
        ]
    }
    assert hooks_missing_require_serial(config) == ["remote-validator"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "hook",
    [
        {"id": "serial-validator", "require_serial": True},
        {"id": "blocked-artifact", "language": "fail"},
    ],
)
def test_serial_and_fail_hooks_are_not_reported(hook: PrecommitHook) -> None:
    """Already serial hooks and fail-language hooks satisfy the policy."""
    config: PrecommitConfig = {"repos": [{"repo": "local", "hooks": [hook]}]}
    assert hooks_missing_require_serial(config) == []
