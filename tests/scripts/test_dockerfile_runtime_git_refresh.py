# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test for OMN-12424.

`Dockerfile.runtime` installs the cross-repo ONEX packages onex_change_control
and omnimarket from *moving* git refs (`@main` / `@dev`) under a persistent
BuildKit uv cache mount (`--mount=type=cache,target=/root/.cache/uv`).

uv keys its git source cache on the resolved commit for the URL, and the URL
itself does not change when the moving branch advances. Therefore a rebuild -
even `docker build --no-cache`, which busts Docker layer cache but not the
BuildKit cache mount - can reinstall a stale wheel from a previous commit of
dev/main. This silently shipped pre-#988 omnimarket code during the OMN-12420
redeploy.

The fix: every release-mode git install of a moving-ref ONEX package must pass
`--refresh-package <name>` so uv re-fetches that package's git source on every
build. This test pins that invariant so the fix cannot regress.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"

# Moving-ref git-sourced ONEX packages installed in release mode, mapped to the
# `--refresh-package` name uv expects (the distribution name, hyphenated).
# omnibase_compat is intentionally excluded: it is installed from a SHA-pinned,
# immutable archive tarball and therefore needs no refresh.
_MOVING_REF_GIT_PACKAGES: dict[str, str] = {
    "onex_change_control": "onex-change-control",
    "omnimarket": "omnimarket",
}


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    raw = _DOCKERFILE.read_text(encoding="utf-8")
    # Collapse shell line continuations so a single `uv pip install` statement
    # that spans `\`-continued physical lines is matchable on one logical line.
    return re.sub(r"\\\n\s*", " ", raw)


@pytest.mark.unit
def test_dockerfile_runtime_exists() -> None:
    assert _DOCKERFILE.is_file()


@pytest.mark.parametrize(
    ("repo_slug", "refresh_name"),
    sorted(_MOVING_REF_GIT_PACKAGES.items()),
)
@pytest.mark.unit
def test_moving_ref_git_install_uses_refresh_package(
    dockerfile_text: str, repo_slug: str, refresh_name: str
) -> None:
    """Each moving-ref git install line must carry the matching --refresh-package."""
    # Find the `uv pip install ... git+https://...<repo_slug>.git@... ` install
    # line (a single logical line may span backslash continuations; the source
    # keeps each install on one physical `uv pip install ...` statement).
    pattern = re.compile(
        r"uv pip install[^\n]*--refresh-package\s+"
        + re.escape(refresh_name)
        + r"[^\n]*git\+https://github\.com/OmniNode-ai/"
        + re.escape(repo_slug)
        + r"\.git@",
    )
    assert pattern.search(dockerfile_text), (
        f"release-mode git install of {repo_slug} must pass "
        f"--refresh-package {refresh_name} (OMN-12424): a moving @dev/@main ref "
        f"under the uv cache mount otherwise ships stale wheels on rebuild."
    )


@pytest.mark.unit
def test_no_moving_ref_git_install_without_refresh(dockerfile_text: str) -> None:
    """No release-mode git+ install of a tracked moving-ref package may omit refresh.

    Guards against someone adding a new `git+...@dev`/`@main` ONEX install that
    reintroduces the staleness bug.
    """
    git_install_re = re.compile(
        r"git\+https://github\.com/OmniNode-ai/([A-Za-z0-9_\-]+)\.git@",
    )
    for match in git_install_re.finditer(dockerfile_text):
        repo_slug = match.group(1)
        if repo_slug not in _MOVING_REF_GIT_PACKAGES:
            continue
        # Look back to the start of this `uv pip install` statement and forward
        # to the URL; the whole statement must contain --refresh-package.
        stmt_start = dockerfile_text.rfind("uv pip install", 0, match.start())
        assert stmt_start != -1, (
            f"git install of {repo_slug} not preceded by a uv pip install statement"
        )
        statement = dockerfile_text[stmt_start : match.end()]
        assert "--refresh-package" in statement, (
            f"git install of moving-ref package {repo_slug} is missing "
            f"--refresh-package (OMN-12424)."
        )
