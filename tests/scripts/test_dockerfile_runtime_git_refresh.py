# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""docker/Dockerfile.runtime installs no ONEX package from a moving git ref.

OMN-12424 found that uv keys its git source cache on the resolved commit of a
moving ref (``@dev`` / ``@main``), so a rebuild under the persistent BuildKit
cache mount reinstalled a stale wheel. The fix was ``--refresh-package`` on the
one moving-ref install the image carried. OMN-21279 removed that install (the
image no longer carries omnimarket). omnibase_compat is retired: it is
no longer installed into the image at all. This guard keeps it that way: a new
``git+https://github.com/OmniNode-ai/...`` install in the image would bring the
staleness bug back, so it is refused rather than refreshed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    raw = _DOCKERFILE.read_text(encoding="utf-8")
    # Drop whole-line comments, then collapse shell line continuations so a
    # single install statement spanning `\`-continued lines is one logical line.
    code = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    return re.sub(r"\\\n\s*", " ", code)


@pytest.mark.unit
def test_dockerfile_runtime_exists() -> None:
    assert _DOCKERFILE.is_file()


@pytest.mark.unit
def test_no_compat_install_in_the_image(dockerfile_text: str) -> None:
    # omnibase_compat is retired; no install of it may remain.
    assert "omnibase_compat" not in dockerfile_text
    assert "OMNIBASE_COMPAT" not in dockerfile_text


@pytest.mark.unit
def test_no_onex_git_install_in_the_image(dockerfile_text: str) -> None:
    installs = re.findall(
        r"git\+https://github\.com/OmniNode-ai/[A-Za-z0-9_\-]+\.git@\S*",
        dockerfile_text,
    )
    assert installs == [], (
        "release-mode git installs of ONEX packages track a moving ref and ship "
        f"stale wheels under the uv cache mount (OMN-12424): {installs}"
    )
