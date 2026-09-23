# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollout gate: the runner image must ship Playwright chromium's system libraries (OMN-19206).

Background. A private repository's CI runs on the omnibase-ci fleet, and the
fleet has no passwordless package manager: the sudoers rule grants one confined
``rm -rf`` and nothing else. So ``playwright install --with-deps`` cannot run
there, and a job that launches Chromium depends on the libraries already being
in the image. They were not, and every Playwright smoke test on the fleet died
at ``browserType.launch`` with::

    chrome-headless-shell: error while loading shared libraries:
    libnspr4.so: cannot open shared object file

This file holds the two halves of the gate. The Dockerfile must install every
package in ``PLAYWRIGHT_CHROMIUM_PACKAGES``, and the image build smoke must
resolve every matching soname with ``ldconfig -p`` inside the image it just
built. The first half catches a package dropped from the list. The second
catches the case a text check cannot: a list that still names the package while
the built image, for whatever reason, no longer carries the library.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
BUILD_SMOKE_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "runner-image-build-smoke.yml"
)

# The library half of `playwright install-deps --dry-run chromium` (playwright
# 1.59.1, ubuntu 22.04), each with the soname the image smoke resolves. The
# same command also lists xvfb and nine font packages. They are left out: the
# fleet runs Chromium headless, and libpango/libcairo pull fontconfig and a
# default font in on their own.
PLAYWRIGHT_CHROMIUM_PACKAGES: dict[str, str] = {
    "libasound2": "libasound.so.2",
    "libatk-bridge2.0-0": "libatk-bridge-2.0.so.0",
    "libatk1.0-0": "libatk-1.0.so.0",
    "libatspi2.0-0": "libatspi.so.0",
    "libcairo2": "libcairo.so.2",
    "libcups2": "libcups.so.2",
    "libdbus-1-3": "libdbus-1.so.3",
    "libdrm2": "libdrm.so.2",
    "libgbm1": "libgbm.so.1",
    "libglib2.0-0": "libglib-2.0.so.0",
    "libnspr4": "libnspr4.so",
    "libnss3": "libnss3.so",
    "libpango-1.0-0": "libpango-1.0.so.0",
    "libwayland-client0": "libwayland-client.so.0",
    "libx11-6": "libX11.so.6",
    "libxcb1": "libxcb.so.1",
    "libxcomposite1": "libXcomposite.so.1",
    "libxdamage1": "libXdamage.so.1",
    "libxext6": "libXext.so.6",
    "libxfixes3": "libXfixes.so.3",
    "libxkbcommon0": "libxkbcommon.so.0",
    "libxrandr2": "libXrandr.so.2",
}


def _apt_installed_packages(source: str) -> set[str]:
    """Every package name an ``apt-get install`` line in the Dockerfile names."""
    # Drop comment lines, so prose naming a package cannot stand in for an
    # install. Then join line continuations and read each install command up
    # to the next shell operator, so a list split across lines is read whole.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    joined = code.replace("\\\n", " ")
    packages: set[str] = set()
    for command in re.findall(r"apt-get install([^\n&;|]*)", joined):
        for token in command.split():
            if re.fullmatch(r"[a-z0-9][a-z0-9.+-]+", token):
                packages.add(token)
    return packages


def _build_smoke_step_scripts() -> list[str]:
    workflow = yaml.safe_load(BUILD_SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["runner-image-build-smoke"]["steps"]
    return [str(step.get("run", "")) for step in steps]


@pytest.mark.parametrize("package", sorted(PLAYWRIGHT_CHROMIUM_PACKAGES))
def test_dockerfile_installs_playwright_chromium_package(package: str) -> None:
    installed = _apt_installed_packages(RUNNER_DOCKERFILE.read_text(encoding="utf-8"))
    assert package in installed, (
        f"runner Dockerfile must apt-install {package}: without it Chromium "
        "cannot launch on the fleet, and a job there has no way to install it "
        "itself (OMN-19206)"
    )


def test_build_smoke_resolves_every_soname_inside_the_built_image() -> None:
    step = next(
        (
            script
            for script in _build_smoke_step_scripts()
            if "ldconfig -p" in script and "docker run --rm" in script
        ),
        None,
    )
    assert step is not None, (
        "runner-image-build-smoke must run `ldconfig -p` inside the image it "
        "built, so an image that lost a Chromium library fails the build smoke "
        "(OMN-19206)"
    )
    missing = [
        soname for soname in PLAYWRIGHT_CHROMIUM_PACKAGES.values() if soname not in step
    ]
    assert not missing, (
        f"the build smoke's ldconfig assertion does not check {missing} (OMN-19206)"
    )
