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

import json
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
# 1.63.0, ubuntu 22.04: `nativeDeps['ubuntu22.04-x64'].chromium`), each with the
# soname the image smoke resolves.
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


# The other half of the same command (`nativeDeps['ubuntu22.04-x64'].tools`,
# playwright 1.63.0, the version omniweb dev pins). OMN-19659: image v9 left
# these out on the theory that libpango/libcairo pull in a default font. They
# pull in DejaVu and nothing else, so a page that shapes emoji, CJK or Thai
# renders tofu, and omniweb still stages fonts-liberation per job by hand
# (OMN-18165) because Chromium aborted on SIGTRAP with no usable font set.
PLAYWRIGHT_TOOLS_PACKAGES: tuple[str, ...] = (
    "fonts-freefont-ttf",
    "fonts-ipafont-gothic",
    "fonts-liberation",
    "fonts-noto-color-emoji",
    "fonts-tlwg-loma-otf",
    "fonts-unifont",
    "fonts-wqy-zenhei",
    "libfontconfig1",
    "libfreetype6",
    "xfonts-cyrillic",
    "xfonts-scalable",
    "xvfb",
)

# The fontconfig family each font package registers. The smoke script requires
# every one of them on an image at or above FONTS_BAKED_IMAGE_VERSION, which is
# the check a package-list grep cannot make: it reads what fontconfig sees.
PLAYWRIGHT_FONT_FAMILIES: tuple[str, ...] = (
    "FreeSans",
    "IPAGothic",
    "Liberation Sans",
    "Noto Color Emoji",
    "Loma",
    "Unifont",
    "WenQuanYi Zen Hei",
)
FONTS_BAKED_IMAGE_VERSION = 10

RUNNER_IMAGE_LOCK = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"
SMOKE_SCRIPT = REPO_ROOT / "docker" / "runners" / "playwright-smoke.sh"
CANARY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-playwright-canary.yml"


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


@pytest.mark.parametrize("package", PLAYWRIGHT_TOOLS_PACKAGES)
def test_dockerfile_installs_playwright_tools_package(package: str) -> None:
    installed = _apt_installed_packages(RUNNER_DOCKERFILE.read_text(encoding="utf-8"))
    assert package in installed, (
        f"runner Dockerfile must apt-install {package}: it is in Playwright's "
        "jammy install-deps set and a fleet job cannot install it itself "
        "(OMN-19659)"
    )


def test_lock_image_version_carries_the_font_set() -> None:
    lock = json.loads(RUNNER_IMAGE_LOCK.read_text(encoding="utf-8"))
    assert lock["image_version"] >= FONTS_BAKED_IMAGE_VERSION, (
        "the font set is baked from image v10; a lock below it would let the "
        "smoke script skip the font assertions it exists to make (OMN-19659)"
    )


def test_smoke_script_requires_every_font_family_and_launches_chromium() -> None:
    script = SMOKE_SCRIPT.read_text(encoding="utf-8")
    missing = [family for family in PLAYWRIGHT_FONT_FAMILIES if family not in script]
    assert not missing, f"smoke script does not require font families {missing}"
    assert f"FONTS_BAKED_IMAGE_VERSION={FONTS_BAKED_IMAGE_VERSION}" in script
    assert "chromium.launch(" in script, "smoke script must launch Chromium"
    assert "@font-face" in script, (
        "smoke script must shape a downloaded web font: that is the path that "
        "aborted Chromium on SIGTRAP with no usable font set (OMN-18165)"
    )
    code = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"\bsudo\b|apt-get", code), (
        "the smoke proves the image needs no package install at job time; it "
        "must not call sudo or apt-get itself"
    )


def test_build_smoke_runs_the_launch_script_as_the_runner_user() -> None:
    step = next(
        (
            script
            for script in _build_smoke_step_scripts()
            if "playwright-smoke.sh" in script
        ),
        None,
    )
    assert step is not None, (
        "runner-image-build-smoke must run docker/runners/playwright-smoke.sh "
        "inside the image it built (OMN-19659)"
    )
    assert "--user runner" in step, (
        "the launch must run as the unprivileged runner user"
    )
    # Fleet runners reach docker through a mounted socket, so a bind mount of a
    # workspace path names a directory the daemon's host does not have. The
    # script goes in on stdin.
    assert "--entrypoint bash" in step
    assert "-s < docker/runners/playwright-smoke.sh" in step


def test_canary_workflow_runs_the_launch_script_on_a_labelled_runner() -> None:
    workflow = yaml.safe_load(CANARY_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get(True, workflow.get("on"))
    assert "workflow_dispatch" in triggers
    job = workflow["jobs"]["playwright-launch"]
    runs_on = [str(label) for label in job["runs-on"]]
    assert runs_on[:2] == ["self-hosted", "omnibase-ci"]
    assert "inputs.runner_label" in runs_on[2], (
        "the canary is targeted by a label added to one recreated runner"
    )
    steps = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert "docker/runners/playwright-smoke.sh" in steps
