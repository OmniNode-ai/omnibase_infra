# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Only the verify pool runs without ``onex`` on its PATH (OMN-24562).

C13, C14 and C29 run on ``[self-hosted, omnibase-verify]`` and fail a job whose PATH
finds ``onex`` after the run. The operator repository's freeze step runs on
``[self-hosted, omnibase-ci]`` and calls ``onex`` by bare name. Both pools run the one
image, which links ``/usr/local/bin/onex`` to the ledger-read shim; the image's ``gosu``
wrapper drops that link when ``RUNNER_LABELS`` carries ``omnibase-verify`` and leaves it
for every other pool.

These tests run the wrapper exactly as the Dockerfile installs it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
HEREDOC = re.compile(
    r"^COPY --chmod=0755 <<'GOSU_POOL_WRAPPER' /usr/local/bin/gosu\n"
    r"(?P<body>.*?)^GOSU_POOL_WRAPPER$",
    re.DOTALL | re.MULTILINE,
)


def _code() -> str:
    return "\n".join(
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def _run_wrapper(tmp_path: Path, labels: str | None) -> tuple[Path, str]:
    """Run the installed wrapper against a stand-in link and a stand-in real gosu."""
    found = HEREDOC.search(DOCKERFILE.read_text(encoding="utf-8"))
    assert found, "the Dockerfile must install gosu as a quoted COPY heredoc"
    wrapper = tmp_path / "gosu"
    wrapper.write_text(found.group("body"), encoding="utf-8")
    wrapper.chmod(0o755)
    real = tmp_path / "gosu-real"
    real.write_text('#!/bin/sh\necho "real-gosu $*"\n', encoding="utf-8")
    real.chmod(0o755)
    link = tmp_path / "onex"
    link.symlink_to("/bin/true")
    env = {
        "PATH": "/usr/bin:/bin",
        "ONEX_PATH_LINK": str(link),
        "GOSU_REAL": str(real),
    }
    if labels is not None:
        env["RUNNER_LABELS"] = labels
    out = subprocess.run(
        [str(wrapper), "runner", "true"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "real-gosu runner true", "hands over to the real gosu"
    return link, out.stdout


def test_verify_pool_loses_onex(tmp_path: Path) -> None:
    link, _ = _run_wrapper(
        tmp_path, "self-hosted,omnibase-verify,host-a,linux,x64,arch-amd64"
    )
    assert not link.is_symlink() and not link.exists()


@pytest.mark.parametrize(
    "labels",
    [
        "self-hosted,omnibase-ci,omnibase-ci-multiarch,linux,x64,arch-amd64",
        "self-hosted,omnibase-ci-mq,linux,x64,arch-amd64",
        "self-hosted,omnibase-deploy,linux,x64,arch-amd64",
        "self-hosted,omnibase-customer-plane,linux,x64,arch-amd64",
        "self-hosted,omnibase-prod-deploy,linux,x64,arch-amd64",
        "self-hosted,not-omnibase-verify-x,omnibase-ci",
        "",
        None,
    ],
)
def test_every_other_pool_keeps_onex(tmp_path: Path, labels: str | None) -> None:
    link, _ = _run_wrapper(tmp_path, labels)
    assert link.is_symlink(), labels


def test_dockerfile_wraps_gosu_and_keeps_the_real_one(tmp_path: Path) -> None:
    code = _code()
    assert "mv /usr/local/bin/gosu /usr/local/libexec/gosu-real" in code
    assert HEREDOC.search(DOCKERFILE.read_text(encoding="utf-8"))
    # The link every pool but verify keeps, and the wrapper test above removing it.
    assert "ln -s /opt/onex-ledger-client/shim/onex /usr/local/bin/onex" in code
    assert code.index("mv /usr/local/bin/gosu") < code.index("<<'GOSU_POOL_WRAPPER'")


def test_the_entrypoint_hands_over_with_gosu() -> None:
    """The wrapper only runs if the entrypoint still drops privileges through ``gosu``."""
    text = (REPO_ROOT / "docker" / "runners" / "entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert 'gosu runner "$@"' in text
