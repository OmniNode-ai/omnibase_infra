# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollout gate: the runner image must ship the AWS CLI v2 (OMN-16444).

Background. ECR-publishing workflows target the ``self-hosted,omnibase-ci``
fleet and run ``aws sts get-caller-identity`` immediately after
``aws-actions/configure-aws-credentials`` (e.g. omniclaude's
``build-and-push-migrate-image.yml``). The runner image never installed the
AWS CLI; some fleet hosts had it out-of-band and some did not, so the same
workflow passed or failed depending on runner placement:

    aws: command not found
    Process completed with exit code 127

omniclaude runs 32678259714 and 32706441145 (2026-08-24) both died there on
``omninode-runner-42`` after the OIDC credential exchange had already
succeeded, masking the real ECR publish result behind a spurious failure.

This test is the static gate that the AWS CLI install stays in the image —
the same enforcement style as ``test_runner_image_buildx_plugin.py`` /
``test_runner_image_compose_plugin.py``. Adding the tool is identity-neutral:
the Dockerfile is not part of the ``runner-image.lock.json`` bound identity.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"


def _dockerfile_source() -> str:
    return RUNNER_DOCKERFILE.read_text(encoding="utf-8")


def _run_blocks(source: str) -> list[str]:
    """Return each top-level RUN block (through its trailing continuations)."""
    return re.findall(
        r"(?ms)^RUN .*?(?=^RUN |^COPY |^USER |^WORKDIR |^ENTRYPOINT |\Z)", source
    )


def test_dockerfile_installs_aws_cli() -> None:
    """The pinned AWS CLI v2 bundle must be installed in the image.

    Without it, any job step that shells out to ``aws`` (``aws sts
    get-caller-identity``, ECR pushes) dies with ``aws: command not found``
    (exit 127) on exactly the hosts that lack an out-of-band install — the
    OMN-16444 placement-dependent failure.
    """
    source = _dockerfile_source()
    assert "awscli-exe-linux-x86_64-${AWSCLI_VERSION}.zip" in source, (
        "runner Dockerfile must install the AWS CLI v2 bundle pinned via "
        "AWSCLI_VERSION so `aws` resolves uniformly across the omnibase-ci "
        "fleet (OMN-16444)"
    )
    assert re.search(r"(?m)^ARG AWSCLI_VERSION=\d+\.\d+\.\d+$", source), (
        "AWSCLI_VERSION must be a pinned ARG (not floating 'latest') so the "
        "image contract is reproducible (OMN-16444)"
    )


def test_aws_cli_install_verifies_the_binary_executes() -> None:
    """The install RUN block must end by executing ``aws --version``.

    A download/unzip that silently produces a broken binary would otherwise
    only surface inside a live CI job. Running ``aws --version`` in the same
    RUN block moves that failure to image-build time.
    """
    blocks = _run_blocks(_dockerfile_source())
    awscli_block = next((b for b in blocks if "awscliv2.zip" in b), None)
    assert awscli_block is not None, (
        "expected a RUN block that installs the AWS CLI v2 bundle"
    )
    assert "aws --version" in awscli_block, (
        "the AWS CLI install RUN block must smoke-test the binary with "
        "`aws --version` so a broken install fails the image build, not a "
        "live job (OMN-16444)"
    )
