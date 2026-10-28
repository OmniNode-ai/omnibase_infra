# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21433: a pull request is proven on a lab slot from a tag at its head.

The operator ruling of 2026-10-28 is that nobody waits for a merge to learn
whether a runtime change works on the lab. The pool already fetches
``pull/<n>/head``; a lane that pushes ``lab/pr-<repo>-<n>-<sha12>`` at the head
it wants proven now names that tag in the params file (``INFRA_TAG`` or
``MARKET_TAG``) and the host fetches the tag instead. The head the tag resolves
to is still compared with ``INFRA_HEAD``, so a tag pushed at the wrong commit is
a ``match=NO`` in the clone output, which the receipt already treats as FAIL.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

RUNTIME_BUILD = Path(__file__).resolve().parents[3] / "scripts" / "runtime_build"
PROVE = (RUNTIME_BUILD / "prepr_pool_prove.sh").read_text(encoding="utf-8")
DRIVER = (RUNTIME_BUILD / "prepr_runtime_pool.py").read_text(encoding="utf-8")


def _function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", PROVE, re.S | re.M)
    assert match is not None, f"{name} is not defined in prepr_pool_prove.sh"
    return match.group(0)


def test_fetch_pr_fetches_the_named_tag_instead_of_the_pull_ref() -> None:
    body = _function("fetch_pr")
    assert "tag=${5:-}" in body
    assert "ref=refs/tags/$tag" in body
    assert "ref=pull/$pr/head" in body, "the pull ref stays the default"


@pytest.mark.parametrize("repo_var", ["INFRA", "MARKET"])
def test_the_clone_phase_passes_the_tag_and_fetches_it_for_the_test_tree(
    repo_var: str,
) -> None:
    assert f'"${{{repo_var}_TAG:-}}"' in PROVE
    expansion = (
        f"${{{repo_var}_TAG:+refs/tags/}}${{{repo_var}_TAG:-pull/${repo_var}_PR/head}}"
    )
    assert expansion in PROVE


def test_the_base_control_drops_the_tag_with_the_other_pr_params() -> None:
    assert '"INFRA_TAG"' in DRIVER
    assert '"MARKET_TAG"' in DRIVER
