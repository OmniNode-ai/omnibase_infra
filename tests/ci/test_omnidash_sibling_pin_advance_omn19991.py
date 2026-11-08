# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omnidash sibling pin and its inline readers carry the advanced sha.

The expected sha lives in a fixture under tests/ci/fixtures, a test-side path
the Repo Evidence Gate overlays at the merge base. The pins there still carry
the old sha, so these tests fail at the merge base and pass at this head, which
is what makes the advance evidence of its own (OMN-19991).

Why by hand: the pin refresh bot advances only the omnimarket contract pin, so
the omnidash line of .github/sibling-pins.yaml had not moved since OMN-20331
added it, and exposure-reader-coverage kept reading the reader surface at a
commit that predates the reader it asks for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA = re.compile(r"\b[0-9a-f]{40}\b")
INLINE_OMNIDASH_REF = re.compile(
    r"repository:\s*OmniNode-ai/omnidash\s*\n\s*ref:\s*([0-9a-f]{40})\b"
)


def _target_sha() -> str:
    fixture = yaml.safe_load(
        (
            REPO_ROOT / "tests/ci/fixtures/omnidash_sibling_pin_advance_omn19991.yaml"
        ).read_text(encoding="utf-8")
    )
    target = fixture["omnidash_ref"]
    assert isinstance(target, str) and SHA.fullmatch(target), target
    return target


def test_sibling_pins_carry_the_advanced_sha() -> None:
    pins = yaml.safe_load(
        (REPO_ROOT / ".github" / "sibling-pins.yaml").read_text(encoding="utf-8")
    )
    assert pins["pins"]["omnidash"] == _target_sha()


def test_contract_names_the_advanced_sha() -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts" / "OMN-19991.yaml").read_text(encoding="utf-8")
    )
    statements = [
        criterion["statement"]
        for requirement in contract["requirements"]
        for criterion in requirement.get("acceptance", ())
        if criterion["id"] == "AC2"
    ]
    assert len(statements) == 1, statements
    found = SHA.findall(statements[0])
    assert found == [_target_sha()], statements[0]


def test_every_inline_omnidash_ref_carries_the_advanced_sha() -> None:
    target = _target_sha()
    refs = {
        f"{path.name}: {ref}"
        for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
        for ref in INLINE_OMNIDASH_REF.findall(path.read_text(encoding="utf-8"))
    }
    assert refs
    assert {r for r in refs if not r.endswith(target)} == set()
