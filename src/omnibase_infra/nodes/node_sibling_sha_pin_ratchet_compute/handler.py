# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure shrink-only decision over internal-sibling commit-sha pins (OMN-18594).

The baseline is the base ref itself: a file may hold no more pin lines after a
change than it held at the base. Today's pins are therefore the baseline, the
count can only fall, and no baseline file exists that a change could raise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationFindingEmbed,
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.models import (
    ModelSiblingShaPinRatchetInput,
)

VALIDATOR_ID: Final[str] = "sibling-sha-pin-ratchet"

# Every OmniNode package and repository name, in either separator form.
_SIBLING: Final[str] = r"(?:omni[a-z0-9]*(?:[-_][a-z0-9]+)*|onex[-_]change[-_]control)"
_SHA: Final[str] = r"[0-9a-f]{40}"
SHA_TOKEN: Final[str] = _SHA

# One pin per line at most; any of these shapes makes the line a pin.
# A uv.lock source resolved from a git TAG reads `<url>?tag=<tag>#<sha>`: the
# lock writes the commit the tag resolved to beside the tag, but the dependency
# is declared by the tag (RULING 2026-10-29T01:50:26Z, OMN-17427), so that
# line is not a commit-sha pin. `?rev=`, `?branch=` and bare `#<sha>` or
# `@<sha>` URLs still are.
_URL_PIN: Final = re.compile(
    rf"OmniNode-ai/{_SIBLING}(?![^\s\"'@#]*\?tag=)[^\s\"'@#]*[@#]{_SHA}(?![0-9a-f])",
    re.IGNORECASE,
)
_MAP_PIN: Final = re.compile(
    rf"^\s*[\"']?{_SIBLING}[\w-]*[\"']?\s*:\s*[\"']?{_SHA}(?![0-9a-f])",
    re.IGNORECASE,
)
_TOML_REV_PIN: Final = re.compile(rf"(?:^|[\s{{,])rev\s*=\s*[\"']{_SHA}[\"']")
# Workflow checkouts (`ref:` or `<x>_ref:`) and pre-commit hook repos (`rev:`)
# holding a sha. Counted only where every such key is a checkout or hook pin.
_YAML_REF_PIN: Final = re.compile(
    rf"(?:^|[\s{{,])(?:(?:[\w-]*_)?ref|rev)\s*:\s*[\"']?{_SHA}(?![0-9a-f])"
)

_OUT_OF_SCOPE: Final = re.compile(r"\.md$|^contracts/OMN-[0-9]+\.ya?ml$")
_OUT_OF_SCOPE_DIR: Final = re.compile(r"(^|/)(tests?|docs?|fixtures)/")


def in_scope(path: str) -> bool:
    """Whether a tracked path can carry a pin the ratchet counts."""
    return not _OUT_OF_SCOPE.search(path) and (
        path.startswith(".github/") or not _OUT_OF_SCOPE_DIR.search(path)
    )


def is_pin_line(path: str, line: str) -> bool:
    """Whether ``line`` of ``path`` pins an internal sibling to a commit sha."""
    if _URL_PIN.search(line) or _MAP_PIN.search(line) or _TOML_REV_PIN.search(line):
        return True
    yaml_pins = path.startswith(".github/") or path == ".pre-commit-config.yaml"
    return yaml_pins and bool(_YAML_REF_PIN.search(line))


def count_pins(lines: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Per-file count of pin lines, for files in scope that carry at least one."""
    counts: dict[str, int] = {}
    for path, line in lines:
        if in_scope(path) and is_pin_line(path, line):
            counts[path] = counts.get(path, 0) + 1
    return counts


class NodeSiblingShaPinRatchetCompute:
    """Definition-B handler: explicit frozen input to the OMN-2362 report."""

    def handle(self, request: ModelSiblingShaPinRatchetInput) -> ModelValidationReport:
        """Decide from the request alone, without filesystem, git, clock or env I/O."""
        before = count_pins(request.base_lines)
        after = count_pins(request.lines)
        findings = [
            ModelValidationFindingEmbed(
                validator_id=VALIDATOR_ID,
                severity="FAIL",
                location=path,
                message=(
                    f"{path}: {after[path]} commit-sha pin(s) of an internal sibling, "
                    f"{before.get(path, 0)} at {request.base_ref}. A new commit-sha pin is "
                    "refused (RULING 2026-10-23T12:01:59Z, OMN-18594)."
                ),
                remediation=(
                    "Depend on a released version through the overnight release train; "
                    "never pin a sibling to a commit. Moving an existing pin keeps the count, "
                    "and needs an operator or cloud-owner grant."
                ),
                rule_id="sibling-sha-pin-added",
            )
            for path in sorted(after)
            if after[path] > before.get(path, 0)
        ]
        return ModelValidationReport.from_findings(
            findings=tuple(findings),
            request=ModelValidationRequestRef(profile="default"),
            validators_run=(VALIDATOR_ID,),
        )
