# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18272 — this repo's cascade generator must emit machine-readable provenance.

``omnibase_core``'s dependency-cascade generator has emitted a ``Cascade
provenance`` block since OMN-16286, and OMN-18235 (``omnibase_core#1685``,
squash ``2b8bf73022193a6c1bb4dfd6e2875184fc886c6a``) made a test and a scoped
pre-commit hook assert it so the generator cannot drop it in an unrelated edit.

``omnibase_infra``'s own ``dependency-cascade.yml`` was never given the
equivalent treatment and emitted **no provenance block at all**. Clause 1 of the
OMN-18233 verified-supersession predicate resolves against that block, so every
cascade bump this repository opened failed the predicate closed: a closed-
unmerged bump could never be proven superseded, and the releasing ticket it
cites stayed blocked by a pull request that can never merge. The predicate fails
in the safe direction — a permanent hold rather than a wrong flip — and still
costs exactly the lane-hours phase 1 of OMN-18232 exists to remove.

**The consumer lives in THIS repository, so this test imports it.** The
precedent in ``omnibase_core`` had to carry a structural copy of the parser's
regexes, because ``omnibase_core`` may not depend on ``omnibase_infra``. Here
there is no such constraint: ``parse_cascade_provenance`` at
``src/omnibase_infra/nodes/node_evidence_autoclose_sweep_effect/cascade_supersession.py``
is the real clause-1 reader, and asserting against it means the generator is
pinned to the schema the closer actually parses rather than to a second copy of
it that can drift. A copied schema that agrees with itself proves nothing.

**Every assertion has a negative control.** A test that asserts a string is
present in a file passes just as happily when it is asserting nothing, so each
structural rule below is also run against a body that violates it and is
required to resolve to ``None``. That is what makes "a cascade bump with no
machine-readable provenance block fails the generator's own test" a measured
property rather than an intention.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.cascade_supersession import (
    parse_cascade_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "dependency-cascade.yml"

#: The step whose `gh pr create --body` heredoc IS the emitted pull request body.
_PR_STEP_NAME = "Open pull request"
_JOB_NAME = "open-bump-pr"

#: Concrete values standing in for the workflow's own resolved outputs. The
#: package and version are this repository's own, because this is the generator
#: that cascades an `omnibase_infra` release into its downstream repositories.
_RENDER_VALUES = {
    "steps.vars.outputs.package": "omnibase_infra",
    "steps.vars.outputs.pkg_hyphen": "omnibase-infra",
    "steps.vars.outputs.version": "0.47.12",
    "steps.vars.outputs.ticket": "OMN-18201",
    "steps.vars.outputs.evidence_source": "OCC#9159",
    "steps.vars.outputs.branch": "automation/bump-omnibase-infra-0.47.12-omn-18201",
    "steps.vars.outputs.base": "dev",
    "inputs.package": "omnibase_infra",
    "inputs.version": "0.47.12",
    "matrix.repo": "omnimarket",
    "github.server_url": "https://github.com",
    "github.repository": "OmniNode-ai/omnibase_infra",
    "github.run_id": "34665021693",
}

_EXPRESSION_RE = re.compile(r"\$\{\{\s*(?P<expr>[^}]+?)\s*\}\}")

#: Used only by the "not a literal" assertion, to locate the two bullets in the
#: UNRENDERED template. The parse itself is delegated to the real consumer.
_PROVENANCE_HEADING_RE = re.compile(r"^#{2,6}\s+cascade\s+provenance\b", re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _pr_create_script() -> str:
    with WORKFLOW_PATH.open() as handle:
        document = yaml.safe_load(handle)
    job = document["jobs"][_JOB_NAME]
    for step in job["steps"]:
        if isinstance(step, dict) and step.get("name") == _PR_STEP_NAME:
            run = step.get("run", "")
            assert isinstance(run, str)
            return run
    raise AssertionError(
        f"job {_JOB_NAME!r} has no step named {_PR_STEP_NAME!r}; the generator's "
        "pull request body could not be located, which is itself the failure "
        "this module exists to catch"
    )


def _body_template() -> str:
    """The heredoc that becomes the bump pull request's body, dedented."""
    script = _pr_create_script()
    opener = "<<'PREOF'"
    start = script.find(opener)
    assert start != -1, (
        "the `gh pr create` step no longer builds its body from a quoted "
        f"heredoc; the emitted body cannot be read:\n{script}"
    )
    start = script.index("\n", start) + 1
    end = script.find("PREOF", start)
    assert end != -1, "unterminated body heredoc in the `gh pr create` step"
    return textwrap.dedent(script[start:end])


def _render(template: str) -> str:
    """The template with its workflow expressions resolved to concrete values.

    An expression this module does not know about renders to a visible marker
    rather than to an empty string: a silently-empty substitution would let a
    field that resolves to nothing pass a presence assertion.
    """

    def substitute(match: re.Match[str]) -> str:
        expression = match.group("expr")
        return _RENDER_VALUES.get(expression, f"<UNRESOLVED:{expression}>")

    return _EXPRESSION_RE.sub(substitute, template)


def _provenance_section_lines(template: str) -> list[str]:
    """The raw template lines inside the provenance section, unrendered."""
    collected: list[str] = []
    in_section = False
    for line in template.splitlines():
        stripped = line.strip()
        if _PROVENANCE_HEADING_RE.match(stripped):
            in_section = True
            continue
        if in_section and _ANY_HEADING_RE.match(stripped):
            break
        if in_section:
            collected.append(line)
    return collected


# ------------------------------------------------------- the assertion -------


@pytest.mark.unit
def test_the_generator_emits_a_provenance_block_the_closer_can_read() -> None:
    """AC1, positive half, measured through the real clause-1 parser."""
    body = _render(_body_template())
    declared = parse_cascade_provenance(body)

    assert declared is not None, (
        "this repository's cascade generator emits no machine-readable "
        "provenance block. Clause 1 of the OMN-18233 supersession predicate "
        "resolves against this block, so without it every closed cascade bump "
        "this generator opens blocks its releasing ticket permanently:\n" + body
    )
    assert declared.source_repo == "OmniNode-ai/omnibase_infra", declared.source_repo
    assert declared.required_version == "0.47.12", declared.required_version
    assert declared.distribution == "omnibase-infra", declared.distribution


@pytest.mark.unit
def test_both_fields_resolve_from_the_workflow_and_are_not_literals() -> None:
    """The fields must carry the run's OWN package and version.

    A block hardcoding a package name would read as present forever and be
    wrong on every bump but one. The version in particular must be the
    `vars`-step output rather than `inputs.version`: the step strips a leading
    `v`, and clause 3 compares the field as a version against a pin.
    """
    section = "\n".join(_provenance_section_lines(_body_template()))
    source_line = ""
    version_line = ""
    for line in section.splitlines():
        if re.match(r"^\s*[-*]\s*Source repo:", line, re.IGNORECASE):
            source_line = line
        if re.match(r"^\s*[-*]\s*Released version:", line, re.IGNORECASE):
            version_line = line

    assert "${{ steps.vars.outputs.package }}" in source_line, source_line
    assert "${{ steps.vars.outputs.version }}" in version_line, version_line
    assert "<UNRESOLVED:" not in _render(source_line + "\n" + version_line)


@pytest.mark.unit
def test_the_source_repo_field_renders_to_an_owner_slash_name() -> None:
    """Clause 1 needs a repository, not a bare package name.

    The consumer derives the pinned distribution from the repository's NAME
    half, so a field carrying only `omnibase_infra` would parse as a repo with
    no owner and resolve to nothing.
    """
    declared = parse_cascade_provenance(_render(_body_template()))
    assert declared is not None
    assert declared.source_repo.count("/") == 1, declared.source_repo
    owner, name = declared.source_repo.split("/")
    assert owner and name, declared.source_repo


# ------------------------------------------------- the negative controls -----


@pytest.mark.unit
def test_a_body_with_the_provenance_heading_removed_fails_the_assertion() -> None:
    """AC1, the half that makes the assertion above mean something.

    This is the exact regression the ticket names: the generator drops the
    block in an unrelated edit. It must resolve to nothing, and this control
    proves the reader is capable of resolving to nothing at all.
    """
    body = _render(_body_template())
    without_heading = "\n".join(
        line for line in body.splitlines() if not _PROVENANCE_HEADING_RE.match(line)
    )

    assert parse_cascade_provenance(without_heading) is None


@pytest.mark.unit
def test_a_body_missing_only_the_version_field_fails_the_assertion() -> None:
    """A half-emitted block is not a block. Clause 1 needs both fields."""
    body = _render(_body_template())
    without_version = "\n".join(
        line
        for line in body.splitlines()
        if not re.match(r"^\s*[-*]\s*Released version:", line, re.IGNORECASE)
    )

    assert parse_cascade_provenance(without_version) is None


@pytest.mark.unit
def test_a_body_missing_only_the_source_repo_field_fails_the_assertion() -> None:
    body = _render(_body_template())
    without_source = "\n".join(
        line
        for line in body.splitlines()
        if not re.match(r"^\s*[-*]\s*Source repo:", line, re.IGNORECASE)
    )

    assert parse_cascade_provenance(without_source) is None


@pytest.mark.unit
def test_fields_outside_the_provenance_section_do_not_count() -> None:
    """The section binding is a rule, not an accident of ordering.

    A generator that moved the two bullets up under `## Summary` would keep
    every substring present while declaring no provenance, and a loose reader
    would call that a pass.
    """
    loose = (
        "## Summary\n\n"
        "- Source repo: `OmniNode-ai/omnibase_infra`\n"
        "- Released version: `0.47.12`\n\n"
        "## Test plan\n\n- [ ] CI passes\n"
    )

    assert parse_cascade_provenance(loose) is None


@pytest.mark.unit
def test_an_ordinary_pull_request_body_declares_no_provenance() -> None:
    """The reader must not turn every pull request into a cascade bump."""
    ordinary = (
        "## Summary\n\nFixes the disk guard so it halts on free space.\n\n"
        "## Test plan\n\n- [x] unit tests\n"
    )

    assert parse_cascade_provenance(ordinary) is None
