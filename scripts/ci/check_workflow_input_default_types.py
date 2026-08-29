# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workflow input default/type parity ratchet (OMN-16906).

What this exists to prevent
---------------------------
``deliver-dev-candidate-to-staging.yml`` — the workflow that makes an
omnibase_infra ``dev`` merge deliver itself to onex-dev (OMN-15796) — returned
GitHub's ``startup_failure`` on **every** invocation from the moment it landed
(runs 33077062502 / 33091593742 / 33106694579 / 33169436998). ``startup_failure``
means the run graph never compiled, so **no job exists and there is no log to
read**; the outage was invisible to every "did the deploy run?" check and cost
a full day of merged-but-undelivered dev commits before anyone noticed.

The cause, isolated by bisect (control run 33224162317 ``startup_failure`` vs.
single-variable run 33224392268 which compiled), was one character class:

.. code-block:: yaml

    # build-workspace-candidate-runtime.yml, workflow_call block
    no-cache:
      required: false
      default: "false"     # <-- a STRING
      type: boolean        # <-- declared BOOLEAN

GitHub type-checks a called workflow's ``workflow_call`` input defaults when it
compiles the **caller's** graph. A string default on a ``type: boolean`` input
fails that compile, and the failure is attributed to the caller, which is why
the callee kept passing its own ``workflow_dispatch`` runs green the whole time
(33107333244 success) while every caller run died at startup.

That asymmetry is the trap this ratchet closes. A bad default under
``workflow_dispatch`` alone is *latent* — GitHub tolerates it, so it sits in the
tree looking fine. It detonates the instant someone adds ``workflow_call`` to
that workflow and copies the input block down, which is exactly what OMN-15796
did. So this checker fails on **both** trigger blocks: the ``workflow_call``
violation is fatal today, and the ``workflow_dispatch`` violation is the same
defect one refactor away from being fatal.

Contract
--------
For every ``.github/workflows/*.y[a]ml`` input under ``workflow_call`` or
``workflow_dispatch`` that declares BOTH a ``type`` and a ``default``, the
default's YAML type must match the declared type:

* ``boolean`` accepts only ``bool``
* ``number`` accepts ``int`` or ``float``, and rejects ``bool`` (which
  subclasses ``int`` in Python and would otherwise slip through)
* ``string`` accepts only ``str``
* ``choice`` accepts only ``str``, and the value must be one of ``options``

Exit codes: ``0`` clean, ``1`` violations found (printed one per line).

Wired as a pre-commit hook AND asserted by
``tests/ci/test_workflow_input_default_type_parity.py`` — per CLAUDE.md rule 5,
a detector that is not a gate is advisory and gets ignored.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRIGGERS_WITH_INPUTS: tuple[str, ...] = ("workflow_call", "workflow_dispatch")


@dataclass(frozen=True)
class Violation:
    """One input whose parsed default does not match its declared type."""

    workflow: str
    trigger: str
    input_name: str
    declared_type: str
    default_repr: str
    default_python_type: str
    detail: str

    def render(self) -> str:
        return (
            f"{self.workflow}: on.{self.trigger}.inputs.{self.input_name} declares "
            f"type: {self.declared_type} but its default is {self.default_repr} "
            f"({self.default_python_type}) — {self.detail}"
        )


def _triggers(document: dict[Any, Any]) -> dict[str, Any]:
    """Return the ``on:`` mapping.

    ``yaml.safe_load`` resolves the bare key ``on`` to the boolean ``True``
    under the YAML 1.1 rules PyYAML implements, so both spellings are checked.
    A workflow whose ``on:`` is a string (``on: push``) or a list has no inputs
    and yields nothing.
    """
    raw = document.get("on", document.get(True))
    return raw if isinstance(raw, dict) else {}


def _check_default(declared: str, default: Any, options: Any) -> str | None:
    """Return a failure detail, or ``None`` when the default is well typed."""
    if declared == "boolean":
        if isinstance(default, bool):
            return None
        return (
            "GitHub rejects this when the workflow is CALLED — quote-strip it to a "
            "bare true/false"
        )
    if declared == "number":
        # bool is an int subclass in Python; a boolean default on a numeric
        # input is a real mismatch, not a passing narrow case.
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            return None
        return "a numeric input's default must be an unquoted number"
    if declared == "string":
        if isinstance(default, str):
            return None
        return "a string input's default must be quoted"
    if declared == "choice":
        if not isinstance(default, str):
            return "a choice input's default must be a quoted string"
        if isinstance(options, list) and default not in options:
            return f"default is not among options {options!r}"
        return None
    # An unknown `type:` value is itself invalid GitHub Actions syntax, but
    # this checker owns default/type parity only — report it rather than
    # silently accepting whatever default sits under it.
    return f"unknown input type {declared!r}"


def scan_document(workflow_name: str, document: dict[Any, Any]) -> list[Violation]:
    """Collect every default/type mismatch in one parsed workflow document."""
    violations: list[Violation] = []
    for trigger, block in _triggers(document).items():
        if trigger not in TRIGGERS_WITH_INPUTS or not isinstance(block, dict):
            continue
        inputs = block.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, spec in inputs.items():
            if not isinstance(spec, dict):
                continue
            declared = spec.get("type")
            # An input with no declared type is untyped-by-omission (GitHub
            # treats it as a string); an input with no default has nothing to
            # mismatch. Neither is this ratchet's concern.
            if not isinstance(declared, str) or "default" not in spec:
                continue
            detail = _check_default(declared, spec["default"], spec.get("options"))
            if detail is None:
                continue
            violations.append(
                Violation(
                    workflow=workflow_name,
                    trigger=trigger,
                    input_name=str(input_name),
                    declared_type=declared,
                    default_repr=repr(spec["default"]),
                    default_python_type=type(spec["default"]).__name__,
                    detail=detail,
                )
            )
    return violations


def scan_paths(paths: list[Path]) -> list[Violation]:
    """Scan concrete workflow files. Unparseable YAML is reported, not skipped."""
    violations: list[Violation] = []
    for path in sorted(paths):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            violations.append(
                Violation(
                    workflow=path.name,
                    trigger="<file>",
                    input_name="<parse>",
                    declared_type="-",
                    default_repr="-",
                    default_python_type="-",
                    detail=f"is not parseable YAML: {exc}",
                )
            )
            continue
        if isinstance(document, dict):
            violations.extend(scan_document(path.name, document))
    return violations


def workflow_files(workflows_dir: Path) -> list[Path]:
    return sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=Path(".github/workflows"),
        help="directory of workflow files to scan (default: .github/workflows)",
    )
    args = parser.parse_args(argv)

    if not args.workflows_dir.is_dir():
        print(
            f"::error::{args.workflows_dir} is not a directory — refusing to report "
            "clean on a scan that inspected nothing.",
            file=sys.stderr,
        )
        return 1

    violations = scan_paths(workflow_files(args.workflows_dir))
    if not violations:
        return 0

    print(
        "Workflow input default/type mismatch (OMN-16906). GitHub fails the "
        "CALLER's run with `startup_failure` — no job, no log — when a called "
        "workflow's input default does not match its declared type:",
        file=sys.stderr,
    )
    for violation in violations:
        print(f"  {violation.render()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
