# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compare committed probe placement variables with their live effective values.

The runner routing policy records the value that each scheduled probe's
``vars.NAME`` expression resolves to in its repository. This check reads one
repository's committed declarations and compares them with the JSON ``vars``
context supplied by a workflow in that repository. It reports every mismatch
so placement changes must be accompanied by a reviewed policy change.

WHY IT EXISTS (OMN-19412 follow-up)
    The required "Lab Probe Windows (OMN-19412)" check resolves a scheduled job
    placed by ``fromJSON(vars.NAME)`` from the committed map
    ``probe_placement_variables``, not from live variables, because reading live
    variables needed the operator's personal CROSS_REPO_PAT (no installed App
    can read Actions variables) and that token is being retired (operator RULING
    2026-09-27T22:46:17Z, item 3). A committed map is only as good as its
    agreement with the live values, so this probe checks that agreement.

WHY THE VARS CONTEXT
    A workflow run reads its own repository's effective variables (repository
    over organisation) through ``toJSON(vars)`` with no token and no App
    permission. ``.github/workflows/probe-placement-drift-reusable.yml`` passes
    that in; each of omnibase_infra, omninode_infra and omnimarket calls it on a
    schedule for its own entry.

Exit codes: 0 every declared value matches, 1 drift (each named), 2 unusable
input (policy, repository entry or vars JSON), never a clean pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2


class PolicyError(ValueError):
    """The policy or live variables input cannot be used safely."""


def load_declared(policy_path: Path, repo: str) -> dict[str, str | None]:
    """Load and validate one repository's committed placement variables."""
    try:
        text = policy_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PolicyError(f"{policy_path}: policy is unreadable: {exc}") from exc

    try:
        document: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"{policy_path}: policy is unreadable YAML: {exc}") from exc

    if not isinstance(document, Mapping):
        raise PolicyError(f"{policy_path}: policy must be a mapping")

    placements = document.get("probe_placement_variables")
    if not isinstance(placements, Mapping):
        raise PolicyError(
            f"{policy_path}: probe_placement_variables is missing or is not a mapping of mappings"
        )

    for repo_name, entry in placements.items():
        if not isinstance(repo_name, str) or not isinstance(entry, Mapping):
            raise PolicyError(
                f"{policy_path}: probe_placement_variables must be a mapping of mappings"
            )

    if repo not in placements:
        raise PolicyError(
            f"{policy_path}: repository {repo!r} is not declared in probe_placement_variables"
        )

    entry = placements[repo]
    if not isinstance(entry, Mapping):
        raise PolicyError(
            f"{policy_path}: probe_placement_variables.{repo} must be a mapping"
        )
    if not entry:
        raise PolicyError(
            f"{policy_path}: probe_placement_variables.{repo} must not be empty"
        )

    declared: dict[str, str | None] = {}
    for name, value in entry.items():
        if not isinstance(name, str):
            raise PolicyError(
                f"{policy_path}: every variable name in "
                f"probe_placement_variables.{repo} must be a string"
            )
        if value is None:
            declared[name] = None
        elif isinstance(value, str) and value:
            declared[name] = value
        else:
            raise PolicyError(
                f"{policy_path}: probe_placement_variables.{repo}.{name} must be "
                "a non-empty string or null"
            )
    return declared


def load_live(path_or_dash: str) -> dict[str, str]:
    """Load a workflow's JSON ``vars`` context from a file or standard input."""
    source = "stdin" if path_or_dash == "-" else path_or_dash
    try:
        if path_or_dash == "-":
            text = sys.stdin.read()
        else:
            text = Path(path_or_dash).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PolicyError(f"{source}: vars JSON is unreadable: {exc}") from exc

    try:
        document: object = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PolicyError(f"{source}: vars input is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise PolicyError(f"{source}: vars JSON must be an object")
    if not document:
        raise PolicyError(
            f"{source}: vars JSON object is empty; in this organisation that means "
            "the vars context was not passed, never that nothing is set"
        )

    live: dict[str, str] = {}
    for name, value in document.items():
        if not isinstance(name, str) or not isinstance(value, str):
            display_name = name if isinstance(name, str) else "<non-string name>"
            raise PolicyError(f"{source}: vars.{display_name} has a non-string value")
        live[name] = value
    return live


def _placement_labels(value: str) -> set[str] | None:
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, str) and parsed:
        return {parsed}
    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(label, str) and label for label in parsed)
    ):
        return set(parsed)
    return None


def placement_equal(a: str, b: str) -> bool:
    """Compare JSON runner labels as sets, falling back to stripped text."""
    a_labels = _placement_labels(a)
    b_labels = _placement_labels(b)
    if a_labels is not None and b_labels is not None:
        return a_labels == b_labels
    return a.strip() == b.strip()


def _display(value: str | None) -> str:
    return "unset" if value is None else repr(value)


def drift(
    repo: str,
    declared: Mapping[str, str | None],
    live: Mapping[str, str],
) -> list[str]:
    """Return one diagnostic for every declared variable that has drifted."""
    differences: list[str] = []
    for name in sorted(declared):
        committed_value = declared[name]
        live_value = live.get(name) or None
        agrees = (committed_value is None and live_value is None) or (
            committed_value is not None
            and live_value is not None
            and placement_equal(committed_value, live_value)
        )
        if agrees:
            continue
        differences.append(
            f"DRIFT: {repo} vars.{name}: committed {_display(committed_value)}, "
            f"live {_display(live_value)}. Fix: change the variable back, or change "
            "config/runner_routing_policy.yaml "
            f"probe_placement_variables.{repo}.{name} in a reviewed pull request "
            "(rule 14: same action as the variable write)."
        )
    return differences


def main(argv: Sequence[str] | None = None) -> int:
    """Run the drift check for one repository."""
    parser = argparse.ArgumentParser(
        description="Compare committed probe placement variables with live values."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--vars-json-file", required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args(argv)

    try:
        declared = load_declared(args.policy, args.repo)
        live = load_live(args.vars_json_file)
    except PolicyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    differences = drift(args.repo, declared, live)
    for line in differences:
        print(line, file=sys.stderr)

    total = len(declared)
    if differences:
        print(
            f"probe placement drift: {len(differences)} of {total} declared "
            f"variable(s) in {args.repo} differ from the live value"
        )
        return EXIT_DRIFT

    print(
        f"probe placement drift: OK; {total} declared variable(s) in "
        f"{args.repo} match the live value"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
