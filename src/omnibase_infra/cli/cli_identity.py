# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex identity`` — inspect a runtime-identity stamp, and assert a probe's
target (OMN-17310 / OMN-17312, epic OMN-17306).

Two subcommands:

``onex identity stamp``
    Print this process's stamp without running a dispatch. Lets a shell probe
    capture "what am I about to run" before it runs anything, and lets an
    operator answer "which omnimarket is this venv on" in one command instead
    of by reading ``direct_url.json`` by hand.

``onex identity assert-target``
    Compare a stamp against a target's own declaration and FAIL CLOSED on
    disagreement — the mechanical form of the manual check that separated the
    valid 2026-08-31 in-lane probe from the invalid one two hours earlier
    (OMN-16932 record; OMN-17295). The invalid probe published over the ``.201``
    broker, so it looked addressed to the lane, while its orchestrator resolved
    out of the operator's local venv on pre-fix code; the lane's own logs had
    zero hits for the correlation id. Both probes printed a confident result.

The declaration must come from a surface the TARGET emits about itself — the
deployed ``build-provenance.json``, a ``direct_url.json`` read back out of the
running container. ``--from-build-provenance`` reads exactly that manifest, so
the common case needs no hand-authored declaration file. A caller-supplied
"I meant the dev lane" is not a declaration; it is the intent that was already
wrong in OMN-17295.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_core.models.runtime.model_declared_target_identity import (
    ModelDeclaredTargetIdentity,
)
from omnibase_core.models.runtime.model_runtime_identity import ModelRuntimeIdentity
from omnibase_core.validation.validator_probe_target import (
    ProbeTargetMismatchError,
    assert_probe_target,
)
from omnibase_infra.runtime_identity import (
    collect_runtime_identity,
    render_identity_line,
)

__all__ = [
    "assert_target_command",
    "declaration_from_build_provenance",
    "identity_group",
    "stamp_command",
]

_COMMIT_LENGTH = 40


@click.group("identity")
def identity_group() -> None:  # stub-ok: click group, subcommands added below
    """Runtime-identity stamp: what code ran, where, against what config."""


@click.command("stamp")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the full block as JSON on stdout instead of the one-line form.",
)
def stamp_command(as_json: bool) -> None:
    """Print this process's runtime-identity stamp.

    The one-line default is the glance-check; ``--json`` is the full block, and
    round-trips through ``ModelRuntimeIdentity`` so a probe script can diff it.
    """
    identity = collect_runtime_identity()
    if as_json:
        click.echo(identity.model_dump_json(indent=2))
    else:
        click.echo(render_identity_line(identity))


def declaration_from_build_provenance(
    manifest_path: Path,
    *,
    target_name: str,
) -> ModelDeclaredTargetIdentity:
    """Build a declaration from a runtime image's own build-provenance manifest.

    Reads ``per_repo_vcs_provenance.siblings.<repo>.vcs_ref`` — the block
    ``scripts/runtime_build/compute_workspace_provenance.py`` writes at image
    build, and the ONLY place a workspace-mode image's vendored SHAs survive
    (``.git`` is stripped from the staged tree). Shape verified live against
    ``omninode-runtime`` on ``.201``, 2026-08-31.

    A sibling marked ``vcs_dirty`` is DROPPED rather than declared: a dirty
    tree's HEAD does not identify its content, so asserting against it would
    manufacture precision the manifest does not have.

    Raises:
        click.ClickException: the manifest is unreadable, or declares nothing.
    """
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(
            f"could not read build-provenance manifest {manifest_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    siblings = {}
    if isinstance(parsed, dict):
        block = parsed.get("per_repo_vcs_provenance")
        if isinstance(block, dict) and isinstance(block.get("siblings"), dict):
            siblings = block["siblings"]

    packages: dict[str, str] = {}
    for repo, entry in siblings.items():
        if not isinstance(entry, dict) or entry.get("vcs_dirty"):
            continue
        ref = entry.get("vcs_ref")
        if isinstance(ref, str) and len(ref) == _COMMIT_LENGTH:
            packages[str(repo)] = ref

    if not packages:
        raise click.ClickException(
            f"{manifest_path} declares no clean sibling commits "
            "(per_repo_vcs_provenance.siblings empty, or every entry dirty). "
            "An empty declaration cannot be asserted against — it would "
            "compare nothing and pass unconditionally."
        )

    return ModelDeclaredTargetIdentity(
        target_name=target_name,
        declared_by=str(manifest_path),
        packages=packages,
    )


def _load_stamp(receipt_path: Path | None) -> ModelRuntimeIdentity:
    """Return the stamp to assert: a receipt's, or this process's own."""
    if receipt_path is None:
        return collect_runtime_identity()
    raw = (
        sys.stdin.read()
        if str(receipt_path) == "-"
        else receipt_path.read_text(encoding="utf-8")
    )
    try:
        receipt: ModelSkillResult[object] = ModelSkillResult.model_validate_json(raw)
    except ValueError as exc:
        raise click.ClickException(
            f"{receipt_path} did not parse as a skill-dispatch receipt: {exc}"
        ) from exc
    if receipt.runtime_identity is None:
        raise click.ClickException(
            f"{receipt_path} carries no runtime_identity block, so there is "
            "nothing to assert. A receipt that does not identify the process "
            "that produced it cannot prove a target (OMN-17308)."
        )
    return receipt.runtime_identity


@click.command("assert-target")
@click.option(
    "--declared",
    "declared_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help=(
        "A ModelDeclaredTargetIdentity JSON document read from the target's "
        "own surface."
    ),
)
@click.option(
    "--from-build-provenance",
    "provenance_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help=(
        "A runtime image's /app/build-provenance.json, read back out of the "
        "running container. The declaration is derived from its own vendored "
        "SHAs — never from the caller's intent."
    ),
)
@click.option(
    "--target-name",
    "target_name",
    default=None,
    help="Name for the target in the failure message (with --from-build-provenance).",
)
@click.option(
    "--receipt",
    "receipt_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Receipt whose runtime_identity to assert ('-' for stdin). Omit to "
        "assert THIS process's own stamp."
    ),
)
def assert_target_command(
    declared_path: Path | None,
    provenance_path: Path | None,
    target_name: str | None,
    receipt_path: Path | None,
) -> None:
    """Assert a stamp satisfies a target's declaration; exit non-zero if not.

    Fails closed on UNKNOWN as loudly as on MISMATCH. In every incident this
    exists for, the honest answer was "I cannot tell" and every surface
    rendered it as "fine".
    """
    if (declared_path is None) == (provenance_path is None):
        raise click.UsageError(
            "pass exactly one of --declared or --from-build-provenance"
        )

    if provenance_path is not None:
        if not target_name:
            raise click.UsageError(
                "--target-name is required with --from-build-provenance"
            )
        declared = declaration_from_build_provenance(
            provenance_path, target_name=target_name
        )
    else:
        assert declared_path is not None  # narrowed by the XOR check above
        try:
            declared = ModelDeclaredTargetIdentity.model_validate_json(
                declared_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"could not read declaration {declared_path}: {exc}"
            ) from exc

    stamped = _load_stamp(receipt_path)

    try:
        verdict = assert_probe_target(stamped=stamped, declared=declared)
    except ProbeTargetMismatchError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(
        f"probe-target assertion PASSED for {verdict.target_name!r} "
        f"(declared by {verdict.declared_by}); compared: "
        f"{', '.join(verdict.compared_fields)}"
    )


identity_group.add_command(stamp_command)
identity_group.add_command(assert_target_command)
