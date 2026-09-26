# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A producer must say which tenant its event belongs to (OMN-16831, item 2).

THE RULING THIS ENFORCES
------------------------
The 2026-08-28 operator ruling on OMN-16831 adopted option (D) and its governing
principle: *"``VerifiedProjectionTenantAuthority`` is an authorization artifact
and must never be the source of ``tenant_id``. Attribution is recorded by the
producer at write time; the authority, when present, only verifies it."* Item 2
of the ruled change set is write-path stamping -- families with a tenant in
scope record it, and **families with no tenant concept record explicitly-none
rather than nothing**.

WHY A GATE AND NOT A CONVENTION
-------------------------------
``ModelEventEnvelope.tenant_id`` defaults to ``None``, so at the model level an
omission is indistinguishable from a deliberate "no tenant here". The
declaration at the construction site is the ONLY thing that separates them, and
an unenforced declaration is how the platform got here: the gateway heartbeat
had its deploy-bound tenant identity in hand and wrote it to the payload only;
the two dispatch re-materialization helpers carried correlation, timestamp and
event type across a hop and dropped the tenant; the projection terminal
publisher bypasses the result applier by construction and so was never reached
by the OMN-16831 carriage fix. Each was silent, and each ended in a
tenant-classified projection write refused fail-closed at the far end, because
a writer under ``FORCE ROW LEVEL SECURITY`` cannot discover a row's tenant by
reading.

WHAT IT REFUSES
---------------
Any construction of ``ModelEventEnvelope`` in this package's shipped source
(``src/omnibase_infra``) that does not pass ``tenant_id`` explicitly.
``tenant_id=None`` passes -- that IS the declaration. A ``**kwargs`` expansion
is refused rather than waved through: it cannot be read statically, and a
checker that passed it would ship a one-character bypass.

WHAT IT DOES NOT COVER, DELIBERATELY
------------------------------------
* **Tests.** A fixture envelope is not a write path.
* **``omnibase_core`` and ``omnimarket``.** Both carry the same surface (24 and
  40 construction sites as of 2026-09-18) and neither is covered here. Ruled
  item 2 names the runtime, which is this package. The other two are recorded
  on OMN-16831 as named follow-up, not quietly skipped.
* **Whether the recorded tenant is TRUE.** This proves a producer answered the
  question. The projection authority's fail-closed verification is what judges
  the answer. Same honest limit as every other blast-radius gate here.

There is no ``--baseline`` flag and there must never be one (OMN-18013;
``no-baseline-refreeze`` refuses the return of every baseline it burned).

Usage (pre-commit / CI):
    uv run python -m omnibase_infra.validators.envelope_tenant_dimension src/omnibase_infra
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: The shipped source this gate governs, relative to the repo root.
PACKAGE_ROOT = Path("src/omnibase_infra")

#: The constructor whose tenant dimension is under enforcement.
ENVELOPE_CLASS = "ModelEventEnvelope"

#: The field a producer must declare.
TENANT_FIELD = "tenant_id"


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class UndeclaredTenantFinding:
    """One envelope construction that did not say which tenant it belongs to."""

    path: str
    line: int
    detail: str


def _constructor_name(func: ast.expr) -> str | None:
    """The class name a call targets, seeing through the subscripted form.

    ``ModelEventEnvelope[dict[str, object]](...)`` is the common spelling in
    this package; a checker matching only the bare ``ast.Name`` would miss most
    real sites.
    """
    if isinstance(func, ast.Subscript):
        func = func.value
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _findings_for_source(text: str, rel_path: str) -> list[UndeclaredTenantFinding]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover - a repo that does not parse
        return [
            UndeclaredTenantFinding(
                path=rel_path,
                line=exc.lineno or 0,
                detail=f"file does not parse, so its producers cannot be checked: {exc.msg}",
            )
        ]

    out: list[UndeclaredTenantFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _constructor_name(node.func) != ENVELOPE_CLASS:
            continue
        keywords = {kw.arg for kw in node.keywords}
        if TENANT_FIELD in keywords:
            continue
        if None in keywords:
            out.append(
                UndeclaredTenantFinding(
                    path=rel_path,
                    line=node.lineno,
                    detail=(
                        f"{ENVELOPE_CLASS} built from a `**` expansion, so the "
                        f"`{TENANT_FIELD}` declaration cannot be read here. Pass "
                        f"`{TENANT_FIELD}=` explicitly at this call."
                    ),
                )
            )
            continue
        out.append(
            UndeclaredTenantFinding(
                path=rel_path,
                line=node.lineno,
                detail=(
                    f"{ENVELOPE_CLASS} built without declaring `{TENANT_FIELD}`. "
                    "Record the tenant this event belongs to, or declare "
                    f"`{TENANT_FIELD}=None` to state that this family has none "
                    "(OMN-16831 ruled item 2). An omission and a deliberate none "
                    "are indistinguishable to every reader."
                ),
            )
        )
    return out


def findings_paths(paths: Sequence[Path]) -> list[UndeclaredTenantFinding]:
    """Every envelope construction in the supplied files or directories."""
    out: list[UndeclaredTenantFinding] = []
    source_paths: set[Path] = set()
    for candidate in paths:
        if candidate.is_file() and candidate.suffix == ".py":
            source_paths.add(candidate)
        elif candidate.is_dir():
            source_paths.update(candidate.rglob("*.py"))
    for path in sorted(source_paths):
        out.extend(_findings_for_source(path.read_text(encoding="utf-8"), str(path)))
    return out


def findings(root: Path) -> list[UndeclaredTenantFinding]:
    """Every envelope construction under ``root`` that declares no tenant."""
    return findings_paths([root])


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refuse an event envelope built without declaring its tenant "
            "dimension (OMN-16831 ruled item 2)."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[PACKAGE_ROOT])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    paths = list(args.paths)
    for path in paths:
        if not path.exists():
            sys.stderr.write(
                f"[envelope-tenant-dimension] FAIL: {path} does not exist, so "
                "the requested subject was not checked.\n"
            )
            return 1
    found = findings_paths(paths)
    if found:
        sys.stderr.write(
            "[envelope-tenant-dimension] FAIL: a producer did not say which "
            "tenant its event belongs to (OMN-16831 item 2):\n"
        )
        for finding in found:
            sys.stderr.write(
                f"  - {finding.path}:{finding.line}\n      {finding.detail}\n"
            )
        return 1
    sys.stderr.write(
        f"[envelope-tenant-dimension] OK: every {ENVELOPE_CLASS} construction "
        f"in {len(paths)} requested path(s) declares its tenant dimension.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
