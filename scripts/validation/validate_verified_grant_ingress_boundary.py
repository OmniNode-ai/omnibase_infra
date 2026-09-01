# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Enforce that only safe composition can create VERIFIED grant rows."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "src" / "omnibase_infra"
_COMPOSITION = _SOURCE / "runtime/first_effect_ledger/composition.py"
_REMOVED = {
    "record_verified_grant",
    "_record_verified_grant",
    "ModelFirstEffectVerifiedGrantProjection",
    "_ModelVerifiedFirstEffectGrantProjection",
    "PostgresVerifiedFirstEffectRecorder",
    "_PostgresVerifiedFirstEffectRecorder",
}


def _violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Name):
            names = (node.id,)
        elif isinstance(node, ast.Attribute):
            names = (node.attr,)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = (node.value,)
        for name in names:
            if name in _REMOVED:
                result.append(
                    f"{getattr(node, 'lineno', 0)}: removed verified-row capability"
                )
    if (
        "public.first_effect_verified_grant_ledger" in source
        and "INSERT INTO {_TABLE}" in source
        and path != _COMPOSITION
    ):
        result.append("0: VERIFIED insert outside safe composition")
    return result


def main() -> int:
    failures = {
        path.relative_to(_ROOT): _violations(path) for path in _SOURCE.rglob("*.py")
    }
    for path, errors in sorted(failures.items()):
        for error in errors:
            print(f"{path}: {error}", file=sys.stderr)
    return int(any(failures.values()))


if __name__ == "__main__":
    raise SystemExit(main())
