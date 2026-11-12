# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI boundary of node_sibling_clone_pin_effect (OMN-23947).

    python runtime_sibling_clone_pin.py --package omnibase-core --repo-root <clone>

Prints the node's result as one JSON object and exits 0, or prints the refusal to
stderr and exits 2. It runs from its file path as well as with ``-m``: the
workspace-build interpreter holds pydantic and nothing else, so when the
``omnibase_infra`` package cannot import, only the packages on the way to the
handler and its models are mounted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path


def _mount_bare_packages() -> None:
    """Mount the packages on the way to the handler, without their inits.

    A bare interpreter cannot import the omnibase_infra package inits (they import
    omnibase_core).
    """
    base = Path(__file__).resolve().parents[2]
    for parts in (("handlers",), ("models",), ()):
        name = ".".join(("omnibase_infra", *parts))
        mod = types.ModuleType(name)
        mod.__path__ = [str(base.joinpath(*parts))]
        sys.modules[name] = mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from omnibase_infra.handlers.handler_sibling_clone_pin import (
            resolve_clone_pin,
        )
        from omnibase_infra.models.sibling_pin import ModelSiblingClonePinRequest
    except ModuleNotFoundError:
        _mount_bare_packages()
        from omnibase_infra.handlers.handler_sibling_clone_pin import (
            resolve_clone_pin,
        )
        from omnibase_infra.models.sibling_pin import ModelSiblingClonePinRequest
    try:
        result = resolve_clone_pin(
            ModelSiblingClonePinRequest(package=args.package, repo_root=args.repo_root)
        )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    sys.stdout.write(result.model_dump_json() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
