# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command line of the atomic in-place venv install (OMN-22712).

The host reconcile and ``install-node-skill-package.sh`` run this under the
interpreter of the venv they repair, before it is known to be healthy, so it
imports nothing from the package: the engine modules are loaded by path. The
typed form of the same operation is ``HandlerVenvAtomicInstall``.

    exchange A B | supported | installed-commit | verify
    stage --python P --stage-dir D [--no-deps] [--label L] -- REQUIREMENT ...
    commit --python P --stage-dir D | rollback --stage-dir D | finalize --stage-dir D
    staging-lock-acquire --path P --pid N | staging-lock-release --path P --pid N
    staging-lock-held --path P
    promote-floor --venv V --uv-lock L --workspace-root H --floor F [--changed]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "handlers" / "venv_atomic_install"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ENGINE_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """Parse one command and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "supported",
            "exchange",
            "stage",
            "commit",
            "rollback",
            "finalize",
            "verify",
            "installed-commit",
            "staging-lock-acquire",
            "staging-lock-release",
            "staging-lock-held",
            "promote-floor",
        ],
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--uv-lock", type=Path)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--floor", type=Path)
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--python")
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--no-deps", action="store_true")
    parser.add_argument("--label", default="install")
    parser.add_argument("rest", nargs="*")
    args = parser.parse_intermixed_args(argv)
    if args.command in (
        "staging-lock-acquire",
        "staging-lock-release",
        "staging-lock-held",
        "promote-floor",
    ):
        return int(
            _load("dispatch_venv_promotion").run_promotion_command(
                args.command.replace("-", "_"),
                path=args.path,
                pid=args.pid,
                venv=args.venv,
                uv_lock=args.uv_lock,
                workspace_root=args.workspace_root,
                floor=args.floor,
                changed=args.changed,
            )
        )
    if args.command in ("verify", "installed-commit"):
        verify = _load("skill_install_verify")
        if args.command == "installed-commit":
            return int(verify.installed_market_commit())
        return int(verify.verify_nodes() or verify.verify_cli())
    engine = _load("atomic_venv_install")
    return int(
        engine.run_command(
            args.command,
            python_bin=args.python,
            stage_dir=args.stage_dir,
            requirements=args.rest,
            no_deps=args.no_deps,
            label=args.label,
            paths=args.rest,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
