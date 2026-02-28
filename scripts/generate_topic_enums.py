#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
#
# generate_topic_enums.py — CLI entry point for contract-driven topic enum generation.
#
# Wires together ContractTopicExtractor (OMN-2963) and TopicEnumGenerator (OMN-2964)
# to produce per-producer Python enum files under src/omnibase_infra/enums/generated/.
#
# Ticket: OMN-2966
#
# Usage:
#   # Generate (or regenerate) enum files from current contracts
#   uv run python scripts/generate_topic_enums.py --generate
#
#   # Check that generated files are up to date (CI/pre-commit)
#   uv run python scripts/generate_topic_enums.py --check
#
#   # Override defaults
#   uv run python scripts/generate_topic_enums.py --generate \
#       --contracts-root src/omnibase_infra/nodes/ \
#       --output-dir src/omnibase_infra/enums/generated/
#
# Exit Codes:
#   --generate:
#     0  Success
#     2  Hard-stop (inconsistent parse — RuntimeError from extractor)
#     3  Unexpected internal error
#
#   --check:
#     0  Clean — generated files match current contracts
#     1  Drift detected (content changed, missing file, or stale extra file)
#     2  Hard-stop (inconsistent parse — RuntimeError from extractor)
#     3  Unexpected internal error
#
# Design decisions:
#   - Writes are atomic: temp file → os.replace(). A killed run never truncates.
#   - Stale cleanup: only removes files matching enum_*_topic.py in enums/generated/.
#     Never removes __init__.py or files outside the generated/ directory.
#   - Producer list is derived entirely from extractor output — no hardcoded list.
#   - Repo-root discovered via Path(__file__).resolve(), not CWD.
#   - Extractor warnings (to stderr) are informational — do NOT escalate to failures
#     in either --generate or --check mode.

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root and output-dir discovery
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CONTRACTS_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "enums" / "generated"

# Pattern for stale-file detection (only these may be removed)
_STALE_PATTERN = "enum_*_topic.py"


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    """
    Write content to path atomically: write to a temp file in the same directory,
    then os.replace(temp, target). A killed run never leaves truncated files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".py")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        tmp_path.replace(path)
    except Exception:
        # Clean up temp file on error
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_topic_enums.py",
        description=(
            "Contract-driven Kafka topic enum generator. "
            "Reads topics from contract.yaml files and renders per-producer Python enum files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  --generate:
    0  Success
    2  Hard-stop (inconsistent parse)
    3  Unexpected internal error

  --check:
    0  Clean
    1  Drift detected (content or file set mismatch)
    2  Hard-stop (inconsistent parse)
    3  Unexpected internal error

Examples:
  # Generate / regenerate all enum files
  uv run python scripts/generate_topic_enums.py --generate

  # Check for drift (CI / pre-commit)
  uv run python scripts/generate_topic_enums.py --check

  # Override contract root
  uv run python scripts/generate_topic_enums.py --generate \\
      --contracts-root src/omnibase_infra/nodes/
""",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="Generate (or regenerate) per-producer enum files from current contracts.",
    )
    mode_group.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Check that generated files are up to date. "
            "Exits 1 on drift, missing file, or stale extra enum_*_topic.py."
        ),
    )
    parser.add_argument(
        "--contracts-root",
        metavar="PATH",
        default=None,
        help=(
            f"Root directory to scan for contract.yaml files. "
            f"Default: {_DEFAULT_CONTRACTS_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help=(
            f"Output directory for generated enum files. Default: {_DEFAULT_OUTPUT_DIR}"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Core logic — generate
# ---------------------------------------------------------------------------


def _run_generate(contracts_root: Path, output_dir: Path) -> int:
    """
    Generate per-producer enum files.

    Returns 0 on success, 2 on hard-stop (inconsistent parse), 3 on unexpected error.
    """
    try:
        from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor
        from omnibase_infra.tools.topic_enum_generator import TopicEnumGenerator
    except ImportError as exc:
        print(f"ERROR: Failed to import tools: {exc}", file=sys.stderr)
        return 3

    try:
        extractor = ContractTopicExtractor()
        entries = extractor.extract(contracts_root)
    except RuntimeError as exc:
        # Hard-stop: inconsistent parsed components (parser bug)
        print(f"ERROR (hard-stop): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: Unexpected error during extraction: {exc}", file=sys.stderr)
        return 3

    try:
        generator = TopicEnumGenerator()
        rendered = generator.render(entries, output_dir=output_dir)
    except RuntimeError as exc:
        # Hard-stop: enum key collision
        print(f"ERROR (hard-stop): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: Unexpected error during generation: {exc}", file=sys.stderr)
        return 3

    if not rendered:
        print(
            f"WARNING: No topics found — no enum files generated. "
            f"Contracts root: {contracts_root}",
            file=sys.stderr,
        )
        return 0

    # Determine which enum_*_topic.py files should exist after this run
    expected_paths: set[Path] = set(rendered.keys())
    expected_enum_filenames: set[str] = {
        p.name for p in expected_paths if fnmatch.fnmatch(p.name, _STALE_PATTERN)
    }

    # Write generated files atomically
    try:
        for file_path, content in rendered.items():
            _atomic_write(file_path, content)
            print(f"  wrote: {file_path.relative_to(_REPO_ROOT)}")
    except Exception as exc:
        print(f"ERROR: Failed to write generated files: {exc}", file=sys.stderr)
        return 3

    # Apply ruff format so generated files are stable under the project's linter.
    # This prevents a pre-commit ruff pass from modifying the files after generation,
    # which would make --check report drift on a clean state.
    try:
        format_result = subprocess.run(
            ["uv", "run", "ruff", "format", str(output_dir)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if format_result.returncode != 0:
            print(
                f"WARNING: ruff format exited {format_result.returncode} — "
                f"generated files may not be ruff-stable.\n{format_result.stderr}",
                file=sys.stderr,
            )
    except FileNotFoundError:
        print(
            "WARNING: 'uv' not found — skipping ruff format. "
            "Generated files may not be ruff-stable.",
            file=sys.stderr,
        )

    # Remove stale enum_*_topic.py files (not in current output, not __init__.py)
    stale_removed: list[Path] = []
    if output_dir.exists():
        for existing in output_dir.iterdir():
            if (
                fnmatch.fnmatch(existing.name, _STALE_PATTERN)
                and existing.name not in expected_enum_filenames
                and existing.is_file()
            ):
                existing.unlink()
                stale_removed.append(existing)
                print(f"  removed stale: {existing.relative_to(_REPO_ROOT)}")

    print(
        f"\nDone: {len([p for p in expected_paths if fnmatch.fnmatch(p.name, _STALE_PATTERN)])} "
        f"producer enum file(s) + __init__.py written."
    )
    if stale_removed:
        print(f"Removed {len(stale_removed)} stale file(s).")

    return 0


# ---------------------------------------------------------------------------
# Core logic — check
# ---------------------------------------------------------------------------


def _run_check(contracts_root: Path, output_dir: Path) -> int:
    """
    Check that generated files are up to date with current contracts.

    Returns 0 on clean, 1 on drift, 2 on hard-stop, 3 on unexpected error.
    """
    try:
        from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor
        from omnibase_infra.tools.topic_enum_generator import TopicEnumGenerator
    except ImportError as exc:
        print(f"ERROR: Failed to import tools: {exc}", file=sys.stderr)
        return 3

    try:
        extractor = ContractTopicExtractor()
        entries = extractor.extract(contracts_root)
    except RuntimeError as exc:
        print(f"ERROR (hard-stop): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: Unexpected error during extraction: {exc}", file=sys.stderr)
        return 3

    try:
        generator = TopicEnumGenerator()
        rendered = generator.render(entries, output_dir=output_dir)
    except RuntimeError as exc:
        print(f"ERROR (hard-stop): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: Unexpected error during generation: {exc}", file=sys.stderr)
        return 3

    # Apply ruff format to in-memory content via temp files so comparison is stable.
    # The generator emits unformatted content; ruff reformats it on --generate.
    # --check must compare the ruff-formatted version to what's on disk.
    ruff_formatted: dict[Path, str] = {}
    with tempfile.TemporaryDirectory() as _tmp_dir:
        tmp_root = Path(_tmp_dir)
        for expected_path, expected_content in rendered.items():
            tmp_file = tmp_root / expected_path.name
            tmp_file.write_text(expected_content, encoding="utf-8")
        try:
            subprocess.run(
                ["uv", "run", "ruff", "format", str(tmp_root)],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            pass  # ruff not available; fall back to raw comparison
        for expected_path in rendered:
            tmp_file = tmp_root / expected_path.name
            if tmp_file.exists():
                ruff_formatted[expected_path] = tmp_file.read_text(encoding="utf-8")
            else:
                ruff_formatted[expected_path] = rendered[expected_path]

    drift_detected = False
    issues: list[str] = []

    # Check 1: Expected files exist on disk with correct content
    for expected_path, expected_content in ruff_formatted.items():
        if not expected_path.exists():
            issues.append(f"  MISSING: {expected_path.relative_to(_REPO_ROOT)}")
            drift_detected = True
            continue

        actual_content = expected_path.read_text(encoding="utf-8")
        if actual_content != expected_content:
            issues.append(f"  DRIFT:   {expected_path.relative_to(_REPO_ROOT)}")
            drift_detected = True

    # Check 2: No stale enum_*_topic.py files on disk
    expected_enum_filenames: set[str] = {
        p.name for p in ruff_formatted if fnmatch.fnmatch(p.name, _STALE_PATTERN)
    }
    if output_dir.exists():
        for existing in output_dir.iterdir():
            if (
                fnmatch.fnmatch(existing.name, _STALE_PATTERN)
                and existing.name not in expected_enum_filenames
                and existing.is_file()
            ):
                issues.append(
                    f"  STALE:   {existing.relative_to(_REPO_ROOT)} "
                    f"(no longer produced by current contracts)"
                )
                drift_detected = True

    if drift_detected:
        print(
            "CHECK FAILED: Generated enum files are out of date.\n"
            "Run: uv run python scripts/generate_topic_enums.py --generate\n"
        )
        for issue in issues:
            print(issue)
        return 1

    print(f"CHECK PASSED: {len(ruff_formatted)} generated file(s) are up to date.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve paths
    contracts_root: Path
    if args.contracts_root is not None:
        contracts_root = Path(args.contracts_root).resolve()
    else:
        contracts_root = _DEFAULT_CONTRACTS_ROOT

    output_dir: Path
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = _DEFAULT_OUTPUT_DIR

    if not contracts_root.exists():
        print(
            f"ERROR: contracts root does not exist: {contracts_root}",
            file=sys.stderr,
        )
        return 3

    if args.generate:
        return _run_generate(contracts_root, output_dir)
    else:  # args.check
        return _run_check(contracts_root, output_dir)


if __name__ == "__main__":
    sys.exit(main())
