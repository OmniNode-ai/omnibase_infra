# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Strip non-market ONEX runtime entry points from installed distributions."""

from __future__ import annotations

import argparse
import configparser
import json
import site
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from email.parser import Parser
from pathlib import Path

DEFAULT_ALLOWED_DISTRIBUTIONS = frozenset({"omnibase-infra", "omnimarket"})
DEFAULT_STRIPPED_GROUPS = frozenset(
    {"onex.domain_plugins", "onex.node_package", "onex.nodes"}
)


@dataclass(frozen=True)
class StrippedEntryPointDist:
    """Entry-point groups stripped from one installed distribution."""

    distribution: str
    path: str
    groups: tuple[str, ...]
    removed_file: bool


@dataclass(frozen=True)
class StripEntryPointReport:
    """Summary of runtime entry-point metadata stripping."""

    allowed_distributions: tuple[str, ...]
    stripped_groups: tuple[str, ...]
    stripped_distributions: tuple[StrippedEntryPointDist, ...]


def normalize_distribution_name(name: str) -> str:
    """Normalize a Python distribution name for comparison."""

    return name.strip().lower().replace("_", "-").replace(".", "-")


def _iter_site_packages() -> list[Path]:
    paths: list[Path] = []
    for raw_path in site.getsitepackages():
        path = Path(raw_path)
        if path.exists():
            paths.append(path)
    usersite = site.getusersitepackages()
    if usersite:
        path = Path(usersite)
        if path.exists():
            paths.append(path)
    return paths


def _dist_name(dist_info_dir: Path) -> str:
    metadata_path = dist_info_dir / "METADATA"
    if metadata_path.exists():
        metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
        name = metadata.get("Name")
        if name:
            return normalize_distribution_name(name)

    raw_name = dist_info_dir.name.removesuffix(".dist-info")
    if "-" in raw_name:
        raw_name = raw_name.rsplit("-", 1)[0]
    return normalize_distribution_name(raw_name)


def _read_entry_points(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[assignment]
    parser.read(path, encoding="utf-8")
    return parser


def _write_entry_points(path: Path, parser: configparser.ConfigParser) -> bool:
    if not parser.sections():
        path.unlink()
        return True

    with path.open("w", encoding="utf-8") as file:
        parser.write(file, space_around_delimiters=True)
    return False


def strip_runtime_entry_points(
    site_package_paths: Iterable[Path],
    *,
    allowed_distributions: Iterable[str] = DEFAULT_ALLOWED_DISTRIBUTIONS,
    stripped_groups: Iterable[str] = DEFAULT_STRIPPED_GROUPS,
) -> StripEntryPointReport:
    """Remove ONEX runtime discovery groups from non-allowed distributions."""

    allowed = frozenset(
        normalize_distribution_name(name) for name in allowed_distributions
    )
    groups = frozenset(stripped_groups)
    stripped: list[StrippedEntryPointDist] = []

    for site_package_path in site_package_paths:
        for dist_info_dir in sorted(site_package_path.glob("*.dist-info")):
            dist_name = _dist_name(dist_info_dir)
            if dist_name in allowed:
                continue

            entry_points_path = dist_info_dir / "entry_points.txt"
            if not entry_points_path.exists():
                continue

            parser = _read_entry_points(entry_points_path)
            removed_groups = tuple(
                group for group in sorted(groups) if parser.remove_section(group)
            )
            if not removed_groups:
                continue

            removed_file = _write_entry_points(entry_points_path, parser)
            stripped.append(
                StrippedEntryPointDist(
                    distribution=dist_name,
                    path=str(entry_points_path),
                    groups=removed_groups,
                    removed_file=removed_file,
                )
            )

    return StripEntryPointReport(
        allowed_distributions=tuple(sorted(allowed)),
        stripped_groups=tuple(sorted(groups)),
        stripped_distributions=tuple(stripped),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove ONEX runtime plugin entry points from installed distributions "
            "that are not allowed in the runtime discovery surface."
        )
    )
    parser.add_argument(
        "--site-packages",
        action="append",
        type=Path,
        help="Site-packages path to scan. Defaults to the active interpreter paths.",
    )
    parser.add_argument(
        "--allowed-distribution",
        action="append",
        default=[],
        help=(
            "Distribution allowed to keep ONEX runtime entry points. May be passed "
            "multiple times. Defaults to omnibase-infra and omnimarket."
        ),
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Entry-point group to strip from non-allowed distributions. May be "
            "passed multiple times. Defaults to ONEX runtime discovery groups."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    site_package_paths = args.site_packages or _iter_site_packages()
    allowed_distributions = args.allowed_distribution or DEFAULT_ALLOWED_DISTRIBUTIONS
    stripped_groups = args.group or DEFAULT_STRIPPED_GROUPS
    report = strip_runtime_entry_points(
        site_package_paths,
        allowed_distributions=allowed_distributions,
        stripped_groups=stripped_groups,
    )
    sys.stdout.write(f"{json.dumps(asdict(report), sort_keys=True)}\n")


if __name__ == "__main__":
    main()
