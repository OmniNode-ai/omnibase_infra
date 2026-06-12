#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render Bifrost lane overlay YAML to dotenv sidecar (OMN-12864).

Usage:
    # Render dev lane overlay to dev.bifrost.env (default)
    uv run python scripts/render_bifrost_lane_overlay_env.py

    # Check that the rendered env file is in sync with the YAML source
    uv run python scripts/render_bifrost_lane_overlay_env.py --check

    # Render a specific overlay file
    uv run python scripts/render_bifrost_lane_overlay_env.py \\
        --overlay docker/lane-overlays/dev.bifrost.yaml \\
        --output docker/lane-overlays/dev.bifrost.env

CI contract: the rendered .env sidecar MUST be committed alongside the YAML
source. The --check flag fails with exit code 1 when the sidecar is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

_DEFAULT_OVERLAY = _REPO_ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "docker" / "lane-overlays" / "dev.bifrost.env"


def load_overlay(path: Path) -> ModelBifrostLaneOverlay:
    """Load and validate the Bifrost lane overlay YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelBifrostLaneOverlay.model_validate(raw)


def render_env_file(overlay: ModelBifrostLaneOverlay) -> str:
    """Render the env-var binding as a dotenv file."""
    env = overlay.as_env_dict()
    lines = [
        f"# Generated from docker/lane-overlays/{overlay.lane}.bifrost.yaml.",
        "# Do not edit directly — edit the YAML source and re-render:",
        "#   uv run python scripts/render_bifrost_lane_overlay_env.py",
        "#",
        "# OMN-12864: committed endpoint authority for BIFROST_LOCAL_* vars.",
        "# OMN-12815: every URL is the COMPLETE final chat-completions URL.",
    ]
    lines.extend(f"{key}={value}" for key, value in sorted(env.items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=_DEFAULT_OVERLAY,
        help="Path to the bifrost lane overlay YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to write the rendered dotenv sidecar.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the sidecar is in sync with the YAML; exit 1 if not.",
    )
    args = parser.parse_args(argv)

    overlay = load_overlay(args.overlay)
    rendered = render_env_file(overlay)

    if args.check:
        if not args.output.exists():
            print(
                f"ERROR: {args.output} does not exist. Run without --check to generate.",
                file=sys.stderr,
            )
            return 1
        existing = args.output.read_text(encoding="utf-8")
        if rendered != existing:
            print(
                f"ERROR: {args.output} is out of sync with {args.overlay}. "
                "Re-render: uv run python scripts/render_bifrost_lane_overlay_env.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {args.output} matches {args.overlay}")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Rendered: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
