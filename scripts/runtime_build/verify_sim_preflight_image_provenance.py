# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail closed unless a local sim image matches captured source hashes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

_HASHES = (
    "infra_head",
    "infra_snapshot_sha256",
    "core_head",
    "core_snapshot_sha256",
    "market_head",
    "market_snapshot_sha256",
    "compat_head",
    "compat_snapshot_sha256",
    "canonical_dockerfile_sha256",
    "staged_dockerfile_sha256",
    "pyrage_wheel_sha256",
)
_PREFIX = "io.omninode.sim-preflight."
_DEPENDENCY_PROFILE = "sim-preflight-runtime-active-base-v1"


def verify(path: Path, image_ref: str, pins: Path) -> None:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("profile") != "sim-preflight-image-provenance-v2"
        or raw.get("dependency_profile") != _DEPENDENCY_PROFILE
    ):
        raise ValueError("invalid sim image provenance")
    image_id = raw.get("image_id")
    if (
        raw.get("image_ref") != image_ref
        or not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(c not in "0123456789abcdef" for c in image_id[7:])
    ):
        raise ValueError("sim image provenance does not name requested image")
    for key in _HASHES:
        value = raw.get(key)
        length = 64 if key.endswith("sha256") else 40
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("sim image provenance has invalid source hash")
    expected: Any = json.loads(pins.read_text(encoding="utf-8"))
    sources = expected.get("sources") if isinstance(expected, dict) else None
    if (
        not isinstance(sources, dict)
        or raw["core_head"] != sources.get("omnibase_core")
        or raw["infra_head"] != sources.get("omnibase_infra")
    ):
        raise ValueError("sim image provenance differs from declared source pins")
    images = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", image_ref],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("docker did not identify exactly one local image")
    image = images[0]
    if image.get("Id") != raw["image_id"]:
        raise ValueError("local image id differs from provenance")
    labels = image.get("Config", {}).get("Labels", {})
    if not isinstance(labels, dict):
        raise ValueError("local image has no labels")
    if any(labels.get(_PREFIX + key) != raw[key] for key in _HASHES):
        raise ValueError("local image labels differ from provenance")
    if labels.get(_PREFIX + "dependency_profile") != _DEPENDENCY_PROFILE:
        raise ValueError("local image lacks the declared dependency profile")
    if (
        labels.get("com.omninode.build_source") != "workspace"
        or labels.get("com.omninode.promotion_class") != "stability-candidate"
        or labels.get("com.omninode.non_main_lineage") != "true"
    ):
        raise ValueError("sim image lacks required non-production workspace identity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--source-pins", type=Path, required=True)
    args = parser.parse_args()
    verify(args.provenance, args.image_ref, args.source_pins)


if __name__ == "__main__":
    main()
