# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every endpoint_from_overlay slot has a value on overlay home main (OMN-26709).

omnibase_infra#6321 flagged slot coder-5090 endpoint_from_overlay while the
overlay home repository's main (51f0563f) held a value for embeddings-201 only, and
every dev build then failed at compose_gen.
ONE test body judges the repository's real contract against an injected overlay
source. Unit sources are synthetic clones (documentation range 192.0.2.0/24): the
known bad holds embeddings-201 only, the known good holds every flagged slot. When
``OVERLAY_MAIN_CLONE`` names the real overlay home checkout (the CI job, the lab run;
``OVERLAY_MAIN_REF`` names the ref, default main), the same body runs against it and
expects no missing slot, so a fixture can never pass while the real overlay lacks
the value.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.contracts.model_llm_endpoint_env_file import load_endpoints
from omnibase_infra.models.contracts.model_llm_endpoint_overlay import (
    OVERLAY_HOME_RELPATH,
    ModelLlmEndpointOverlay,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"


def overlay_main_missing_slots(
    endpoints: list[dict[str, object]], clone: Path, ref: str
) -> list[str]:
    """Slot ids flagged ``endpoint_from_overlay`` that the overlay at ``ref`` holds no value for.

    Reads the overlay home clone at ``ref`` (never its working tree). An unreadable
    clone raises ``ValueError`` (``read_clone``); a ref holding no overlay file misses
    every flagged slot. Lives in the test, not the package, so the check adds no
    packaged source and needs no release.
    """
    overlay = ModelLlmEndpointOverlay.read_clone(clone, ref)
    held = overlay.endpoint_urls if overlay is not None else {}
    return [
        str(endpoint.get("slot_id", ""))
        for endpoint in endpoints
        if endpoint.get("endpoint_from_overlay")
        and str(endpoint.get("slot_id", "")) not in held
    ]


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_main(tmp_path: Path, urls: dict[str, str]) -> Path:
    repo = tmp_path / "overlay-home"
    overlay_file = repo / OVERLAY_HOME_RELPATH
    overlay_file.parent.mkdir(parents=True)
    overlay_file.write_text(
        yaml.safe_dump({"schema_version": 1, "endpoint_urls": urls}),
        encoding="utf-8",
    )
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "overlay", cwd=repo)
    return repo


_EMBEDDINGS_ONLY = {"embeddings-201": "http://192.0.2.10:8100"}


_FLAGGED = [
    str(e["slot_id"])
    for e in load_endpoints(_CONTRACT)
    if e.get("endpoint_from_overlay")
]

#: (id, urls the source holds, slots expected missing)
_FIXTURE_SOURCES = [
    ("known-bad-coder-5090-absent", _EMBEDDINGS_ONLY, ["coder-5090"]),
    (
        "known-good-every-slot",
        dict.fromkeys(_FLAGGED, "http://192.0.2.10:8100"),
        [],
    ),
]


def _sources() -> list[object]:
    params: list[object] = [
        pytest.param(urls, expected, None, id=name)
        for name, urls, expected in _FIXTURE_SOURCES
    ]
    real = os.environ.get("OVERLAY_MAIN_CLONE", "")
    if real:
        params.append(pytest.param(None, [], Path(real), id="real-overlays-main"))
    return params


@pytest.mark.parametrize(("urls", "expected_missing", "real_clone"), _sources())
def test_every_flagged_slot_has_a_value_on_overlay_main(
    tmp_path: Path,
    urls: dict[str, str] | None,
    expected_missing: list[str],
    real_clone: Path | None,
) -> None:
    clone = (
        real_clone if real_clone is not None else _overlay_main(tmp_path, urls or {})
    )
    ref = os.environ.get("OVERLAY_MAIN_REF", "main") if real_clone else "main"
    missing = overlay_main_missing_slots(load_endpoints(_CONTRACT), clone, ref)
    assert missing == expected_missing, (
        f"slots flagged endpoint_from_overlay with no value in the overlay home at {ref}: {missing}"
    )


def test_flagged_slots_include_the_known_bad_and_known_good() -> None:
    assert {"coder-5090", "embeddings-201"} <= set(_FLAGGED)


def test_unreadable_clone_is_refused_not_clean(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        overlay_main_missing_slots(
            load_endpoints(_CONTRACT), tmp_path / "absent", "main"
        )
