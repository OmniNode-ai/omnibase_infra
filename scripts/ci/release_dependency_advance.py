# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Say whether a package's dev branch pins a sibling ahead of its latest published release (OMN-19655).

Why this exists
---------------
omnibase_infra pins its sibling packages exactly (``omnibase-core==X``). A core
release reaches infra's dev through the dependency cascade's bump PR, and then
nothing cuts infra: the release train runs once a night, and a hand cut waits
for someone to notice. Until infra is released, the newest published infra
still pins the OLD core exactly, so every downstream package that raises its
core floor to the new version cannot be installed from PyPI. That is what
failed omnimarket's release on every dev push on 2026-09-24 (core 0.47.22) and
again on 2026-09-25 (core 0.47.23; infra 0.38.57 pinned ==0.47.22).

This script measures that state and nothing more: the exact sibling pins in the
dev ``pyproject.toml`` against the exact sibling pins in the newest release of
the same package on PyPI. A pin that runs ahead is a stranding advance, and
``release-train-dependency-advance.yml`` then dispatches the release train for
the package. The train's own premises still decide whether it cuts; this adds
a trigger, never an override.

Standard library only, so it runs on any runner's ``python3`` with nothing
installed.

Exit codes: ``0`` decided (stranded or not; read the JSON or the step output);
``2`` the index or the pyproject could not be read, in which case no
``stranded=`` output is written, because an unread index is not "in step".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Distribution-name prefixes that are ours. An exact pin on anything else is a
#: third-party pin and cannot strand a sibling release.
_OUR_PREFIXES = ("omnibase-", "omninode-", "omnimarket", "omniclaude", "onex-")

_EXACT_PIN = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s,;]+)\s*$"
)

_PYPI_JSON = "https://pypi.org/pypi/{package}/json"  # url-authority-ok: PyPI public JSON API, fixed canonical third-party endpoint (no ONEX routing authority)

Fetch = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class ModelPinAdvance:
    """One sibling the dev branch pins at a version the published release does not."""

    name: str
    published: str
    dev: str


@dataclass(frozen=True)
class ModelAdvanceVerdict:
    """The measured state for one package."""

    package: str
    published_version: str
    advances: tuple[ModelPinAdvance, ...]

    @property
    def stranded(self) -> bool:
        return bool(self.advances)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def exact_sibling_pins(requirements: Iterable[str]) -> dict[str, str]:
    """Return ``{name: version}`` for every unconditional ``==`` pin on our packages.

    A requirement with an environment marker is conditional (an extra, a
    platform) and is not what a plain ``pip install`` resolves, so it is left out.
    """
    pins: dict[str, str] = {}
    for raw in requirements:
        if ";" in raw:
            continue
        match = _EXACT_PIN.match(raw)
        if match is None:
            continue
        name = _normalize(match.group(1))
        if name.startswith(_OUR_PREFIXES):
            pins[name] = match.group(2)
    return pins


def _release_key(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _runs_ahead(dev: str, published: str) -> bool:
    if not published:
        return True
    dev_key, published_key = _release_key(dev), _release_key(published)
    if dev_key is None or published_key is None:
        return dev != published
    return dev_key > published_key


def fetch_pypi_json(package: str) -> dict[str, Any]:
    """Read ``package``'s JSON document from PyPI; raise on any failure."""
    url = _PYPI_JSON.format(package=package)
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310  # nosec B310 - fixed https literal
        payload: dict[str, Any] = json.load(response)
    return payload


def _dev_requirements(pyproject: Path) -> list[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError(f"{pyproject}: [project].dependencies is not a list")
    return [str(d) for d in deps]


def decide(*, package: str, pyproject: Path, fetch: Fetch) -> ModelAdvanceVerdict:
    """Compare dev's exact sibling pins with the newest published release's."""
    document = fetch(package)
    info = document["info"]
    published_version = str(info["version"])
    published = exact_sibling_pins(info.get("requires_dist") or [])
    dev = exact_sibling_pins(_dev_requirements(pyproject))
    advances = tuple(
        ModelPinAdvance(name=name, published=published.get(name, ""), dev=version)
        for name, version in sorted(dev.items())
        if _runs_ahead(version, published.get(name, ""))
    )
    return ModelAdvanceVerdict(
        package=package, published_version=published_version, advances=advances
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, help="PyPI distribution name")
    parser.add_argument("--pyproject", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verdict = decide(
            package=args.package, pyproject=args.pyproject, fetch=fetch_pypi_json
        )
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(
            f"ERROR: could not measure {args.package}'s sibling pins against PyPI: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    payload = {
        "package": verdict.package,
        "published_version": verdict.published_version,
        "stranded": verdict.stranded,
        "advances": [asdict(a) for a in verdict.advances],
    }
    print(json.dumps(payload, indent=2))
    for item in verdict.advances:
        print(
            f"STRANDED: published {verdict.package}=={verdict.published_version} pins "
            f"{item.name}=={item.published or '<absent>'}; dev pins {item.name}=={item.dev}. "
            f"A downstream floor raise on {item.name} cannot resolve until "
            f"{verdict.package} releases.",
            file=sys.stderr,
        )
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"stranded={'true' if verdict.stranded else 'false'}\n")
            handle.write(f"published_version={verdict.published_version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
