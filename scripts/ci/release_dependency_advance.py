# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Say whether a package's dev branch pins a sibling ahead of its latest published release (OMN-19655).

Why this exists
---------------
Up to 0.38.58 omnibase_infra published its sibling requirements as exact pins
(``omnibase-core==X``). A core release reaches infra's dev through the
dependency cascade's bump PR, and then nothing cuts infra: the release train
runs once a night, and a hand cut waits for someone to notice. Until infra is
released, the newest published infra still pins the OLD core exactly, so every
downstream package that raises its core floor to the new version cannot be
installed from PyPI. That is what failed omnimarket's release on every dev push
on 2026-09-24 (core 0.47.22) and again on 2026-09-25 (core 0.47.23; infra
0.38.57 pinned ==0.47.22).

Since OMN-19655 part 1 infra publishes a compatible range for omnibase-core
(``>=FLOOR,<NEXT_MINOR``) and keeps the exact version it builds and tests
against only in ``[tool.uv] override-dependencies``. So the two sides compared
here are no longer both exact pins:

* **dev** is the exact version dev resolves for each sibling: the
  ``override-dependencies`` pin when there is one (that is what ``uv.lock``
  and the runtime image carry), else an exact pin in ``[project.dependencies]``.
* **published** is whatever requirement the newest release on PyPI declares for
  that sibling, in any specifier shape.

A sibling is a stranding advance when dev resolves a version ABOVE everything
the published requirement admits (or the release does not require the sibling
at all). A core patch inside the published range is therefore not stranding:
a downstream floor raise to it resolves against the published infra. A
specifier this script cannot evaluate counts as an advance, since the answer
only dispatches the train and the train's own premises still decide whether it
cuts; this adds a trigger, never an override.

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

#: ``name[extras] specifier`` of one requirement, marker already stripped. The
#: specifier may be wrapped in parentheses (the older Requires-Dist spelling).
_REQUIREMENT = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*\(?\s*([^()]*?)\s*\)?\s*$"
)

_CLAUSE = re.compile(r"^(===|==|!=|~=|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)*)$")

_PYPI_JSON = "https://pypi.org/pypi/{package}/json"  # url-authority-ok: PyPI public JSON API, fixed canonical third-party endpoint (no ONEX routing authority)

Fetch = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class ModelPinAdvance:
    """One sibling dev resolves above what the published release admits.

    ``published`` is the published requirement's specifier as PyPI serves it
    (``==0.47.22``, ``<0.48.0,>=0.47.23``), empty when the release does not
    require the sibling. ``dev`` is the exact version dev resolves.
    """

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


def published_sibling_requirements(requirements: Iterable[str]) -> dict[str, str]:
    """Return ``{name: specifier}`` for every unconditional requirement on our packages.

    Unlike :func:`exact_sibling_pins`, any specifier shape is kept, since a
    published release may declare a range. A requirement with an environment
    marker is conditional and left out, as there.
    """
    specs: dict[str, str] = {}
    for raw in requirements:
        if ";" in raw:
            continue
        match = _REQUIREMENT.match(raw)
        if match is None:
            continue
        name = _normalize(match.group(1))
        if name.startswith(_OUR_PREFIXES):
            specs[name] = re.sub(r"\s+", "", match.group(2))
    return specs


def _release_key(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _padded(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    width = max(len(a), len(b))
    return tuple(t + (0,) * (width - len(t)) for t in (a, b))


def _clause_admits(operator: str, bound: tuple[int, ...], v: tuple[int, ...]) -> bool:
    if operator == "~=":
        upper = (*bound[:-2], bound[-2] + 1)
        return _clause_admits(">=", bound, v) and _clause_admits("<", upper, v)
    left, right = _padded(v, bound)
    return {
        "==": left == right,
        "!=": left != right,
        ">=": left >= right,
        "<=": left <= right,
        ">": left > right,
        "<": left < right,
    }[operator]


def specifier_admits(specifier: str, version: str) -> bool | None:
    """Say whether ``version`` satisfies every clause of ``specifier``.

    Numeric release versions and the ``== != ~= >= <= > <`` operators only,
    which is every shape our packages publish. ``None`` means the specifier or
    the version is outside that and this function will not guess.
    """
    v = _release_key(version)
    if v is None:
        return None
    admitted = True
    for clause in filter(None, (c.strip() for c in specifier.split(","))):
        match = _CLAUSE.match(clause)
        if match is None or match.group(1) == "===":
            return None
        bound = _release_key(match.group(2))
        if bound is None or (match.group(1) == "~=" and len(bound) < 2):
            return None
        admitted = admitted and _clause_admits(match.group(1), bound, v)
    return admitted


def _strands(dev: str, published: str | None) -> bool:
    """Whether dev's exact ``dev`` version runs past the published requirement."""
    if published is None:
        return True
    admitted = specifier_admits(published, dev)
    if admitted is None:
        return True
    if admitted:
        return False
    # Not admitted. Stranding only when dev is ABOVE what is admitted: a dev
    # version below the published floor is a downgrade and strands no
    # downstream floor raise. With no lower bound to compare, count it.
    floors = [
        key
        for clause in published.split(",")
        if (m := _CLAUSE.match(clause.strip()))
        and m.group(1) in {">=", ">", "==", "~="}
        if (key := _release_key(m.group(2))) is not None
    ]
    dev_key = _release_key(dev)
    if not floors or dev_key is None:
        return True
    left, right = _padded(dev_key, max(floors))
    return left >= right


def fetch_pypi_json(package: str) -> dict[str, Any]:
    """Read ``package``'s JSON document from PyPI; raise on any failure."""
    url = _PYPI_JSON.format(package=package)
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310  # nosec B310 - fixed https literal
        payload: dict[str, Any] = json.load(response)
    return payload


def dev_sibling_versions(pyproject: Path) -> dict[str, str]:
    """Return ``{name: exact version}`` dev resolves for each sibling.

    ``[tool.uv] override-dependencies`` wins over ``[project.dependencies]``,
    as it does for uv: since OMN-19655 infra's published omnibase-core
    requirement is a range, and the exact version lives only in the override.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError(f"{pyproject}: [project].dependencies is not a list")
    overrides = data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    if not isinstance(overrides, list):
        raise ValueError(f"{pyproject}: [tool.uv].override-dependencies is not a list")
    versions = exact_sibling_pins(str(d) for d in deps)
    versions.update(exact_sibling_pins(str(o) for o in overrides))
    return versions


def decide(*, package: str, pyproject: Path, fetch: Fetch) -> ModelAdvanceVerdict:
    """Compare the versions dev resolves with what the newest release requires."""
    document = fetch(package)
    info = document["info"]
    published_version = str(info["version"])
    published = published_sibling_requirements(info.get("requires_dist") or [])
    dev = dev_sibling_versions(pyproject)
    advances = tuple(
        ModelPinAdvance(name=name, published=published.get(name, ""), dev=version)
        for name, version in sorted(dev.items())
        if _strands(version, published.get(name))
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
            f"STRANDED: published {verdict.package}=={verdict.published_version} requires "
            f"{item.name}{item.published or ' <absent>'}; dev resolves {item.name}=={item.dev}. "
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
