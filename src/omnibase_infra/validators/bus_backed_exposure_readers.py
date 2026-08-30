# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every ``bus_backed: true`` exposure must have a declared READER (OMN-17199).

THE DEFECT CLASS THIS GATE CLOSES
---------------------------------
Every ratchet this platform has built around projection exposures guards the WRITE
end. Nothing anywhere asserted the READ end, and that one asymmetry is the mechanical
cause of three separate incidents:

* OMN-14440 — a job fired every 30 minutes for three months, failed every time, wrote
  348KB of errors. Nobody ever read it.
* Phoenix (OMN-16782) — a healthy UI on a pinned image with working env, health checks
  and a contract overlay, attached to a pipe nothing writes to. Re-probed 2026-08-30:
  ``spans 0, traces 0``, 79 days.
* ``onex.snapshot.projection.consumer-flow.v1`` — declared ``bus_backed: true`` with a
  live writer producing 2.55M rows at 43-second freshness, and zero widgets on zero
  surfaces. 79 ``STALLED`` consumer-windows in ten minutes, rendered nowhere. That one
  happened INSIDE the Phase 1 deliverable of the epic written to prevent it.

WHY THE EXISTING RATCHETS ALL PASS ON THAT SHAPE
------------------------------------------------
* OMN-15864 asserts every ``bus_backed`` exposure topic has a scheduled PRODUCER. It
  never asks whether anything reads it.
* OMN-16795 / ``scripts/check_subscribe_wiring_health.py`` asserts a declared
  ``subscribe`` topic has a PUBLISHER. Same end of the pipe: bus-level wiring, not the
  render boundary.
* OMN-16783 asserts a contract declaring subscribe+publish declares an
  ``expected_flow``. A runtime flow expectation, not consumption by a surface.

An exposure is a promise that somebody looks. This is the only place that promise is
checked.

WHAT COUNTS AS A READER
-----------------------
1. A component in the omnidash component registry
   (``src/registry/component-registry.json``) declaring the topic in its
   ``dataSources``.
2. That component placed in a shipped omnidash dashboard layout
   (``src/templates/*.ts`` -- ``DASHBOARD_TEMPLATES``, NOT the gitignored
   ``dashboard-layouts/``). Strictly stronger than (1); either satisfies the gate, and
   the report says which.
3. An explicit ``consumers: none`` on the exposure carrying a non-empty
   ``consumers_reason``.

(1) is not a CI-only field. ``dataSources[].topic`` is emitted from the very ``TOPICS``
symbols the widgets pass to ``useProjectionQuery``, and the generated manifest is loaded
into ``ComponentRegistry`` at app boot (``src/main.tsx`` -> ``RegistryProvider``). The
render layer and this gate resolve ONE declaration, which is the whole point: a
CI-only mirror would be a second source of truth and the exact drift shape this epic
keeps finding (three omnidash widgets still declare ``llm_cost.v1``, which no contract
has exposed since OMN-14896).

THE ESCAPE HATCH IS NOT SILENT
------------------------------
``consumers: none`` with no reason FAILS. Per
``feedback_optional_input_means_the_check_does_not_exist``, an escape hatch that can be
taken silently is not a gate. Two further rules stop the opt-out rotting into an
amnesty list by another name:

* ``consumers_reason`` without ``consumers: none`` FAILS — a reason with nothing to
  justify is a leftover.
* ``consumers: none`` on an exposure that DOES have a reader FAILS — once somebody
  renders it, the opt-out is a lie and must be deleted, not left standing.
* Any other ``consumers`` value FAILS closed. ``consumers: tbd`` must not become a
  third, undocumented escape hatch.

NO GRANDFATHERED RATCHET
------------------------
This module deliberately ships no companion file to freeze today's violations into, and
``tests/unit/validators/test_omn17199_bus_backed_exposure_readers.py`` asserts over this
module's own identifiers that it never grows one. OMN-17068 records what happened to the
sibling subscribe-has-publisher ratchet that did: a 682-entry grandfathered list with
expiry dates nothing ever read. A large violation count is the finding, not a reason for
a bypass file.

FAILS CLOSED ON A MISSING READER SURFACE
----------------------------------------
An absent registry, an unparseable one, or an absent layouts directory raises
:class:`ReaderSurfaceError` rather than yielding "no readers found" and reporting every
exposure as violating — or, worse, an empty scan reporting compliance. A gate whose
evidence can silently disappear is the fail-open shape this epic exists to close.

USAGE
-----
::

    python -m omnibase_infra.validators.bus_backed_exposure_readers \
        <contracts-dir> [<contracts-dir> ...] \
        [--extra-contracts-dir <dir> ...] \
        --registry <path/to/component-registry.json> \
        --layouts-dir <path/to/omnidash/src/templates>

Exit ``0`` when every served ``bus_backed`` exposure has a reader or a reasoned opt-out;
``1`` otherwise, including when the scan found no contracts at all.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import yaml

# Single source of exposure-parsing truth. `_projection_api_served_exposures` is the
# helper the topic extractor already uses to decide which exposures are actually
# served -- section-level `expose: true` AND per-exposure `bus_backed: true` -- and it
# mirrors omnimarket's own `build_projection_topic_map` / `SnapshotCache` gates.
# Re-implementing that predicate here is precisely how a gate and the runtime drift
# apart, which is the failure mode this module exists to catch. So it is imported, not
# copied.
from omnibase_infra.tools.contract_topic_extractor import (
    _projection_api_served_exposures,
)

# The literal an exposure declares to opt out of having a reader. A bare YAML null is
# NOT accepted: null is what a half-written field looks like, and this opt-out has to be
# typed on purpose.
_OPT_OUT_LITERAL = "none"

# A reason has to actually say something. Ten characters rejects "n/a", "-", "todo" and
# "none" without pretending to judge prose quality.
_MIN_REASON_CHARS = 10

_MISSING = object()

# A shipped layout entry names the component it places. The templates are hand-written
# TypeScript object literals with a uniform `componentName: '<name>'` field
# (`src/templates/*.ts`), so the name is lifted textually rather than by standing up a
# TypeScript parser inside a Python gate. Over-matching is harmless here: an extra name
# can only ever resolve to zero topics through the registry, never invent a reader.
_RE_LAYOUT_COMPONENT = re.compile(r"componentName:\s*['\"]([^'\"]+)['\"]")


class ReaderSurfaceError(RuntimeError):
    """The reader surface could not be resolved, so no verdict is possible.

    Raised instead of degrading to an empty reader set. "No readers found" and "the
    registry was not checked out" are indistinguishable downstream, and one of them is a
    false accusation against every exposure in the tree.
    """


@dataclass(frozen=True)
class Exposure:
    """One served ``bus_backed: true`` projection exposure."""

    contract: Path
    topic: str
    consumers: object
    consumers_reason: object


@dataclass(frozen=True)
class Finding:
    """One exposure that fails the reader assertion."""

    contract: Path
    topic: str
    code: str
    reason: str


# ---------------------------------------------------------------------------
# Reader surface (omnidash)
# ---------------------------------------------------------------------------


def collect_registry_readers(registry: Path) -> dict[str, set[str]]:
    """Map each projection topic to the omnidash components declaring it.

    Reads ``dataSources[].topic`` from the generated component registry -- the same
    manifest ``src/main.tsx`` imports and hands to ``ComponentRegistry`` at boot.
    """
    try:
        raw_text = registry.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReaderSurfaceError(
            f"cannot read the omnidash component registry at {registry}: {exc}. "
            "Without it no reader can be resolved and every exposure would be reported "
            "as unread, so this fails closed instead of guessing."
        ) from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ReaderSurfaceError(
            f"{registry} is not valid JSON: {exc}. The component registry is the reader "
            "source of truth; an unparseable one is an infrastructure failure, not an "
            "absence of readers."
        ) from exc

    components = raw.get("components") if isinstance(raw, dict) else None
    if not isinstance(components, dict):
        raise ReaderSurfaceError(
            f"{registry} has no `.components` mapping. That is the shape this gate "
            "resolves readers from; refusing to report an empty reader set as fact."
        )

    readers: dict[str, set[str]] = {}
    for name, manifest in components.items():
        if not isinstance(manifest, dict):
            continue
        for source in manifest.get("dataSources") or ():
            if not isinstance(source, dict):
                continue
            topic = source.get("topic")
            if isinstance(topic, str) and topic:
                readers.setdefault(topic, set()).add(str(name))
    return readers


def collect_layout_readers(
    layouts_dir: Path, registry_readers: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Map each projection topic to the shipped layouts that place a reader of it.

    A layout entry names a ``componentName``; the topic it reads is the one that
    component declares in the registry. Layout placement is therefore resolved THROUGH
    the registry rather than from a parallel topic list -- one declaration, not two.

    The shipped layouts are ``omnidash/src/templates/*.ts`` (``DASHBOARD_TEMPLATES``),
    NOT ``omnidash/dashboard-layouts/``. That directory is in omnidash's ``.gitignore``
    (line 27) and holds a user's locally-saved dashboards; nothing there reaches a
    checkout, so treating it as the shipped surface would have made this gate resolve
    zero layout readers on every CI run while looking like it worked.
    """
    if not layouts_dir.is_dir():
        raise ReaderSurfaceError(
            f"the omnidash shipped-templates directory {layouts_dir} does not exist. "
            "Layout placement is part of the reader surface; a missing directory is an "
            "infrastructure failure, not an absence of readers."
        )

    component_to_topics: dict[str, set[str]] = {}
    for topic, names in registry_readers.items():
        for name in names:
            component_to_topics.setdefault(name, set()).add(topic)

    readers: dict[str, set[str]] = {}
    for layout_file in sorted(layouts_dir.glob("*.ts")):
        try:
            text = layout_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReaderSurfaceError(
                f"cannot read the shipped layout {layout_file}: {exc}. Layouts are read "
                "to resolve readers; silently skipping one would understate the reader "
                "surface and produce a false violation."
            ) from exc
        for component_name in _RE_LAYOUT_COMPONENT.findall(text):
            for topic in component_to_topics.get(component_name, ()):
                readers.setdefault(topic, set()).add(layout_file.name)
    return readers


# ---------------------------------------------------------------------------
# Exposure surface (contracts)
# ---------------------------------------------------------------------------


def _iter_contract_files(dirs: Sequence[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("contract.yaml")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def collect_bus_backed_exposures(dirs: Sequence[Path]) -> list[Exposure]:
    """Return every served ``bus_backed: true`` exposure under the given roots."""
    found: list[Exposure] = []
    for contract_path in _iter_contract_files(dirs):
        try:
            data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ReaderSurfaceError(
                f"cannot parse {contract_path}: {exc}. A contract this gate cannot read "
                "is a contract whose exposures it cannot see; refusing to pass over it."
            ) from exc
        if not isinstance(data, dict):
            continue
        for exposure in _projection_api_served_exposures(data.get("projection_api")):
            topic = exposure.get("topic")
            found.append(
                Exposure(
                    contract=contract_path,
                    topic=topic if isinstance(topic, str) and topic else "<missing>",
                    consumers=exposure.get("consumers", _MISSING),
                    consumers_reason=exposure.get("consumers_reason", _MISSING),
                )
            )
    return found


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def evaluate(
    exposures: Sequence[Exposure], readers: dict[str, set[str]]
) -> list[Finding]:
    """Return one :class:`Finding` per exposure that has no reader and no reasoned
    opt-out. An empty list means every exposure is accounted for."""
    findings: list[Finding] = []
    for exposure in exposures:
        finding = _judge(exposure, readers)
        if finding is not None:
            findings.append(finding)
    return findings


def _judge(exposure: Exposure, readers: dict[str, set[str]]) -> Finding | None:
    if exposure.topic == "<missing>":
        return Finding(
            contract=exposure.contract,
            topic="<missing>",
            code="exposure_without_topic",
            reason=(
                "a `bus_backed: true` exposure declares no `topic`, so no reader can "
                "ever be resolved for it."
            ),
        )

    seen_by = sorted(readers.get(exposure.topic, ()))
    declared = exposure.consumers
    raw_reason = exposure.consumers_reason
    reason_text = raw_reason.strip() if isinstance(raw_reason, str) else ""

    if declared is _MISSING:
        if raw_reason is not _MISSING:
            return Finding(
                contract=exposure.contract,
                topic=exposure.topic,
                code="reason_without_opt_out",
                reason=(
                    "`consumers_reason` is declared but `consumers: none` is not. A "
                    "consumers_reason with nothing to justify is a leftover -- delete "
                    "it, or declare the opt-out it belongs to."
                ),
            )
        if seen_by:
            return None
        return Finding(
            contract=exposure.contract,
            topic=exposure.topic,
            code="no_reader",
            reason=(
                "declared `bus_backed: true` and NO omnidash component declares this "
                "topic in its `dataSources`, and no shipped layout places one. The "
                "exposure is a promise that somebody looks, and nobody does."
            ),
        )

    if declared != _OPT_OUT_LITERAL:
        return Finding(
            contract=exposure.contract,
            topic=exposure.topic,
            code="unrecognised_consumers_value",
            reason=(
                f"`consumers: {declared!r}` is not a recognised declaration. The only "
                f"accepted value is the literal `{_OPT_OUT_LITERAL}`; readers are "
                "declared on the reader side, in the omnidash component registry, so "
                "that the render layer and this gate resolve one field and not two."
            ),
        )

    if len(reason_text) < _MIN_REASON_CHARS:
        return Finding(
            contract=exposure.contract,
            topic=exposure.topic,
            code="opt_out_without_reason",
            reason=(
                "`consumers: none` carries no usable `consumers_reason`. An escape "
                "hatch that can be taken silently is not a gate -- say, in a sentence, "
                "why nothing renders this exposure."
            ),
        )

    if seen_by:
        return Finding(
            contract=exposure.contract,
            topic=exposure.topic,
            code="stale_opt_out",
            reason=(
                "`consumers: none` is declared, but "
                f"{', '.join(seen_by)} now read this topic. Delete the opt-out; a "
                "standing opt-out over a live reader is how an escape hatch turns into "
                "a permanent amnesty entry."
            ),
        )

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bus_backed_exposure_readers",
        description=(
            "Assert every served `bus_backed: true` projection exposure has a declared "
            "reader or a reasoned `consumers: none` (OMN-17199)."
        ),
    )
    parser.add_argument(
        "contracts_dirs",
        nargs="*",
        type=Path,
        help="roots to scan for contract.yaml files.",
    )
    parser.add_argument(
        "--extra-contracts-dir",
        action="append",
        default=[],
        type=Path,
        dest="extra_contracts_dirs",
        help=(
            "additional contract root, repeatable. Used for the sibling omnimarket "
            "checkout, where every bus_backed exposure in the platform lives today."
        ),
    )
    parser.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="path to omnidash src/registry/component-registry.json.",
    )
    parser.add_argument(
        "--layouts-dir",
        required=True,
        type=Path,
        help="path to omnidash src/templates/ (the shipped DASHBOARD_TEMPLATES).",
    )
    return parser


def _report(
    findings: Sequence[Finding],
    exposures: Sequence[Exposure],
    readers: dict[str, set[str]],
    stream: IO[str],
) -> None:
    if findings:
        stream.write(
            "[exposure-reader-coverage] FAIL: a `bus_backed: true` exposure is a "
            "promise that somebody looks at it. These exposures have no reader and no "
            "reasoned opt-out (OMN-17199):\n"
        )
        for finding in sorted(findings, key=lambda f: (f.topic, str(f.contract))):
            stream.write(f"  - {finding.topic}\n")
            stream.write(f"      contract: {finding.contract}\n")
            stream.write(f"      code:     {finding.code}\n")
            stream.write(f"      reason:   {finding.reason}\n")
        stream.write(
            "\n  Fix, in order of preference:\n"
            "    1. Render it. Add an omnidash component whose `dataSources` declares\n"
            "       the topic, and place it on a shipped dashboard layout.\n"
            "    2. If nothing should ever render it, declare on the exposure:\n"
            "         consumers: none\n"
            '         consumers_reason: "<why nothing renders this>"\n'
            "    3. If nothing reads it and nothing should expose it, delete the\n"
            "       exposure. An exposure nobody wants is not a thing to silence.\n"
            "\n  This gate has no companion file to record a violation in, on purpose\n"
            "  (OMN-17068). Do not add one.\n"
        )
        return

    stream.write(
        f"[exposure-reader-coverage] OK: {len(exposures)} served `bus_backed` "
        "exposure(s), 0 without a reader.\n"
    )
    for exposure in sorted(exposures, key=lambda e: e.topic):
        seen_by = sorted(readers.get(exposure.topic, ()))
        detail = ", ".join(seen_by) if seen_by else "opted out with a stated reason"
        stream.write(f"  - {exposure.topic} :: {detail}\n")


def check_exposure_readers(
    contracts_dirs: Sequence[Path],
    registry: Path,
    layouts_dir: Path,
    stream: IO[str] | None = None,
) -> int:
    out = stream if stream is not None else sys.stderr

    registry_readers = collect_registry_readers(registry)
    layout_readers = collect_layout_readers(layouts_dir, registry_readers)
    readers: dict[str, set[str]] = {
        topic: set(names) for topic, names in registry_readers.items()
    }
    for topic, names in layout_readers.items():
        readers.setdefault(topic, set()).update(names)

    scanned = list(_iter_contract_files(contracts_dirs))
    if not scanned:
        out.write(
            "[exposure-reader-coverage] FAIL: scanned 0 contract.yaml files across "
            f"{[str(d) for d in contracts_dirs]}. An empty scan is not compliance -- it "
            "is a gate that has lost its input, which is exactly how a required check "
            "turns into a green rubber stamp.\n"
        )
        return 1

    exposures = collect_bus_backed_exposures(contracts_dirs)
    findings = evaluate(exposures, readers)
    _report(findings, exposures, readers, out)
    return 1 if findings else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dirs = [*args.contracts_dirs, *args.extra_contracts_dirs]
    try:
        return check_exposure_readers(dirs, args.registry, args.layouts_dir)
    except ReaderSurfaceError as exc:
        sys.stderr.write(f"[exposure-reader-coverage] FAIL (fail-closed): {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
