# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The declared settle budget for a lab-pass probe, and the job's affordance.

WHAT THIS REPLACES (OMN-18436)
------------------------------
``runtime-rebuild-trigger.yml`` computed the settle budget in shell as
``job ceiling - elapsed - reserved tail`` and handed the remainder to
``lab_pass_receipt.py probe-lane``. Three consequences, in increasing order of
severity:

1. The lane's boot budget became a function of how long the deploy agent queue
   happened to be, which it is not.
2. The arithmetic was untested, because it lived in a ``run:`` block.
3. The job's own three declared constants could not all hold. A 30-minute
   ceiling, a ``--wait-timeout 25m`` convergence wait and a 120-second reserved
   tail leave **180 seconds** of settle budget in the worst case, against a lane
   boot measured at 463-668 seconds. Every convergence slower than roughly
   twenty minutes produced a receipt that FAILED on timing alone, on a healthy
   lane correctly running the merged sha -- and rule 24(b) fails closed on a
   FAIL receipt, so that receipt then refused the sha for staging delivery.

So the budget is DECLARED (``config/lab_pass_settle_budget.yaml``) and the job
ceiling is DERIVED from it, rather than the other way round. What the job can
AFFORD is still computed, because it is a real fact about the run, but it is no
longer the budget: it is compared against the declaration, and a job that cannot
afford the declared budget says so in a named receipt check instead of reporting
the shortfall as a lane-health failure.

FAIL-CLOSED, EVERYWHERE
-----------------------
A missing declaration file, a missing lane entry, a non-integer budget, a
declared budget above the lane's own compose ``start_period`` or below its worst
observed boot are each a raised :class:`SettleBudgetError`. None of them falls
back to a remainder: the remainder IS the defect, and a fallback to it would
reintroduce it silently on exactly the runs where the declaration was
unreadable.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The MODEL lives in the receipt module, not here, and the dependency points
# this way on purpose. ``lab_pass_receipt.py`` is held to a stdlib-only import
# surface -- run 34235502322 died at import on a bare runner and took a whole
# dev-candidate delivery with it -- and this module needs PyYAML to read the
# declaration. So the reader depends on the writer's vocabulary; the writer
# depends on nothing.
from scripts.ci.lab_pass_receipt import ModelSettleBudget

__all__ = [
    "DEFAULT_DECLARATION_PATH",
    "ModelLaneSettleDeclaration",
    "SettleBudgetError",
    "affordable_seconds",
    "assert_declaration_within_bounds",
    "derive_settle_budget",
    "load_declaration",
    "max_declared_start_period_seconds",
    "parse_compose_duration",
]

#: Resolved relative to this file and never from an environment variable: a
#: declaration whose location can be repointed is a declaration that can be
#: widened without review.
DEFAULT_DECLARATION_PATH: Final[Path] = (
    REPO_ROOT / "config" / "lab_pass_settle_budget.yaml"
)

SCHEMA_VERSION: Final[int] = 1

#: ``start_period: 1800s``. Compose accepts a duration string; the receipt needs
#: seconds. Only the units compose itself documents are accepted -- an
#: unrecognised suffix raises rather than being coerced to a number that would
#: silently be wrong by three orders of magnitude.
_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>us|ms|s|m|h)?$"
)
_UNIT_SECONDS: Final[dict[str, float]] = {
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


class SettleBudgetError(RuntimeError):
    """The settle budget could not be resolved from the declaration.

    Its own class so a caller cannot catch it alongside a transport error and
    fall through to a default. There is no default.
    """


def parse_compose_duration(raw: Any) -> float:
    """Seconds from a compose duration string such as ``1800s`` or ``2m``."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if not isinstance(raw, str):
        msg = f"not a compose duration: {raw!r}"
        raise SettleBudgetError(msg)
    match = _DURATION_RE.match(raw.strip())
    if match is None:
        msg = (
            f"not a compose duration: {raw!r}. Accepted units are "
            f"{sorted(_UNIT_SECONDS)}; an unrecognised suffix is refused rather "
            "than coerced."
        )
        raise SettleBudgetError(msg)
    unit = match.group("unit") or "s"
    return float(match.group("value")) * _UNIT_SECONDS[unit]


@dataclass(frozen=True)
class ModelLaneSettleDeclaration:
    """One lane's declared settle budget and the two bounds that constrain it."""

    lane: str
    settle_budget_seconds: int
    compose_file: str
    readiness_services: tuple[str, ...]
    #: ``(seconds, sha)`` pairs, each a ``lane ready after Ns`` line from a real
    #: emitted receipt. The lower bound is their MAX, so adding a slower
    #: observation without raising the budget turns the bounds check red.
    observed_boot_seconds: tuple[tuple[int, str], ...]

    @property
    def worst_observed_seconds(self) -> int:
        return max(seconds for seconds, _ in self.observed_boot_seconds)


def _require(payload: Any, key: str, where: str) -> Any:
    if not isinstance(payload, dict) or key not in payload:
        msg = f"{where}: missing required key {key!r}"
        raise SettleBudgetError(msg)
    return payload[key]


def load_declaration(lane: str, path: Path | None = None) -> ModelLaneSettleDeclaration:
    """Read one lane's declaration. Every failure raises; none defaults."""
    declaration_path = path or DEFAULT_DECLARATION_PATH
    try:
        raw_text = declaration_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = (
            f"the settle-budget declaration at {declaration_path} is unreadable "
            f"({exc}). Refusing to fall back to a job-ceiling remainder: the "
            "remainder is the defect OMN-18436 removed."
        )
        raise SettleBudgetError(msg) from exc

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        msg = f"{declaration_path} is not valid YAML: {exc}"
        raise SettleBudgetError(msg) from exc

    version = _require(payload, "schema_version", str(declaration_path))
    if version != SCHEMA_VERSION:
        msg = (
            f"{declaration_path}: schema_version={version!r} is not "
            f"{SCHEMA_VERSION}. Refusing to interpret a declaration written "
            "against a different contract."
        )
        raise SettleBudgetError(msg)

    lanes = _require(payload, "lanes", str(declaration_path))
    if not isinstance(lanes, dict) or lane not in lanes:
        msg = (
            f"{declaration_path} declares no settle budget for lane {lane!r} "
            f"(declared: {sorted(lanes) if isinstance(lanes, dict) else lanes}). "
            "A lane whose boot budget nobody has measured does not get a "
            "remainder; it gets a refusal."
        )
        raise SettleBudgetError(msg)

    entry = lanes[lane]
    where = f"{declaration_path} lanes.{lane}"
    budget = _require(entry, "settle_budget_seconds", where)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        msg = (
            f"{where}: settle_budget_seconds must be a positive integer, got {budget!r}"
        )
        raise SettleBudgetError(msg)

    services = _require(entry, "readiness_services", where)
    if not isinstance(services, list) or not services:
        msg = f"{where}: readiness_services must be a non-empty list"
        raise SettleBudgetError(msg)

    observations_raw = _require(entry, "observed_boot_seconds", where)
    if not isinstance(observations_raw, list) or not observations_raw:
        msg = (
            f"{where}: observed_boot_seconds must be a non-empty list. A budget "
            "with no observation behind it is a guess, and the lower bound that "
            "keeps it honest has nothing to compare against."
        )
        raise SettleBudgetError(msg)
    observations: list[tuple[int, str]] = []
    for row in observations_raw:
        seconds = _require(row, "seconds", f"{where}.observed_boot_seconds")
        sha = _require(row, "sha", f"{where}.observed_boot_seconds")
        if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds <= 0:
            msg = f"{where}.observed_boot_seconds: seconds must be a positive integer, got {seconds!r}"
            raise SettleBudgetError(msg)
        observations.append((seconds, str(sha)))

    return ModelLaneSettleDeclaration(
        lane=lane,
        settle_budget_seconds=budget,
        compose_file=str(_require(entry, "compose_file", where)),
        readiness_services=tuple(str(s) for s in services),
        observed_boot_seconds=tuple(observations),
    )


def max_declared_start_period_seconds(
    declaration: ModelLaneSettleDeclaration, repo_root: Path | None = None
) -> float:
    """The lane's own declared upper bound, parsed from its compose model.

    A readiness service that declares no ``start_period`` raises. Skipping it
    would silently lower the ceiling the bounds check compares against, which is
    the direction that lets a too-large budget through.
    """
    root = repo_root or REPO_ROOT
    compose_path = root / declaration.compose_file
    try:
        model = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"cannot read the compose model at {compose_path}: {exc}"
        raise SettleBudgetError(msg) from exc

    services = (model or {}).get("services")
    if not isinstance(services, dict):
        msg = f"{compose_path} declares no services mapping"
        raise SettleBudgetError(msg)

    periods: list[float] = []
    for name in declaration.readiness_services:
        service = services.get(name)
        if not isinstance(service, dict):
            msg = (
                f"{compose_path} declares no service {name!r}, but the settle "
                "declaration names it as readiness-gated"
            )
            raise SettleBudgetError(msg)
        healthcheck = service.get("healthcheck")
        if not isinstance(healthcheck, dict) or "start_period" not in healthcheck:
            msg = (
                f"{compose_path} service {name!r} declares no healthcheck "
                "start_period, so the lane states no upper bound on its own boot"
            )
            raise SettleBudgetError(msg)
        periods.append(parse_compose_duration(healthcheck["start_period"]))
    return max(periods)


def assert_declaration_within_bounds(
    declaration: ModelLaneSettleDeclaration, repo_root: Path | None = None
) -> None:
    """Both bounds, checked against parsed models rather than against prose."""
    upper = max_declared_start_period_seconds(declaration, repo_root)
    if declaration.settle_budget_seconds > upper:
        msg = (
            f"lane {declaration.lane}: declared settle budget "
            f"{declaration.settle_budget_seconds}s exceeds the lane's own "
            f"compose start_period of {upper:.0f}s. Waiting past the point "
            "docker stops treating a slow boot as a boot proves nothing."
        )
        raise SettleBudgetError(msg)
    worst = declaration.worst_observed_seconds
    if declaration.settle_budget_seconds < worst:
        msg = (
            f"lane {declaration.lane}: declared settle budget "
            f"{declaration.settle_budget_seconds}s is below the worst boot this "
            f"lane has been observed to take ({worst}s). An observation is a "
            "FLOOR on the real cost, so a budget under it fails on timing alone."
        )
        raise SettleBudgetError(msg)


def affordable_seconds(
    job_ceiling_seconds: int, elapsed_seconds: int, reserved_tail_seconds: int
) -> int:
    """What is left of this job's ceiling once the tail is reserved.

    The tail exists because the receipt still has to be BUILT, VALIDATED and
    UPLOADED after the probe returns; a settle wait that consumed it would
    produce no receipt at all, and rule 24(b) cannot tell "no receipt" from
    "nobody ran it".
    """
    return max(0, job_ceiling_seconds - elapsed_seconds - reserved_tail_seconds)


def derive_settle_budget(
    lane: str,
    job_ceiling_seconds: int,
    elapsed_seconds: int,
    reserved_tail_seconds: int,
    path: Path | None = None,
    repo_root: Path | None = None,
) -> ModelSettleBudget:
    """Resolve the declared budget and this job's affordance for it."""
    declaration = load_declaration(lane, path)
    assert_declaration_within_bounds(declaration, repo_root)
    source_path = path or DEFAULT_DECLARATION_PATH
    return ModelSettleBudget(
        lane=lane,
        declared_seconds=declaration.settle_budget_seconds,
        affordable_seconds=affordable_seconds(
            job_ceiling_seconds, elapsed_seconds, reserved_tail_seconds
        ),
        job_ceiling_seconds=job_ceiling_seconds,
        elapsed_seconds=elapsed_seconds,
        reserved_tail_seconds=reserved_tail_seconds,
        source=str(source_path.relative_to(REPO_ROOT))
        if source_path.is_relative_to(REPO_ROOT)
        else str(source_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve one lane's DECLARED settle budget and this job's "
            "affordance for it, and print it as one line of JSON for "
            "lab_pass_receipt.py probe-lane --settle-budget-json."
        )
    )
    parser.add_argument("--lane", required=True)
    parser.add_argument(
        "--job-ceiling-seconds",
        type=int,
        required=True,
        help="the emitting job's own timeout-minutes, in seconds",
    )
    parser.add_argument(
        "--elapsed-seconds",
        type=int,
        required=True,
        help="seconds this job has already spent, for the affordability read",
    )
    parser.add_argument(
        "--reserved-tail-seconds",
        type=int,
        default=120,
        help=(
            "seconds held back so the receipt can still be built, validated and "
            "uploaded after the probe returns. An unwritten receipt is the one "
            "outcome worse than a failing one."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        budget = derive_settle_budget(
            lane=args.lane,
            job_ceiling_seconds=args.job_ceiling_seconds,
            elapsed_seconds=args.elapsed_seconds,
            reserved_tail_seconds=args.reserved_tail_seconds,
        )
    except SettleBudgetError as exc:
        # Fail closed, and print NOTHING on stdout: a caller that captured a
        # partial line and passed it on would be back to guessing a budget.
        print(f"::error::settle budget unresolved: {exc}", file=sys.stderr)
        return 1
    print(budget.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
