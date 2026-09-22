# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19195 -- the C12 producer: the deployed provider catalogue, graded.

WHAT THIS IS
    Beta criterion C12 reads "Provider catalogue equals the handler-backed
    set; no Claude entry; no house entry", proven by "catalogue readback +
    negative test". This producer takes both, on the deployed lane, every run,
    and its EXIT CODE is the verdict -- the shape C11 and C15 already use, so
    the board reads a workflow conclusion and needs no second grader.

    ``0``  every check below held on the deployed code
    ``1``  at least one did not; the record names each one and why
    ``2``  the probe could not run at all (no container, docker refused, the
           observer could not import the subject). Deliberately distinct from
           ``1``: "I could not look" is not "the catalogue is wrong".

HOW IT READS THE SUBJECT
    ``c12_provider_catalogue_observe.py`` is piped to
    ``docker exec -i -u appuser <container> python - /app`` on the lane's host,
    so it runs in the DEPLOYED interpreter against the DEPLOYED package and the
    DEPLOYED intake route, as the container's own unprivileged user. It prints
    what the deployed code returned and grades nothing; the grading is here,
    against expectations declared in THIS file, and is tested offline over
    recorded observations. The observer writes nothing: every injection is an
    in-memory copy handed to a pure function.

WHAT IS CHECKED, BY CLAUSE
    parity      the house-keyed slug set derived from the deployed platform
                rungs equals offered + not_offered, the deployed parity
                function agrees, and neither set is empty. The grader
                re-derives the equality itself rather than trusting the
                deployed function's own "clean".
    no_claude   no catalogue id matches ``anthropic|claude`` by the grader's
                own pattern, the deployed scan over every string VALUE on
                every row found nothing, and the intake model refuses two
                Claude spellings on the ``provider`` field.
    no_house    the deployed OMN-18311 validator returns zero findings.
    surface     the intake route's resolved body model IS the model that
                validates against the catalogue; it ACCEPTS every offered
                provider (the positive control: a gate that refuses
                everything also refuses Claude) and refuses every not-offered
                one and an unbacked one.
    negative    each clause is shown to bite in the same process: an unbacked
                row, a dropped row, a Claude row, a laundered house reference,
                a house rung identity collision, an un-registerable provider,
                and an empty rung set that must fail closed.

    The finding-class names are declared HERE. Reading them back from the
    observer would let a renamed class grade itself.

WHY THE NEGATIVE ARM IS GRADED, NOT LOGGED
    The shipped reading alone is exactly what a checker that always answers
    "clean" produces. The injections run the SAME deployed functions in the
    SAME process, so a checker that stopped biting reds this probe instead of
    reading as a proof.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

OBSERVER: Final[Path] = (
    Path(__file__).resolve().parent / "c12_provider_catalogue_observe.py"
)

DEFAULT_CONTAINER: Final[str] = "onex-api"
DEFAULT_USER: Final[str] = "appuser"
DEFAULT_APP_ROOT: Final[str] = "/app"

# The grader's own Claude pattern. Independent of the deployed
# FORBIDDEN_PROVIDER_PATTERN on purpose: if that constant were weakened, a scan
# using it would weaken with it.
CLAUDE_RE: Final[re.Pattern[str]] = re.compile(r"anthropic|claude", re.IGNORECASE)

# Must match the observer's constants; pinned by a test.
UNBACKED_PROVIDER: Final[str] = "openai"
CLAUDE_PROVIDERS: Final[tuple[str, ...]] = ("anthropic", "Claude-3")
SYNTHETIC_TOKEN_PROVIDER: Final[str] = "c12probe"
ROUTE_PREFIX: Final[str] = "/v1/tenants/me/inference-credentials"

DECLARED_HOUSE_REF: Final[str] = "declared_house_ref"
HOUSE_RUNG_IDENTITY_COLLISION: Final[str] = "house_rung_identity_collision"
NO_CUSTOMER_REGISTERABLE_KEY: Final[str] = "no_customer_registerable_key"
FAIL_CLOSED_EXCEPTION: Final[str] = "HouseCatalogueError"

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2

_ABSENT: Final[object] = object()


class ProbeInputError(RuntimeError):
    """The probe could not run. Exit 2, never exit 1."""


@dataclass(frozen=True)
class Check:
    name: str
    clause: str
    ok: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "clause": self.clause,
            "ok": self.ok,
            "evidence": self.evidence,
        }


@dataclass
class Record:
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [f"{c.clause}/{c.name}: {c.evidence}" for c in self.checks if not c.ok]

    @property
    def verdict(self) -> str:
        return "pass" if self.checks and not self.failures else "fail"

    @property
    def exit_code(self) -> int:
        return EXIT_OK if self.verdict == "pass" else EXIT_FINDINGS

    @property
    def detail(self) -> str:
        if self.verdict == "pass":
            return (
                f"all {len(self.checks)} checks held: the deployed catalogue equals "
                "the handler-backed set, carries no Claude entry and no house "
                "entry, is bound to the intake route, and every injection was "
                "refused in the same process"
            )
        return f"{len(self.failures)} of {len(self.checks)} checks failed"

    def to_dict(
        self, *, target: dict[str, Any], as_of: str, observed: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "criterion": "C12",
            "ticket": "OMN-19195",
            "as_of": as_of,
            "target": target,
            "verdict": self.verdict,
            "detail": self.detail,
            "checks": [c.to_dict() for c in self.checks],
            "failures": self.failures,
            "observed": observed,
        }


def _get(obs: Any, *path: str) -> Any:
    cur = obs
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return _ABSENT
        cur = cur[key]
    return cur


def _str_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def _has_finding(entry: Any, provider: str, finding_class: str) -> bool:
    findings = _get(entry, "findings")
    if not isinstance(findings, list):
        return False
    return any(
        isinstance(f, dict)
        and f.get("provider") == provider
        and f.get("finding_class") == finding_class
        for f in findings
    )


def _refused_on_provider(entry: Any) -> bool:
    if not isinstance(entry, dict) or entry.get("accepted") is not False:
        return False
    errors = entry.get("errors")
    if not isinstance(errors, list):
        return False
    return any(isinstance(e, dict) and e.get("loc") == ["provider"] for e in errors)


def grade(obs: dict[str, Any]) -> Record:
    """Grade observer output. Pure; every expectation is declared here."""
    rec = Record()

    def add(clause: str, name: str, ok: bool, evidence: str) -> None:
        rec.checks.append(
            Check(name=name, clause=clause, ok=bool(ok), evidence=evidence)
        )

    offered = _str_list(_get(obs, "shipped", "offered"))
    not_offered = _str_list(_get(obs, "shipped", "not_offered"))
    house = _str_list(_get(obs, "shipped", "house_keyed_slugs"))
    gap_missing = _str_list(
        _get(obs, "shipped", "parity_gap", "missing_from_catalogue")
    )
    gap_unbacked = _str_list(
        _get(obs, "shipped", "parity_gap", "unbacked_in_catalogue")
    )

    # ---- parity -------------------------------------------------------------
    add(
        "parity",
        "catalogue_non_empty",
        bool(offered),
        f"offered={offered!r} -- an empty catalogue is not parity, it is an outage",
    )
    add(
        "parity",
        "handler_backed_set_non_empty",
        bool(house),
        f"house_keyed_slugs={house!r} over rung_count={_get(obs, 'shipped', 'rung_count')!r}",
    )
    declared = set(offered or []) | set(not_offered or [])
    add(
        "parity",
        "declared_equals_handler_backed",
        offered is not None
        and not_offered is not None
        and house is not None
        and declared == set(house),
        f"offered+not_offered={sorted(declared)!r} handler_backed={house!r}",
    )
    add(
        "parity",
        "offered_and_not_offered_disjoint",
        offered is not None
        and not_offered is not None
        and not (set(offered) & set(not_offered)),
        f"overlap={sorted(set(offered or []) & set(not_offered or []))!r}",
    )
    add(
        "parity",
        "deployed_parity_gap_clean",
        gap_missing == [] and gap_unbacked == [],
        f"missing_from_catalogue={gap_missing!r} unbacked_in_catalogue={gap_unbacked!r}",
    )

    # ---- no Claude entry ----------------------------------------------------
    ids = [*(offered or []), *(not_offered or [])]
    add(
        "no_claude",
        "no_claude_catalogue_id",
        offered is not None and not any(CLAUDE_RE.search(p) for p in ids),
        f"ids={ids!r}",
    )
    hits = _get(obs, "shipped", "claude_hits")
    add(
        "no_claude",
        "deployed_value_scan_empty",
        hits == [],
        f"claude_hits={hits if hits is not _ABSENT else '<absent>'!r}",
    )
    for provider in CLAUDE_PROVIDERS:
        entry = _get(obs, "intake", provider)
        add(
            "no_claude",
            f"intake_refuses_{provider}",
            _refused_on_provider(entry),
            f"intake[{provider!r}]={entry if entry is not _ABSENT else '<absent>'!r}"[
                :500
            ],
        )

    # ---- no house entry -----------------------------------------------------
    house_entry = _get(obs, "shipped", "house_entry")
    add(
        "no_house",
        "house_validator_zero_findings",
        isinstance(house_entry, dict)
        and house_entry.get("findings") == []
        and "raised" not in house_entry,
        f"house_entry={house_entry if house_entry is not _ABSENT else '<absent>'!r}",
    )

    # ---- customer surface ---------------------------------------------------
    add(
        "surface",
        "route_is_the_intake_route",
        _get(obs, "route", "prefix") == ROUTE_PREFIX,
        f"route.prefix={_get(obs, 'route', 'prefix')!r}",
    )
    add(
        "surface",
        "route_body_is_catalogue_model",
        _get(obs, "route", "bound_to_catalogue_model") is True,
        f"body_models={_get(obs, 'route', 'body_models')!r}",
    )
    for provider in offered or []:
        entry = _get(obs, "intake", provider)
        add(
            "surface",
            f"intake_accepts_{provider}",
            isinstance(entry, dict) and entry.get("accepted") is True,
            f"intake[{provider!r}]={entry if entry is not _ABSENT else '<absent>'!r}"[
                :500
            ],
        )
    for provider in [*(not_offered or []), UNBACKED_PROVIDER]:
        entry = _get(obs, "intake", provider)
        add(
            "surface",
            f"intake_refuses_{provider}",
            _refused_on_provider(entry),
            f"intake[{provider!r}]={entry if entry is not _ABSENT else '<absent>'!r}"[
                :500
            ],
        )

    # ---- negative test ------------------------------------------------------
    neg = _get(obs, "negative")
    pu = _get(neg, "parity_unbacked")
    add(
        "negative",
        "unbacked_row_detected",
        _get(pu, "unbacked_in_catalogue") == [UNBACKED_PROVIDER]
        and _get(pu, "missing_from_catalogue") == [],
        f"parity_unbacked={pu if pu is not _ABSENT else '<absent>'!r}",
    )
    pm = _get(neg, "parity_missing")
    dropped = _get(pm, "dropped")
    add(
        "negative",
        "dropped_row_detected",
        isinstance(dropped, str)
        and _get(pm, "missing_from_catalogue") == [dropped]
        and _get(pm, "unbacked_in_catalogue") == [],
        f"parity_missing={pm if pm is not _ABSENT else '<absent>'!r}",
    )
    cr = _get(neg, "claude_row")
    cr_hits = _get(cr, "claude_hits")
    add(
        "negative",
        "claude_row_detected",
        isinstance(cr_hits, list) and CLAUDE_PROVIDERS[0] in cr_hits,
        f"claude_row={cr if cr is not _ABSENT else '<absent>'!r}",
    )
    for key, klass in (
        ("house_declared_ref", DECLARED_HOUSE_REF),
        ("house_rung_collision", HOUSE_RUNG_IDENTITY_COLLISION),
    ):
        entry = _get(neg, key)
        provider = _get(entry, "provider")
        add(
            "negative",
            f"{key}_detected",
            isinstance(provider, str) and _has_finding(entry, provider, klass),
            f"{key}={entry if entry is not _ABSENT else '<absent>'!r}",
        )
    nr = _get(neg, "no_registerable_key")
    add(
        "negative",
        "no_registerable_key_detected",
        _has_finding(nr, SYNTHETIC_TOKEN_PROVIDER, NO_CUSTOMER_REGISTERABLE_KEY),
        f"no_registerable_key={nr if nr is not _ABSENT else '<absent>'!r}",
    )
    fc = _get(neg, "empty_rungs_fail_closed")
    add(
        "negative",
        "empty_rungs_fail_closed",
        _get(fc, "raised", "type") == FAIL_CLOSED_EXCEPTION,
        f"empty_rungs_fail_closed={fc if fc is not _ABSENT else '<absent>'!r}",
    )

    return rec


# ---------------------------------------------------------------------------
# Live half
# ---------------------------------------------------------------------------


def _docker(
    docker_bin: str, args: list[str], *, stdin: str | None, timeout: float
) -> str:
    try:
        proc = subprocess.run(
            [docker_bin, *args],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeInputError(
            f"docker {args[0]} could not run: {type(exc).__name__}"
        ) from exc
    if proc.returncode != 0:
        raise ProbeInputError(
            f"docker {args[0]} exited {proc.returncode}: {proc.stderr.strip()[:400]}"
        )
    return proc.stdout


def observe_live(
    *, docker_bin: str, container: str, user: str, app_root: str, timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    inspect = _docker(
        docker_bin,
        [
            "inspect",
            "--format",
            "{{.Config.Image}}|{{.State.Status}}|{{.State.StartedAt}}",
            container,
        ],
        stdin=None,
        timeout=timeout,
    ).strip()
    image, _, rest = inspect.partition("|")
    status, _, started = rest.partition("|")
    target = {
        "container": container,
        "image": image,
        "status": status,
        "started_at": started,
    }
    if status != "running":
        raise ProbeInputError(f"container {container!r} is {status!r}, not running")
    try:
        source = OBSERVER.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeInputError(f"observer not readable at {OBSERVER}: {exc}") from exc
    out = _docker(
        docker_bin,
        ["exec", "-i", "-u", user, container, "python", "-", app_root],
        stdin=source,
        timeout=timeout,
    )
    return parse_observer_output(out), target


def parse_observer_output(out: str) -> dict[str, Any]:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise ProbeInputError("the observer printed nothing")
    try:
        payload = json.loads(lines[-1])
    except ValueError as exc:
        raise ProbeInputError("the observer's last line is not JSON") from exc
    return checked_payload(payload)


def checked_payload(payload: Any) -> dict[str, Any]:
    """Refuse a payload that is not an observation. Exit 2, never a verdict."""
    if not isinstance(payload, dict):
        raise ProbeInputError("the observer's output is not a JSON object")
    if "error" in payload:
        raise ProbeInputError(
            f"the observer could not read the subject: {payload['error']!r}"
        )
    return payload


def _observed_summary(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "omnimarket_version": obs.get("omnimarket_version"),
        "files_sha256": obs.get("files"),
        "shipped": obs.get("shipped"),
        "route": obs.get("route"),
        "negative": obs.get("negative"),
    }


def _render_summary(record: Record, *, target: dict[str, Any], as_of: str) -> str:
    lines = [
        "## C12 -- the deployed provider catalogue (OMN-19195)",
        "",
        f"**Verdict: `{record.verdict}`** on `{target.get('container')}` "
        f"(`{target.get('image')}`) at `{as_of}`.",
        "",
        record.detail,
        "",
        "| Clause | Check | Pass |",
        "| --- | --- | --- |",
    ]
    for c in record.checks:
        lines.append(f"| {c.clause} | {c.name} | {'yes' if c.ok else '**no**'} |")
    if record.failures:
        lines += ["", "Failures:", ""]
        lines += [f"- `{f}`" for f in record.failures]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--app-root", default=DEFAULT_APP_ROOT)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--replay",
        default="",
        help="Grade recorded observer output (a JSON object) instead of reading the lane.",
    )
    parser.add_argument("--record", default="", help="Write the JSON record here.")
    parser.add_argument("--summary", default="", help="Append a markdown summary here.")
    args = parser.parse_args(argv)

    as_of = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if args.replay:
            try:
                replayed = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ProbeInputError(
                    f"replay file {args.replay!r} is not JSON"
                ) from exc
            obs = checked_payload(replayed)
            target: dict[str, Any] = {"container": "<replay>", "image": None}
        else:
            obs, target = observe_live(
                docker_bin=args.docker_bin,
                container=args.container,
                user=args.user,
                app_root=args.app_root,
                timeout=args.timeout,
            )
    except (ProbeInputError, OSError) as exc:
        print(f"::error::C12 probe could not run: {exc}", file=sys.stderr)
        return EXIT_INPUT

    record = grade(obs)
    if args.record:
        Path(args.record).write_text(
            json.dumps(
                record.to_dict(
                    target=target, as_of=as_of, observed=_observed_summary(obs)
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(_render_summary(record, target=target, as_of=as_of))

    print(f"container: {target.get('container')}  image: {target.get('image')}")
    for c in record.checks:
        print(f"{c.clause:<10} {c.name:<40} pass={c.ok}")
    if record.failures:
        for f in record.failures:
            print(f"::error::{f}", file=sys.stderr)
        print(f"::error::C12 RED -- {record.detail}", file=sys.stderr)
    else:
        print(f"C12 GREEN -- {record.detail}")
    return record.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
