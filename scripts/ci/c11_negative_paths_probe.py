# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19175 -- the C11 producer: four negative paths, each a typed refusal.

WHAT THIS IS
    A probe that drives the four negative paths of beta criterion C11 against a
    deployed lane's tenant-bearing API and grades each one against an
    expectation declared HERE, in the grader, not read off the response. Its
    EXIT CODE is the criterion's verdict -- the C15 chain-canary shape -- and
    it additionally writes a small JSON record so a reader can see the four
    rows without reading a log.

    ``0``  every case returned its declared typed refusal and the control passed
    ``1``  at least one case did not; the record names which and why
    ``2``  the probe could not run at all (unresolvable credential, unreadable
           replay file). Deliberately distinct from ``1``: "I could not run" is
           not "the platform refused wrongly", and collapsing them is how a
           configuration failure gets reported as a product failure.

THE POSITIVE CONTROL IS NOT DECORATION
    Four refusals are exactly what an unreachable endpoint produces. The
    control (``GET /health``) runs in the SAME invocation against the SAME base
    URL, so an all-refused result from a dead lane grades FAIL rather than
    PASS. This is the criterion's own "a zero that is not proven" failure class
    and it is the reason the control is graded rather than logged.

THE HEADER FORM IS PINNED
    ``x-api-key`` and not ``Authorization: Bearer``. Measured on the lab dev
    lane 2026-09-22: a bearer credential returns the same
    ``auth.credential.invalid`` code with a different ``detail``, so a producer
    sending both forms would have two arms collide on one code and would then
    have to grade on ``detail`` prose. One form, pinned, and the pin has a test.

THE MALFORMED ARM CARRIES NO CODE, BY OBSERVATION
    Payload refusals on this API are the framework's default 422 envelope --
    ``{"detail":[{"type":"missing","loc":["body","<field>"],...}]}`` -- and
    carry no ``code`` field at all. The criterion asks for "a typed refusal
    naming the offending field", which that envelope does: ``type`` types it
    and ``loc`` names the field. So this arm is graded on ``type`` and ``loc``.
    A producer must not synthesize a code the platform never returned; that is
    a grader grading itself.

THE THREE CODED ARMS ARE GRADED ON DISTINCTNESS TOO
    OMN-18042 landed these codes precisely because wrong-key and absent-key had
    previously been a byte-identical bare ``401 {"detail":"Unauthorized"}``. A
    regression to that state is a PASS under per-case grading alone, so the
    codes are additionally checked pairwise distinct.

WHY A REPLAY MODE EXISTS
    ``--replay`` grades recorded observations instead of making requests. That
    is what makes the grading logic falsifiable offline, in the ordinary test
    suite, rather than only on a lane -- including the cases a live run cannot
    produce on demand (a cross-tenant read that succeeds, two codes collapsing
    onto one). The live half is the workflow's job.

NO CREDENTIAL REACHES ARGV, A LOG OR THE RECORD
    The valid credential is read from the environment by NAME. ``/proc`` is
    world-readable, so a value on a command line is readable by every process
    on the host; the record is an uploaded artifact, so a value in it is a real
    exposure. Only the variable's NAME is ever printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

# The one header form this producer sends. See the module docstring.
HEADER_NAME: Final[str] = "x-api-key"

# A well-formed credential that is never valid anywhere. It is a literal on
# purpose: the wrong-key arm must present SOMETHING, and generating a random
# value would make a failing run un-reproducible.
SYNTHETIC_INVALID_KEY: Final[str] = "onx_c11_probe_never_valid_omn19175"

# The field the malformed body omits, and therefore the field the 422 envelope
# must name back. Both halves are declared here so they cannot drift apart.
MALFORMED_BODY: Final[dict[str, Any]] = {"omn19175_not_a_field": True}
MALFORMED_MISSING_FIELD: Final[str] = "name"

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2


class ProbeInputError(RuntimeError):
    """The probe could not run. Exit 2, never exit 1."""


@dataclass(frozen=True)
class CaseSpec:
    """One declared expectation. Pre-stated, never derived from a response."""

    name: str
    method: str
    path: str
    # "none" -> send no credential; "invalid" -> the synthetic one above;
    # "valid" -> the credential resolved from the environment by name.
    credential: str
    expected_status: int
    expected_code: str | None = None
    expected_missing_field: str | None = None
    body: dict[str, Any] | None = None
    why: str = ""


CONTROL: Final[CaseSpec] = CaseSpec(
    name="positive_control",
    method="GET",
    path="/health",
    credential="none",
    expected_status=200,
    why=(
        "Runs in the same invocation as the four refusals so an all-refused "
        "result from a dead endpoint cannot grade PASS."
    ),
)

CASES: Final[tuple[CaseSpec, ...]] = (
    CaseSpec(
        name="wrong_key",
        method="GET",
        path="/v1/whoami",
        credential="invalid",
        expected_status=401,
        expected_code="auth.credential.invalid",
        why="A credential was presented and did not resolve.",
    ),
    CaseSpec(
        name="absent_key",
        method="GET",
        path="/v1/whoami",
        credential="none",
        expected_status=401,
        expected_code="auth.credential.absent",
        why="No credential was presented at all -- distinct from wrong_key.",
    ),
    CaseSpec(
        name="wrong_tenant",
        method="GET",
        # Formatted with the other tenant's slug at request time.
        path="/v1/tenants/{other_tenant_slug}/api-keys",
        credential="valid",
        expected_status=403,
        expected_code="auth.tenant.forbidden",
        why=(
            "A VALID credential against a tenant it does not own. Auth resolves "
            "before the tenant comparison, so this arm is the only one that "
            "needs a working credential -- which is also what makes it the arm "
            "that proves the credential works."
        ),
    ),
    CaseSpec(
        name="malformed",
        method="POST",
        path="/v1/api-keys",
        credential="valid",
        expected_status=422,
        expected_code=None,
        expected_missing_field=MALFORMED_MISSING_FIELD,
        body=MALFORMED_BODY,
        why=(
            "A schema-invalid body on an AUTHENTICATED route, so the refusal is "
            "a validation refusal and not an auth one. Request validation runs "
            "before the endpoint function, so nothing is created."
        ),
    ),
)


@dataclass(frozen=True)
class Observation:
    """What came back. ``error`` is set when no HTTP response was obtained."""

    status: int | None
    body: Any = None
    error: str | None = None


@dataclass
class CaseResult:
    name: str
    expected_status: int
    expected_code: str | None
    observed_status: int | None
    observed_code: str | None
    observed_detail: str
    ok: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.name,
            "expected_status": self.expected_status,
            "expected_code": self.expected_code,
            "observed_status": self.observed_status,
            "observed_code": self.observed_code,
            "observed_detail": self.observed_detail,
            "pass": self.ok,
            "reason": self.reason,
        }


@dataclass
class Record:
    positive_control: CaseResult
    cases: list[CaseResult]
    failures: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "pass" if not self.failures else "fail"

    @property
    def exit_code(self) -> int:
        return EXIT_OK if not self.failures else EXIT_FINDINGS

    @property
    def detail(self) -> str:
        if not self.failures:
            return (
                "four negative paths each returned their declared typed refusal, "
                "the three coded refusals are pairwise distinct, and the positive "
                "control passed in the same invocation"
            )
        return "; ".join(self.failures)

    def to_dict(self, *, base_url: str, as_of: str) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "criterion": "C11",
            "ticket": "OMN-19175",
            "as_of": as_of,
            "base_url": base_url,
            "header_form": HEADER_NAME,
            "verdict": self.verdict,
            "detail": self.detail,
            "positive_control": self.positive_control.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "failures": list(self.failures),
        }


def _summarise(body: Any, limit: int = 240) -> str:
    """A short, value-free rendering of a response body for the record."""
    if body is None:
        return ""
    try:
        text = body if isinstance(body, str) else json.dumps(body, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(body)
    return text[:limit]


def _code_of(body: Any) -> str | None:
    if isinstance(body, dict):
        code = body.get("code")
        if isinstance(code, str) and code:
            return code
    return None


def _names_missing_field(body: Any, field_name: str) -> bool:
    """Is this the 422 envelope, and does it name ``field_name`` as missing?

    DECISION 3's whole content. ``type == "missing"`` types the refusal and
    ``loc`` names the offending field; both must hold, on the same entry.
    """
    if not isinstance(body, dict):
        return False
    detail = body.get("detail")
    if not isinstance(detail, list) or not detail:
        return False
    for entry in detail:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "missing":
            continue
        loc = entry.get("loc")
        if isinstance(loc, list) and any(part == field_name for part in loc):
            return True
    return False


def grade_case(spec: CaseSpec, observation: Observation) -> CaseResult:
    """Grade one observation against its pre-stated expectation."""
    observed_code = _code_of(observation.body)
    detail = _summarise(observation.body)

    if observation.error is not None:
        return CaseResult(
            spec.name,
            spec.expected_status,
            spec.expected_code,
            observation.status,
            observed_code,
            detail,
            False,
            f"no response obtained: {observation.error}",
        )

    if observation.status != spec.expected_status:
        return CaseResult(
            spec.name,
            spec.expected_status,
            spec.expected_code,
            observation.status,
            observed_code,
            detail,
            False,
            f"expected status {spec.expected_status}, observed {observation.status}",
        )

    if spec.expected_code is not None:
        if observed_code != spec.expected_code:
            return CaseResult(
                spec.name,
                spec.expected_status,
                spec.expected_code,
                observation.status,
                observed_code,
                detail,
                False,
                (
                    f"expected a typed refusal carrying code "
                    f"{spec.expected_code!r}, observed {observed_code!r}"
                ),
            )
    elif spec.expected_missing_field is not None:
        if observed_code is not None:
            return CaseResult(
                spec.name,
                spec.expected_status,
                spec.expected_code,
                observation.status,
                observed_code,
                detail,
                False,
                (
                    "this arm is graded on the validation envelope and declares "
                    f"no code, yet the response carried code {observed_code!r}: "
                    "the platform contract moved and the grader must be "
                    "re-derived from it, not patched around"
                ),
            )
        if not _names_missing_field(observation.body, spec.expected_missing_field):
            return CaseResult(
                spec.name,
                spec.expected_status,
                spec.expected_code,
                observation.status,
                observed_code,
                detail,
                False,
                (
                    "the refusal does not name the offending field: expected a "
                    f"detail entry with type 'missing' whose loc names "
                    f"{spec.expected_missing_field!r}"
                ),
            )

    return CaseResult(
        spec.name,
        spec.expected_status,
        spec.expected_code,
        observation.status,
        observed_code,
        detail,
        True,
        "returned the declared typed refusal",
    )


def grade(observations: Mapping[str, Observation]) -> Record:
    """Grade the control and all four cases, then check code distinctness."""
    missing = [spec.name for spec in (CONTROL, *CASES) if spec.name not in observations]
    if missing:
        raise ProbeInputError(
            "no observation recorded for: " + ", ".join(sorted(missing))
        )

    control = grade_case(CONTROL, observations[CONTROL.name])
    results = [grade_case(spec, observations[spec.name]) for spec in CASES]

    failures: list[str] = []
    if not control.ok:
        failures.append(f"positive_control: {control.reason}")
    for result in results:
        if not result.ok:
            failures.append(f"{result.name}: {result.reason}")

    # Pairwise distinctness over the coded arms. Checked even when every arm
    # passed its own expectation, because the expectations are what make the
    # codes distinct and a future edit could quietly make two of them equal.
    seen: dict[str, str] = {}
    for result in results:
        code = result.observed_code
        if code is None:
            continue
        if code in seen:
            failures.append(
                f"{result.name}: code {code!r} is not distinct from "
                f"{seen[code]!r} -- two refusals the caller cannot tell apart"
            )
        else:
            seen[code] = result.name

    return Record(positive_control=control, cases=results, failures=failures)


# ---------------------------------------------------------------------------
# Live half
# ---------------------------------------------------------------------------


def resolve_credential(env_name: str, environ: Mapping[str, str]) -> str:
    """Resolve the valid credential by NAME. Absent is exit 2, not a verdict."""
    if not env_name:
        raise ProbeInputError(
            "no credential environment variable NAME was given; this probe "
            "never takes a credential VALUE on its command line"
        )
    value = environ.get(env_name, "")
    if not value:
        raise ProbeInputError(
            f"the credential environment variable {env_name!r} is empty or "
            "unset, so the wrong_tenant and malformed arms cannot be driven. "
            "This is a configuration failure and is reported as one"
        )
    return value


def _request(
    *,
    base_url: str,
    spec: CaseSpec,
    other_tenant_slug: str,
    valid_key: str,
    timeout: float,
) -> Observation:
    path = spec.path.format(other_tenant_slug=other_tenant_slug)
    url = base_url.rstrip("/") + path
    data: bytes | None = None
    headers: dict[str, str] = {"accept": "application/json"}
    if spec.body is not None:
        data = json.dumps(spec.body).encode("utf-8")
        headers["content-type"] = "application/json"
    if spec.credential == "invalid":
        headers[HEADER_NAME] = SYNTHETIC_INVALID_KEY
    elif spec.credential == "valid":
        headers[HEADER_NAME] = valid_key

    # Scheme-checked rather than suppressed: urlopen honours `file:` and any
    # registered handler, and this base URL comes from a workflow input.
    if not url.startswith(("http://", "https://")):
        raise ProbeInputError(f"refusing a non-HTTP base URL: {spec.path!r}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked directly above
        url, data=data, headers=headers, method=spec.method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
            return Observation(status=response.status, body=_maybe_json(raw))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return Observation(status=exc.code, body=_maybe_json(raw))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # The exception type, never its message verbatim: a URLError can echo
        # the request, and the request for two arms carries a credential.
        return Observation(status=None, error=type(exc).__name__)


def _maybe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def observe_live(
    *,
    base_url: str,
    other_tenant_slug: str,
    valid_key: str,
    timeout: float,
) -> dict[str, Observation]:
    return {
        spec.name: _request(
            base_url=base_url,
            spec=spec,
            other_tenant_slug=other_tenant_slug,
            valid_key=valid_key,
            timeout=timeout,
        )
        for spec in (CONTROL, *CASES)
    }


def observations_from_replay(payload: Mapping[str, Any]) -> dict[str, Observation]:
    """Read recorded observations. The offline half of the grader's proof."""
    raw = payload.get("observations")
    if not isinstance(raw, dict):
        raise ProbeInputError("replay payload has no 'observations' mapping")
    out: dict[str, Observation] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ProbeInputError(f"replay observation {name!r} is not an object")
        status = entry.get("status")
        if status is not None and not isinstance(status, int):
            raise ProbeInputError(
                f"replay observation {name!r} has a non-integer status"
            )
        error = entry.get("error")
        if error is not None and not isinstance(error, str):
            raise ProbeInputError(f"replay observation {name!r} has a non-string error")
        out[str(name)] = Observation(status=status, body=entry.get("body"), error=error)
    return out


def _render_summary(record: Record, *, base_url: str, as_of: str) -> str:
    lines = [
        "## C11 -- the four negative paths (OMN-19175)",
        "",
        f"**Verdict: `{record.verdict}`** against `{base_url}` at `{as_of}`.",
        "",
        record.detail,
        "",
        "| Case | Expected | Observed | Code | Pass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in (record.positive_control, *record.cases):
        expected = str(case.expected_status)
        if case.expected_code:
            expected += f" / `{case.expected_code}`"
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                case.name,
                expected,
                case.observed_status,
                f"`{case.observed_code}`" if case.observed_code else "—",
                "yes" if case.ok else "**no**",
            )
        )
    lines += [
        "",
        "The positive control runs in the same invocation as the four refusals, "
        "so an all-refused result from an unreachable endpoint grades FAIL.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        default="",
        help="Tenant-bearing API base URL. Required unless --replay is given.",
    )
    parser.add_argument(
        "--credential-env",
        default="",
        help=(
            "NAME of the environment variable holding the valid credential. "
            "Never a value: argv is world-readable."
        ),
    )
    parser.add_argument(
        "--other-tenant-slug",
        default="",
        help="A tenant slug the credential does NOT own; drives the wrong_tenant arm.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--replay",
        default="",
        help="Grade recorded observations from this JSON file instead of making requests.",
    )
    parser.add_argument("--record", default="", help="Write the JSON record here.")
    parser.add_argument(
        "--summary",
        default="",
        help="Append a markdown summary here (the job step summary).",
    )
    args = parser.parse_args(argv)

    as_of = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        if args.replay:
            base_url = args.base_url or "<replay>"
            payload = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            observations = observations_from_replay(payload)
        else:
            if not args.base_url:
                raise ProbeInputError("--base-url is required unless --replay is given")
            if not args.other_tenant_slug:
                raise ProbeInputError("--other-tenant-slug is required")
            base_url = args.base_url
            valid_key = resolve_credential(args.credential_env, os.environ)
            print(f"base url:        {base_url}")
            print(
                f"credential var:  {args.credential_env}  (a NAME; the value is never printed)"
            )
            print(f"header form:     {HEADER_NAME}")
            print(f"other tenant:    {args.other_tenant_slug}")
            observations = observe_live(
                base_url=base_url,
                other_tenant_slug=args.other_tenant_slug,
                valid_key=valid_key,
                timeout=args.timeout,
            )
        record = grade(observations)
    except ProbeInputError as exc:
        print(f"::error::C11 probe could not run: {exc}", file=sys.stderr)
        return EXIT_INPUT

    payload_out = record.to_dict(base_url=base_url, as_of=as_of)
    if args.record:
        Path(args.record).write_text(
            json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(_render_summary(record, base_url=base_url, as_of=as_of))

    for case in (record.positive_control, *record.cases):
        print(
            f"{case.name:<17} expected={case.expected_status} "
            f"observed={case.observed_status} code={case.observed_code} "
            f"pass={case.ok}"
        )
    if record.failures:
        print(f"::error::C11 RED -- {record.detail}", file=sys.stderr)
    else:
        print(f"C11 GREEN -- {record.detail}")
    return record.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
