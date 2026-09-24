# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16987 -- the provider-rung liveness canary, graded.

WHAT THIS IS
    Every declared cloud backend of the dev lane's deployed delegation contract
    that carries an endpoint and a key reference gets ONE authenticated minimal
    chat completion, sent from inside the runtime with the runtime's own
    contract, resolver and transport, and a typed per-rung verdict. The job's
    EXIT CODE is the canary's verdict, the shape the C11, C12 and C15 producers
    already use:

    ``0``  every probed rung is LIVE and every wrong-key control was refused
    ``1``  at least one rung is not LIVE, or a control did not bite; the
           record names each one and why
    ``2``  the probe could not run at all (no container, docker refused, the
           observer could not load the contract). Deliberately distinct from
           ``1``: "I could not look" is not "a rung is dead".

    Resolvability is not liveness. A key NAME that resolves to some value says
    nothing about whether that value authenticates and has quota, and until
    this canary nothing in the estate asked the second question on a schedule.

HOW IT READS THE SUBJECT
    ``provider_rung_canary_observe.py`` is piped to
    ``docker exec -i -u <user> <container> python - <contract>`` on the lane's
    host. It reports raw facts per request and grades nothing. The verdicts are
    decided HERE, from those facts, by rules declared in this file and tested
    offline over recorded observations.

THE VERDICTS
    LIVE            2xx, a chat-completion body (non-empty ``choices``) and a
                    model id echoed back.
    AUTH_DEAD       401 or 403, or the Gemini surface's own invalid-key shape
                    (400 ``INVALID_ARGUMENT`` naming the API key, observed live
                    on 2026-09-24 against a deliberately wrong key).
    QUOTA_DEAD      429. Sub-classified by the deployed contract's own
                    ``provider_quota_policy``: the rule whose host and code
                    match names the disposition (z.ai ``1310`` reset-window vs
                    ``1113`` unfunded), so the two read differently.
    UNREACHABLE     connect, DNS, TLS or timeout; the exception class is kept.
    UNRESOLVED      the key reference is declared and the runtime's resolver
                    returned nothing. No request is sent. Never LIVE, never a
                    pass.
    PROTOCOL_ERROR  2xx that is not a chat completion, or a body that is not
                    JSON.
    HTTP_ERROR      any other status (a 404 path, a 400 model name, a 5xx).
    PROBE_ERROR     the transport raised something outside those families.
    SKIPPED_NO_ENDPOINT / SKIPPED_NO_SECRET_REF
                    not probed, not claimed live, recorded as themselves.

WHY THERE IS A WRONG-KEY CONTROL IN EVERY RUN
    A probe that reads 200 whatever credential it sends is not measuring the
    credential. Each distinct endpoint receives one request with a deliberately
    invalid key in the same run, and it must grade AUTH_DEAD. A control that
    reads LIVE fails the run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

OBSERVER: Final[Path] = (
    Path(__file__).resolve().parent / "provider_rung_canary_observe.py"
)

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2

LIVE: Final[str] = "LIVE"
AUTH_DEAD: Final[str] = "AUTH_DEAD"
QUOTA_DEAD: Final[str] = "QUOTA_DEAD"
UNREACHABLE: Final[str] = "UNREACHABLE"
UNRESOLVED: Final[str] = "UNRESOLVED"
PROTOCOL_ERROR: Final[str] = "PROTOCOL_ERROR"
HTTP_ERROR: Final[str] = "HTTP_ERROR"
PROBE_ERROR: Final[str] = "PROBE_ERROR"
SKIPPED_NO_ENDPOINT: Final[str] = "SKIPPED_NO_ENDPOINT"
SKIPPED_NO_SECRET_REF: Final[str] = "SKIPPED_NO_SECRET_REF"
PROBE: Final[str] = "PROBE"

AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
# The Gemini OpenAI-compatible surface refuses an invalid key with 400
# INVALID_ARGUMENT "Please pass a valid API key" (observed 2026-09-24), not 401.
INVALID_KEY_STATUS: Final[str] = "INVALID_ARGUMENT"
INVALID_KEY_MARKER: Final[str] = "api key"


class ProbeInputError(RuntimeError):
    """The probe could not run. Exit 2, never exit 1."""


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "evidence": self.evidence}


@dataclass
class Record:
    checks: list[Check] = field(default_factory=list)
    rungs: list[dict[str, Any]] = field(default_factory=list)
    controls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [f"{c.name}: {c.evidence}" for c in self.checks if not c.ok]

    @property
    def verdict(self) -> str:
        return "pass" if self.checks and not self.failures else "fail"

    @property
    def exit_code(self) -> int:
        return EXIT_OK if self.verdict == "pass" else EXIT_FINDINGS

    @property
    def detail(self) -> str:
        probed = [r for r in self.rungs if r["disposition"] == PROBE]
        live = [r for r in probed if r["verdict"] == LIVE]
        return (
            f"{len(live)} of {len(probed)} probed backends LIVE; "
            f"{len(self.failures)} of {len(self.checks)} checks failed"
        )

    def to_dict(
        self, *, target: dict[str, Any], as_of: str, observed: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "ticket": "OMN-16987",
            "as_of": as_of,
            "target": target,
            "verdict": self.verdict,
            "detail": self.detail,
            "rungs": self.rungs,
            "controls": self.controls,
            "checks": [c.to_dict() for c in self.checks],
            "failures": self.failures,
            "observed": observed,
        }


def expected_disposition(row: dict[str, Any]) -> str:
    """Re-derived here, not read from the observer: a probe that silently skips
    a credentialed backend must not be able to grade its own coverage."""
    endpoint = row.get("endpoint_url")
    if not isinstance(endpoint, str) or not endpoint.strip():
        return SKIPPED_NO_ENDPOINT
    ref = row.get("secret_ref")
    if not isinstance(ref, str) or not ref.strip():
        return SKIPPED_NO_SECRET_REF
    return PROBE


def _host_matches(host: Any, rule_host: Any) -> bool:
    """The runtime's own matching: case-insensitive, exact or a subdomain."""
    if not isinstance(host, str) or not isinstance(rule_host, str) or not rule_host:
        return False
    h, r = host.lower(), rule_host.lower()
    return h == r or h.endswith("." + r)


def quota_class(obs: dict[str, Any], host: Any, code: Any) -> str | None:
    for rule in obs.get("quota_policy") or []:
        if not isinstance(rule, dict) or not _host_matches(
            host, rule.get("match_endpoint_host")
        ):
            continue
        for c in rule.get("codes") or []:
            if isinstance(c, dict) and str(c.get("code")) == str(code):
                return str(c.get("disposition"))
    return None


def classify(fact: dict[str, Any]) -> str:
    """One observed request, one verdict. Pure; the whole grading rule."""
    if fact.get("secret_resolved") is False:
        return UNRESOLVED
    family = fact.get("exception_family")
    if family == "transport":
        return UNREACHABLE
    if family == "decode":
        return PROTOCOL_ERROR
    if fact.get("exception") is not None or family is not None:
        return PROBE_ERROR
    status = fact.get("http_status")
    if not isinstance(status, int):
        return PROBE_ERROR
    if 200 <= status < 300:
        if fact.get("body_is_chat_completion") is True and fact.get("model_echo"):
            return LIVE
        return PROTOCOL_ERROR
    if status in AUTH_STATUSES:
        return AUTH_DEAD
    if (
        status == 400
        and fact.get("provider_error_status") == INVALID_KEY_STATUS
        and INVALID_KEY_MARKER in str(fact.get("provider_error_message", "")).lower()
    ):
        return AUTH_DEAD
    if status == 429:
        return QUOTA_DEAD
    return HTTP_ERROR


def grade(obs: dict[str, Any]) -> Record:
    record = Record()
    rows = obs.get("backends")
    probes = obs.get("probes")
    controls = obs.get("controls")
    if not isinstance(rows, list) or not isinstance(probes, list):
        record.checks.append(
            Check("observation_shape", False, "no backends or probes list")
        )
        return record
    if not isinstance(controls, list):
        controls = []

    record.checks.append(
        Check(
            "contract_is_runtime_binding",
            obs.get("runtime_binds_contract") is True,
            f"read {obs.get('contract_path')!r}; the runtime binds it: "
            f"{obs.get('runtime_binds_contract')!r}",
        )
    )

    by_backend: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for probe in probes:
        for backend_id in probe.get("backend_ids") or []:
            if backend_id in by_backend:
                duplicates.append(str(backend_id))
            by_backend[str(backend_id)] = probe

    expected = [
        str(r.get("backend_id")) for r in rows if expected_disposition(r) == PROBE
    ]
    silently_skipped = [b for b in expected if b not in by_backend]
    unexpected = sorted(set(by_backend) - set(expected))
    record.checks.append(
        Check(
            "every_credentialed_backend_probed",
            bool(expected)
            and not silently_skipped
            and not unexpected
            and not duplicates,
            f"expected {len(expected)} {expected}; not probed {silently_skipped}; "
            f"probed but not credentialed {unexpected}; probed twice {duplicates}",
        )
    )

    for row in rows:
        backend_id = str(row.get("backend_id"))
        disposition = expected_disposition(row)
        entry: dict[str, Any] = {
            "backend_id": backend_id,
            "provider": row.get("provider"),
            "tier": row.get("tier"),
            "model_name": row.get("model_name"),
            "secret_ref": row.get("secret_ref"),
            "disposition": disposition,
        }
        if disposition != PROBE:
            entry["verdict"] = disposition
            record.rungs.append(entry)
            continue
        fact = by_backend.get(backend_id)
        if fact is None:
            entry["verdict"] = PROBE_ERROR
            record.rungs.append(entry)
            continue
        verdict = classify(fact)
        entry.update(
            {
                "verdict": verdict,
                "endpoint_host": fact.get("endpoint_host"),
                "http_status": fact.get("http_status"),
                "latency_ms": fact.get("latency_ms"),
                "model_echo": fact.get("model_echo"),
                "provider_error_code": fact.get("provider_error_code"),
                "provider_error_status": fact.get("provider_error_status"),
                "provider_error_message": fact.get("provider_error_message"),
                "exception": fact.get("exception"),
                "resolver_error": fact.get("resolver_error"),
                "secret_resolved": fact.get("secret_resolved"),
                "shared_probe_with": [
                    b for b in fact.get("backend_ids") or [] if b != backend_id
                ],
            }
        )
        if verdict == QUOTA_DEAD:
            entry["quota_class"] = quota_class(
                obs, fact.get("endpoint_host"), fact.get("provider_error_code")
            )
        record.rungs.append(entry)
        record.checks.append(
            Check(
                f"rung/{backend_id}",
                verdict == LIVE,
                f"{verdict} (http {fact.get('http_status')}, "
                f"code {fact.get('provider_error_code')}, "
                f"exception {fact.get('exception')})",
            )
        )
        if verdict == UNRESOLVED:
            record.checks.append(
                Check(
                    f"no_request_without_key/{backend_id}",
                    fact.get("request_sent") is False,
                    f"request_sent={fact.get('request_sent')!r}",
                )
            )

    probed_endpoints = sorted(
        {str(p.get("endpoint_url")) for p in probes if p.get("endpoint_url")}
    )
    controlled = {str(c.get("endpoint_url")) for c in controls}
    record.checks.append(
        Check(
            "wrong_key_control_per_endpoint",
            bool(probed_endpoints) and set(probed_endpoints) <= controlled,
            f"endpoints {len(probed_endpoints)}; controlled {len(controlled)}; "
            f"missing {sorted(set(probed_endpoints) - controlled)}",
        )
    )
    for ctl in controls:
        verdict = classify(ctl)
        host = ctl.get("endpoint_host")
        record.controls.append(
            {
                "endpoint_host": host,
                "verdict": verdict,
                "http_status": ctl.get("http_status"),
                "provider_error_code": ctl.get("provider_error_code"),
                "provider_error_status": ctl.get("provider_error_status"),
                "exception": ctl.get("exception"),
            }
        )
        record.checks.append(
            Check(
                f"control/{host}",
                verdict == AUTH_DEAD,
                f"a deliberately invalid key read {verdict} "
                f"(http {ctl.get('http_status')}); it must read {AUTH_DEAD}",
            )
        )

    redactions = obs.get("redactions", 0)
    echoes = {
        str(p.get("backend_ids")): p.get("key_echoes")
        for p in probes
        if p.get("key_echoes")
    }
    record.checks.append(
        Check(
            "no_key_material_echoed",
            redactions in (0, None) and not echoes,
            f"{redactions!r} occurrences of a resolved key scrubbed from the "
            f"observation line; per-rung provider echoes {echoes}",
        )
    )
    return record


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
    *,
    docker_bin: str,
    container: str,
    user: str,
    contract_path: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The format names three non-secret fields only. `docker inspect` without a
    # format would return the container's environment, which holds the keys.
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
        "user": user,
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
        ["exec", "-i", "-u", user, container, "python", "-", contract_path],
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
        "config_version": obs.get("config_version"),
        "contract_path": obs.get("contract_path"),
        "overlay_path": obs.get("overlay_path"),
        "runtime_binds_contract": obs.get("runtime_binds_contract"),
        "quota_policy": obs.get("quota_policy"),
    }


def _render_summary(record: Record, *, target: dict[str, Any], as_of: str) -> str:
    lines = [
        "## Provider-rung liveness canary (OMN-16987)",
        "",
        f"**Verdict: `{record.verdict}`** on `{target.get('container')}` "
        f"(`{target.get('image')}`) at `{as_of}`.",
        "",
        record.detail,
        "",
        "| Backend | Verdict | HTTP | Latency ms | Provider code | Quota class | Shares probe with |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in record.rungs:
        lines.append(
            f"| {r['backend_id']} | {r['verdict']} | {r.get('http_status') or ''} | "
            f"{r.get('latency_ms') or ''} | {r.get('provider_error_code') or ''} | "
            f"{r.get('quota_class') or ''} | "
            f"{', '.join(r.get('shared_probe_with') or [])} |"
        )
    lines += [
        "",
        "Wrong-key controls (each must read AUTH_DEAD):",
        "",
        "| Endpoint host | Verdict | HTTP |",
        "| --- | --- | --- |",
    ]
    for c in record.controls:
        lines.append(
            f"| {c.get('endpoint_host')} | {c['verdict']} | {c.get('http_status')} |"
        )
    if record.failures:
        lines += ["", "Failures:", ""]
        lines += [f"- `{f}`" for f in record.failures]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--container", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--contract-path",
        required=True,
        help="The merged delegation contract the runtime binds as BIFROST_CONTRACT_PATH.",
    )
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--timeout", type=float, default=300.0)
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
                contract_path=args.contract_path,
                timeout=args.timeout,
            )
    except (ProbeInputError, OSError) as exc:
        print(f"::error::provider-rung canary could not run: {exc}", file=sys.stderr)
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
    for r in record.rungs:
        print(f"{r['backend_id']:<32} {r['verdict']:<22} http={r.get('http_status')}")
    for c in record.controls:
        print(f"control {c.get('endpoint_host')!s:<40} {c['verdict']}")
    if record.failures:
        for f in record.failures:
            print(f"::error::{f}", file=sys.stderr)
        print(f"::error::provider-rung canary RED -- {record.detail}", file=sys.stderr)
    else:
        print(f"provider-rung canary GREEN -- {record.detail}")
    return record.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
