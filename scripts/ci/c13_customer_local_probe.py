# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19179 -- the C13 producer: customer-local delegation on a no-checkout machine.

WHAT THIS IS
    The producer for beta criterion C13: "Customer-local delegation on a
    no-checkout machine: three files, local model named, zero provider calls".
    It drives one real ``onex delegate`` run as a customer would on a machine
    that holds no OmniNode repository, observes it from outside the process,
    and grades four clauses against expectations declared HERE. Its EXIT CODE
    is the criterion's verdict -- the C11 and C15 shape -- and it also writes a
    JSON record so a reader can see each clause without reading a log.

    ``0``  all four clauses proven, and every control behaved
    ``1``  at least one clause unproven or one control misbehaved; the record
           names which and why
    ``2``  the probe could not run at all (no ``onex`` on the customer PATH, no
           model server, no strace, unreadable replay). Deliberately distinct
           from ``1``: "I could not run" is not "the product failed".

THE FOUR CLAUSES
    no_checkout      every OmniNode distribution the run imported came from the
                     package index (the CLI's own runtime identity says
                     ``source=registry`` with no commit and no import path, and
                     no ``direct_url.json`` exists, which pip writes for any
                     VCS, local-path or editable install), the working
                     directory is not inside a git work tree, and no source
                     tree of any of the five distributions exists under the
                     scanned roots.
    three_files      ``result.txt``, ``receipt.json`` and ``run.json`` exist
                     under the run's own directory, parse, agree with each
                     other and with stdout on the correlation and run ids, and
                     ``result.txt`` is the accepted response.
    local_model      the receipt's routing tier is ``local``, its model is the
                     id the local server itself reports serving, its endpoint
                     is that server's declared address and port, and every
                     attempt on the ladder was a local one.
    zero_provider    the whole customer session made no connect() to any
                     address other than loopback and the declared model
                     server, and performed no name lookup.

THE MODEL SERVER THE CUSTOMER DECLARES (OMN-19805)
    A customer's own model may run on the machine itself or on another machine
    in the customer's own private network (a GPU box beside the laptop). The
    probe takes that server's address as ``--model-host``: an IP literal, so the
    run needs no name lookup, and loopback or a private address only. A public
    address is refused before anything runs, because a model reached over the
    internet is a provider. Only a connect to exactly that host and port counts
    as the model; every other non-loopback connect is external. The default is
    loopback, which is the original shape of this criterion.

AN UNPROVEN ZERO IS A FAILURE, NOT A PASS
    "No provider call was recorded" means nothing unless the instrument that
    would have recorded one was running and could see this process tree. Two
    positive controls make that falsifiable in the same invocation:

    1. the configured run MUST show a connect() to the model server's own
       declared address and port, AND the server's own token counter must have moved.
       Zero model connects means strace was blind to the process that did the
       work, so the zero it reports for provider calls is unproven.
    2. a deliberate outbound connect, traced by the SAME wrapper and classified
       by the SAME parser, MUST be classified external. If it is not, the
       parser cannot see the thing it is grading for.

    And one negative control: the same prompt, before the customer
    configuration exists, must be a refusal that reaches no model at all. That
    is what shows the configured run's model came from the configuration this
    probe wrote and not from some ambient default.

WHY STRACE AND NOT A FIREWALL
    A firewall that blocks outbound traffic proves the traffic did not leave;
    it does not prove it was not attempted, and an attempted provider call that
    fails reads, in the receipt, like an ordinary rung refusal. connect() is
    where an attempt becomes visible whether or not it succeeds.

WHY A REPLAY MODE EXISTS
    ``grade --observations`` grades a recorded observation file instead of
    running anything. That makes every failure branch falsifiable offline in
    the ordinary test suite -- including the ones a live run cannot produce on
    demand (an external connect, a blind tracer, a checkout-installed package).

STANDARD LIBRARY ONLY
    Same reason as the C11 producer: a probe whose job is to be readable should
    not acquire a dependency resolution step that can fail for reasons unrelated
    to the thing it probes.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2

#: The five distributions a delegation imports. The no-checkout clause requires
#: every one of them to have come from the package index.
ONEX_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "omnibase_compat",
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omnimarket",
)

#: Project names whose source tree, if found on the machine, means it is not a
#: no-checkout machine. Both spellings, because a pyproject may use either.
SOURCE_TREE_PROJECT_NAMES: Final[frozenset[str]] = frozenset(
    name for dist in ONEX_DISTRIBUTIONS for name in (dist, dist.replace("_", "-"))
)

#: The three files OMN-16999 promises a local delegation writes.
RUN_FILES: Final[tuple[str, ...]] = ("result.txt", "receipt.json", "run.json")

#: The one documented customer configuration file (OMN-16200, omnimarket
#: 0.4.203): the machine-local overlay, relative to the customer's HOME. A clean
#: install that has not written it is refused before any model call, naming it.
OVERLAY_RELATIVE_PATH: Final[str] = ".omninode/delegation/bifrost_overrides.yaml"

#: The deliberate outbound connect for positive control 2. A literal address so
#: the control itself needs no name lookup.
OUTBOUND_CONTROL_HOST: Final[str] = "1.1.1.1"
OUTBOUND_CONTROL_PORT: Final[int] = 443

#: The model host when the customer declares none: their own machine.
LOOPBACK_MODEL_HOST: Final[str] = "127.0.0.1"

#: The shipped local backends the customer overlay points at their own model.
#: The shipped routing sends code classes to ``local-coder`` and prose classes
#: (``document`` among them) to ``local-heavy-reasoning``, so both are declared.
LOCAL_BACKEND_IDS: Final[tuple[str, ...]] = ("local-coder", "local-heavy-reasoning")

_INET_RE: Final = re.compile(
    r"sa_family=AF_INET, sin_port=htons\((?P<port>\d+)\), "
    r'sin_addr=inet_addr\("(?P<host>[^"]+)"\)'
)
_INET6_RE: Final = re.compile(
    r"sa_family=AF_INET6, sin6_port=htons\((?P<port>\d+)\).*?"
    r'inet_pton\(AF_INET6, "(?P<host>[^"]+)"'
)
_UNIX_RE: Final = re.compile(r'sa_family=AF_UNIX, sun_path=@?"(?P<path>[^"]*)"')
_FAMILY_RE: Final = re.compile(r"sa_family=(?P<family>AF_[A-Z0-9]+)")


class ProbeInputError(RuntimeError):
    """The probe could not run; exit 2, never a verdict."""


# --------------------------------------------------------------------------
# connect() classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Connect:
    """One connect() syscall parsed from a strace log."""

    family: str
    host: str | None
    port: int | None
    path: str | None
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "kind": self.kind,
        }


def _is_loopback(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped
    return address


def validate_model_host(host: str) -> str:
    """The declared model host, or ProbeInputError when it cannot be one.

    An IP literal only (a hostname would need the name lookup this probe
    grades as a failure), and loopback or private only: a public address is a
    provider, not the customer's own model.
    """
    address = _ip(host)
    if address is None:
        raise ProbeInputError(
            f"model host {host!r} is not an IP literal; a customer-local model is "
            "declared by address so the run needs no name lookup"
        )
    if not (address.is_loopback or (address.is_private and not address.is_global)):
        raise ProbeInputError(
            f"model host {host!r} is neither loopback nor a private address; a model "
            "reached over the public internet is a provider, not a customer-local model"
        )
    return str(address)


def _is_model_host(host: str, model_host: str) -> bool:
    """Whether ``host`` is the declared model host. Any loopback matches loopback."""
    address, declared = _ip(host), _ip(model_host)
    if address is None or declared is None:
        return False
    if declared.is_loopback:
        return address.is_loopback
    return address == declared


def classify(
    family: str,
    host: str | None,
    port: int | None,
    path: str | None,
    *,
    model_port: int,
    model_host: str = LOOPBACK_MODEL_HOST,
) -> str:
    """Name what a connect() was for. The grader's whole vocabulary."""
    if family in ("AF_INET", "AF_INET6") and host is not None:
        if port == model_port and _is_model_host(host, model_host):
            return "model"
        if not _is_loopback(host):
            return "external"
        if port == 53:
            return "name_lookup"
        return "loopback_other"
    if family == "AF_UNIX":
        # glibc asks nscd before it asks DNS; a connect to its socket is a
        # name lookup whether or not nscd is running.
        if path is not None and "nscd" in path:
            return "name_lookup"
        return "unix_local"
    return "other_family"


def parse_connects(
    strace_text: str, *, model_port: int, model_host: str = LOOPBACK_MODEL_HOST
) -> list[Connect]:
    """Every connect() in a ``strace -f -e trace=connect`` log, classified.

    ``<... connect resumed>`` continuation lines carry no address and are
    skipped: the address is always on the line that opened the call.
    """
    connects: list[Connect] = []
    for line in strace_text.splitlines():
        if "connect(" not in line or "resumed>" in line:
            continue
        family_match = _FAMILY_RE.search(line)
        if family_match is None:
            continue
        family = family_match.group("family")
        host: str | None = None
        port: int | None = None
        path: str | None = None
        if family == "AF_INET":
            match = _INET_RE.search(line)
            if match:
                host, port = match.group("host"), int(match.group("port"))
        elif family == "AF_INET6":
            match = _INET6_RE.search(line)
            if match:
                host, port = match.group("host"), int(match.group("port"))
        elif family == "AF_UNIX":
            match = _UNIX_RE.search(line)
            if match:
                path = match.group("path")
        if family in ("AF_INET", "AF_INET6") and host is None:
            # An inet connect whose address this parser cannot read is graded
            # as external: an unreadable address is not a proven-local one.
            connects.append(Connect(family, None, None, None, "external"))
            continue
        connects.append(
            Connect(
                family,
                host,
                port,
                path,
                classify(
                    family,
                    host,
                    port,
                    path,
                    model_port=model_port,
                    model_host=model_host,
                ),
            )
        )
    return connects


def count_kinds(connects: Sequence[Connect]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for connect in connects:
        counts[connect.kind] = counts.get(connect.kind, 0) + 1
    return counts


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------


@dataclass
class Clause:
    name: str
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }


def _load_json(text: str | None) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def declared_model_host(obs: Mapping[str, Any]) -> str:
    """The model host a recorded session declared. A record without one used loopback."""
    return str(obs.get("model_host") or LOOPBACK_MODEL_HOST)


def _session_connects(
    obs: Mapping[str, Any], step: str, model_port: int
) -> list[Connect]:
    return parse_connects(
        obs.get("steps", {}).get(step, {}).get("strace", "") or "",
        model_port=model_port,
        model_host=declared_model_host(obs),
    )


def grade_no_checkout(obs: Mapping[str, Any], receipt: Any) -> Clause:
    clause = Clause("no_checkout")
    identity = (
        (receipt or {}).get("receipt", {}).get("runtime_identity")
        if isinstance(receipt, dict)
        else None
    )
    if not isinstance(identity, dict):
        clause.reasons.append(
            "receipt carries no runtime_identity, so where the code came from is unproven"
        )
        return clause
    packages = identity.get("packages") or {}
    # Recorded, not graded: locus_kind says whether the interpreter is a venv,
    # a container or the system Python, which is orthogonal to whether the code
    # it imports came from a checkout. The per-package source fields below are
    # the claim that matters.
    clause.evidence["locus_kind"] = identity.get("locus_kind")
    per_package: dict[str, Any] = {}
    for dist in ONEX_DISTRIBUTIONS:
        entry = packages.get(dist)
        if not isinstance(entry, dict):
            clause.reasons.append(f"{dist} absent from the runtime identity")
            continue
        per_package[dist] = {
            "version": entry.get("version"),
            "source": entry.get("source"),
            "commit": entry.get("commit"),
            "import_path": entry.get("import_path"),
        }
        if entry.get("source") != "registry":
            clause.reasons.append(
                f"{dist} source is {entry.get('source')!r}, not the package index"
            )
        if entry.get("commit") or entry.get("import_path"):
            clause.reasons.append(
                f"{dist} names a commit or import path, which only a checkout has"
            )
    clause.evidence["packages"] = per_package

    direct_urls = obs.get("direct_url") or {}
    clause.evidence["direct_url"] = direct_urls
    for dist in ONEX_DISTRIBUTIONS:
        if dist not in direct_urls:
            clause.reasons.append(f"no install-origin readback for {dist}")
        elif direct_urls[dist] is not None:
            clause.reasons.append(
                f"{dist} was installed from {direct_urls[dist]!r}, not the index"
            )

    source_trees = obs.get("source_trees")
    if source_trees is None:
        clause.reasons.append(
            "no source-tree scan was recorded, so the machine is unproven clean"
        )
    elif source_trees:
        clause.reasons.append(f"OmniNode source tree(s) on the machine: {source_trees}")
    clause.evidence["source_trees"] = source_trees
    clause.evidence["scan_roots"] = obs.get("scan_roots")

    git_ancestor = obs.get("workdir_git_ancestor", "unrecorded")
    if git_ancestor == "unrecorded":
        clause.reasons.append(
            "whether the working directory sits in a git work tree was not recorded"
        )
    elif git_ancestor:
        clause.reasons.append(
            f"working directory is inside a git work tree at {git_ancestor}"
        )
    clause.evidence["customer_env_keys"] = obs.get("customer_env_keys")
    return clause


def grade_three_files(obs: Mapping[str, Any], stdout_doc: Any, receipt: Any) -> Clause:
    clause = Clause("three_files")
    files = obs.get("run_files") or {}
    clause.evidence["run_dir"] = obs.get("run_dir")
    clause.evidence["run_dir_listing"] = obs.get("run_dir_listing")
    for name in RUN_FILES:
        content = files.get(name)
        if content is None:
            clause.reasons.append(f"{name} missing")
        elif not content.strip():
            clause.reasons.append(f"{name} is empty")
    run_doc = _load_json(files.get("run.json"))
    if files.get("receipt.json") is not None and not isinstance(receipt, dict):
        clause.reasons.append("receipt.json does not parse as a JSON object")
    if files.get("run.json") is not None and not isinstance(run_doc, dict):
        clause.reasons.append("run.json does not parse as a JSON object")
    if not isinstance(stdout_doc, dict):
        clause.reasons.append("stdout was not exactly one JSON result")
    if clause.reasons:
        return clause

    assert (
        isinstance(receipt, dict)
        and isinstance(run_doc, dict)
        and isinstance(stdout_doc, dict)
    )
    correlation = {
        "stdout": stdout_doc.get("correlation_id"),
        "receipt": receipt.get("correlation_id"),
        "run": run_doc.get("correlation_id"),
    }
    run_ids = {
        "stdout": stdout_doc.get("run_id"),
        "receipt": receipt.get("run_id"),
        "run": run_doc.get("run_id"),
    }
    clause.evidence["correlation_id"] = correlation
    clause.evidence["run_id"] = run_ids
    if len(set(correlation.values())) != 1 or None in correlation.values():
        clause.reasons.append(
            f"correlation ids disagree across stdout and the files: {correlation}"
        )
    if len(set(run_ids.values())) != 1 or None in run_ids.values():
        clause.reasons.append(
            f"run ids disagree across stdout and the files: {run_ids}"
        )
    response = receipt.get("receipt", {}).get("result", {}).get("response")
    result_text = files["result.txt"]
    if not isinstance(response, str) or not response.strip():
        clause.reasons.append("receipt carries no accepted response")
    elif result_text.strip() != response.strip():
        clause.reasons.append("result.txt is not the response the receipt accepted")
    clause.evidence["result_chars"] = len(result_text)
    if obs.get("steps", {}).get("configured", {}).get("returncode") != 0:
        clause.reasons.append(
            f"onex delegate exited {obs.get('steps', {}).get('configured', {}).get('returncode')}, not 0"
        )
    if receipt.get("status") != "success":
        clause.reasons.append(
            f"receipt status is {receipt.get('status')!r}, not 'success'"
        )
    return clause


def grade_local_model(obs: Mapping[str, Any], receipt: Any) -> Clause:
    clause = Clause("local_model")
    served = obs.get("served_models")
    model_port = obs.get("model_port")
    model_host = declared_model_host(obs)
    clause.evidence["served_models"] = served
    clause.evidence["model_port"] = model_port
    clause.evidence["model_host"] = model_host
    if not served:
        clause.reasons.append("the local server's own /v1/models readback is absent")
        return clause
    if not isinstance(receipt, dict):
        clause.reasons.append("no receipt to read the answering model from")
        return clause
    model = receipt.get("model")
    endpoint = receipt.get("endpoint")
    clause.evidence.update(
        {
            "receipt_model": model,
            "receipt_endpoint": endpoint,
            "receipt_backend_id": receipt.get("backend_id"),
            "receipt_routing_tier": receipt.get("routing_tier"),
        }
    )
    if receipt.get("routing_tier") != "local":
        clause.reasons.append(
            f"routing tier is {receipt.get('routing_tier')!r}, not 'local'"
        )
    if not model:
        clause.reasons.append("receipt names no model")
    elif model not in served:
        clause.reasons.append(
            f"receipt model {model!r} is not an id the local server serves ({served})"
        )
    parsed = urllib.parse.urlparse(endpoint or "")
    if not parsed.hostname or not _is_model_host(parsed.hostname, model_host):
        clause.reasons.append(
            f"receipt endpoint {endpoint!r} is not the declared model host {model_host}"
        )
    elif parsed.port != model_port:
        clause.reasons.append(
            f"receipt endpoint port {parsed.port} is not the model server's {model_port}"
        )
    attempts = receipt.get("receipt", {}).get("result", {}).get("attempts") or []
    clause.evidence["attempts"] = [
        {k: a.get(k) for k in ("tier", "backend_id", "model_id", "acceptance_decision")}
        for a in attempts
    ]
    if not attempts:
        clause.reasons.append("receipt records no routing attempt")
    non_local = [a for a in attempts if a.get("tier") != "local"]
    if non_local:
        clause.reasons.append(f"{len(non_local)} attempt(s) left the local tier")
    accepted = [a for a in attempts if a.get("acceptance_decision") == "accept"]
    if len(accepted) != 1:
        clause.reasons.append(
            f"{len(accepted)} accepted attempts, expected exactly one"
        )
    elif accepted[0].get("model_id") != model:
        clause.reasons.append("the accepted attempt's model is not the receipt's model")
    return clause


def grade_zero_provider(obs: Mapping[str, Any]) -> Clause:
    clause = Clause("zero_provider")
    model_port = int(obs.get("model_port") or 0)
    model_host = declared_model_host(obs)
    clause.evidence["model_host"] = model_host
    steps = obs.get("steps") or {}
    session_steps = [s for s in ("init", "unconfigured", "configured") if s in steps]
    clause.evidence["traced_steps"] = session_steps
    for required in ("init", "unconfigured", "configured"):
        if required not in steps:
            clause.reasons.append(f"customer step {required!r} was not traced")
    per_step: dict[str, Any] = {}
    for step in session_steps:
        connects = _session_connects(obs, step, model_port)
        counts = count_kinds(connects)
        per_step[step] = counts
        external = [c.as_dict() for c in connects if c.kind == "external"]
        lookups = [c.as_dict() for c in connects if c.kind == "name_lookup"]
        if external:
            clause.reasons.append(
                f"{step}: {len(external)} connect(s) to an address that is neither loopback "
                f"nor the declared model server: {external[:5]}"
            )
        if lookups:
            clause.reasons.append(
                f"{step}: {len(lookups)} name lookup(s), which a loopback-only run never needs: {lookups[:5]}"
            )
    clause.evidence["connects_by_step"] = per_step

    # Positive control 1: the instrument saw the process that did the work.
    configured = per_step.get("configured", {})
    if configured.get("model", 0) < 1:
        clause.reasons.append(
            "positive control failed: the configured run shows no connect to the model server, so the "
            "tracer did not see the process that answered and its zero is unproven"
        )
    metrics = obs.get("model_metrics") or {}
    clause.evidence["model_metrics"] = metrics
    delta = metrics.get("configured_tokens_predicted_delta")
    if not isinstance(delta, int) or delta < 1:
        clause.reasons.append(
            f"positive control failed: the model server's own token counter moved by {delta!r} over the configured run"
        )

    # Positive control 2: the parser recognises an outbound connect.
    control = steps.get("outbound_control")
    if control is None:
        clause.reasons.append(
            "outbound control was not run, so the parser's ability to see a provider call is unproven"
        )
    else:
        control_counts = count_kinds(
            parse_connects(
                control.get("strace", "") or "",
                model_port=model_port,
                model_host=model_host,
            )
        )
        clause.evidence["outbound_control"] = control_counts
        if control_counts.get("external", 0) < 1:
            clause.reasons.append(
                "positive control failed: a deliberate outbound connect was not classified external"
            )

    # Negative control: before configuration exists, nothing answers.
    unconfigured = steps.get("unconfigured")
    if unconfigured is not None:
        clause.evidence["unconfigured_returncode"] = unconfigured.get("returncode")
        clause.evidence["unconfigured_names_overlay_file"] = (
            "bifrost_overrides.yaml"
            in ((unconfigured.get("stdout") or "") + (unconfigured.get("stderr") or ""))
        )
        if unconfigured.get("returncode") == 0:
            clause.reasons.append(
                "negative control failed: an unconfigured run succeeded, so the configured model is not proven to come from the configuration"
            )
        if per_step.get("unconfigured", {}).get("model", 0):
            clause.reasons.append(
                "negative control failed: an unconfigured run reached the model server"
            )
        # The server's token counter over the unconfigured run is RECORDED, not
        # graded. On the lab customer machine the model server is shared
        # with other sessions, so a non-zero delta there can be someone else's
        # request. The traced connect count above is exclusive to this process
        # tree and is the negative control's proof.
    return clause


@dataclass
class Record:
    verdict: str
    clauses: list[Clause]
    as_of: str
    context: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_version": RECORD_VERSION,
            "criterion": "C13",
            "ticket": "OMN-19179",
            "verdict": self.verdict,
            "as_of": self.as_of,
            "clauses": {c.name: c.as_dict() for c in self.clauses},
            "context": self.context,
        }


def grade(obs: Mapping[str, Any], *, as_of: str | None = None) -> Record:
    configured = (obs.get("steps") or {}).get("configured") or {}
    stdout_doc = _load_json(configured.get("stdout"))
    receipt = _load_json((obs.get("run_files") or {}).get("receipt.json"))
    clauses = [
        grade_no_checkout(obs, receipt),
        grade_three_files(obs, stdout_doc, receipt),
        grade_local_model(obs, receipt),
        grade_zero_provider(obs),
    ]
    verdict = "PASS" if all(c.passed for c in clauses) else "FAIL"
    context = {
        "installed": obs.get("installed"),
        "model_artifact": obs.get("model_artifact"),
        "config_surface": obs.get("config_surface"),
        "customer_config": obs.get("customer_config"),
        "prompt": obs.get("prompt"),
    }
    return Record(
        verdict=verdict,
        clauses=clauses,
        as_of=as_of
        or obs.get("as_of")
        or datetime.datetime.now(datetime.UTC).isoformat(),
        context=context,
    )


# --------------------------------------------------------------------------
# live observation
# --------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - validated loopback or private model URL from argv
        body: bytes = response.read()
    return body.decode("utf-8", "replace")


def served_model_ids(base_url: str) -> list[str]:
    payload = json.loads(_http_get(f"{base_url}/v1/models"))
    return sorted({str(item["id"]) for item in payload.get("data", []) if "id" in item})


#: The server's own generated-token counter, per server build. llama-server
#: exposes the first with ``--metrics``; vLLM exposes the second, once per
#: engine and model label set, so every sample is summed.
TOKEN_COUNTERS: Final[tuple[str, ...]] = (
    "llamacpp:tokens_predicted_total",
    "vllm:generation_tokens_total",
)


def tokens_predicted_from_metrics(text: str) -> int:
    """Sum the first token counter present in a Prometheus text exposition."""
    for counter in TOKEN_COUNTERS:
        samples = [
            line
            for line in text.splitlines()
            if line.startswith(counter)
            and line[len(counter) : len(counter) + 1] in ("{", " ")
        ]
        if samples:
            return sum(int(float(line.split()[-1])) for line in samples)
    raise ProbeInputError(
        f"model server exposes none of {list(TOKEN_COUNTERS)}; a llama-server "
        "needs --metrics"
    )


def tokens_predicted(base_url: str) -> int:
    """The model server's own generated-token counter."""
    return tokens_predicted_from_metrics(_http_get(f"{base_url}/metrics"))


def model_base_url(model_host: str, model_port: int) -> str:
    """``http://host:port`` for a validated IP literal, bracketing IPv6."""
    host = f"[{model_host}]" if ":" in model_host else model_host
    return f"http://{host}:{model_port}"


def bifrost_overlay_yaml(
    served_model: str,
    model_port: int,
    max_tokens: int,
    model_host: str = LOOPBACK_MODEL_HOST,
) -> str:
    """The customer's overlay: the shipped local backends, pointed at their model."""
    endpoint = f"{model_base_url(model_host, model_port)}/v1/chat/completions"
    lines = [
        'config_version: "2.1.0"',
        'schema_version: "bifrost_delegation.v1"',
        "backends:",
    ]
    for backend_id in LOCAL_BACKEND_IDS:
        lines += [
            f"  - backend_id: {backend_id}",
            # A customer has no contract resolver: their own model is what C13 names.
            f'    endpoint_url: "{endpoint}"',
            f'    model_name: "{served_model}"',
            "    tier: local",
            "    timeout_ms: 240000",
            f"    max_tokens: {max_tokens}",
        ]
    return "\n".join(lines) + "\n"


_DIRECT_URL_SNIPPET: Final[str] = (
    "import importlib.metadata as m, json\n"
    "out = {}\n"
    "for d in %r:\n"
    "    try:\n"
    "        out[d] = m.distribution(d).read_text('direct_url.json')\n"
    "    except m.PackageNotFoundError:\n"
    "        out[d] = 'NOT INSTALLED'\n"
    "print(json.dumps(out))\n"
)


def find_source_trees(roots: Sequence[Path], max_depth: int = 5) -> list[str]:
    """Directories holding a pyproject.toml that declares an OmniNode distribution."""
    skip = {
        "site-packages",
        "node_modules",
        ".cache",
        "hostedtoolcache",
        ".git",
        "proc",
        "sys",
    }
    found: list[str] = []
    name_re = re.compile(r'^\s*name\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
    for root in roots:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for current, dirs, files in os.walk(root, onerror=lambda _e: None):
            depth = len(Path(current).parts) - base_depth
            dirs[:] = [d for d in dirs if d not in skip and depth < max_depth]
            if "pyproject.toml" in files:
                try:
                    text = (Path(current) / "pyproject.toml").read_text(
                        errors="replace"
                    )
                except OSError:
                    continue
                match = name_re.search(text)
                if match and match.group(1) in SOURCE_TREE_PROJECT_NAMES:
                    found.append(current)
    return sorted(set(found))


def git_ancestor(path: Path) -> str | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return None


def _traced(strace: str, trace_file: Path, argv: Sequence[str]) -> list[str]:
    return [
        strace,
        "-f",
        "-qq",
        "-e",
        "trace=connect",
        "-o",
        str(trace_file),
        "--",
        *argv,
    ]


def _run_step(
    name: str,
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    strace: str,
    trace_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    trace_file = trace_dir / f"{name}.strace"
    completed = subprocess.run(
        _traced(strace, trace_file, argv),
        env=dict(env),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr[-20000:],
        "strace": trace_file.read_text(errors="replace") if trace_file.exists() else "",
    }


def observe_live(args: argparse.Namespace) -> dict[str, Any]:
    # First: a public model host is refused before anything else is checked or run.
    model_host = validate_model_host(args.model_host)
    strace = shutil.which("strace")
    if strace is None:
        raise ProbeInputError(
            "strace is not installed; the zero-provider clause has no instrument"
        )
    customer_home = Path(args.customer_home).resolve()
    customer_bin = Path(args.customer_bin).resolve()
    workdir = Path(args.workdir).resolve()
    trace_dir = Path(args.trace_dir).resolve()
    for directory in (workdir, trace_dir):
        directory.mkdir(parents=True, exist_ok=True)
    onex = customer_bin / "onex"
    if not onex.exists():
        raise ProbeInputError(f"no onex at {onex}; the customer install did not happen")
    base_url = model_base_url(model_host, int(args.model_port))
    try:
        served = served_model_ids(base_url)
        tokens_before_unconfigured = tokens_predicted(base_url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ProbeInputError(
            f"local model server at {base_url} is not answering: {exc}"
        ) from exc
    if args.served_model not in served:
        raise ProbeInputError(
            f"model server serves {served}, not the declared {args.served_model!r}"
        )

    # The customer's environment is built from nothing. No variable from the
    # job leaks into it: not OMNI_HOME, not a broker address, not a key.
    base_env = {
        "HOME": str(customer_home),
        "PATH": f"{customer_bin}:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    # The documented surface only (OMN-16200): one file under the customer's
    # HOME, and no environment binding of any kind. The configured run uses the
    # SAME environment as the negative control; the file is the only difference.
    overlay_path = customer_home / OVERLAY_RELATIVE_PATH
    configured_env = dict(base_env)
    timeout = int(args.step_timeout)
    steps: dict[str, Any] = {}
    steps["init"] = _run_step(
        "init",
        [str(onex), "local", "init", "--json"],
        env=base_env,
        cwd=workdir,
        strace=strace,
        trace_dir=trace_dir,
        timeout=timeout,
    )
    steps["unconfigured"] = _run_step(
        "unconfigured",
        [str(onex), "delegate", args.prompt],
        env=base_env,
        cwd=workdir,
        strace=strace,
        trace_dir=trace_dir,
        timeout=timeout,
    )
    tokens_after_unconfigured = tokens_predicted(base_url)

    # Written only now, AFTER the negative control, so the refusal above was
    # taken on a machine with no customer configuration at all.
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(
        bifrost_overlay_yaml(
            args.served_model, int(args.model_port), int(args.max_tokens), model_host
        )
    )
    # The negative control's own run directory is set aside so the configured
    # run's files are the only ones the three_files clause can find.
    runs_root = workdir / ".onex_state" / "runs"
    if runs_root.exists():
        runs_root.rename(workdir / ".onex_state" / "runs.unconfigured")

    tokens_before_configured = tokens_predicted(base_url)
    steps["configured"] = _run_step(
        "configured",
        [str(onex), "delegate", args.prompt],
        env=configured_env,
        cwd=workdir,
        strace=strace,
        trace_dir=trace_dir,
        timeout=timeout,
    )
    tokens_after_configured = tokens_predicted(base_url)

    steps["outbound_control"] = _run_step(
        "outbound_control",
        [
            sys.executable,
            "-c",
            "import socket; s = socket.socket(); s.settimeout(5); "
            f"s.connect_ex(({OUTBOUND_CONTROL_HOST!r}, {OUTBOUND_CONTROL_PORT}))",
        ],
        env=base_env,
        cwd=workdir,
        strace=strace,
        trace_dir=trace_dir,
        timeout=60,
    )

    stdout_doc = _load_json(steps["configured"]["stdout"]) or {}
    run_id = stdout_doc.get("run_id")
    run_dir = runs_root / str(run_id) if run_id else None
    run_files: dict[str, str | None] = {}
    listing: list[str] | None = None
    if run_dir is not None and run_dir.is_dir():
        listing = sorted(p.name for p in run_dir.iterdir())
        for name in RUN_FILES:
            path = run_dir / name
            run_files[name] = (
                path.read_text(errors="replace") if path.exists() else None
            )
    else:
        run_files = dict.fromkeys(RUN_FILES)

    receipt = _load_json(run_files.get("receipt.json")) or {}
    interpreter = (
        receipt.get("receipt", {}).get("runtime_identity", {}).get("interpreter")
        if isinstance(receipt, dict)
        else None
    )
    direct_url: dict[str, Any] = {}
    installed: dict[str, Any] = {}
    if interpreter:
        readback = subprocess.run(
            [interpreter, "-c", _DIRECT_URL_SNIPPET % (ONEX_DISTRIBUTIONS,)],
            env=base_env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        direct_url = _load_json(readback.stdout) or {}
        installed = {
            dist: entry.get("version")
            for dist, entry in (
                receipt.get("receipt", {}).get("runtime_identity", {}).get("packages")
                or {}
            ).items()
        }

    # /tmp is scanned, not written: a clone left there would make this a
    # checkout machine, and it is where a stray one would most likely sit.
    scan_roots = [customer_home, workdir, Path.home(), Path("/tmp")]  # noqa: S108
    for env_name in ("GITHUB_WORKSPACE", "RUNNER_TEMP"):
        if os.environ.get(env_name):
            scan_roots.append(Path(os.environ[env_name]))
    return {
        "as_of": datetime.datetime.now(datetime.UTC).isoformat(),
        "prompt": args.prompt,
        "model_port": int(args.model_port),
        "model_host": model_host,
        "served_models": served,
        "model_artifact": {
            "file": args.model_file,
            "sha256": args.model_sha256,
            "server": args.server_build,
        },
        "model_metrics": {
            "unconfigured_tokens_predicted_delta": tokens_after_unconfigured
            - tokens_before_unconfigured,
            "configured_tokens_predicted_delta": tokens_after_configured
            - tokens_before_configured,
        },
        "customer_env_keys": sorted(configured_env),
        "config_surface": (
            f"the documented machine-local overlay ~/{OVERLAY_RELATIVE_PATH} "
            "(OMN-16200) and nothing else: no environment binding, shipped routing tiers"
        ),
        "customer_config": {
            f"~/{OVERLAY_RELATIVE_PATH}": overlay_path.read_text(),
        },
        "steps": steps,
        "run_dir": str(run_dir) if run_dir else None,
        "run_dir_listing": listing,
        "run_files": run_files,
        "direct_url": direct_url,
        "installed": installed,
        "scan_roots": [str(r) for r in scan_roots],
        "source_trees": find_source_trees(scan_roots),
        "workdir_git_ancestor": git_ancestor(workdir),
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def render_summary(record: Record) -> str:
    lines = [
        f"## C13 customer-local delegation — {record.verdict}",
        "",
        f"as of `{record.as_of}`",
        "",
        "| clause | result | first reason |",
        "|---|---|---|",
    ]
    for clause in record.clauses:
        first = clause.reasons[0] if clause.reasons else ""
        lines.append(
            f"| `{clause.name}` | {'PASS' if clause.passed else 'FAIL'} | {first} |"
        )
    lines += ["", f"config surface: {record.context.get('config_surface')}", ""]
    return "\n".join(lines)


def _write_outputs(record: Record, args: argparse.Namespace) -> None:
    payload = json.dumps(record.as_dict(), indent=2, sort_keys=True)
    Path(args.record).write_text(payload + "\n")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(render_summary(record) + "\n")
    print(render_summary(record))
    for clause in record.clauses:
        for reason in clause.reasons:
            print(f"FAIL {clause.name}: {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="drive a live customer session and grade it")
    run.add_argument("--customer-home", required=True)
    run.add_argument("--customer-bin", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--trace-dir", required=True)
    run.add_argument("--model-port", required=True, type=int)
    # The customer's own model server: loopback, or a private address in the
    # customer's own network (OMN-19805). Never a public address.
    run.add_argument("--model-host", default=LOOPBACK_MODEL_HOST)
    run.add_argument("--served-model", required=True)
    run.add_argument("--model-file", default="")
    run.add_argument("--model-sha256", default="")
    run.add_argument("--server-build", default="")
    run.add_argument("--prompt", default="explain what a calendar app needs")
    run.add_argument("--step-timeout", default="600")
    # The response budget the customer overlay gives the local backend. A
    # reasoning model spends tokens before it answers, so a small budget turns a
    # healthy model into an empty answer (measured on the lab customer machine, 2026-09-23: 512
    # tokens exhausted, quality 0.0). It must also fit the server's context.
    run.add_argument("--max-tokens", default="8192")
    run.add_argument("--observations-out", required=True)
    run.add_argument("--record", required=True)
    run.add_argument("--summary", default="")

    replay = sub.add_parser("grade", help="grade a recorded observation file")
    replay.add_argument("--observations", required=True)
    replay.add_argument("--record", required=True)
    replay.add_argument("--summary", default="")

    args = parser.parse_args(argv)
    try:
        if args.mode == "run":
            observations = observe_live(args)
            Path(args.observations_out).write_text(
                json.dumps(observations, indent=2, sort_keys=True) + "\n"
            )
        else:
            try:
                observations = json.loads(Path(args.observations).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ProbeInputError(f"unreadable observations file: {exc}") from exc
    except ProbeInputError as exc:
        print(f"C13 probe could not run: {exc}", file=sys.stderr)
        Path(args.record).write_text(
            json.dumps(
                {
                    "record_version": RECORD_VERSION,
                    "criterion": "C13",
                    "verdict": "COULD_NOT_RUN",
                    "reason": str(exc),
                },
                indent=2,
            )
            + "\n"
        )
        return EXIT_INPUT

    record = grade(observations)
    _write_outputs(record, args)
    return EXIT_OK if record.verdict == "PASS" else EXIT_FINDINGS


if __name__ == "__main__":
    sys.exit(main())
