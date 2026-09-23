# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19200 -- the C29 producer: a customer's own provider key, no OmniNode cloud.

WHAT THIS IS
    The producer for beta criterion C29: "Customer-local delegation to a
    provider with the customer's own key: clean machine, one provider key the
    customer supplies, the receipt names that provider and model, and no
    OmniNode cloud endpoint is contacted." It drives one real customer session
    -- install, key registration, delegation -- on a machine that holds no
    OmniNode repository, observes it from outside the process, and grades four
    clauses against expectations declared HERE. Its EXIT CODE is the
    criterion's verdict, the C11 / C13 / C15 shape.

    ``0``  all four clauses proven, and every control behaved
    ``1``  at least one clause unproven or one control misbehaved; the record
           names which and why
    ``2``  the probe could not run at all (no ``onex`` on the customer PATH, no
           strace, no key supplied to the probe). Deliberately distinct from
           ``1``: "I could not run" is not "the product failed", and it is
           never ``0``.

THE FOUR CLAUSES
    clean_machine    every OmniNode distribution the run imported came from the
                     package index, the working directory is not inside a git
                     work tree, and no source tree of any of the five
                     distributions exists under the scanned roots.
    customer_key     the key reached the product through the customer surface
                     and nothing else: ``onex secret set`` read it on stdin,
                     the customer environment was built from nothing (no key
                     variable exists in it), every reference the machine holds
                     afterwards names the one provider, the delegation's key was
                     answered by this machine's own store, and the key's value
                     appears in no output the session produced.
    names_provider   the receipt names the provider (its backend id carries the
                     provider slug and its endpoint host is that provider's
                     host) and the model (the accepted attempt's model is the
                     receipt's model), every attempt stayed on that provider,
                     and ``result.txt`` is the accepted response.
    no_omninode      across the WHOLE customer session -- identity init, the
                     keyless refusal, key registration and the delegation --
                     no name under an OmniNode domain was looked up and every
                     external connect() is attributable, through a DNS answer
                     the same session received, to the provider's host. An
                     external connect the grader cannot attribute is a failure:
                     an address nobody looked up could be anyone's, ours
                     included.

AN UNPROVEN ZERO IS A FAILURE, NOT A PASS
    "No OmniNode host was contacted" means nothing unless the instrument could
    see a contact. Two positive controls make that falsifiable in the same
    invocation:

    1. the keyed delegation MUST show a connect() to an address the session's
       own DNS answer gave for the provider host. Zero such connects means the
       tracer was blind to the process that did the work, or the DNS decoder
       could not read the answer, and either way the zero is unproven.
    2. a deliberate lookup of an OmniNode host (``api.omninode.ai``), traced by
       the SAME wrapper and read by the SAME decoder, MUST be classified as
       OmniNode cloud. If it is not, the grader cannot see the thing it grades
       for.

    And one negative control: the same prompt, on the same configured machine
    BEFORE any key is registered, must be a typed refusal that reaches no
    provider. That is what shows the delegation's key came from the
    registration this probe performed and not from anything ambient.

THE KEY IS REGISTERED THE WAY THE PRODUCT SAYS
    ``onex secret set llm.<provider>.api_key``: the example in ``onex secret``'s
    own module docstring and the exact remediation the secret resolver prints
    when that reference is unregistered. The probe does not register it any
    other way. If the product's own instruction does not produce a working
    delegation, this probe is red, which is the finding.
    On 2026-09-22 it does not: the local store files ``llm.glm.api_key`` under
    provider ``llm.glm.api``, the BYOK substitution never fires, and the house-
    shaped reference is refused on the customer-local surface (OMN-19205). From
    omnimarket 0.4.203 a provider-only customer sees that same refusal worded
    as "No local model is declared on this machine" (the OMN-16200 gate re-words
    it, and changes nothing else).

THE CUSTOMER WRITES NO CONFIGURATION
    Since omnimarket 0.4.203 a clean install resolves its routing contracts
    from the packaged defaults. The customer environment is HOME, PATH, LANG and
    TERM and nothing else, and this customer declares no local model: C29 is the
    provider-only case. The only act is ``onex local init`` and handing over the
    key.

WHY STRACE, AND WHY DNS
    connect() is where an attempt becomes visible whether or not it succeeds,
    but connect() carries an address, not a name, and "is this address
    OmniNode's" cannot be answered from an address alone. The glibc stub
    resolver sends its query and receives its answer through ``sendmmsg`` /
    ``recvfrom`` on a UDP socket, so the same trace carries the name that was
    asked for and the addresses the answer returned. The grader joins the two.
    A resolver that bypasses that path (nss-resolve over varlink, DNS over
    HTTPS) leaves every connect unattributed, which grades FAIL, never PASS.

THE KEY NEVER LEAVES MEMORY
    It reaches this process in one environment variable, which is popped before
    anything is spawned, and is written only to ``onex secret set``'s stdin. It
    is compared against every captured output and, if found, redacted before
    anything is written and graded as a failure. It is never printed.

STANDARD LIBRARY ONLY
    Same reason as the C11 and C13 producers: a probe whose job is to be
    readable should not acquire a dependency resolution step that can fail for
    reasons unrelated to the thing it probes.
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
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2


@dataclass(frozen=True)
class ProviderSpec:
    """What the grader expects of one provider, declared here and nowhere else."""

    slug: str
    host: str
    #: The reference ``onex secret set`` is told to store the key under -- the
    #: form the CLI's docstring and the resolver's remediation both print.
    registration_ref: str


PROVIDERS: Final[Mapping[str, ProviderSpec]] = {
    "glm": ProviderSpec("glm", "api.z.ai", "llm.glm.api_key"),
    "openrouter": ProviderSpec("openrouter", "openrouter.ai", "llm.openrouter.api_key"),
}

#: A name equal to, or under, one of these is OmniNode cloud.
OMNINODE_DOMAINS: Final[tuple[str, ...]] = ("omninode.ai", "omninode.com")

#: Positive control 2: looked up under the tracer, must be classified OmniNode.
OMNINODE_CONTROL_HOST: Final[str] = "api.omninode.ai"

#: The keyless negative control must be a TYPED refusal: an ONEX error code in
#: the result, not a fallthrough and not an untyped crash. Which code is not
#: pinned -- on omnimarket 0.4.198-0.4.202 it was the customer-key terminus
#: (``ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT``), from 0.4.203 the OMN-16200
#: gate re-words the same refusal as ``ONEX_CORE_041_INVALID_CONFIGURATION``.
#: The code seen is recorded.
KEYLESS_REFUSAL_CODE_RE: Final = re.compile(r"\[?(ONEX_[A-Z0-9_]+)\]?")

#: The customer environment's whole vocabulary. Anything else in it -- above
#: all a variable carrying a key -- would mean the key had a second way in.
ALLOWED_CUSTOMER_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {
        "HOME",
        "PATH",
        "LANG",
        "TERM",
    }
)

#: The customer session's steps, every one traced, in order.
SESSION_STEPS: Final[tuple[str, ...]] = ("init", "keyless", "secret_set", "keyed")

#: The five distributions a delegation imports.
ONEX_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "omnibase_compat",
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omnimarket",
)

SOURCE_TREE_PROJECT_NAMES: Final[frozenset[str]] = frozenset(
    name for dist in ONEX_DISTRIBUTIONS for name in (dist, dist.replace("_", "-"))
)

RUN_FILES: Final[tuple[str, ...]] = ("result.txt", "receipt.json", "run.json")

REDACTED: Final[str] = "[C29-REDACTED-KEY]"


class ProbeInputError(RuntimeError):
    """The probe could not run; exit 2, never a verdict."""


# --------------------------------------------------------------------------
# strace decoding
# --------------------------------------------------------------------------

_HEX_RE: Final = re.compile(r"\\x([0-9a-fA-F]{2})")
_BUFFER_RE: Final = re.compile(r'"((?:\\x[0-9a-fA-F]{2})+)"')
_SYSCALL_RE: Final = re.compile(
    r"^(?:\[pid\s+)?(?P<pid>\d+)\]?\s+(?P<call>[a-z_0-9]+)\("
)
#: A call strace split across two lines. A blocking ``recvfrom`` prints its
#: buffer on the RESUMED line, so the DNS answer is only ever there.
_RESUMED_RE: Final = re.compile(r"<\.\.\. (?P<call>[a-z_0-9]+) resumed>")
_INET_RE: Final = re.compile(
    r"sa_family=AF_INET, sin_port=htons\((?P<port>\d+)\), "
    r'sin_addr=inet_addr\("(?P<host>[^"]+)"\)'
)
_INET6_RE: Final = re.compile(
    r"sa_family=AF_INET6, sin6_port=htons\((?P<port>\d+)\).*?"
    r'inet_pton\(AF_INET6, "(?P<host>[^"]+)"'
)
_DNS_CALLS: Final[frozenset[str]] = frozenset(
    {"sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg"}
)
_HOSTNAME_LABEL_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,63}$")


def _unescape(text: str) -> str:
    """strace ``-xx`` hex-escapes every string, addresses included."""
    return _HEX_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


@dataclass(frozen=True)
class DnsMessage:
    is_response: bool
    qname: str
    #: (owner name, rdata) for every A, AAAA and CNAME answer record.
    addresses: tuple[tuple[str, str], ...] = ()
    cnames: tuple[tuple[str, str], ...] = ()


def _read_name(buf: bytes, offset: int, depth: int = 0) -> tuple[str, int] | None:
    """A DNS name at ``offset``, following compression pointers."""
    labels: list[str] = []
    while True:
        if offset >= len(buf) or depth > 16:
            return None
        length = buf[offset]
        if length == 0:
            return ".".join(labels).lower(), offset + 1
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(buf):
                return None
            pointer = ((length & 0x3F) << 8) | buf[offset + 1]
            target = _read_name(buf, pointer, depth + 1)
            if target is None:
                return None
            suffix = target[0]
            return ".".join([*labels, suffix] if suffix else labels).lower(), offset + 2
        if length > 63 or offset + 1 + length > len(buf):
            return None
        label = buf[offset + 1 : offset + 1 + length].decode("ascii", "replace")
        if not _HOSTNAME_LABEL_RE.match(label):
            return None
        labels.append(label)
        offset += 1 + length


def parse_dns(buf: bytes) -> DnsMessage | None:
    """A strictly-read DNS message, or ``None`` for anything that is not one."""
    if len(buf) < 17:
        return None
    flags = int.from_bytes(buf[2:4], "big")
    qdcount = int.from_bytes(buf[4:6], "big")
    ancount = int.from_bytes(buf[6:8], "big")
    opcode = (flags >> 11) & 0xF
    if qdcount != 1 or opcode != 0:
        return None
    is_response = bool(flags & 0x8000)
    if not is_response and ancount != 0:
        return None
    name = _read_name(buf, 12)
    if name is None or not name[0]:
        return None
    qname, offset = name
    if offset + 4 > len(buf):
        return None
    qclass = int.from_bytes(buf[offset + 2 : offset + 4], "big")
    if qclass != 1:
        return None
    offset += 4
    addresses: list[tuple[str, str]] = []
    cnames: list[tuple[str, str]] = []
    for _ in range(ancount if is_response else 0):
        owner = _read_name(buf, offset)
        if owner is None or owner[1] + 10 > len(buf):
            break
        offset = owner[1]
        rtype = int.from_bytes(buf[offset : offset + 2], "big")
        rdlength = int.from_bytes(buf[offset + 8 : offset + 10], "big")
        rdata_at = offset + 10
        if rdata_at + rdlength > len(buf):
            break
        if rtype == 1 and rdlength == 4:
            addresses.append(
                (owner[0], str(ipaddress.IPv4Address(buf[rdata_at : rdata_at + 4])))
            )
        elif rtype == 28 and rdlength == 16:
            addresses.append(
                (owner[0], str(ipaddress.IPv6Address(buf[rdata_at : rdata_at + 16])))
            )
        elif rtype == 5:
            target = _read_name(buf, rdata_at)
            if target is not None:
                cnames.append((owner[0], target[0]))
        offset = rdata_at + rdlength
    return DnsMessage(is_response, qname, tuple(addresses), tuple(cnames))


@dataclass(frozen=True)
class Connect:
    family: str
    host: str | None
    port: int | None

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "host": self.host, "port": self.port}


@dataclass
class Trace:
    """Everything one strace log says about names and addresses."""

    connects: list[Connect] = field(default_factory=list)
    queried: list[str] = field(default_factory=list)
    #: address -> the QUESTION name whose answer carried it.
    answered: dict[str, str] = field(default_factory=dict)
    #: every name any answer carried (owners and CNAME targets).
    answer_names: set[str] = field(default_factory=set)


def parse_trace(strace_text: str) -> Trace:
    trace = Trace()
    for line in strace_text.splitlines():
        resumed = _RESUMED_RE.search(line)
        if resumed is not None:
            if resumed.group("call") == "connect":
                # The address is always on the line that opened the call.
                continue
            call: str | None = resumed.group("call")
        else:
            match = _SYSCALL_RE.match(line)
            call = match.group("call") if match else None
        if call is None:
            # Unprefixed form (strace without -f) -- read the call name directly.
            head = line.split("(", 1)[0].strip()
            call = head if head.isidentifier() else None
        if call == "connect":
            text = _unescape(line)
            if "sa_family=AF_INET6" in text:
                inet6 = _INET6_RE.search(text)
                trace.connects.append(
                    Connect(
                        "AF_INET6",
                        inet6.group("host") if inet6 else None,
                        int(inet6.group("port")) if inet6 else None,
                    )
                )
            elif "sa_family=AF_INET," in text:
                inet = _INET_RE.search(text)
                trace.connects.append(
                    Connect(
                        "AF_INET",
                        inet.group("host") if inet else None,
                        int(inet.group("port")) if inet else None,
                    )
                )
            continue
        if call not in _DNS_CALLS:
            continue
        for hex_body in _BUFFER_RE.findall(line):
            message = parse_dns(bytes.fromhex(hex_body.replace("\\x", "")))
            if message is None:
                continue
            if not message.is_response:
                trace.queried.append(message.qname)
                continue
            trace.answer_names.add(message.qname)
            for owner, target in message.cnames:
                trace.answer_names.update((owner, target))
            for owner, address in message.addresses:
                trace.answer_names.add(owner)
                trace.answered[address] = message.qname
    return trace


def is_omninode(name: str) -> bool:
    name = name.lower().rstrip(".")
    return any(name == d or name.endswith("." + d) for d in OMNINODE_DOMAINS)


def _is_loopback(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def classify_connects(
    trace: Trace, *, provider_host: str, nameservers: Sequence[str]
) -> list[dict[str, Any]]:
    """Name what every inet connect() was for. The grader's whole vocabulary.

    ``resolver``     port 53 on a nameserver the machine's resolv.conf declares
    ``loopback``     a loopback address -- never leaves the machine
    ``provider``     an address the session's own DNS answer gave for the
                     provider host
    ``omninode``     an address the session's own DNS answer gave for an
                     OmniNode name
    ``other_named``  attributable, but to a host that is neither
    ``unattributed`` no traced DNS answer produced it, or it could not be read
    """
    rows: list[dict[str, Any]] = []
    for connect in trace.connects:
        row = connect.as_dict()
        host = connect.host
        if host is None:
            row["kind"] = "unattributed"
        elif _is_loopback(host):
            row["kind"] = "loopback"
        elif connect.port == 53 and host in nameservers:
            row["kind"] = "resolver"
        elif host in trace.answered:
            name = trace.answered[host]
            row["name"] = name
            if is_omninode(name):
                row["kind"] = "omninode"
            elif name == provider_host:
                row["kind"] = "provider"
            else:
                row["kind"] = "other_named"
        else:
            row["kind"] = "unattributed"
        rows.append(row)
    return rows


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


def _provider(obs: Mapping[str, Any]) -> ProviderSpec | None:
    return PROVIDERS.get(str(obs.get("provider") or ""))


def _ref_names_provider(ref: str, slug: str) -> bool:
    """A reference belongs to ``slug`` in either shape the local store holds."""
    if ref.startswith("cred_"):
        parts = ref[len("cred_") :].rsplit("_", 2)
        return len(parts) == 3 and parts[1] == slug
    parts = ref.split(".")
    return len(parts) == 3 and parts[0] == "llm" and parts[1] == slug


def grade_clean_machine(obs: Mapping[str, Any], receipt: Any) -> Clause:
    clause = Clause("clean_machine")
    identity = (
        receipt.get("receipt", {}).get("runtime_identity")
        if isinstance(receipt, dict)
        else None
    )
    if not isinstance(identity, dict):
        clause.reasons.append(
            "the keyed receipt carries no runtime_identity, so where the code came from is unproven"
        )
    else:
        packages = identity.get("packages") or {}
        clause.evidence["locus_kind"] = identity.get("locus_kind")
        per_package: dict[str, Any] = {}
        for dist in ONEX_DISTRIBUTIONS:
            entry = packages.get(dist)
            if not isinstance(entry, dict):
                clause.reasons.append(f"{dist} absent from the runtime identity")
                continue
            per_package[dist] = {
                k: entry.get(k) for k in ("version", "source", "commit", "import_path")
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
    return clause


def grade_customer_key(
    obs: Mapping[str, Any], spec: ProviderSpec | None, receipt: Any
) -> Clause:
    clause = Clause("customer_key")
    steps = obs.get("steps") or {}
    env_keys = obs.get("customer_env_keys")
    clause.evidence["customer_env_keys"] = env_keys
    if env_keys is None:
        clause.reasons.append("the customer environment's keys were not recorded")
    else:
        extra = sorted(set(env_keys) - ALLOWED_CUSTOMER_ENV_KEYS)
        if extra:
            clause.reasons.append(
                f"the customer environment carries variables beyond the allowed set: {extra}"
            )
    registration = steps.get("secret_set")
    if registration is None:
        clause.reasons.append("the key registration step was not run")
    else:
        clause.evidence["registration_argv"] = registration.get("argv")
        clause.evidence["registration_returncode"] = registration.get("returncode")
        if registration.get("returncode") != 0:
            clause.reasons.append(
                f"onex secret set exited {registration.get('returncode')}, not 0"
            )
        if registration.get("stdin") != "key":
            clause.reasons.append(
                "the key did not reach onex secret set on stdin, the only surface the customer has"
            )
    refs = obs.get("registered_refs")
    clause.evidence["registered_refs"] = refs
    if spec is None:
        clause.reasons.append(f"unknown provider {obs.get('provider')!r}")
    elif not refs:
        clause.reasons.append(
            "onex secret list shows no reference, so no key is registered"
        )
    else:
        foreign = [r for r in refs if not _ref_names_provider(r, spec.slug)]
        if foreign:
            clause.reasons.append(
                f"references for a provider other than {spec.slug!r} are registered: {foreign}"
            )
    result = (
        receipt.get("receipt", {}).get("result", {})
        if isinstance(receipt, dict)
        else {}
    )
    source = result.get("secret_source")
    used_ref = result.get("secret_ref")
    clause.evidence["receipt_secret_source"] = source
    clause.evidence["receipt_secret_ref"] = used_ref
    if source != "store":
        clause.reasons.append(
            f"the delegation's key came from {source!r}, not this machine's own store"
        )
    if refs and used_ref not in refs:
        clause.reasons.append(
            f"the delegation authenticated with {used_ref!r}, which is not a reference this session registered"
        )
    leaks = obs.get("key_leaks")
    clause.evidence["key_leaks"] = leaks
    if leaks is None:
        clause.reasons.append("the captured outputs were not checked for the key")
    elif leaks:
        clause.reasons.append(
            f"the key's value appeared in captured output (redacted before writing): {leaks}"
        )
    return clause


def grade_names_provider(
    obs: Mapping[str, Any], spec: ProviderSpec | None, receipt: Any
) -> Clause:
    clause = Clause("names_provider")
    steps = obs.get("steps") or {}
    keyed = steps.get("keyed") or {}
    files = obs.get("run_files") or {}
    clause.evidence["keyed_returncode"] = keyed.get("returncode")
    if keyed.get("returncode") != 0:
        clause.reasons.append(
            f"the keyed delegation exited {keyed.get('returncode')}, not 0"
        )
    if not isinstance(receipt, dict):
        clause.reasons.append("no receipt.json for the keyed run")
        stdout_doc = _load_json(keyed.get("stdout"))
        refusal = _find_refusal(stdout_doc)
        if refusal:
            clause.evidence["keyed_refusal"] = refusal
            clause.reasons.append(f"the keyed delegation was refused: {refusal}")
        return clause
    result = receipt.get("receipt", {}).get("result", {}) or {}
    backend_id = receipt.get("backend_id")
    endpoint = receipt.get("endpoint")
    model = receipt.get("model")
    clause.evidence.update(
        {
            "receipt_status": receipt.get("status"),
            "receipt_backend_id": backend_id,
            "receipt_endpoint": endpoint,
            "receipt_model": model,
            "receipt_routing_tier": receipt.get("routing_tier"),
            "result_provider_field": result.get("provider"),
            "result_model_name": result.get("model_name"),
        }
    )
    if receipt.get("status") != "success":
        refusal = result.get("error_message") or _find_refusal(receipt)
        clause.reasons.append(
            f"receipt status is {receipt.get('status')!r}, not 'success'"
            + (f": {refusal}" if refusal else "")
        )
    if spec is None:
        clause.reasons.append(f"unknown provider {obs.get('provider')!r}")
        return clause
    if not isinstance(backend_id, str) or spec.slug not in backend_id.split("-"):
        clause.reasons.append(
            f"receipt backend id {backend_id!r} does not name provider {spec.slug!r}"
        )
    host = urllib.parse.urlparse(endpoint or "").hostname
    if host != spec.host:
        clause.reasons.append(
            f"receipt endpoint host {host!r} is not the provider's host {spec.host!r}"
        )
    if not model:
        clause.reasons.append("receipt names no model")
    attempts = result.get("attempts") or []
    clause.evidence["attempts"] = [
        {k: a.get(k) for k in ("tier", "backend_id", "model_id", "acceptance_decision")}
        for a in attempts
    ]
    if not attempts:
        clause.reasons.append("receipt records no routing attempt")
    elsewhere = [
        a.get("backend_id")
        for a in attempts
        if spec.slug not in str(a.get("backend_id") or "").split("-")
    ]
    if elsewhere:
        clause.reasons.append(f"attempt(s) left provider {spec.slug!r}: {elsewhere}")
    accepted = [a for a in attempts if a.get("acceptance_decision") == "accept"]
    if len(accepted) != 1:
        clause.reasons.append(
            f"{len(accepted)} accepted attempts, expected exactly one"
        )
    elif accepted[0].get("model_id") != model:
        clause.reasons.append("the accepted attempt's model is not the receipt's model")
    response = result.get("response")
    result_text = files.get("result.txt")
    if not isinstance(response, str) or not response.strip():
        clause.reasons.append("receipt carries no accepted response")
    elif result_text is None or result_text.strip() != response.strip():
        clause.reasons.append("result.txt is not the response the receipt accepted")
    return clause


def _find_refusal(doc: Any) -> str | None:
    """The first ``error_message`` anywhere in a result document."""
    if isinstance(doc, dict):
        message = doc.get("error_message")
        if isinstance(message, str) and message:
            return message[:400]
        for value in doc.values():
            found = _find_refusal(value)
            if found:
                return found
    elif isinstance(doc, list):
        for value in doc:
            found = _find_refusal(value)
            if found:
                return found
    return None


def grade_no_omninode(obs: Mapping[str, Any], spec: ProviderSpec | None) -> Clause:
    clause = Clause("no_omninode")
    steps = obs.get("steps") or {}
    nameservers = obs.get("nameservers") or []
    provider_host = spec.host if spec else ""
    clause.evidence["nameservers"] = nameservers
    per_step: dict[str, Any] = {}
    for step in SESSION_STEPS:
        if step not in steps:
            clause.reasons.append(f"customer step {step!r} was not traced")
            continue
        trace = parse_trace(steps[step].get("strace", "") or "")
        rows = classify_connects(
            trace, provider_host=provider_host, nameservers=nameservers
        )
        kinds: dict[str, int] = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        per_step[step] = {
            "queried": sorted(set(trace.queried)),
            "connect_kinds": kinds,
        }
        omninode_names = sorted(
            {n for n in (*trace.queried, *trace.answer_names) if is_omninode(n)}
        )
        if omninode_names:
            clause.reasons.append(
                f"{step}: OmniNode name(s) looked up: {omninode_names}"
            )
        for kind in ("omninode", "other_named", "unattributed"):
            bad = [r for r in rows if r["kind"] == kind]
            if bad:
                clause.reasons.append(
                    f"{step}: {len(bad)} {kind} external connect(s): {bad[:5]}"
                )
    clause.evidence["steps"] = per_step

    # Positive control 1: the tracer saw the process that reached the provider,
    # and the decoder attributed the address to the provider's own lookup.
    keyed = per_step.get("keyed", {})
    if keyed.get("connect_kinds", {}).get("provider", 0) < 1:
        clause.reasons.append(
            "positive control failed: the keyed delegation shows no connect to an address the "
            f"session's own DNS answer gave for {provider_host!r}, so the zero OmniNode contacts "
            "it reports are unproven"
        )

    # Positive control 2: the decoder recognises an OmniNode lookup.
    control = steps.get("omninode_control")
    if control is None:
        clause.reasons.append(
            "the OmniNode lookup control was not run, so the decoder's ability to see an OmniNode "
            "name is unproven"
        )
    else:
        control_trace = parse_trace(control.get("strace", "") or "")
        seen = sorted({n for n in control_trace.queried if is_omninode(n)})
        clause.evidence["omninode_control_seen"] = seen
        if OMNINODE_CONTROL_HOST not in seen:
            clause.reasons.append(
                f"positive control failed: a deliberate lookup of {OMNINODE_CONTROL_HOST!r} was not "
                "decoded and classified as OmniNode cloud"
            )

    # Negative control: before a key exists, a typed refusal and no provider.
    keyless = steps.get("keyless")
    if keyless is not None:
        refusal = _find_refusal(_load_json(keyless.get("stdout"))) or ""
        code = KEYLESS_REFUSAL_CODE_RE.search(refusal)
        typed = code is not None
        clause.evidence["keyless_returncode"] = keyless.get("returncode")
        clause.evidence["keyless_typed_refusal"] = typed
        clause.evidence["keyless_refusal_code"] = code.group(1) if code else None
        clause.evidence["keyless_refusal"] = refusal[:400]
        if keyless.get("returncode") == 0:
            clause.reasons.append(
                "negative control failed: a delegation with no key registered succeeded, so the "
                "keyed run's key is not proven to be the one this session registered"
            )
        if not typed:
            clause.reasons.append(
                "negative control failed: the keyless run's result carries no typed ONEX "
                "refusal code, so a refusal is not distinguishable from a crash"
            )
        keyless_kinds = per_step.get("keyless", {}).get("connect_kinds", {})
        if keyless_kinds.get("provider", 0):
            clause.reasons.append(
                "negative control failed: the keyless delegation reached the provider"
            )
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
            "criterion": "C29",
            "ticket": "OMN-19200",
            "verdict": self.verdict,
            "as_of": self.as_of,
            "clauses": {c.name: c.as_dict() for c in self.clauses},
            "context": self.context,
        }


def grade(obs: Mapping[str, Any], *, as_of: str | None = None) -> Record:
    spec = _provider(obs)
    receipt = _load_json((obs.get("run_files") or {}).get("receipt.json"))
    clauses = [
        grade_clean_machine(obs, receipt),
        grade_customer_key(obs, spec, receipt),
        grade_names_provider(obs, spec, receipt),
        grade_no_omninode(obs, spec),
    ]
    verdict = "PASS" if all(c.passed for c in clauses) else "FAIL"
    context = {
        "provider": obs.get("provider"),
        "provider_host": spec.host if spec else None,
        "registration_ref": spec.registration_ref if spec else None,
        "installed": obs.get("installed"),
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


def read_nameservers(resolv_conf: Path = Path("/etc/resolv.conf")) -> list[str]:
    try:
        text = resolv_conf.read_text()
    except OSError:
        return []
    return [
        line.split()[1]
        for line in text.splitlines()
        if line.strip().startswith("nameserver") and len(line.split()) > 1
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
    stdin_text: str | None = None,
) -> dict[str, Any]:
    trace_file = trace_dir / f"{name}.strace"
    completed = subprocess.run(
        [
            strace,
            "-f",
            "-qq",
            "-s",
            "1024",
            "-xx",
            "-e",
            "trace=connect,sendto,sendmsg,sendmmsg,recvfrom,recvmsg,recvmmsg",
            "-o",
            str(trace_file),
            "--",
            *argv,
        ],
        env=dict(env),
        cwd=cwd,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "argv": list(argv),
        "stdin": "key" if stdin_text is not None else None,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr[-20000:],
        "strace": trace_file.read_text(errors="replace") if trace_file.exists() else "",
    }


def redact_key(value: Any, key: str, path: str, leaks: list[str]) -> Any:
    """Replace the key wherever it appears, recording where. Never prints it."""
    if isinstance(value, str):
        if key and key in value:
            leaks.append(path)
            return value.replace(key, REDACTED)
        return value
    if isinstance(value, dict):
        return {k: redact_key(v, key, f"{path}.{k}", leaks) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_key(v, key, f"{path}[{i}]", leaks) for i, v in enumerate(value)]
    return value


def observe_live(args: argparse.Namespace) -> dict[str, Any]:
    # Popped before anything is spawned, so no child ever inherits it.
    key = os.environ.pop(args.key_env, "")
    if not key:
        raise ProbeInputError(
            f"no provider key in {args.key_env}; the probe has nothing to register "
            "and a run without one is not a verdict"
        )
    spec = PROVIDERS.get(args.provider)
    if spec is None:
        raise ProbeInputError(
            f"unknown provider {args.provider!r}; know {sorted(PROVIDERS)}"
        )
    strace = shutil.which("strace")
    if strace is None:
        raise ProbeInputError(
            "strace is not installed; the no_omninode clause has no instrument"
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

    base_env = {
        "HOME": str(customer_home),
        "PATH": f"{customer_bin}:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    customer_env = dict(base_env)
    timeout = int(args.step_timeout)

    def run(
        name: str, argv: Sequence[str], stdin_text: str | None = None
    ) -> dict[str, Any]:
        """One traced customer step, in the customer's environment."""
        return _run_step(
            name,
            argv,
            env=customer_env,
            cwd=workdir,
            strace=strace,
            trace_dir=trace_dir,
            timeout=timeout,
            stdin_text=stdin_text,
        )

    steps: dict[str, Any] = {}
    steps["init"] = run("init", [str(onex), "local", "init", "--json"])
    steps["keyless"] = run("keyless", [str(onex), "delegate", args.prompt])
    runs_root = workdir / ".onex_state" / "runs"
    if runs_root.exists():
        runs_root.rename(workdir / ".onex_state" / "runs.keyless")
    steps["secret_set"] = run(
        "secret_set",
        [str(onex), "secret", "set", spec.registration_ref],
        stdin_text=key,
    )
    steps["keyed"] = run("keyed", [str(onex), "delegate", args.prompt])
    listing = subprocess.run(
        [str(onex), "secret", "list"],
        env=customer_env,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    registered_refs = [
        line.split()[0]
        for line in listing.stdout.splitlines()
        if line.startswith("  ") and line.split()
    ]
    steps["omninode_control"] = _run_step(
        "omninode_control",
        [
            sys.executable,
            "-c",
            "import socket\ntry:\n    socket.getaddrinfo("
            f"{OMNINODE_CONTROL_HOST!r}, 443)\nexcept OSError:\n    pass\n",
        ],
        env=base_env,
        cwd=workdir,
        strace=strace,
        trace_dir=trace_dir,
        timeout=60,
    )

    stdout_doc = _load_json(steps["keyed"]["stdout"]) or {}
    run_id = stdout_doc.get("run_id")
    run_dir = runs_root / str(run_id) if run_id else None
    run_files: dict[str, str | None] = dict.fromkeys(RUN_FILES)
    if run_dir is not None and run_dir.is_dir():
        for name in RUN_FILES:
            path = run_dir / name
            run_files[name] = (
                path.read_text(errors="replace") if path.exists() else None
            )
    receipt = _load_json(run_files.get("receipt.json")) or {}
    installed = {
        dist: entry.get("version")
        for dist, entry in (
            (receipt.get("receipt", {}) if isinstance(receipt, dict) else {})
            .get("runtime_identity", {})
            .get("packages")
            or {}
        ).items()
    }
    scan_roots = [customer_home, workdir, Path.home(), Path("/tmp")]  # noqa: S108 - scanned, never written
    for env_name in ("GITHUB_WORKSPACE", "RUNNER_TEMP"):
        if os.environ.get(env_name):
            scan_roots.append(Path(os.environ[env_name]))

    observations: dict[str, Any] = {
        "as_of": datetime.datetime.now(datetime.UTC).isoformat(),
        "provider": spec.slug,
        "prompt": args.prompt,
        "customer_env_keys": sorted(customer_env),
        "nameservers": read_nameservers(),
        "steps": steps,
        "registered_refs": registered_refs,
        "secret_list": listing.stdout,
        "run_dir": str(run_dir) if run_dir else None,
        "run_files": run_files,
        "installed": installed,
        "scan_roots": [str(r) for r in scan_roots],
        "source_trees": find_source_trees(scan_roots),
        "workdir_git_ancestor": git_ancestor(workdir),
    }
    leaks: list[str] = []
    observations = redact_key(observations, key, "observations", leaks)
    # The store file is the one place the value is meant to live; every other
    # file under the customer's HOME and working directory is checked too.
    for root in (customer_home, workdir):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix in (
                ".sqlite",
                ".sqlite-journal",
                ".sqlite-wal",
            ):
                continue
            try:
                if key in path.read_text(errors="ignore"):
                    leaks.append(str(path))
            except OSError:
                continue
    observations["key_leaks"] = leaks
    del key
    return observations


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def render_summary(record: Record) -> str:
    lines = [
        f"## C29 customer-local delegation on the customer's own key — {record.verdict}",
        "",
        f"as of `{record.as_of}`, provider `{record.context.get('provider')}`",
        "",
        "| clause | result | first reason |",
        "|---|---|---|",
    ]
    for clause in record.clauses:
        first = clause.reasons[0] if clause.reasons else ""
        lines.append(
            f"| `{clause.name}` | {'PASS' if clause.passed else 'FAIL'} | {first} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(record: Record, args: argparse.Namespace) -> None:
    Path(args.record).write_text(
        json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n"
    )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(render_summary(record) + "\n")
    print(render_summary(record))
    for clause in record.clauses:
        for reason in clause.reasons:
            print(f"FAIL {clause.name}: {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="drive a live customer session and grade it")
    run.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    run.add_argument(
        "--key-env", required=True, help="NAME of the variable holding the key"
    )
    run.add_argument("--customer-home", required=True)
    run.add_argument("--customer-bin", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--trace-dir", required=True)
    run.add_argument("--prompt", default="explain what a calendar app needs")
    run.add_argument("--step-timeout", default="600")
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
        print(f"C29 probe could not run: {exc}", file=sys.stderr)
        Path(args.record).write_text(
            json.dumps(
                {
                    "record_version": RECORD_VERSION,
                    "criterion": "C29",
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
