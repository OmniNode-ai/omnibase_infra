# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Content hash of a contract file (OMN-18709).

The runtime manifest carries one hash per discovered contract. Before this
module that hash was ``sha256("{name}:{version}")`` -- a hash of the contract's
*identity*, not of its content -- so a contract whose body was rewritten kept
the same hash as long as its name and version strings were untouched. The field
exists for drift detection and could not detect the drift class it exists for.

This module owns the one canonical definition. There is deliberately no second
implementation: ``manifest_builder`` calls this function, and any future
consumer in this repository calls this function.

Canonical form -- declared, not implied
---------------------------------------

An outside consumer (a build step stamping an image label, a board resolver in
another repository, a shell one-liner) has to be able to compute the identical
value without importing this module. The rule is therefore deliberately small:

1. Read the file's bytes exactly as they are on disk. No YAML parse, no
   re-serialisation, no key sorting, no comment stripping.
2. Normalise line endings in those bytes: ``\\r\\n`` becomes ``\\n``, then any
   remaining lone ``\\r`` becomes ``\\n``.
3. SHA-256 the normalised bytes.
4. Render the digest as lowercase hexadecimal, 64 characters, with no prefix.

Reference reproductions, both of which return the same 64 characters this
function returns::

    python3 -c 'import hashlib,sys; b=open(sys.argv[1],"rb").read(); \\
        print(hashlib.sha256(b.replace(b"\\r\\n",b"\\n").replace(b"\\r",b"\\n")).hexdigest())' contract.yaml

    perl -pe 's/\\r\\n/\\n/g; s/\\r/\\n/g' contract.yaml | shasum -a 256

Why raw bytes rather than a YAML round-trip. A round-trip would make the hash
insensitive to comments and formatting, which reads as a feature until you ask
who can reproduce it: every consumer would then need this repository's exact
YAML library, its exact dump settings and its exact version, and a library
upgrade would move every hash in the fleet with no contract having changed. Raw
bytes cost one property -- a whitespace-only edit moves the hash -- and buy the
property the field is for, which is that a consumer with no Python at all can
verify it. A moved hash with an unchanged body is a visible, cheap false
positive; an unmoved hash with a changed body is the silent failure this ticket
exists to remove.

Why line endings are normalised at all. The bytes on disk depend on the
checkout's ``core.autocrlf`` setting, so without step 2 the same commit could
hash differently in a container built on a Windows runner than in the
repository. That is the one difference that is never a contract change.

Failure behaviour. An unreadable contract file raises. It does not hash the
empty string, and it does not fall back to the identity hash: a hash that is
wrong is worse than a manifest that is absent, because the wrong one is
indistinguishable from a correct one downstream. The kernel's manifest-emission
boundary already degrades an exception here to a logged warning and an unemitted
manifest (``service_kernel`` step 9.8), which is a visible gap rather than a
false fact.

Related:
    - OMN-18709: this change.
    - OMN-18708 (M1): the image label that carries these hashes; blocked on this.
    - ``beta/plans/2026-09-18-program-board-capabilities-and-value.md`` M3.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from omnibase_infra.errors.error_contract_content_hash import ContractContentHashError

__all__ = [
    "CONTRACT_CONTENT_HASH_ALGORITHM",
    "canonical_contract_bytes",
    "contract_content_hash",
]

#: The digest algorithm named in the canonical form above. Consumers reproducing
#: the hash outside this repository read this name from the module docstring;
#: the constant exists so a test can assert the two agree.
CONTRACT_CONTENT_HASH_ALGORITHM = "sha256"


def canonical_contract_bytes(raw: bytes) -> bytes:
    """Apply step 2 of the canonical form to already-read bytes.

    Split out from :func:`contract_content_hash` so a caller that already holds
    the bytes -- a build step reading them out of a layer, a test constructing
    them in memory -- normalises them the same way rather than approximating it.

    Args:
        raw: The contract file's bytes exactly as they are on disk.

    Returns:
        The same bytes with ``\\r\\n`` and lone ``\\r`` both rendered as ``\\n``.
    """
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def contract_content_hash(
    contract_path: Path,
    *,
    correlation_id: UUID | None = None,
) -> str:
    """Return the canonical content hash of one contract file.

    Args:
        contract_path: Filesystem path to the ``contract.yaml``. This is the
            ``contract_path`` field the runtime's discovery pass already records
            on every ``ModelDiscoveredContract``, so the hash is taken over the
            same file the running process was wired from.
        correlation_id: Optional correlation id carried into the error context.

    Returns:
        The lowercase hexadecimal SHA-256 digest of the canonical bytes, 64
        characters, with no algorithm prefix.

    Raises:
        ContractContentHashError: The file could not be read -- it is absent, it
            is a directory, or the process cannot read it. Never returns a hash
            in this case.
    """
    try:
        raw = Path(contract_path).read_bytes()
    except OSError as exc:
        raise ContractContentHashError(
            f"Contract file could not be read for content hashing: {contract_path}",
            contract_path=str(contract_path),
            correlation_id=correlation_id,
            original_error_type=type(exc).__name__,
        ) from exc

    return hashlib.sha256(canonical_contract_bytes(raw)).hexdigest()
