# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Package source-hash attestation — OMN-9139.

This module carries the git commit SHA that produced this wheel.
The value is stamped at build time by ``release-python-reusable.yml``
and the ``attest-source-hash`` CI job.

Build tooling writes::

    __source_hash__ = "a1b2c3d4..."  # full git SHA

For local development (un-stamped), the value is ``"dev"``.
For a published wheel without attestation, the attest-source-hash CI
job will reject the build before it reaches PyPI.

Usage::

    from omnibase_infra import __about__
    print(__about__.__source_hash__)   # e.g. "a1b2c3d4"
"""

from __future__ import annotations

#: Git commit SHA stamped by CI at build time.
#: Local development default: "dev".
#: Never "unknown" in a published wheel — attest-source-hash rejects it.
__source_hash__: str = "dev"

__all__: list[str] = ["__source_hash__"]
