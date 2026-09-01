# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelCredentialStoreWrite -- what onboarding hands the credential store (OMN-17028).

WHY THIS MODEL EXISTS AT ALL
    Before it, an onboarding policy could only emit two shapes: ``env_output``
    (a dict of env var names) and ``credentials_output`` (a dict of secret refs).
    Neither is the shape ``StoreGatewayCredential`` reads, so ``connect_cloud``
    wrote two files the credential reader never opens and the machine finished
    onboarding unauthenticated. The fix is not a third file -- it is handing the
    collected values to the store that owns BOTH files, and this model is that
    handoff, typed.

WHY THE FIELDS ARE EXACTLY THE STORE'S OWN WRITE SIGNATURE
    ``tenant_slug`` / ``base_url`` / ``api_key`` are the three arguments of
    ``StoreGatewayCredential.save_api_key``. Keeping them identical is what
    makes a policy that names a field the store does not accept fail at policy
    LOAD time (``extra="forbid"`` plus the policy validator) rather than at the
    end of a live onboarding run, after the operator has already typed a
    credential in.

WHY api_key IS A SecretStr HERE
    This model travels inside ``ModelInteractiveResult``, which is returned as
    onboarding provenance and is therefore repr'd, logged, and model-dumped into
    receipts. Only the store unwraps it, on the one line that writes the 0600
    file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr

__all__ = ["ModelCredentialStoreWrite"]


class ModelCredentialStoreWrite(BaseModel):
    """The tenant API credential an onboarding policy collected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_slug: str = Field(
        min_length=1,
        description=(
            "The tenant this key belongs to. Names the ref the key is filed "
            "under in credentials.json, so it is not decorative."
        ),
    )
    base_url: str = Field(
        min_length=1,
        description=(
            "Gateway origin the key authenticates against. No default exists "
            "anywhere in this path: a substituted origin sends a live customer "
            "key to whatever host the release happened to ship with."
        ),
    )
    api_key: SecretStr = Field(
        description="The dashboard-minted key. Written only to the 0600 file."
    )
