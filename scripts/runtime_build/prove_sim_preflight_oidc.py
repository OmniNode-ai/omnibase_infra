# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise real disposable authorization-code login with the Gateway verifier.

Run inside the pinned Gateway image on the isolated clone network. No Gateway
server or workflow route is started. Only the approved test realm is contacted.
Credentials and the resulting token are private files, never console output.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import stat
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

ISSUER = "http://auth.localhost:28080/realms/omninode"
REDIRECT = "http://localhost:3000/sim-preflight-auth-proof"  # url-authority-ok: exact isolated test-client redirect, never a product default.


def read_seeded_identity(path: Path) -> dict[str, str]:
    """Bind the auth proof to the actual test user seeded in the clone database."""
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("identity receipt must be a private regular file")
    identity = json.loads(path.read_text())
    if not isinstance(identity, dict) or set(identity) != {
        "tenant_id",
        "subject",
        "principal_id",
    }:
        raise ValueError("unexpected identity receipt schema")
    if not all(isinstance(value, str) and value for value in identity.values()):
        raise ValueError("identity receipt fields must be nonempty strings")
    return {
        "tenant_id": str(UUID(identity["tenant_id"])),
        "subject": str(UUID(identity["subject"])),
        "principal_id": identity["principal_id"],
    }


class LoginForm(HTMLParser):
    """Accept only Keycloak's unique login form on the approved issuer."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fields = dict(attrs)
        if tag == "form" and fields.get("id") == "kc-form-login":
            self.actions.append(fields.get("action") or "")

    def action(self) -> str:
        if len(self.actions) != 1:
            raise ValueError("expected one login form")
        parsed = urlsplit(self.actions[0])
        if (parsed.scheme, parsed.hostname, parsed.port) != (
            "http",
            "auth.localhost",
            28080,
        ):
            raise ValueError("login form escaped disposable issuer")
        if not parsed.path.startswith("/realms/omninode/login-actions/"):
            raise ValueError("login form has unexpected realm action")
        return self.actions[0]


def prove(credentials: Path, realm_file: Path, token_output: Path) -> dict[str, bool]:
    # These imports intentionally resolve from the running Gateway artifact.
    import requests

    logging.disable(logging.CRITICAL)
    private_dir = credentials.parent
    if (
        not private_dir.is_absolute()
        or private_dir.is_symlink()
        or stat.S_IMODE(private_dir.stat().st_mode) != 0o700
        or realm_file.parent != private_dir
        or token_output.parent != private_dir
    ):
        raise ValueError("auth proof files must share the private directory")
    for path in (credentials, realm_file):
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise ValueError("auth proof input must be a private regular file")
    env = dict(line.split("=", 1) for line in credentials.read_text().splitlines())
    expected_tenant, identity = _identity_context(realm_file, private_dir)
    if token_output.exists() or token_output.is_symlink():
        raise ValueError("token output must not already exist")
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(32)
    session = requests.Session()
    session.trust_env = False
    page = session.get(
        ISSUER + "/protocol/openid-connect/auth",
        params={
            "client_id": "omnidash",
            "redirect_uri": REDIRECT,
            "response_type": "code",
            "scope": "openid tenant",
            "state": state,
            "nonce": secrets.token_urlsafe(32),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        timeout=15,
        allow_redirects=False,
    )
    if page.status_code != 200:
        raise ValueError("realm did not serve login form")
    form = LoginForm()
    form.feed(page.text)
    login = session.post(
        form.action(),
        data={
            "username": env["SIM_PREFLIGHT_DEMO_USERNAME"],
            "password": env["SIM_PREFLIGHT_DEMO_PASSWORD"],
            "credentialId": "",
        },
        timeout=15,
        allow_redirects=False,
    )
    location = login.headers.get("Location", "")
    callback = urlsplit(location)
    expected = urlsplit(REDIRECT)
    query = parse_qs(callback.query)
    if (
        login.status_code != 302
        or callback[:3] != expected[:3]
        or query.get("state") != [state]
        or len(query.get("code", [])) != 1
    ):
        raise ValueError("authorization-code login did not complete")
    response = session.post(
        ISSUER + "/protocol/openid-connect/token",
        data={
            "grant_type": "authorization_code",
            "code": query["code"][0],
            "redirect_uri": REDIRECT,
            "client_id": "omnidash",
            "client_secret": env["SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET"],
            "code_verifier": verifier,
        },
        timeout=15,
        allow_redirects=False,
    )
    if response.status_code != 200:
        raise ValueError("authorization-code exchange failed")
    token = response.json()["access_token"]
    _verify_gateway_token(token, expected_tenant, identity)
    descriptor = os.open(
        token_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(token)
    return {
        "real_auth_code_login": True,
        "gateway_oidc_verified": True,
        "workflow_identity_claims_present": True,
        "owner_tenant_bound": True,
        "seeded_user_subject_bound": True,
        "wrong_audience_refused": True,
        "tampered_signature_refused": True,
    }


def _identity_context(
    realm_file: Path, private_dir: Path
) -> tuple[str, dict[str, str]]:
    from tenant_identity import derive_principal_id

    realm = json.loads(realm_file.read_text())
    expected_tenant = str(UUID(realm["users"][0]["attributes"]["tenant_id"][0]))
    identity = read_seeded_identity(private_dir / "identity.json")
    if identity["tenant_id"] != expected_tenant or identity[
        "principal_id"
    ] != derive_principal_id(UUID(expected_tenant)):
        raise ValueError("seeded identity does not match captured owner")
    return expected_tenant, identity


def _verify_gateway_token(
    token: str, expected_tenant: str, identity: dict[str, str]
) -> None:
    from auth_oidc import OIDCAuth, OIDCAuthError
    from tenant_identity import derive_principal_id

    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token):
        raise ValueError("browser token has an invalid wire shape")
    auth = OIDCAuth(issuer=ISSUER, audience="onex-api")
    result = auth.decode_and_resolve(token)
    if (
        str(result.tenant.tenant_id) != expected_tenant
        or result.subject != identity["subject"]
        or result.claims.get("sub") != identity["subject"]
        or result.claims.get("azp") != "omnidash"
        or result.tenant.tenant_slug != "sim-preflight"
        or result.claims.get("principal_id")
        != derive_principal_id(UUID(expected_tenant))
    ):
        raise ValueError("verified user identity does not match captured owner")
    try:
        OIDCAuth(
            issuer=ISSUER, audience="wrong-disposable-audience"
        ).decode_and_resolve(token)
    except OIDCAuthError:
        pass
    else:
        raise ValueError("wrong audience was accepted")
    header, payload, signature = token.split(".")
    tampered = (
        header
        + "."
        + payload
        + "."
        + ("A" if signature[0] != "A" else "B")
        + signature[1:]
    )
    try:
        auth.decode_and_resolve(tampered)
    except OIDCAuthError:
        pass
    else:
        raise ValueError("tampered signature was accepted")


def verify_browser_token(token_file: Path, realm_file: Path) -> dict[str, bool]:
    """Verify the token already issued to the real browser-backed OmniDash session."""
    logging.disable(logging.CRITICAL)
    private_dir = token_file.parent
    if (
        not private_dir.is_absolute()
        or private_dir.is_symlink()
        or stat.S_IMODE(private_dir.stat().st_mode) != 0o700
        or realm_file.parent != private_dir
    ):
        raise ValueError("browser proof files must share the private directory")
    for path in (token_file, realm_file):
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise ValueError("browser proof input must be a private regular file")
    expected_tenant, identity = _identity_context(realm_file, private_dir)
    token = token_file.read_text(encoding="ascii").strip()
    _verify_gateway_token(token, expected_tenant, identity)
    return {
        "browser_user_token_consumed": True,
        "gateway_oidc_verified": True,
        "workflow_identity_claims_present": True,
        "owner_tenant_bound": True,
        "seeded_user_subject_bound": True,
        "wrong_audience_refused": True,
        "tampered_signature_refused": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--credentials", type=Path)
    mode.add_argument("--browser-token", type=Path)
    parser.add_argument("--realm", type=Path, required=True)
    parser.add_argument("--token-output", type=Path)
    args = parser.parse_args()
    try:
        if args.browser_token is not None:
            if args.token_output is not None:
                raise ValueError("browser token verification does not write a token")
            result = verify_browser_token(args.browser_token, args.realm)
        else:
            if args.token_output is None:
                raise ValueError("authorization-code mode requires token output")
            result = prove(args.credentials, args.realm, args.token_output)
    except Exception:  # noqa: BLE001 -- never expose credential-bearing HTTP exceptions
        print("disposable OIDC proof failed; no token or credentials disclosed")
        return 65
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
