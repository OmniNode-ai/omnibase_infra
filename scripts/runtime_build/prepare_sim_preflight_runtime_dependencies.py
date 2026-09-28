# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stage the local-only, full-closure runtime recipe from frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

CANONICAL_DOCKERFILE_SHA256 = (
    "31372055e8e67e19c68000e00472dfd53a3b40891ddf75400c4fc116090010e6"
)
PYRAGE_WHEEL = (
    "pyrage-1.4.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
)
PYRAGE_SHA256 = "f868b360ccd3f836326d42b532fae3e8eb8340a88cca0a39c0ae10fce68f6cc5"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"runtime recipe anchor absent or nonunique: {old[:80]!r}")
    return source.replace(old, new, 1)


def _locked_pyrage_url(market: Path) -> str:
    lock = tomllib.loads((market / "uv.lock").read_text(encoding="utf-8"))
    packages = [p for p in lock["package"] if p["name"] == "pyrage"]
    if len(packages) != 1 or packages[0]["version"] != "1.4.0":
        raise ValueError("Market lock lacks unique pyrage 1.4.0")
    wheels = [
        wheel
        for wheel in packages[0]["wheels"]
        if Path(urlparse(wheel["url"]).path).name == PYRAGE_WHEEL
    ]
    if len(wheels) != 1 or wheels[0]["hash"] != f"sha256:{PYRAGE_SHA256}":
        raise ValueError("Market lock pyrage arm64 wheel differs from approved digest")
    url = wheels[0]["url"]
    if not isinstance(url, str):
        raise ValueError("Market lock pyrage wheel has invalid URL")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise ValueError("Market lock pyrage wheel has unexpected registry")
    return url


def _recipe(raw: str) -> str:
    plugin_install = (
        "RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \\\n"
        "    uv-with-retry pip install --no-deps \\\n"
        '    "omninode-claude>=0.25.1,<1.0.0" \\\n'
        '    "omninode-memory>=0.18.2,<1.0.0" \\\n'
        '    "omninode-intelligence>=0.24.0,<1.0.0"\n'
    )
    raw = _replace_once(
        raw,
        plugin_install,
        "# Local active-base profile omits inactive top-level plugins.\n",
    )
    raw = _replace_once(raw, '    "adaptive-classifier>=0.1.2" \\\n', "")
    raw = _replace_once(
        raw,
        '    "qdrant-client>=1.7.0,<1.17.0" \\\n',
        '    "qdrant-client>=1.18.0,<1.20.0" \\\n',
    )
    wheel_path = f"/opt/sim-preflight-artifacts/{PYRAGE_WHEEL}"
    closure = (
        "# Local active-base closure: resolve every declared Market/Memory dependency\n"
        "# against the exact installed workspace siblings, with locked pyrage only.\n"
        f"COPY docker/sim-preflight-artifacts/{PYRAGE_WHEEL} {wheel_path}\n"
        "RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \\\n"
        "    ( cd / && VIRTUAL_ENV=/app/.venv uv-with-retry pip install \\\n"
        "        --python /app/.venv/bin/python --no-sources --no-cache \\\n"
        "        --reinstall-package omnimarket \\\n"
        f'        "pyrage @ file://{wheel_path}" \\\n'
        '        "omnimarket @ file:///workspace/sibling-repos/omnimarket" )\n\n'
    )
    raw = _replace_once(
        raw, "# Security: upgrade protobuf", closure + "# Security: upgrade protobuf"
    )
    gate = (
        "# Local active-base profile must preserve source installation and full closure.\n"
        "# The dependency resolver can reinstall distributions after the earlier strip.\n"
        "RUN /app/.venv/bin/python -m omnibase_infra.runtime.strip_runtime_entry_points\n"
        "RUN /app/.venv/bin/python /workspace/compute_workspace_provenance.py && \\\n"
        "    uv pip check --python /app/.venv/bin/python && \\\n"
        '    /app/.venv/bin/python -c "import importlib.metadata as m; '
        "from packaging.version import Version; "
        "assert m.version('omninode-memory') == '0.18.0'; "
        "assert m.version('pyrage') == '1.4.0'; "
        "assert Version(m.version('cryptography')) >= Version('50.0.1'); "
        "assert Version(m.version('aiohttp')) >= Version('3.14.3'); "
        "assert Version(m.version('pyasn1')) >= Version('0.6.4'); "
        "assert Version(m.version('protobuf')) >= Version('5.29.6'); "
        "assert Version(m.version('setuptools')) >= Version('78.1.1'); "
        "assert Version(m.version('transformers')) >= Version('5.5.0')\"\n\n"
    )
    raw = _replace_once(
        raw, "# Verify torch is CPU-only", gate + "# Verify torch is CPU-only"
    )
    raw = _replace_once(
        raw,
        'ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint-runtime.sh"]',
        "COPY --chown=omniinfra:omniinfra docker/sim-preflight-local-profile.json "
        "/app/sim-preflight-local-profile.json\n"
        'ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint-runtime.sh"]',
    )
    return raw


def prepare(context: Path, *, wheel_bytes: bytes | None = None) -> dict[str, str]:
    dockerfile = context / "docker/Dockerfile.runtime"
    raw = dockerfile.read_bytes()
    if _sha256(raw) != CANONICAL_DOCKERFILE_SHA256:
        raise ValueError(
            "staged canonical Dockerfile.runtime differs from approved hash"
        )
    market = context / "workspace/sibling-repos/omnimarket"
    url = _locked_pyrage_url(market)
    if wheel_bytes is None:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            wheel_bytes = response.read()
    if _sha256(wheel_bytes) != PYRAGE_SHA256:
        raise ValueError("downloaded pyrage differs from Market lock hash")
    staged = _recipe(raw.decode("utf-8")).encode("utf-8")
    artifacts = context / "docker/sim-preflight-artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    (artifacts / PYRAGE_WHEEL).write_bytes(wheel_bytes)
    dockerfile.write_bytes(staged)
    manifest = {
        "profile": "sim-preflight-runtime-active-base-v1",
        "canonical_dockerfile_sha256": CANONICAL_DOCKERFILE_SHA256,
        "staged_dockerfile_sha256": _sha256(staged),
        "pyrage_source_kind": "market_lock_wheel",
        "pyrage_version": "1.4.0",
        "pyrage_wheel": PYRAGE_WHEEL,
        "pyrage_wheel_sha256": PYRAGE_SHA256,
    }
    (context / "docker/sim-preflight-local-profile.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.context), sort_keys=True))


if __name__ == "__main__":
    main()
