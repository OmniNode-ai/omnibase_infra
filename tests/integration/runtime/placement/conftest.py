# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A throwaway single-member etcd and a TCP proxy that can black-hole it (OMN-21421).

The etcd binary is ``$ONEX_TEST_ETCD_BIN``, else ``etcd`` on PATH, else the
v3.7.3 release archive for this platform, downloaded once into
``~/.cache/onex-test-etcd`` and checked against the release's SHA256SUMS. A
failed download fails the tests; they are never skipped for a missing etcd.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import platform
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest

ETCD_VERSION = "v3.7.3"
_RELEASES = f"https://github.com/etcd-io/etcd/releases/download/{ETCD_VERSION}"


def _download_etcd() -> str:
    """Fetch and verify the etcd release binary for this platform; returns its path."""
    system = platform.system().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}[
        platform.machine().lower()
    ]
    stem = f"etcd-{ETCD_VERSION}-{system}-{arch}"
    archive = f"{stem}.zip" if system == "darwin" else f"{stem}.tar.gz"
    cache = Path.home() / ".cache" / "onex-test-etcd"
    binary = cache / stem / "etcd"
    if binary.is_file():
        return str(binary)
    cache.mkdir(parents=True, exist_ok=True)
    sums = httpx.get(f"{_RELEASES}/SHA256SUMS", follow_redirects=True, timeout=60)
    sums.raise_for_status()
    expected = next(
        line.split()[0]
        for line in sums.text.splitlines()
        if line.endswith(f" {archive}")
    )
    body = httpx.get(f"{_RELEASES}/{archive}", follow_redirects=True, timeout=300)
    body.raise_for_status()
    if hashlib.sha256(body.content).hexdigest() != expected:
        raise RuntimeError(f"{archive} does not match the release SHA256SUMS")
    member = f"{stem}/etcd"
    if archive.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(body.content)) as bundle:
            data = bundle.read(member)
    else:
        with tarfile.open(fileobj=io.BytesIO(body.content)) as bundle:
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"{archive} has no {member}")
            data = extracted.read()
    with tempfile.TemporaryDirectory(dir=cache) as staging:
        staged = Path(staging) / stem
        staged.mkdir()
        (staged / "etcd").write_bytes(data)
        (staged / "etcd").chmod(0o755)
        with contextlib.suppress(OSError):
            staged.rename(cache / stem)
    return str(binary)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class EtcdProcess:
    """One etcd member on loopback ports; ``stop`` and ``start`` keep its data."""

    def __init__(self, binary: str, data_dir: Path) -> None:
        self.binary = binary
        self.data_dir = data_dir
        self.client_port = _free_port()
        self.peer_port = _free_port()
        self.endpoint = f"http://127.0.0.1:{self.client_port}"
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        log = (self.data_dir.parent / "etcd.log").open("ab")
        self._proc = subprocess.Popen(
            [
                self.binary,
                "--name=placement-test",
                f"--data-dir={self.data_dir}",
                f"--listen-client-urls={self.endpoint}",
                f"--advertise-client-urls={self.endpoint}",
                f"--listen-peer-urls=http://127.0.0.1:{self.peer_port}",
                f"--initial-advertise-peer-urls=http://127.0.0.1:{self.peer_port}",
                f"--initial-cluster=placement-test=http://127.0.0.1:{self.peer_port}",
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{self.endpoint}/health", timeout=0.5).status_code == 200:
                    return
            time.sleep(0.1)
        raise RuntimeError("test etcd did not become healthy within 15 s")

    def stop(self) -> None:
        if self._proc is not None:
            os.killpg(self._proc.pid, signal.SIGTERM)
            self._proc.wait(timeout=10)
            self._proc = None


@pytest.fixture
def etcd(tmp_path: Path) -> Iterator[EtcdProcess]:
    binary = (
        os.environ.get("ONEX_TEST_ETCD_BIN") or shutil.which("etcd") or _download_etcd()
    )
    member = EtcdProcess(binary, tmp_path / "etcd-data")
    member.start()
    try:
        yield member
    finally:
        member.stop()


class BlackholeProxy:
    """Forwards TCP to etcd until ``cut``; then holds every connection silent."""

    def __init__(self, upstream_port: int) -> None:
        self._upstream_port = upstream_port
        self._cut = False
        self._writers: list[asyncio.StreamWriter] = []
        self._server: asyncio.Server | None = None
        self.port = 0

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = int(self._server.sockets[0].getsockname()[1])

    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(ConnectionError, asyncio.CancelledError):
            while not self._cut:
                chunk = await reader.read(65536)
                if not chunk or self._cut:
                    break
                writer.write(chunk)
                await writer.drain()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.append(writer)
        if self._cut:
            await asyncio.sleep(3600)
            return
        up_reader, up_writer = await asyncio.open_connection(
            "127.0.0.1", self._upstream_port
        )
        self._writers.append(up_writer)
        await asyncio.gather(
            self._pipe(reader, up_writer), self._pipe(up_reader, writer)
        )

    def cut(self) -> None:
        """Black-hole: drop every forwarded byte from now on, never answer."""
        self._cut = True

    async def close(self) -> None:
        for writer in self._writers:
            writer.close()
        if self._server is not None:
            self._server.close()


@pytest.fixture
async def blackhole_proxy(etcd: EtcdProcess) -> AsyncIterator[BlackholeProxy]:
    proxy = BlackholeProxy(etcd.client_port)
    await proxy.start()
    try:
        yield proxy
    finally:
        await proxy.close()
