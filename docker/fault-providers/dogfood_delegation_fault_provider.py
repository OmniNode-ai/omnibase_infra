#!/usr/bin/env python3
"""No-secret deterministic OpenAI-compatible provider faults for dogfood only."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

_SUPPORTED_STATUSES: Final = frozenset({429, 503})


def parse_status(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", type=int, choices=sorted(_SUPPORTED_STATUSES), required=True
    )
    return parser.parse_args(argv).status


def error_payload(status: int) -> dict[str, object]:
    if status not in _SUPPORTED_STATUSES:
        raise ValueError(f"unsupported deterministic fault status: {status}")
    return {
        "error": {
            "code": status,
            "message": f"dogfood deterministic provider fault: HTTP {status}",
            "status": "RESOURCE_EXHAUSTED" if status == 429 else "UNAVAILABLE",
        }
    }


def handler_for(status: int) -> type[BaseHTTPRequestHandler]:
    class FaultHandler(BaseHTTPRequestHandler):
        def _write_json(self, response_status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"code": 404}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"code": 404}})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length:
                self.rfile.read(content_length)
            self._write_json(status, error_payload(status))

        def log_message(self, _format: str, *args: object) -> None:
            return

    return FaultHandler


def main(argv: list[str] | None = None) -> None:
    status = parse_status(argv)
    ThreadingHTTPServer(("0.0.0.0", 8080), handler_for(status)).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
