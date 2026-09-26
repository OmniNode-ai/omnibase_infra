#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# playwright-smoke.sh -- prove the omnibase-ci runner image launches
# Playwright's Chromium with nothing installed at job time (OMN-19206, OMN-19659).
#
# A fleet job has no passwordless package manager, so `playwright install
# --with-deps` cannot run there. Whatever Chromium needs must already be in the
# image. This script is the one place that proves it, and it is run in two:
#
#   * runner-image-build-smoke.yml, inside the image the PR just built, as the
#     unprivileged runner user. The script arrives on stdin (`bash -s`), because
#     fleet runners reach docker through a mounted socket and a bind mount of a
#     workspace path names a directory the daemon's host does not have.
#   * runner-playwright-canary.yml, on a live runner recreated onto a new image,
#     before the rest of the fleet is rolled.
#
# What it does, in order, all without root:
#   1. Refuse to run as root -- a root run would prove nothing about a job.
#   2. On an image at or above FONTS_BAKED_IMAGE_VERSION, require every font
#      family Playwright's jammy install-deps set registers, read through
#      fontconfig by name (not by package), and the Xvfb binary.
#   3. Install the Python Playwright client and its browser into a temp dir.
#   4. Launch headless Chromium and shape a DOWNLOADED web font (an @font-face
#      data URL built from a system font file). That is the path that aborted
#      the browser on SIGTRAP when the image had no usable font set (OMN-18165);
#      a plain page never reaches it.
#   5. On a font-carrying image, launch headed Chromium on a private Xvfb display.
#
# Exit 0 on PASS; any failure exits non-zero with a ::error:: line naming it.
set -euo pipefail

PLAYWRIGHT_VERSION="${PLAYWRIGHT_VERSION:-1.63.0}"
FONTS_BAKED_IMAGE_VERSION=10

fail() {
  echo "::error::playwright smoke: $*" >&2
  exit 1
}

[[ "$(id -u)" != "0" ]] || fail "running as root; run as the runner user, which is what a job runs as"

image_version="${OMNI_RUNNER_IMAGE_VERSION:-0}"
[[ "${image_version}" =~ ^[0-9]+$ ]] || image_version=0
require_fonts=0
if (( image_version >= FONTS_BAKED_IMAGE_VERSION )); then
  require_fonts=1
fi
echo "runner=${RUNNER_NAME:-<none>} host=$(hostname) image_version=${image_version} require_fonts=${require_fonts} playwright=${PLAYWRIGHT_VERSION}"

if (( require_fonts )); then
  missing=()
  for family in "FreeSans" "IPAGothic" "Liberation Sans" "Noto Color Emoji" "Loma" "Unifont" "WenQuanYi Zen Hei"; do
    found="$(fc-list ":family=${family}" family)"
    if [[ -n "${found}" ]]; then
      echo "font family resolves: ${family}"
    else
      missing+=("${family}")
    fi
  done
  (( ${#missing[@]} == 0 )) || fail "image v${image_version} lacks font families: ${missing[*]}"
  command -v Xvfb >/dev/null || fail "image v${image_version} has no Xvfb binary"
  echo "font files visible to fontconfig: $(fc-list | wc -l)"
fi

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
export PLAYWRIGHT_BROWSERS_PATH="${work}/browsers"

uv venv --quiet --python python3.12 "${work}/venv"
uv pip install --quiet --python "${work}/venv/bin/python" "playwright==${PLAYWRIGHT_VERSION}"
if (( require_fonts )); then
  # the full build, because step 5 launches headed
  "${work}/venv/bin/python" -m playwright install chromium
else
  "${work}/venv/bin/python" -m playwright install --only-shell chromium
fi

font_file="$(fc-match -f '%{file}' sans-serif)"
[[ -r "${font_file}" ]] || fail "fc-match returned no readable sans-serif font file (${font_file:-empty})"
echo "web font source: ${font_file}"

if (( require_fonts )); then
  display=":$(( 90 + RANDOM % 900 ))"
  Xvfb "${display}" -screen 0 1280x800x24 -nolisten tcp &
  xvfb_pid=$!
  trap 'kill "${xvfb_pid}" 2>/dev/null || true; rm -rf "${work}"' EXIT
  export SMOKE_DISPLAY="${display}"
  sleep 1
  kill -0 "${xvfb_pid}" 2>/dev/null || fail "Xvfb did not start on ${display}"
fi

SMOKE_FONT_FILE="${font_file}" SMOKE_HEADED="${require_fonts}" "${work}/venv/bin/python" - <<'PY'
import base64
import os
import sys

from playwright.sync_api import sync_playwright

font = base64.b64encode(open(os.environ["SMOKE_FONT_FILE"], "rb").read()).decode()
page_html = f"""<!doctype html><html><head><style>
@font-face {{ font-family: SmokeWebFont; src: url(data:font/ttf;base64,{font}); }}
body {{ font-family: SmokeWebFont, sans-serif; font-size: 24px; }}
</style></head><body>
<p id="latin">The runner renders text</p>
<p>emoji \U0001F680 ✅</p><p>CJK 中文 日本語</p><p>Thai ภาษาไทย</p>
</body></html>"""


def run(headless: bool) -> None:
    mode = "headless" if headless else "headed"
    env = dict(os.environ)
    if not headless:
        env["DISPLAY"] = os.environ["SMOKE_DISPLAY"]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, env=env)
        page = browser.new_page()
        page.set_content(page_html)
        loaded = page.evaluate(
            """async () => {
                await document.fonts.load("24px SmokeWebFont");
                await document.fonts.ready;
                return [...document.fonts]
                    .filter(f => f.family === "SmokeWebFont")
                    .map(f => f.status);
            }"""
        )
        if loaded != ["loaded"]:
            sys.exit(f"::error::{mode}: @font-face web font status {loaded}, expected ['loaded']")
        shot = page.screenshot(full_page=True)
        if len(shot) < 1000:
            sys.exit(f"::error::{mode}: screenshot is {len(shot)} bytes")
        print(f"{mode} chromium {browser.version}: web font loaded, screenshot {len(shot)} bytes")
        browser.close()


run(headless=True)
if os.environ.get("SMOKE_HEADED") == "1":
    run(headless=False)
PY

echo "Playwright smoke PASSED: Chromium launched with no package installed at job time."
