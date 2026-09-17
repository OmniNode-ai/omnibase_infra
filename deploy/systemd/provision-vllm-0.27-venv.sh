#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# provision-vllm-0.27-venv.sh — idempotent toolchain repairs for the venv that
# deploy/systemd/vllm-gpu0-qwen38-27b.service runs from (OMN-18570, 2026-09-17).
#
#   sudo -u jonah bash deploy/systemd/provision-vllm-0.27-venv.sh
#   sudo -u jonah bash deploy/systemd/provision-vllm-0.27-venv.sh --check
#
# WHAT IS REPO-DECLARED AND WHAT IS NOT — read this before assuming the venv is
# reproducible from this repository. It is NOT. /opt/vllm-0.27/venv on .201 was
# built by hand: no repo path creates it, installs vLLM 0.27.1 into it, or pins
# its torch/transformers/flashinfer versions, and this script does not create it
# either. What this script declares is the two TOOLCHAIN REPAIRS that stand
# between a hand-built venv and a vLLM that can actually compile a kernel on
# sm120 — the two things that were fixed by hand on 2026-09-17 and that a later
# pip operation silently undoes. Everything else about that venv remains
# undeclared; that gap is real and is stated here rather than papered over.
#
# Run this AFTER any pip operation in the venv, not only at first install. Both
# repairs are undone by ordinary package management, with no error at the time:
# the failure surfaces later, at kernel-compile time, as a message that does not
# name the cause.
#
# ---------------------------------------------------------------------------
# REPAIR 1 — pin nvidia-nvvm to 13.0.88
# ---------------------------------------------------------------------------
# cicc (the NVVM compiler, shipped in nvidia-nvvm) and ptxas (shipped in
# nvidia-cuda-nvcc, which cuda-toolkit pins) must agree on the PTX ISA version.
# With nvidia-nvvm 13.4.92 installed beside the 13.0.x toolkit, cicc emitted
# PTX ISA 9.4 and ptxas accepted at most 9.0, so every FlashInfer JIT compile
# failed. cuda-toolkit 13.0.3.0 pins nvidia-nvvm==13.0.88.*, and an
# unconstrained `pip install -U` in this venv silently reintroduces the skew by
# pulling nvvm forward while the rest of the toolkit stays put.
#
# ---------------------------------------------------------------------------
# REPAIR 2 — the two CUDA library symlinks FlashInfer links against
# ---------------------------------------------------------------------------
# FlashInfer's build links against <cu13>/lib64 and against an unversioned
# libcudart.so. The pip-installed nvidia-cuda-runtime wheel provides neither:
# it ships lib/ (not lib64/) and libcudart.so.13 (no unversioned alias). Two
# symlinks close the gap. A reinstall or upgrade of nvidia-cuda-runtime
# replaces that directory and wipes both.
#
# Neither repair is a workaround for a bug in this repo's code. Both are
# packaging gaps in the upstream wheels, made durable here so the next person
# who runs pip in this venv does not spend the afternoon rediscovering them.
set -euo pipefail

VENV="${VLLM_027_VENV:-/opt/vllm-0.27/venv}"
NVVM_PIN="nvidia-nvvm==13.0.88"
SITE_PACKAGES="${VENV}/lib/python3.12/site-packages"
CU13="${SITE_PACKAGES}/nvidia/cu13"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
fi

fail() {
    echo "provision-vllm-0.27-venv: $*" >&2
    exit 1
}

[[ -x "${VENV}/bin/python" ]] || fail "no interpreter at ${VENV}/bin/python. This script repairs an existing venv; it does not build one. See the header."
[[ -d "${CU13}/lib" ]] || fail "no ${CU13}/lib — nvidia-cuda-runtime is not installed in ${VENV}."

# --- Repair 1: nvidia-nvvm pin -------------------------------------------
installed_nvvm="$("${VENV}/bin/pip" show nvidia-nvvm 2>/dev/null | awk '/^Version:/ {print $2}')"
if [[ "${installed_nvvm}" == "13.0.88" ]]; then
    echo "ok   nvidia-nvvm ${installed_nvvm} (pinned, matches ptxas PTX ISA <= 9.0)"
elif [[ "${CHECK_ONLY}" == "1" ]]; then
    fail "nvidia-nvvm is '${installed_nvvm:-absent}', expected 13.0.88. cicc would emit a PTX ISA ptxas cannot read. Re-run without --check."
else
    echo "fix  installing ${NVVM_PIN} (was '${installed_nvvm:-absent}')"
    "${VENV}/bin/pip" install --no-deps "${NVVM_PIN}"
fi

# --- Repair 2: lib64 -> lib, libcudart.so -> libcudart.so.13 --------------
ensure_symlink() {
    local link="$1" target="$2"
    if [[ -L "${link}" && "$(readlink "${link}")" == "${target}" ]]; then
        echo "ok   ${link} -> ${target}"
        return 0
    fi
    if [[ "${CHECK_ONLY}" == "1" ]]; then
        fail "${link} does not point at ${target}. FlashInfer's link step would fail. Re-run without --check."
    fi
    [[ -e "${link}" && ! -L "${link}" ]] && fail "${link} exists and is not a symlink; refusing to replace a real file."
    echo "fix  ${link} -> ${target}"
    ln -sfn "${target}" "${link}"
}

ensure_symlink "${CU13}/lib64" "lib"
ensure_symlink "${CU13}/lib/libcudart.so" "libcudart.so.13"

echo "provision-vllm-0.27-venv: done (${VENV})"
