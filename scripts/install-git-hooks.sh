#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# install-git-hooks.sh — Install onex-git-hook-relay git hooks into a repo.
#
# Usage:
#   ./scripts/install-git-hooks.sh                  # Install into current repo
#   ./scripts/install-git-hooks.sh /path/to/repo    # Install into target repo
#
# Installs the following hooks:
#   - pre-commit: Emits a git hook event before every commit
#   - post-receive: Emits a git hook event after every push (server-side)
#
# Requirements:
#   - onex-git-hook-relay must be installed and on PATH
#   - GITHUB_TOKEN, KAFKA_BOOTSTRAP_SERVERS should be set in environment
#
# Related Tickets:
#   - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)

set -euo pipefail

REPO_DIR="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
HOOKS_DIR="${REPO_DIR}/.git/hooks"

if [ ! -d "${HOOKS_DIR}" ]; then
    echo "ERROR: ${HOOKS_DIR} does not exist. Is ${REPO_DIR} a git repository?" >&2
    exit 1
fi

# Determine repo identity from git remote
get_repo_identity() {
    local remote_url
    remote_url=$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null || echo "")
    if [ -z "${remote_url}" ]; then
        echo "unknown/unknown"
        return
    fi
    # Normalize: extract {owner}/{name} from SSH or HTTPS remote URL
    # SSH:   git@github.com:OmniNode-ai/omniclaude.git
    # HTTPS: https://github.com/OmniNode-ai/omniclaude.git
    echo "${remote_url}" \
        | sed -E 's|.*github\.com[:/]||' \
        | sed -E 's|\.git$||'
}

REPO_IDENTITY=$(get_repo_identity)
echo "Installing git hooks for repo: ${REPO_IDENTITY}"

# ─── pre-commit hook ────────────────────────────────────────────────────────

PRE_COMMIT_HOOK="${HOOKS_DIR}/pre-commit"

# Preserve existing pre-commit hook if present
if [ -f "${PRE_COMMIT_HOOK}" ] && ! grep -q "onex-git-hook-relay" "${PRE_COMMIT_HOOK}" 2>/dev/null; then
    echo "Appending onex-git-hook-relay to existing pre-commit hook"
    cat >> "${PRE_COMMIT_HOOK}" << 'HOOK_APPEND'

# onex-git-hook-relay — emit pre-commit event to ONEX event bus
_ONEX_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
# Prefer GITHUB_USER env var or git config github.user (GitHub username, not PII)
_ONEX_AUTHOR="${GITHUB_USER:-$(git config github.user 2>/dev/null || echo "")}"
if [ -z "${_ONEX_AUTHOR}" ]; then _ONEX_AUTHOR="unknown"; fi
if command -v onex-git-hook-relay >/dev/null 2>&1; then
    onex-git-hook-relay emit \
        --hook pre-commit \
        --repo "REPO_IDENTITY_PLACEHOLDER" \
        --branch "${_ONEX_BRANCH}" \
        --author "${_ONEX_AUTHOR}" \
        --outcome pass \
        --gates-json '[]' \
        2>/dev/null || true
fi
HOOK_APPEND
    # Replace placeholder with actual repo identity
    sed -i.bak "s|REPO_IDENTITY_PLACEHOLDER|${REPO_IDENTITY}|g" "${PRE_COMMIT_HOOK}"
    rm -f "${PRE_COMMIT_HOOK}.bak"
else
    cat > "${PRE_COMMIT_HOOK}" << HOOK_EOF
#!/usr/bin/env bash
# pre-commit hook — managed by onex-git-hook-relay (OMN-2656)
set -euo pipefail

_ONEX_BRANCH=\$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
# Prefer GITHUB_USER env var or git config github.user (GitHub username, not PII)
_ONEX_AUTHOR="\${GITHUB_USER:-\$(git config github.user 2>/dev/null || echo "")}"
if [ -z "\${_ONEX_AUTHOR}" ]; then _ONEX_AUTHOR="unknown"; fi

if command -v onex-git-hook-relay >/dev/null 2>&1; then
    onex-git-hook-relay emit \\
        --hook pre-commit \\
        --repo "${REPO_IDENTITY}" \\
        --branch "\${_ONEX_BRANCH}" \\
        --author "\${_ONEX_AUTHOR}" \\
        --outcome pass \\
        --gates-json '[]' \\
        2>/dev/null || true
fi
HOOK_EOF
fi

chmod +x "${PRE_COMMIT_HOOK}"
echo "  ✓ pre-commit hook installed"

# ─── post-receive hook (server-side) ────────────────────────────────────────

POST_RECEIVE_HOOK="${HOOKS_DIR}/post-receive"

if [ -f "${POST_RECEIVE_HOOK}" ] && ! grep -q "onex-git-hook-relay" "${POST_RECEIVE_HOOK}" 2>/dev/null; then
    cat >> "${POST_RECEIVE_HOOK}" << 'HOOK_APPEND'

# onex-git-hook-relay — emit post-receive event to ONEX event bus
while IFS=' ' read -r _old_rev _new_rev _refname; do
    _ONEX_BRANCH=$(echo "${_refname}" | sed 's|refs/heads/||')
    if command -v onex-git-hook-relay >/dev/null 2>&1; then
        onex-git-hook-relay emit \
            --hook post-receive \
            --repo "REPO_IDENTITY_PLACEHOLDER" \
            --branch "${_ONEX_BRANCH}" \
            --author "${REMOTE_USER:-unknown}" \
            --outcome pass \
            --gates-json '[]' \
            2>/dev/null || true
    fi
done
HOOK_APPEND
    sed -i.bak "s|REPO_IDENTITY_PLACEHOLDER|${REPO_IDENTITY}|g" "${POST_RECEIVE_HOOK}"
    rm -f "${POST_RECEIVE_HOOK}.bak"
else
    cat > "${POST_RECEIVE_HOOK}" << HOOK_EOF
#!/usr/bin/env bash
# post-receive hook — managed by onex-git-hook-relay (OMN-2656)

while IFS=' ' read -r _old_rev _new_rev _refname; do
    _ONEX_BRANCH=\$(echo "\${_refname}" | sed 's|refs/heads/||')
    if command -v onex-git-hook-relay >/dev/null 2>&1; then
        onex-git-hook-relay emit \\
            --hook post-receive \\
            --repo "${REPO_IDENTITY}" \\
            --branch "\${_ONEX_BRANCH}" \\
            --author "\${REMOTE_USER:-unknown}" \\
            --outcome pass \\
            --gates-json '[]' \\
            2>/dev/null || true
    fi
done
HOOK_EOF
fi

chmod +x "${POST_RECEIVE_HOOK}"
echo "  ✓ post-receive hook installed"

echo ""
echo "Hooks installed successfully for ${REPO_IDENTITY}"
echo "Events will be published to: onex.evt.git.hook.v1"
echo "Spool path (Kafka unavailable): ~/.onex/spool/git-hooks.jsonl"
