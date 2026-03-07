# GitHub Actions self-hosted runner image for OmniNode CI
# Ticket: OMN-3275 / Epic: OMN-3273
#
# NOT pre-baked: ruff, mypy — repos pin via pyproject.toml; workflows call `uv run ruff`.
# Pre-baking causes tool drift. Install at workflow time via `uv run`.

ARG RUNNER_VERSION=2.323.0
ARG GH_VERSION=2.67.0
ARG KUBECTL_VERSION=1.32.1
ARG UV_VERSION=0.6.2

# Pinned digest for reproducible builds — update via: docker pull ubuntu:22.04
FROM ubuntu:22.04@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751

ARG RUNNER_VERSION
ARG GH_VERSION
ARG KUBECTL_VERSION
ARG UV_VERSION

# Version labels — read by runner-status skill
LABEL org.omninode.runner.version="${RUNNER_VERSION}"
LABEL org.omninode.gh.version="${GH_VERSION}"
LABEL org.omninode.kubectl.version="${KUBECTL_VERSION}"
LABEL org.omninode.uv.version="${UV_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNER_HOME=/home/runner/actions-runner

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    tar \
    unzip \
    libicu-dev \
    libkrb5-3 \
    zlib1g \
    libssl3 \
    lsb-release \
    gnupg \
    apt-transport-https \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 via deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Install uv
RUN curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv \
    && chmod +x /usr/local/bin/uv

# Install Docker CLI (no daemon — socket mounted at runtime)
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install gh CLI
RUN curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin --strip-components=2 "gh_${GH_VERSION}_linux_amd64/bin/gh" \
    && chmod +x /usr/local/bin/gh

# Install kubectl
RUN curl -fsSLo /usr/local/bin/kubectl \
    "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl

# Create non-root runner user and add to docker group for socket access
RUN groupadd --gid 1001 runner \
    && useradd --uid 1001 --gid runner --shell /bin/bash --create-home runner \
    && groupadd --force docker \
    && usermod -aG docker runner

# Download and SHA256-verify GitHub Actions runner binary
# SHA256 for actions-runner-linux-x64-2.323.0.tar.gz
ENV RUNNER_SHA256=0dbc9bf5a58620fc52cb6cc0448abcca964a8d74b5f39773b7afcad9ab691e19

RUN mkdir -p "${RUNNER_HOME}" \
    && cd "${RUNNER_HOME}" \
    && curl -fsSLo runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && echo "${RUNNER_SHA256}  runner.tar.gz" | sha256sum --check --strict \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && chown -R runner:runner "${RUNNER_HOME}"

# Credential cache directory (for entrypoint.sh re-registration cache)
RUN mkdir -p /home/runner/.runner-creds && chown runner:runner /home/runner/.runner-creds

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Install gosu for privilege de-escalation in entrypoint.
# The entrypoint starts as root to fix Docker socket GID, then drops to runner.
RUN curl -fsSLo /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/1.17/gosu-amd64" \
    && chmod +x /usr/local/bin/gosu \
    && gosu nobody true

# Run as root — entrypoint fixes Docker socket GID then drops to runner via gosu
WORKDIR ${RUNNER_HOME}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
