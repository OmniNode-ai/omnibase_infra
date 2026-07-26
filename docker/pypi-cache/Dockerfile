# PyPI pull-through cache (devpi-server) for the self-hosted runner fleet.
# Ticket: OMN-14027 (Layer-B egress caching) | Parent: OMN-13932
#
# STATUS: SHOVEL-READY / INERT. Nothing builds this image until an operator runs
# `docker compose -f docker/docker-compose.pypi-cache.yml up -d` at rollout, which
# is soak-gated (OMN-14027 is DESIGN-ONLY until the execution gate trips). See
# docs/runbooks/pypi-cache-egress-rollout.md.
#
# devpi's built-in `root/pypi` index is an on-demand pull-through mirror: the first
# runner to request a wheel fetches it from PyPI once; the cache then serves all 64
# runners from the LAN. This removes the redundant per-runner PyPI egress that
# saturates the single home uplink under concurrent merge_group load (OMN-14017).

# Base image is pinned by TAG here; freeze it to a @sha256 digest at rollout
# (see the runbook) to match the runner-image reproducibility discipline.
FROM python:3.12-slim

# devpi versions are range-pinned for the shovel-ready spec; freeze to exact
# versions at rollout together with the base-image digest.
ARG DEVPI_SERVER_SPEC="devpi-server>=6.9,<7"
ARG DEVPI_WEB_SPEC="devpi-web>=4.1,<5"

ENV DEVPI_SERVERDIR=/devpi/server \
    DEVPI_HOST=0.0.0.0 \
    DEVPI_PORT=3141

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "${DEVPI_SERVER_SPEC}" "${DEVPI_WEB_SPEC}"

RUN useradd --create-home --uid 1002 devpi \
    && mkdir -p /devpi \
    && chown -R devpi:devpi /devpi

COPY entrypoint.sh /usr/local/bin/pypi-cache-entrypoint.sh
RUN chmod +x /usr/local/bin/pypi-cache-entrypoint.sh

USER devpi
WORKDIR /devpi
EXPOSE 3141

ENTRYPOINT ["/usr/local/bin/pypi-cache-entrypoint.sh"]
