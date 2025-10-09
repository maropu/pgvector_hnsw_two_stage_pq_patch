# syntax=docker/dockerfile:1
#
# pgvector build w/ candidate pruning patch
#
# Requirements:
#   - Docker BuildKit (for `ADD <git-url>#<ref>`)
#
# Build (override versions via --build-arg as needed):
#   docker build \
#     --build-arg PG_MAJOR=17 \
#     --build-arg PGVECTOR_REF=v0.8.0 \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch \
#     -t pgvector-artifacts:pg17-v0.8.0 .
#
# Extract artifacts via `docker cp` (no container start required):
#   cid=$(docker create pgvector-artifacts:pg17-v0.8.0)
#   docker cp "$cid":/artifacts ./artifacts
#   docker rm "$cid"
#   ls -al ./artifacts   # contains lib/vector.so, extension/*.sql|.control, doc/*
#
# (Optional) Direct-to-local output via Buildx:
#   docker buildx build \
#     --build-arg PG_MAJOR=17 \
#     --build-arg PGVECTOR_REF=v0.8.0 \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch \
#     --target out \
#     --output type=local,dest=./dist \
#     .
#
# Notes:
#   - ABI compatibility: match the PostgreSQL MAJOR (e.g., 17.x) on the target host.
#   - Install locations follow `pg_config`:
#       * $libdir              -> vector.so
#       * $sharedir/extension  -> vector.control, vector--*.sql

ARG PG_MAJOR=17
ARG PGVECTOR_REF=v0.8.0
ARG PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch

# ---- Build stage (keeps ABI with target Postgres by using postgres:<major>-bookworm) ----
FROM postgres:${PG_MAJOR}-bookworm AS build
ARG PG_MAJOR
ARG PGVECTOR_REF
ARG PATCH_FILE

# Fetch source at the specified git ref (tag/SHA) using BuildKit remote git context
ADD https://github.com/pgvector/pgvector.git#${PGVECTOR_REF} /tmp/pgvector

# Copy the specified patch file from build context
COPY ${PATCH_FILE} /tmp/pgvector/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- Install build dependencies ---
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      postgresql-server-dev-"${PG_MAJOR}" \
      ca-certificates \
      patch; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# --- Prepare source workspace ---
WORKDIR /tmp/pgvector

# --- Patch (dry-run for diagnostics) ---
RUN set -eux; \
    patch -p1 --forward --dry-run < "${PATCH_FILE}"

# --- Apply patch ---
RUN set -eux; \
    patch -p1 --forward < "${PATCH_FILE}"

# --- Build extension ---
RUN set -eux; \
    make clean; \
    make USE_PGXS=1 OPTFLAGS=""

# --- Staged install into DESTDIR ---
RUN set -eux; \
    DESTDIR="/pkgroot"; \
    make USE_PGXS=1 install DESTDIR="${DESTDIR}"

# --- Collect artifacts into /artifacts ---
RUN set -eux; \
    PKGLIBDIR="$(pg_config --pkglibdir)"; \
    EXT_DIR="$(pg_config --sharedir)/extension"; \
    DESTDIR="/pkgroot"; \
    mkdir -p /artifacts/lib /artifacts/extension /artifacts/doc; \
    cp "${DESTDIR}${PKGLIBDIR}/vector.so" /artifacts/lib/; \
    cp "${DESTDIR}${EXT_DIR}/vector.control" /artifacts/extension/; \
    cp "${DESTDIR}${EXT_DIR}/vector--"*.sql /artifacts/extension/; \
    cp LICENSE README.md /artifacts/doc/

# --- Package tarball (avoid self-inclusion by writing to /tmp first) ---
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    ARTIFACT_NAME="pgvector-${PGVECTOR_REF}-pg${PG_MAJOR}-bookworm-${ARCH}.tar.gz"; \
    TMP_TARBALL="/tmp/${ARTIFACT_NAME}"; \
    tar -C /artifacts -czf "${TMP_TARBALL}" .; \
    mv "${TMP_TARBALL}" "/artifacts/${ARTIFACT_NAME}"

# --- Cleanup build residues ---
RUN set -eux; \
    rm -rf /tmp/pgvector "/pkgroot"; \
    apt-get update; \
    apt-get purge -y --auto-remove build-essential postgresql-server-dev-"${PG_MAJOR}" patch libpq-dev; \
    rm -rf /var/lib/apt/lists/*

# ---- Optional out stage (for Buildx local output) ----
FROM scratch AS out
COPY --from=build /artifacts /

# ---- Final artifacts image (default build target) ----
FROM debian:bookworm-slim AS artifacts
COPY --from=build /artifacts /artifacts
LABEL org.opencontainers.image.title="pgvector artifacts" \
      org.opencontainers.image.description="pgvector build outputs under /artifacts" \
      org.opencontainers.image.source="https://github.com/pgvector/pgvector"
