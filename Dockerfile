# syntax=docker/dockerfile:1
# Reproducible build of refusion. Normally run via scripts/docker-build.sh:
#   build:  docker buildx build --target export --output type=local,dest=dist .
#   test:   docker buildx build --target test .
# Debian bookworm (glibc 2.36) so the binary runs on most current distros.

ARG RUST_VERSION=1.99.0
FROM rust:${RUST_VERSION}-bookworm AS base
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*

# ---- OpenCASCADE (cached until the build script changes)
FROM base AS occt
COPY scripts/build-occt.sh /tmp/build-occt.sh
RUN OCCT_PREFIX=/opt/occt OCCT_WORK=/tmp/occt sh /tmp/build-occt.sh \
 && mkdir -p /opt/occt/share/licenses \
 && cp /tmp/occt/src/OCCT-7_8_1/LICENSE_LGPL_21.txt /tmp/occt/src/OCCT-7_8_1/OCCT_LGPL_EXCEPTION.txt /opt/occt/share/licenses/ \
 && rm -rf /tmp/occt

# ---- refusion
FROM base AS build
COPY --from=occt /opt/occt /opt/occt
ENV OCCT_DIR=/opt/occt \
    CARGO_PROFILE_RELEASE_DEBUG=0 \
    CARGO_PROFILE_RELEASE_STRIP=true
WORKDIR /refusion
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/refusion/target \
    cargo build --release --locked -p refusion-app \
 && cp target/release/refusion /usr/local/bin/refusion

FROM build AS test
RUN rustup component add rustfmt
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/refusion/target \
    cargo fmt --all --check \
 && cargo test --workspace --locked

# ---- output: just the binary and examples (docker build --output)
FROM scratch AS export
COPY --from=build /usr/local/bin/refusion /refusion
COPY examples/ /examples/
COPY LICENSE /licenses/refusion-MIT.txt
COPY --from=occt /opt/occt/share/licenses/ /licenses/
