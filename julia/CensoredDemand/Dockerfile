# syntax=docker/dockerfile:1
#
# CensoredDemand.jl — headless job-runner image.
#
# Usage (config + data are MOUNTED, not baked into the image):
#
#   docker build -t censored-demand .
#   docker run --rm -v "$PWD/work:/work" censored-demand /work/config.json
#
# bin/run.jl reads a JSON config from the path given as the first arg (a path
# inside the container, typically under the /work mount) and writes its results
# JSON to wherever the config says — also under /work so it lands on the host.
#
# Pin a precise patch in production if you want fully reproducible rebuilds
# (e.g. julia:1.11.6-bookworm). 1.11 tracks the latest 1.11.x.
FROM julia:1.11-bookworm

# Fail fast and loud inside RUN steps.
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /opt/CensoredDemand

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# - JULIA_NUM_THREADS=auto  -> use all available cores for the M5 threading.
#   (Override at runtime with `docker run -e JULIA_NUM_THREADS=N ...` or by
#    constraining the container with --cpus.)
# - JULIA_DEPOT_PATH        -> pin the depot so the precompile cache produced in
#   the build layers is the same one used at runtime.
# - JULIA_PROJECT           -> default active project (entrypoint also sets it
#   explicitly via --project for belt-and-suspenders).
# - JULIA_CPU_TARGET=generic-> portable native code so the image runs across a
#   range of host CPUs (drop this for a same-host build to squeeze out perf).
ENV JULIA_NUM_THREADS=auto \
    JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PROJECT=/opt/CensoredDemand \
    JULIA_CPU_TARGET=generic

# ---------------------------------------------------------------------------
# 1. Dependency layer (cached) — copy only the env files, then instantiate.
#    This layer is re-used on every rebuild that doesn't change Project/Manifest.
# ---------------------------------------------------------------------------
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ---------------------------------------------------------------------------
# 2. Source + entrypoint, then PRECOMPILE the package itself so the first
#    `docker run` does not pay the compile cost. `using CensoredDemand` forces
#    the package + its full dependency graph to be compiled and cached into the
#    pinned depot above.
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY bin/ ./bin/
RUN julia --project=. -e 'using Pkg; Pkg.precompile()' \
 && julia --project=. -e 'using CensoredDemand'

# ---------------------------------------------------------------------------
# Entrypoint: pass a config path as the docker-run argument, e.g.
#   docker run --rm -v "$PWD/work:/work" censored-demand /work/config.json
# ---------------------------------------------------------------------------
ENTRYPOINT ["julia", "--project=/opt/CensoredDemand", "/opt/CensoredDemand/bin/run.jl"]
