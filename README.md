# Censored Demand Systems — AIDS & QUAIDS

Cloud + MCP infrastructure to estimate censored Almost Ideal (AIDS) **and**
Quadratic Almost Ideal (QUAIDS) demand systems by maximum likelihood, handling
zero-expenditure (censored) observations via the Wales–Woodland likelihood.

## Both AIDS and QUAIDS

This package supports **both** demand systems from a single estimator, selected
via the `quaids` flag:

- `quaids = false` → linear **AIDS** (Almost Ideal Demand System / AI)
- `quaids = true`  → quadratic **QUAIDS** (Quadratic Almost Ideal Demand System / QUAI)

The same censored log-likelihood, elasticity, and estimation routines serve both
specifications, so AIDS and QUAIDS results are produced by one consistent code path.

## Layout

- `julia/CensoredDemand/` — the Julia estimation package (`CensoredDemand.jl`).
- `mcp/` — MCP server fronting the estimator.
- `aws/` — cloud job + storage infrastructure.
- `scripts/` — orchestration and helper scripts.

## Status

Early scaffold. The Julia module loads and exposes stubs (`aids_shares`,
`censored_loglike`, `censored_elasticity`, `estimate`) that throw until
implemented. Method source of truth is the R `censoredAIDS` package; we port,
we do not reinvent.

## More

See [PLAN.md](PLAN.md) for the full project plan, architecture, and the win
condition (Julia results must match the original R package numerically).
