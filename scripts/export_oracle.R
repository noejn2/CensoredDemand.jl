# export_oracle.R
# Generate a MULTI-MODE golden oracle for the Julia port of aidsCalculate.
# Covers all four code paths (quaids x demos) plus random stress cases.
# Base R only (no extra packages).

src_dir    <- "/tmp/censoredAIDS/tests/testthat"
params_dir <- file.path(src_dir, "testing_params")
base_dir   <- "/Users/noejnava/Desktop/julia-mcp-censored-quaids/julia/CensoredDemand"
out_dir    <- file.path(base_dir, "test/fixtures/extra")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("/tmp/censoredAIDS/R/aidsCalculate.R")

## ---- Source data + real params --------------------------------------------
testing_data <- readRDS(file.path(src_dir, "testing_data.RDS"))
full_alpha <- readRDS(file.path(params_dir, "full_alpha.rds"))
full_beta  <- readRDS(file.path(params_dir, "full_beta.rds"))
full_gamma <- readRDS(file.path(params_dir, "full_gamma.rds"))
full_lamda <- readRDS(file.path(params_dir, "full_lamda.rds"))
full_theta <- readRDS(file.path(params_dir, "full_theta.rds"))

m <- 4L  # number of goods
t <- 4L  # number of demographics

Prices    <- as.matrix(testing_data[, c("lnp1", "lnp2", "lnp3", "lnp4")])
Budget    <- matrix(testing_data$lnw)
# Demographics column order [age, size, educ, sex] -- same as the M0 fixture.
Demos     <- as.matrix(testing_data[, c("age", "size", "educ", "sex")])
ShareNames <- c("SSB", "Juice", "Milk", "Water")

storage.mode(Prices) <- "double"
storage.mode(Budget) <- "double"
storage.mode(Demos)  <- "double"

## ---- Full-precision (verified round-trip) writer --------------------------
# Emit sprintf("%.17g", v): 17 significant digits uniquely identify any binary64
# and round-trip exactly under a correctly-rounded parser -- which is what the
# consumer (Julia's parse(Float64, .)) uses. We do NOT emit C99 hex floats, as a
# generic CSV -> Float64 reader would not parse them. R's as.numeric() is not
# correctly rounded, so round-trip is verified with Julia (the real consumer)
# via verify_roundtrip.jl, comparing %.17g vs exact %a representations.
verifier_path <- function()
  "/Users/noejnava/Desktop/julia-mcp-censored-quaids/scripts/verify_roundtrip.jl"

fmt_full <- function(x) vapply(x, function(v) {
  if (is.na(v) || !is.finite(v)) return(as.character(v))
  sprintf("%.17g", v)
}, character(1))

julia_roundtrip_check <- function(df, path) {
  cols <- names(df)[vapply(df, is.numeric, logical(1))]
  if (length(cols) == 0) return(invisible(TRUE))
  hex <- as.data.frame(lapply(df[cols], function(col)
    vapply(col, function(v) if (is.na(v) || !is.finite(v)) "NaN" else sprintf("%a", v),
           character(1))), check.names = FALSE, stringsAsFactors = FALSE)
  dec <- as.data.frame(lapply(df[cols], function(col)
    vapply(col, function(v) if (is.na(v) || !is.finite(v)) "NaN" else sprintf("%.17g", v),
           character(1))), check.names = FALSE, stringsAsFactors = FALSE)
  dec_path <- tempfile(fileext = ".csv"); hex_path <- tempfile(fileext = ".csv")
  utils::write.csv(dec, dec_path, row.names = FALSE, quote = FALSE)
  utils::write.csv(hex, hex_path, row.names = FALSE, quote = FALSE)
  res <- system2("julia", c(verifier_path(), dec_path, hex_path),
                 stdout = TRUE, stderr = TRUE)
  unlink(c(dec_path, hex_path))
  if (!any(grepl("OK", res)))
    stop(sprintf("Julia round-trip verification failed for %s:\n%s",
                 basename(path), paste(res, collapse = "\n")))
  invisible(TRUE)
}

write_full_precision <- function(df, path) {
  out <- df
  for (nm in names(out)) if (is.numeric(out[[nm]])) out[[nm]] <- fmt_full(out[[nm]])
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  julia_roundtrip_check(df, path)
}

## ---- Expected parameter lengths per mode ----------------------------------
nalpha <- m - 1                 # 3
nbeta  <- m - 1                 # 3
ngamma <- 0.5 * (m - 1) * m     # 6
ntheta <- (m - 1) * t           # 12
nlamda <- m - 1                 # 3
len_for <- function(quaids, demos)
  nalpha + nbeta + ngamma + (if (demos) ntheta else 0) + (if (quaids) nlamda else 0)

## ---- Build the "real" subset params for a given mode ----------------------
g3 <- full_gamma[1:3, 1:3]
gamma_vec <- g3[upper.tri(g3, diag = TRUE)]          # length 6
theta_vec <- as.numeric(full_theta[1:4, 1:3])        # length 12 (column-major)
real_params <- function(quaids, demos) {
  p <- c(full_alpha[1:3], full_beta[1:3], gamma_vec)
  if (demos)  p <- c(p, theta_vec)
  if (quaids) p <- c(p, full_lamda[1:3])
  as.numeric(p)
}

## ---- Run aidsCalculate for a mode + param vector --------------------------
run_case <- function(params, quaids, demos) {
  Dem <- if (demos) Demos else matrix()
  aidsCalculate(Prices = Prices, Budget = Budget, ShareNames = ShareNames,
                Demographics = Dem, Params = params, quaids = quaids)
}

## ---- Mode table -----------------------------------------------------------
modes <- list(
  list(key = "quaids_demos",   quaids = TRUE,  demos = TRUE),
  list(key = "nodemos",        quaids = FALSE, demos = TRUE),   # quaids=FALSE, demos=TRUE
  list(key = "quaids_nodemos", quaids = TRUE,  demos = FALSE),
  list(key = "linear_nodemos", quaids = FALSE, demos = FALSE)
)
# Friendlier explicit names that encode (quaids, demos) plainly.
mode_name <- function(quaids, demos) {
  q <- if (quaids) "quaids" else "linear"
  d <- if (demos)  "demos"  else "nodemos"
  paste0(q, "_", d)
}

manifest <- list()
files_written <- character(0)
add_manifest <- function(name, quaids, demos, plen, pfile, sfile) {
  manifest[[length(manifest) + 1]] <<- list(
    name = name, quaids = quaids, demos = demos, params_len = plen,
    params_csv = file.path("test/fixtures/extra", pfile),
    expected_shares_csv = file.path("test/fixtures/extra", sfile)
  )
}

write_case <- function(name, params, quaids, demos) {
  expected <- run_case(params, quaids, demos)
  stopifnot(nrow(expected) == 615, ncol(expected) == 4)
  pfile <- paste0(name, "_params.csv")
  sfile <- paste0(name, "_shares.csv")

  params_df <- data.frame(value = as.numeric(params))
  shares_df <- as.data.frame(expected)
  colnames(shares_df) <- ShareNames

  write_full_precision(params_df, file.path(out_dir, pfile))
  write_full_precision(shares_df, file.path(out_dir, sfile))

  add_manifest(name, quaids, demos, length(params), pfile, sfile)
  files_written <<- c(files_written, file.path(out_dir, pfile), file.path(out_dir, sfile))
  cat(sprintf("  %-22s quaids=%-5s demos=%-5s len=%2d  q[1,1]=%s\n",
              name, quaids, demos, length(params), sprintf("%.15g", expected[1, 1])))
}

## ---- Generate the 12 cases ------------------------------------------------
# i indexes modes 1..4; seeds for random cases are 101+i with i counting cases.
cat("Generating multi-mode oracle cases:\n")
rand_seed_counter <- 0L
for (md in modes) {
  q <- md$quaids; d <- md$demos
  nm <- mode_name(q, d)
  L  <- len_for(q, d)

  # (1) real case
  write_case(paste0(nm, "_real"), real_params(q, d), q, d)

  # (2,3) two random cases per mode: set.seed(101 + i)
  for (r in 1:2) {
    rand_seed_counter <- rand_seed_counter + 1L
    set.seed(101L + rand_seed_counter)
    rp <- runif(L, -0.05, 0.05)
    write_case(paste0(nm, "_rand", r), rp, q, d)
  }
}

## ---- Write manifest.json (hand-built, base R only) ------------------------
jbool <- function(b) if (isTRUE(b)) "true" else "false"
jstr  <- function(s) paste0("\"", s, "\"")
entries <- vapply(manifest, function(e) {
  paste0(
    "  {\n",
    "    \"name\": ", jstr(e$name), ",\n",
    "    \"quaids\": ", jbool(e$quaids), ",\n",
    "    \"demos\": ", jbool(e$demos), ",\n",
    "    \"params_len\": ", e$params_len, ",\n",
    "    \"params_csv\": ", jstr(e$params_csv), ",\n",
    "    \"expected_shares_csv\": ", jstr(e$expected_shares_csv), "\n",
    "  }"
  )
}, character(1))
manifest_json <- paste0("[\n", paste(entries, collapse = ",\n"), "\n]\n")
manifest_path <- file.path(out_dir, "manifest.json")
writeLines(manifest_json, manifest_path)
files_written <- c(files_written, manifest_path)

cat("\nWrote", length(manifest), "cases +", "manifest to:", out_dir, "\n")
cat("manifest:", manifest_path, "\n")
