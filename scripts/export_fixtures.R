# export_fixtures.R
# Export R golden-test fixtures to language-neutral CSV for the Julia port.
# Base R only (no extra packages).

src_dir     <- "/tmp/censoredAIDS/tests/testthat"
params_dir  <- file.path(src_dir, "testing_params")
out_dir     <- "/Users/noejnava/Desktop/julia-mcp-censored-quaids/julia/CensoredDemand/test/fixtures"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## ---- Read source fixtures -------------------------------------------------
# Note: data file ships as testing_data.RDS (uppercase ext) in the source tree.
testing_data <- readRDS(file.path(src_dir, "testing_data.RDS"))
qshares      <- readRDS(file.path(src_dir, "qshares.rds"))
loglikes     <- readRDS(file.path(src_dir, "loglikes.rds"))

full_alpha <- readRDS(file.path(params_dir, "full_alpha.rds"))
full_beta  <- readRDS(file.path(params_dir, "full_beta.rds"))
full_gamma <- readRDS(file.path(params_dir, "full_gamma.rds"))
full_lamda <- readRDS(file.path(params_dir, "full_lamda.rds"))
full_theta <- readRDS(file.path(params_dir, "full_theta.rds"))
full_sigma <- readRDS(file.path(params_dir, "full_sigma.rds"))

## ---- (A) SHARES params (reproduces qshares.rds) ---------------------------
# NO sigma, theta NOT transposed. length 27.
params_shares <- c(full_alpha[1:3],
                   full_beta[1:3],
                   full_gamma[1:3, 1:3][upper.tri(full_gamma[1:3, 1:3], diag = TRUE)],
                   full_theta[1:4, 1:3],
                   full_lamda[1:3])

## ---- (B) LOGLIKE params (reproduces loglikes.rds) -------------------------
# theta IS transposed; sigma appended as chol upper-tri. length 33.
full_theta_t <- t(readRDS(file.path(params_dir, "full_theta.rds")))
full_sigma_c <- chol(readRDS(file.path(params_dir, "full_sigma.rds")))
params_loglike <- c(full_alpha[1:3],
                    full_beta[1:3],
                    full_gamma[1:3, 1:3][upper.tri(full_gamma[1:3, 1:3], diag = TRUE)],
                    full_theta_t[1:4, 1:3],
                    full_lamda[1:3],
                    full_sigma_c[upper.tri(full_sigma_c, diag = TRUE)])

## ---- Coerce outputs -------------------------------------------------------
testing_data_df <- as.data.frame(testing_data)
testing_data_df <- testing_data_df[, c("s1", "s2", "s3", "s4",
                                        "lnp1", "lnp2", "lnp3", "lnp4",
                                        "lnw", "age", "size", "sex", "educ")]

qshares_df <- as.data.frame(qshares)
colnames(qshares_df) <- c("SSB", "Juice", "Milk", "Water")

loglikes_df <- data.frame(loglike = as.numeric(loglikes))

params_shares_df  <- data.frame(value = as.numeric(params_shares))
params_loglike_df <- data.frame(value = as.numeric(params_loglike))

## ---- Full-precision writer ------------------------------------------------
# Every numeric value is written with sprintf("%.17g", v): 17 significant
# decimal digits. This is the standard shortest representation that uniquely
# identifies any IEEE-754 binary64 and round-trips exactly under a correctly
# rounded parser -- which is what the consumer (Julia's parse(Float64, .);
# likewise Python's float()) uses. We deliberately do NOT emit C99 hex floats
# ("%a"): a hex literal would not be parsed as a number by a generic CSV ->
# Float64 reader and would break the Julia port.
#
# NOTE: R's own as.numeric() is not correctly rounded, so it cannot be used to
# verify round-trip for ~2% of values (a 17-digit decimal can land on a
# neighbouring double under R's parser even though it is exact under a correct
# one). We therefore verify round-trip with Julia (see julia_roundtrip_check),
# the actual consumer.
fmt_full <- function(x) vapply(x, function(v) {
  if (is.na(v) || !is.finite(v)) return(as.character(v))
  sprintf("%.17g", v)
}, character(1))

# Path to the Julia round-trip verifier script (lives next to this R script).
verifier_path <- function()
  "/Users/noejnava/Desktop/julia-mcp-censored-quaids/scripts/verify_roundtrip.jl"

# Verify, using Julia's correctly-rounded parser, that every value in `path`
# parses back to the exact bit pattern in `df`. We pass the originals to Julia
# as exact hex-float literals (lossless) and compare against parse(Float64,csv).
julia_roundtrip_check <- function(df, path) {
  cols <- names(df)[vapply(df, is.numeric, logical(1))]
  if (length(cols) == 0) return(invisible(TRUE))
  hex <- as.data.frame(lapply(df[cols], function(col)
    vapply(col, function(v) if (is.na(v) || !is.finite(v)) "NaN" else sprintf("%a", v),
           character(1))), check.names = FALSE, stringsAsFactors = FALSE)
  # Only compare the numeric columns of `path`; write a numeric-only copy.
  num_only <- as.data.frame(lapply(df[cols], function(col)
    vapply(col, function(v) if (is.na(v) || !is.finite(v)) "NaN" else sprintf("%.17g", v),
           character(1))), check.names = FALSE, stringsAsFactors = FALSE)
  dec_path <- tempfile(fileext = ".csv")
  hex_path <- tempfile(fileext = ".csv")
  utils::write.csv(num_only, dec_path, row.names = FALSE, quote = FALSE)
  utils::write.csv(hex,      hex_path, row.names = FALSE, quote = FALSE)
  res <- system2("julia", c(verifier_path(), dec_path, hex_path),
                 stdout = TRUE, stderr = TRUE)
  unlink(c(dec_path, hex_path))
  if (!any(grepl("OK", res))) {
    stop(sprintf("Julia round-trip verification failed for %s:\n%s",
                 basename(path), paste(res, collapse = "\n")))
  }
  invisible(TRUE)
}

write_full_precision <- function(df, path) {
  out <- df
  for (nm in names(out)) {
    if (is.numeric(out[[nm]])) out[[nm]] <- fmt_full(out[[nm]])
  }
  # quote = FALSE: the %.17g strings contain no commas/quotes, so unquoted is
  # safe and keeps every value trivially parseable as a number by any reader.
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  julia_roundtrip_check(df, path)
}

## ---- Write CSVs -----------------------------------------------------------
write_full_precision(testing_data_df,  file.path(out_dir, "testing_data.csv"))
write_full_precision(params_shares_df, file.path(out_dir, "params_shares.csv"))
write_full_precision(params_loglike_df,file.path(out_dir, "params_loglike.csv"))
write_full_precision(qshares_df,       file.path(out_dir, "qshares.csv"))
write_full_precision(loglikes_df,      file.path(out_dir, "loglikes.csv"))

## ---- Write meta JSON (hand-built, base R only) ----------------------------
json <- paste0(
  "{\n",
  "  \"n_obs\": 615,\n",
  "  \"n_goods\": 4,\n",
  "  \"n_demographics\": 4,\n",
  "  \"share_names\": [\"SSB\", \"Juice\", \"Milk\", \"Water\"],\n",
  "  \"demographic_order\": [\"age\", \"size\", \"educ\", \"sex\"],\n",
  "  \"quaids\": true,\n",
  "  \"params_shares_len\": 27,\n",
  "  \"params_loglike_len\": 33,\n",
  "  \"note\": \"Demographics column order is [age, size, educ, sex] -- the data order used to build theta. The theta matrix columns follow this demographic ordering; loglike params transpose theta while shares params do not.\"\n",
  "}\n"
)
writeLines(json, file.path(out_dir, "fixtures_meta.json"))

## ---- Report --------------------------------------------------------------
cat("Wrote fixtures to:", out_dir, "\n")
cat("  testing_data.csv   :", nrow(testing_data_df), "x", ncol(testing_data_df), "\n")
cat("  params_shares.csv  : length", nrow(params_shares_df), "\n")
cat("  params_loglike.csv : length", nrow(params_loglike_df), "\n")
cat("  qshares.csv        :", nrow(qshares_df), "x", ncol(qshares_df), "\n")
cat("  loglikes.csv       : length", nrow(loglikes_df), " sum =", sum(loglikes_df$loglike), "\n")
cat("  fixtures_meta.json written.\n")
