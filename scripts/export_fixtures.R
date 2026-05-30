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

## ---- Write CSVs -----------------------------------------------------------
write.csv(testing_data_df,  file.path(out_dir, "testing_data.csv"),   row.names = FALSE)
write.csv(params_shares_df, file.path(out_dir, "params_shares.csv"),  row.names = FALSE)
write.csv(params_loglike_df,file.path(out_dir, "params_loglike.csv"), row.names = FALSE)
write.csv(qshares_df,       file.path(out_dir, "qshares.csv"),        row.names = FALSE)
write.csv(loglikes_df,      file.path(out_dir, "loglikes.csv"),       row.names = FALSE)

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
