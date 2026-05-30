#!/usr/bin/env Rscript
# =============================================================================
# make_elast_oracle.R
#
# Generate an R elasticity ORACLE for validating a Julia port of
# censoredAIDS::censoredElasticity.
#
# Design goal: NEAR-DETERMINISTIC cross-validation.
#   censoredElasticity uses Monte-Carlo simulation of demand-system errors
#   (mvtnorm::rmvnorm). We make the result reproducible in Julia by EXPORTING
#   the exact epsilons matrix R drew (reps x (m-1), pre-cbind). The Julia port
#   then INJECTS this same matrix instead of re-drawing, so the draw count and
#   RNG implementation become irrelevant to matching.
#
# Model: SSB 4-good QUAIDS (m = 4) with 4 demographics.
#   params = the 33-length params_loglike.csv (quaids = TRUE; last 6 = sigma).
#   Demographics in MODEL order [age, size, educ, sex]; func = mean.
#
# Only writes under .../test/fixtures/elast/ .
# =============================================================================

suppressWarnings(suppressMessages({
  library(mvtnorm)
  # matrixcalc is only used by the original for stopifnot() PSD/symmetry checks;
  # we load it if available but do not depend on it for the math.
  have_matrixcalc <- requireNamespace("matrixcalc", quietly = TRUE)
}))

# ---- Paths ------------------------------------------------------------------
R_SRC      <- "/tmp/censoredAIDS/R"
FIX_DIR    <- "/Users/noejnava/Desktop/julia-mcp-censored-quaids/julia/CensoredDemand/test/fixtures"
OUT_DIR    <- file.path(FIX_DIR, "elast")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Full %.17g precision writer (no scientific-notation truncation issues).
write_full <- function(x, path) {
  m <- as.matrix(x)
  con <- file(path, "w")
  on.exit(close(con))
  for (i in seq_len(nrow(m))) {
    writeLines(paste(formatC(m[i, ], format = "g", digits = 17), collapse = ","), con)
  }
}

# ---- Source ORIGINAL package functions --------------------------------------
# muaidsCalculate (the share-equation kernel used by the elasticity routine).
source(file.path(R_SRC, "muaidsCalculate.R"))
# The original censoredElasticity is sourced too, then OVERRIDDEN below so we
# can guarantee we are reproducing its exact body with instrumentation.
source(file.path(R_SRC, "censoredElasticity.R"))

# =============================================================================
# INSTRUMENTED COPY of censoredElasticity
#
# This is a faithful copy of the original body with these changes ONLY:
#   * reps        : 1e5 -> 10000   (cheap to export; injection makes count moot)
#   * epsilons    : captured to the global `CAPTURE` env (pre-cbind matrix)
#   * mu inputs   : muPrices / muBudget / muDemogs captured
#   * outputs     : E_Uobs, Elasticities, SE captured (unchanged math)
# Everything inside the math is byte-for-byte the same logic as the original.
# =============================================================================
CAPTURE <- new.env()

censoredElasticity_instrumented <- function(
    Prices = matrix(),
    Budget = matrix(),
    ShareNames = NULL,
    Demographics = matrix(),
    DemographicNames = NULL,
    Params = matrix(),
    quaids = FALSE,
    vcov = matrix(),
    func,
    ...) {

  reps  <- 10000   # <-- INSTRUMENTED: was 1e+5
  delta <- 1e-5
  demos <- !all(dim(Demographics) == c(1, 1))

  # ----: checks (matrixcalc-based, kept but made optional) :----
  if (have_matrixcalc) {
    stopifnot({matrixcalc::is.positive.semi.definite(vcov) &
               matrixcalc::is.symmetric.matrix(vcov)})
  }
  stopifnot({dim(vcov)[1] == length(Params)})

  m <- ncol(Prices)
  n <- nrow(Prices)
  t <- ncol(Demographics)
  j <- 0.5*(m - 1)*m

  nalpha <- m - 1
  nbeta  <- m - 1
  ngamma <- 0.5*(m - 1)*m
  if (demos) {ntheta <- (m - 1)*t}else{ntheta <- 0}
  if (quaids) {nlamda <- (m - 1)}else{nlamda <- 0}
  stopifnot({length(Params) == (nalpha + nbeta + ngamma + ntheta + nlamda + j)})

  # ----: point estimates :----
  muPrices <- apply(Prices, 2, func, ...)
  muBudget <- apply(Budget, 2, func, ...)
  muDemogs <- apply(Demographics, 2, func, ...)

  # ----: Sigma & error simulations :----
  Sigmma <- matrix(0, ncol = (m - 1), nrow = (m - 1))
  Sigmma[upper.tri(Sigmma, diag = TRUE)] <- Params[c((length(Params) - j + 1):length(Params))]
  Sigmma <- t(Sigmma) %*% Sigmma

  # >>> INSTRUMENTED: seed BEFORE the draw, exactly as instructed. <<<
  set.seed(424242)
  epsilons <- mvtnorm::rmvnorm(n = reps, sigma = Sigmma, method = "chol")
  # Capture the reps x (m-1) matrix BEFORE the cbind(-rowSums) step.
  CAPTURE$epsilons   <- epsilons
  CAPTURE$Sigmma     <- Sigmma
  CAPTURE$muPrices   <- muPrices
  CAPTURE$muBudget   <- muBudget
  CAPTURE$muDemogs   <- muDemogs

  epsilons <- cbind(epsilons, -rowSums(epsilons))

  # ----: Amemiya-Tobin truncation mapping :----
  treatTruncations <- function(x) {
    x[x < 0] <- 0
    rsums <- rowSums(x)
    apply(x, 2, function(L) L/rsums)
  }

  # ----: E(X,b) :----
  U <- muaidsCalculate(muPrices = muPrices,
                       muBudget = muBudget,
                       muDemographics = muDemogs,
                       Params = Params[-c((length(Params) - j + 1):length(Params))],
                       quaids = quaids, m = m, t = t, dems = demos)

  Ulat   <- t(apply(epsilons, 1, function(x) U + x))
  Uobs   <- treatTruncations(Ulat)
  E_Uobs <- apply(Uobs, 2, mean)

  # ----: E(X + delta, b) :----
  U_dx <- sapply(1:(1 + length(muPrices)), function(x) {
    if(x > length(muPrices)) {
      muBudget_delta <- exp(muBudget)
      muBudget_delta <- log(muBudget_delta + delta)
      muaidsCalculate(muPrices = muPrices, muBudget = muBudget_delta,
                      muDemographics = muDemogs,
                      Params = Params[-c((length(Params) - j + 1):length(Params))],
                      quaids = quaids, m = m, t = t, dems = demos)
    }else{
      muPrices_delta <- exp(muPrices)
      muPrices_delta[x] <- muPrices_delta[x] + delta
      muPrices_delta <- log(muPrices_delta)
      muaidsCalculate(muPrices = muPrices_delta, muBudget = muBudget,
                      muDemographics = muDemogs,
                      Params = Params[-c((length(Params) - j + 1):length(Params))],
                      quaids = quaids, m = m, t = t, dems = demos)
    }
  })
  U_dx <- t(U_dx)

  EUobs_dx <- lapply(split(U_dx,1:nrow(U_dx)), function(u) {
    Ulat_dx <- t(apply(epsilons, 1, function(e) u + e))
    Uobs_dx <- treatTruncations(Ulat_dx)
    apply(Uobs_dx, 2, mean)
  })

  # ----: E(X, b + delta) :----
  U_db <- sapply(1:length(Params[-c((length(Params) - j + 1):length(Params))]), function(x) {
    b_delta <- Params[-c((length(Params) - j + 1):length(Params))]
    b_delta[x] <- b_delta[x] + delta
    muaidsCalculate(muPrices = muPrices, muBudget = muBudget,
                    muDemographics = muDemogs, Params = b_delta,
                    quaids = quaids, m = m, t = t, dems = demos)
  })
  U_db <- t(U_db)

  EUobs_db <- lapply(split(U_db,1:nrow(U_db)), function(u) {
    Ulat_db <- t(apply(epsilons, 1, function(e) u + e))
    Uobs_db <- treatTruncations(Ulat_db)
    apply(Uobs_db, 2, mean)
  })
  EUobs_db <- do.call(rbind, EUobs_db)

  # ----: E(X + delta, b + delta) :----
  U_dxb <- lapply(1:(1 + length(muPrices)), function(x) {
    mat_out <- matrix(0, nrow = m, ncol = length(Params[-c((length(Params) - j + 1):length(Params))]))
    for(p in 1:length(Params[-c((length(Params) - j + 1):length(Params))])) {
      b_delta <- Params[-c((length(Params) - j + 1):length(Params))]
      b_delta[p] <- b_delta[p] + delta
      if(x > length(muPrices)) {
        muBudget_delta <- exp(muBudget)
        muBudget_delta <- log(muBudget_delta + delta)
        out <- muaidsCalculate(muPrices = muPrices, muBudget = muBudget_delta,
                               muDemographics = muDemogs, Params = b_delta,
                               quaids = quaids, m = m, t = t, dems = demos)
        mat_out[,p] <- out
      }else{
        muPrices_delta <- exp(muPrices)
        muPrices_delta[x] <- muPrices_delta[x] + delta
        muPrices_delta <- log(muPrices_delta)
        out <- muaidsCalculate(muPrices = muPrices_delta, muBudget = muBudget,
                               muDemographics = muDemogs, Params = b_delta,
                               quaids = quaids, m = m, t = t, dems = demos)
        mat_out[,p] <- out
      }
    }
    mat_out
  })

  EUobs_dxb <- lapply(1:(1 + length(muPrices)), function(x) {
    focus_dxb <- t(U_dxb[[x]])
    EUobs_dxb <- lapply(split(focus_dxb,1:nrow(focus_dxb)), function(l) {
      Ulat_dxb <- t(apply(epsilons, 1, function(xx) l + xx))
      Uobs_dxb <- treatTruncations(Ulat_dxb)
      apply(Uobs_dxb, 2, mean)
    })
    do.call(cbind, EUobs_dxb)
  })

  # ----: Elasticity calculations :----
  m_EUobs_dx <- do.call(cbind, EUobs_dx)
  etas <- sapply(1:m, function(i) {
    p1 <- (E_Uobs[i] - m_EUobs_dx[i,(m + 1)])/delta
    p2_num <- exp(muBudget) + .5*delta
    p2_den <- E_Uobs[i] + .5*(E_Uobs[i] - m_EUobs_dx[i,(m + 1)])
    p2 <- p2_num/p2_den
    income_eta <- p1*p2 + 1
    price_eta <- sapply(1:m, function(j) {
      p1 <- (E_Uobs[i] - m_EUobs_dx[i,j])/delta
      p2_num <- exp(muPrices[j]) + .5*delta
      p2_den <- E_Uobs[i] + .5*(E_Uobs[i] - m_EUobs_dx[i,j])
      p2 <- p2_num/p2_den
      p1*p2
    })
    return(c(price_eta, income_eta))
  })

  etas[1:m, 1:m] <- etas[1:m, 1:m] - diag(1, m)
  etas <- t(etas)
  colnames(etas) <- c(paste0("Price ", as.character(1:m)), "Income")
  rownames(etas) <- paste0("Quantity", as.character(1:m))
  if(!is.null(ShareNames)) {
    colnames(etas)[1:m] <- ShareNames
    rownames(etas) <- ShareNames
  }

  # ----: Standard Errors :----
  etas_SE <- lapply(1:m, function(i) {
    m_EUobs_dxb <- t(EUobs_dxb[[(m + 1)]])
    p1 <- (EUobs_db[,i] - m_EUobs_dxb[,i])/delta
    p2_num <- exp(muBudget) + .5*delta
    p2_den <- EUobs_db[,i] + .5*(EUobs_db[,i] - m_EUobs_dxb[,i])
    p2 <- p2_num/p2_den
    SE_income_eta <- p1*p2 + 1
    SE_price_eta <- sapply(1:m, function(j) {
      m_EUobs_dxb <- t(EUobs_dxb[[j]])
      p1 <- (EUobs_db[,i] - m_EUobs_dxb[,i])/delta
      p2_num <- exp(muPrices[j]) + .5*delta
      p2_den <- EUobs_db[,i] + .5*(EUobs_db[,i] - m_EUobs_dxb[,i])/delta
      p2 <- p2_num/p2_den
      if(i == j) {p1*p2 - 1} else {p1*p2}
    })
    return(cbind(SE_price_eta, SE_income_eta))
  })

  expand_rVector <- function(v, M) matrix(v[col(M)], nrow = nrow(M), ncol = ncol(M))

  etas_SE <- lapply(1:length(etas_SE), function(x) {
    dy <- etas_SE[[x]]
    f <- expand_rVector(etas[x,], etas_SE[[x]])
    J <- (dy - f) / delta
    vcov_dims <- length(Params[-c((length(Params) - j + 1):length(Params))])
    etavcov <- t(J) %*% (vcov[1:vcov_dims,1:vcov_dims]/n) %*% J
    sqrt(diag(etavcov))
  })

  etas_SE <- do.call(rbind, etas_SE)
  colnames(etas_SE) <- c(paste0("Price ", as.character(1:m)), "Income")
  rownames(etas_SE) <- paste0("Quantity", as.character(1:m))
  if(!is.null(ShareNames)) {
    colnames(etas_SE)[1:m] <- ShareNames
    rownames(etas_SE) <- ShareNames
  }

  list(Elasticities = etas, SE = etas_SE, E_Uobs = E_Uobs)
}

# =============================================================================
# DATA SETUP
# =============================================================================
dat <- read.csv(file.path(FIX_DIR, "testing_data.csv"))

# Prices (logged) and budget (logged), in good order 1..4 = SSB,Juice,Milk,Water.
Prices <- as.matrix(dat[, c("lnp1","lnp2","lnp3","lnp4")])
Budget <- as.matrix(dat[, "lnw", drop = FALSE])

# Demographics in MODEL order [age, size, educ, sex] (NOT the CSV column order,
# which is age,size,sex,educ). The theta params follow [age,size,educ,sex].
Demographics <- as.matrix(dat[, c("age","size","educ","sex")])
colnames(Demographics) <- c("age","size","educ","sex")

# Params = full 33-length params_loglike (27 model params + 6 sigma).
Params <- read.csv(file.path(FIX_DIR, "params_loglike.csv"))$value
stopifnot(length(Params) == 33)

# ---- Deterministic PSD 33x33 vcov ------------------------------------------
set.seed(7)
A    <- matrix(runif(33*33), 33, 33)
vcov <- (A + t(A)) / 2 + 33 * diag(33)   # symmetric positive-definite

ShareNames <- c("SSB","Juice","Milk","Water")

# =============================================================================
# RUN
# =============================================================================
res <- censoredElasticity_instrumented(
  Prices       = Prices,
  Budget       = Budget,
  ShareNames   = ShareNames,
  Demographics = Demographics,
  Params       = Params,
  quaids       = TRUE,
  vcov         = vcov,
  func         = mean,
  na.rm        = TRUE
)

# =============================================================================
# EXPORT  (all at full %.17g precision)
# =============================================================================
# epsilons: reps x (m-1) = 10000 x 3  (the pre-cbind draws, the SAME matrix R used)
write_full(CAPTURE$epsilons, file.path(OUT_DIR, "epsilons.csv"))

# vcov: 33 x 33
write_full(vcov, file.path(OUT_DIR, "vcov.csv"))

# elasticities: m x (m+1) = 4 x 5 (4 price cols + 1 income col)
write_full(res$Elasticities, file.path(OUT_DIR, "elasticities_R.csv"))

# E_Uobs: length m = 4 (column vector)
write_full(matrix(res$E_Uobs, ncol = 1), file.path(OUT_DIR, "e_uobs_R.csv"))

# SE: 4 x 5 (same shape as elasticities)
write_full(res$SE, file.path(OUT_DIR, "se_R.csv"))

# mu inputs actually used (column means): muPrices(4), muBudget(1), muDemogs(4)
mu_inputs <- data.frame(
  name  = c("muPrice_SSB","muPrice_Juice","muPrice_Milk","muPrice_Water",
            "muBudget",
            "muDemog_age","muDemog_size","muDemog_educ","muDemog_sex"),
  value = c(CAPTURE$muPrices, CAPTURE$muBudget, CAPTURE$muDemogs)
)
con <- file(file.path(OUT_DIR, "mu_inputs.csv"), "w")
writeLines("name,value", con)
for (i in seq_len(nrow(mu_inputs))) {
  writeLines(paste0(mu_inputs$name[i], ",",
                    formatC(mu_inputs$value[i], format = "g", digits = 17)), con)
}
close(con)

# ---- meta.json --------------------------------------------------------------
meta_lines <- c(
  "{",
  '  "description": "R elasticity oracle for validating a Julia port of censoredAIDS::censoredElasticity (SSB 4-good QUAIDS).",',
  '  "model": "QUAIDS",',
  '  "m": 4,',
  '  "n_demographics": 4,',
  '  "n_obs": 615,',
  '  "quaids": true,',
  '  "reps": 10000,',
  '  "seed_epsilons": 424242,',
  '  "epsilon_draw": "set.seed(424242); mvtnorm::rmvnorm(n=reps, sigma=Sigma, method=\\"chol\\"); epsilons.csv is the reps x (m-1) = 10000 x 3 matrix BEFORE cbind(-rowSums). Julia validates near-deterministically by INJECTING this exact matrix (then appending the -rowSums column itself).",',
  '  "Sigma_construction": "Sigma is (m-1)x(m-1)=3x3; upper.tri(diag=TRUE) filled with the last j=6 params (the sigma block of params_loglike), then Sigma = t(S) %*% S.",',
  '  "vcov_construction": "set.seed(7); A = matrix(runif(33*33),33,33); vcov = (A + t(A))/2 + 33*diag(33). Symmetric positive-definite 33x33. SE path uses the leading 27x27 block (vcov[1:27,1:27]) divided by n=615.",',
  '  "delta": 1e-05,',
  '  "func": "mean (column means, na.rm=TRUE)",',
  '  "demographic_order_model": ["age", "size", "educ", "sex"],',
  '  "demographic_order_note": "MODEL order is [age,size,educ,sex]; the raw testing_data.csv column order is [age,size,sex,educ]. muDemogs and the theta params both follow the MODEL order.",',
  '  "good_order": ["SSB", "Juice", "Milk", "Water"],',
  '  "params": "33-length params_loglike.csv; first 27 are model params (alpha[3], beta[3], gamma[6], theta[12], lambda[3]) passed to muaidsCalculate; last 6 are the sigma block used to build Sigma.",',
  '  "elasticities_layout": {',
  '    "shape": "4 x 5",',
  '    "rows": "quantity i in good order [SSB,Juice,Milk,Water] (uncompensated own/cross response of quantity i)",',
  '    "cols": ["Price SSB", "Price Juice", "Price Milk", "Price Water", "Income"],',
  '    "note": "Columns 1..m are price elasticities (own-price on the diagonal, with the identity correction etas[1:m,1:m]-diag(1,m) already applied); column m+1 is the income/expenditure elasticity. Matrix is returned transposed so rows index quantities."',
  '  },',
  '  "se_layout": "Same 4 x 5 shape and same row/col ordering as elasticities_R.csv; delta-method standard errors using vcov[1:27,1:27]/n.",',
  '  "e_uobs_layout": "length m = 4 column vector, good order [SSB,Juice,Milk,Water]; mean of Amemiya-Tobin-corrected observed shares over reps draws.",',
  '  "mu_inputs_layout": "name,value rows: muPrice_{SSB,Juice,Milk,Water}, muBudget, muDemog_{age,size,educ,sex} (the column means actually used as the evaluation point).",',
  '  "files": ["epsilons.csv", "vcov.csv", "elasticities_R.csv", "e_uobs_R.csv", "se_R.csv", "mu_inputs.csv", "meta.json"],',
  paste0('  "R_version": "', R.version.string, '",'),
  paste0('  "mvtnorm_version": "', as.character(packageVersion("mvtnorm")), '"'),
  "}"
)
writeLines(meta_lines, file.path(OUT_DIR, "meta.json"))

# ---- Console report ---------------------------------------------------------
cat("\n===== ELASTICITIES (4x5) =====\n")
print(res$Elasticities)
cat("\n===== E_Uobs (length 4) =====\n")
print(res$E_Uobs)
cat("\n===== SE (4x5) =====\n")
print(res$SE)
cat("\n===== mu inputs =====\n")
print(mu_inputs)
cat("\nvcov dims:", paste(dim(vcov), collapse = " x "), "\n")
cat("epsilons dims:", paste(dim(CAPTURE$epsilons), collapse = " x "), "\n")
cat("\nFiles written to:", OUT_DIR, "\n")
print(list.files(OUT_DIR))
