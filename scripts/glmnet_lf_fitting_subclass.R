library(tibble)
library(dplyr)
library(readr)
library(purrr)
library(MuMIn)
library(parallel)
library(glmnet)

require(doMC)
registerDoMC(cores = 4)

aicc_cvglmnet <- function (cvfit) {
    fit <- cvfit$glmnet.fit
    K <- fit$df + 1 # adding one for intercept term
    n <- fit$nobs
    aicc <- deviance(fit) + 2 * K + (2 * K * (K + 1)) / (n - K - 1)
    ind <- cvfit$index[1] # lambda.min
    aicc[ind]
}

loglik_cvglmnet <- function (cvfit) {
    fit <- cvfit$glmnet.fit
    ll <- -deviance(fit) / 2
    ind <- cvfit$index[1] # lambda.min
    ll[ind]
}

aicc_glmnet <- function (fit) {
    K <- fit$df + 1 # adding one for intercept term
    n <- fit$nobs
    deviance(fit) + 2 * K + (2 * K * (K + 1)) / (n - K - 1)
}

loglik_glmnet <- function (fit) {
    -deviance(fit) / 2
}

aicc_glm <- function (fit) {
    K <- attr(logLik(fit), "df")
    n <- attr(logLik(fit), "nobs")
    -2 * c(logLik(fit)) + 2 * K + (2 * K * (K + 1)) / (n - K - 1)
}

args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]
output_dir <- args[2]

min_subclass_count <- 10
min_targeting <- 4
cv_folds <- 5

subclass_fileparts <- list(
    "L2/3 IT" = "L23-IT",
    "L4 & L5 IT" = "L4-L5-IT",
    "L6 IT" = "L6-IT",
    "Car3" = "L5L6-IT-Car3",
    "L5 ET" = "L5-ET",
    "L6 CT" = "L6-CT",
    "L6b" = "L6b"
)

df_visp <- read_csv(data_file, show_col_types = FALSE)

# assign subclasses based on MET-types
df_visp <- df_visp |>
    mutate(
        subclass = case_when(
            predicted_met_type %in% c("L2/3 IT") ~ "L2/3 IT",
            predicted_met_type %in% c("L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5") ~ "L4 & L5 IT",
            predicted_met_type %in% c("L6 IT-1", "L6 IT-2", "L6 IT-3") ~ "L6 IT",
            predicted_met_type %in% c("L5/L6 IT Car3") ~ "Car3",
            predicted_met_type %in% c("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3") ~ "L5 ET",
            predicted_met_type %in% c("L6 CT-1", "L6 CT-2") ~ "L6 CT",
            predicted_met_type %in% c("L6b") ~ "L6b"
        )
    )

subclass_counts <- table(df_visp$subclass)
subclasses_to_keep <- names(subclass_counts)[subclass_counts >= min_subclass_count]
print(subclasses_to_keep)

region_cols <- df_visp |>
    select(starts_with("ipsi_") | starts_with("contra_")) |>
    colnames()

# Drop region of origin (VISp) and non-specific regions
drop_regions = c(
    "ipsi_VISp",
    "ipsi_fiber tracts",
    "contra_fiber tracts",
    "ipsi_TH",
    "ipsi_MB",
    "ipsi_P"
)
region_cols <- region_cols[!(region_cols %in% drop_regions)]

# ---- PER SUBCLASS -----
fit_models_subclass <- function (selected_subclass) {
    message(selected_subclass)
    df_subclass <- df_visp |> filter(subclass == selected_subclass)

    n_targeting_area <- colSums(select(df_subclass, all_of(region_cols)) > 0)
    areas_to_fit <- names(n_targeting_area[n_targeting_area >= min_targeting])


    # ----------------------------------------------
    # ------- Null model ---------------------------
    # ----------------------------------------------

    null_fits <- mclapply(
        areas_to_fit,
        function (region) {
            y <- df_subclass[, region] > 0
            null_mod <- glm(y ~ 1, family = binomial)
            null_mod
        }
    )
    names(null_fits) <- areas_to_fit

    # ----------------------------------------------
    # ------- Surface coordinates ------------------
    # ----------------------------------------------

    surf_cols <- df_subclass |> select(starts_with("surface")) |> colnames()
    model_formula <- reformulate(termlabels = surf_cols)
    X <- model.matrix(model_formula, df_subclass)[, -1]

    surf_fits <- lapply(
        areas_to_fit,
        function (region) {
            y <- df_subclass[, region] > 0
            cv.glmnet(X, y, family = binomial, alpha = 0, nfolds = cv_folds, parallel = TRUE)
        }
    )
    names(surf_fits) <- areas_to_fit

    # ----------------------------------------------
    # ------- Latent factors -----------------------
    # ----------------------------------------------

    lf_cols <- df_subclass |>
        select(starts_with("LF") &
        ends_with(subclass_fileparts[[selected_subclass]])) |>
        colnames()

    # Replace hyphens (cause problems with glm variable selection)
    lf_cols_rename <- gsub("-", "_", lf_cols)
    df_subclass_rename <- df_subclass |>
        rename_with(~ gsub("-", "_", .x, fixed = TRUE))

    model_formula <- reformulate(termlabels = lf_cols_rename)
    X <- model.matrix(model_formula, df_subclass_rename)[, -1, drop = FALSE]
	only_one_lf <- dim(X)[2] == 1

    lf_fits <- lapply(
        areas_to_fit,
        function (region) {
            y <- df_subclass[, region] > 0
            if (only_one_lf) {
                df_fit <- data.frame(X)
                df_fit$y <- y
                fit <- glm(y ~ ., data=df_fit, family = binomial)
            } else {
                fit <- cv.glmnet(X, y, family = binomial, alpha = 0,
                	nfolds = cv_folds, parallel = TRUE)
            }
            fit
        }
    )
    names(lf_fits) <- areas_to_fit


    # ----------------------------------------------
    # ------- Full model ---------------------------
    # ----------------------------------------------

    model_formula <- reformulate(termlabels = c(lf_cols_rename, surf_cols))
    X <- model.matrix(model_formula, df_subclass_rename)[, -1]

    full_fits <- lapply(
        areas_to_fit,
        function (region) {
            y <- df_subclass[, region] > 0
            cv.glmnet(X, y, family = binomial, alpha = 0,
            	nfolds = cv_folds, parallel = TRUE)
        }
    )
    names(full_fits) <- areas_to_fit

    # ----------------------------------------------
    # ------- Combine results ----------------------
    # ----------------------------------------------

    # Get AICc and logLik for all models
    null_results <- lapply(null_fits,
        function (fit) {
            data.frame(
                AICc = aicc_glm(fit),
                logLik = c(logLik(fit))
            )
        }
    )
    null_tbl <- null_results |>
        list_rbind(names_to = "region") |>
        as_tibble()
    null_tbl$model_type <- "null"

    null_coefs_tbl <- null_fits |>
        map(coef) |>
        map(t) |>
        map(as_tibble) |>
        list_rbind(names_to = "region")
    null_coefs_tbl$model_type <- "null"


    surf_results <- lapply(surf_fits,
        function (fit) {
            data.frame(
                AICc = aicc_cvglmnet(fit),
                logLik = loglik_cvglmnet(fit)
            )
        }
    )
    surf_tbl <- surf_results |>
        list_rbind(names_to = "region") |>
        as_tibble()
    surf_tbl$model_type <- "surface"

    surf_coefs_tbl <- surf_fits |>
        map(coef, s = "lambda.min") |>
        map(as.matrix) |>
        map(t) |>
        map(as_tibble) |>
        list_rbind(names_to = "region")
    surf_coefs_tbl$model_type <- "surface"


    lf_results <- lapply(lf_fits,
        function (fit) {
            if (inherits(fit, "glm")) {
                result <- data.frame(
                    AICc = aicc_glm(fit),
                    logLik = c(logLik(fit))
                )
            } else {
                result <- data.frame(
                    AICc = aicc_cvglmnet(fit),
                    logLik = loglik_cvglmnet(fit)
                )
            }
            result
        }
    )
    lf_tbl <- lf_results |>
        list_rbind(names_to = "region") |>
        as_tibble()
    lf_tbl$model_type <- "lf"

    lf_coefs_tbl <- lf_fits |>
        map(coef, s = "lambda.min") |>
        map(as.matrix) |>
        map(t) |>
        map(as_tibble) |>
        list_rbind(names_to = "region")
    lf_coefs_tbl$model_type <- "lf"


    full_results <- lapply(full_fits,
        function (fit) {
            data.frame(
                AICc = aicc_cvglmnet(fit),
                logLik = loglik_cvglmnet(fit)
            )
        }
    )
    full_tbl <- full_results |>
        list_rbind(names_to = "region") |>
        as_tibble()
    full_tbl$model_type <- "full"

    full_coefs_tbl <- full_fits |>
        map(coef, s = "lambda.min") |>
        map(as.matrix) |>
        map(t) |>
        map(as_tibble) |>
        list_rbind(names_to = "region")
    full_coefs_tbl$model_type <- "full"


	all_tbl <- bind_rows(null_tbl, surf_tbl, lf_tbl, full_tbl)
	all_coefs_tbl <- bind_rows(null_coefs_tbl, surf_coefs_tbl, lf_coefs_tbl, full_coefs_tbl)
	
	write_csv(all_tbl, file = file.path(output_dir,
		paste0(subclass_fileparts[[selected_subclass]], "_aicc_loglik.csv")))
	write_csv(all_coefs_tbl, file = file.path(output_dir,
		paste0(subclass_fileparts[[selected_subclass]], "_coef.csv")))
}
# fit_models_subclass(subclasses_to_keep[[2]])
subclasses_to_keep |> map(fit_models_subclass)
warnings()