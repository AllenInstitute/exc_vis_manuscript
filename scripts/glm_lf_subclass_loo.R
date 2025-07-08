library(tibble)
library(dplyr)
library(readr)
library(purrr)
library(MuMIn)
library(parallel)
library(glmnet)
library(data.table)

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


subclass_fileparts <- list(
    "L2/3 IT" = "L23-IT",
    "L4 & L5 IT" = "L4-L5-IT",
    "L6 IT" = "L6-IT",
    "Car3" = "L5L6-IT-Car3",
    "L5 ET" = "L5-ET",
    "L6 CT" = "L6-CT",
    "L6b" = "L6b"
)

cv_folds <- 5


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



subclasses_for_loo_fits <- c("L2/3 IT", "L5 ET", "L4 & L5 IT", "L6 CT")

fit_null <- function(region, df_train) {
    y <- df_train[, region] > 0
    null_mod <- glm(y ~ 1, family = binomial)
    null_mod
}

fit_surf <- function(region, df_train) {
    surf_cols <- df_train |> select(starts_with("surface")) |> colnames()
    model_formula <- reformulate(termlabels = surf_cols)
    X <- model.matrix(model_formula, df_train)[, -1]

    y <- df_train[, region] > 0
    cv.glmnet(X, y, family = binomial,
        alpha = 0, nfolds = cv_folds, parallel = TRUE)
}

fit_lf <- function(region, df_train, lf_cols) {
    # Replace hyphens (cause problems with glm variable selection)
    lf_cols_rename <- gsub("-", "_", lf_cols)
    df_train_rename <- df_train |>
        rename_with(~ gsub("-", "_", .x, fixed = TRUE))
    model_formula <- reformulate(termlabels = lf_cols_rename)
    X <- model.matrix(model_formula, df_train_rename)[, -1, drop = FALSE]
    only_one_lf <- dim(X)[2] == 1

    y <- df_train[, region] > 0
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

fit_full <- function(region, df_train, lf_cols) {
    surf_cols <- df_train |> select(starts_with("surface")) |> colnames()
    lf_cols_rename <- gsub("-", "_", lf_cols)
    df_train_rename <- df_train |>
        rename_with(~ gsub("-", "_", .x, fixed = TRUE))
    model_formula <- reformulate(termlabels = c(lf_cols_rename, surf_cols))
    X <- model.matrix(model_formula, df_train_rename)[, -1]

    y <- df_train[, region] > 0
    cv.glmnet(X, y, family = binomial,
    	alpha=0, nfolds = cv_folds, parallel = TRUE)
}

loo_fits_for_subclass <- function(selected_subclass) {
    df_subclass <- df_visp |> filter(subclass == selected_subclass)
    print(selected_subclass)
    filepart <- subclass_fileparts[[selected_subclass]]
    print(filepart)
    results_filename <- file.path(output_dir,
        paste0(subclass_fileparts[[selected_subclass]], "_aicc_loglik.csv"))
    results <- read_csv(results_filename)

    regions_fit <- results$region |> unique()

    loo_results <- lapply(regions_fit, function(r) {
        print(r)
        # find best model by AICc
        best_model <- results |>
            filter(region == r) |>
            slice_min(order_by = AICc) |>
            pull(model_type)
        print(best_model)
        region_loo_results <- lapply(1:dim(df_subclass)[1],
            function(test_ind) {
                df_train <- df_subclass[-test_ind, ]

                null_fit <- fit_null(r, df_train)
                if (best_model == "surface") {
                	if (test_ind == 1) {
                		message("fitting surface model")
                	}
                    fit <- fit_surf(r, df_train)

                    surf_cols <- df_train |>
                        select(starts_with("surface")) |>
                        colnames()
                    model_formula <- reformulate(termlabels = surf_cols)
                    test_X <- model.matrix(model_formula,
                        df_subclass[test_ind, ])[, -1]
                } else if (best_model == "lf") {
                	if (test_ind == 1) {
                		message("fitting latent factor model")
                	}
                    lf_cols <- df_train |>
                        select(starts_with("LF") &
                        ends_with(filepart)) |>
                        colnames()

                    fit <- fit_lf(r, df_train, lf_cols)

                    # Replace hyphens (cause problems with variable selection)
                    lf_cols_rename <- gsub("-", "_", lf_cols)
                    df_test_rename <- df_subclass[test_ind, ] |>
                        rename_with(~ gsub("-", "_", .x, fixed = TRUE))
                    model_formula <- reformulate(termlabels = lf_cols_rename)
                    test_X <- model.matrix(model_formula,
                        df_test_rename)[, -1, drop = FALSE]
                } else if (best_model == "full") {
                	if (test_ind == 1) {
                		message("fitting full model")
                	}
                    surf_cols <- df_train |>
                        select(starts_with("surface")) |>
                        colnames()
                    lf_cols <- df_train |>
                        select(starts_with("LF") &
                        ends_with(filepart)) |>
                        colnames()

                    fit <- fit_full(r, df_train, lf_cols)

                    # Replace hyphens (cause problems with variable selection)
                    lf_cols_rename <- gsub("-", "_", lf_cols)
                    df_test_rename <- df_subclass[test_ind, ] |>
                        rename_with(~ gsub("-", "_", .x, fixed = TRUE))
                    model_formula <- reformulate(termlabels = c(lf_cols_rename, surf_cols))
                    test_X <- model.matrix(model_formula,
                        df_test_rename)[, -1]
                } else {
                	if (test_ind == 1) {
                		message("using null model")
                	}
                
                    fit <- null_fit
                }

                # calculate logLik of held-out point
                resp <- df_subclass[test_ind, r] > 0
                prob <- predict(null_fit,
                    newdata = df_subclass[test_ind, ],
                    type = "response")
                ll_null <- sum(log(prob * resp + (1 - prob) * (1 - resp)))

                if (inherits(fit, "glm")) {
                    if (best_model == "lf") {
                        prob <- predict(fit,
                            newdata = data.frame(test_X),
                            type = "response")
                    } else {
                        prob <- predict(fit,
                            newdata = df_subclass[test_ind, ],
                            type = "response")
                    }
                } else {
                    prob <- predict(fit,
                        newx = test_X,
                        s = "lambda.min",
                        type = "response")
                }
                ll_fit <- sum(log(prob * resp + (1 - prob) * (1 - resp)))

                list(
                    specimen_id = df_subclass$`...1`[test_ind],
                    region = r,
                    pred_prob = prob,
                    ll_null = ll_null,
                    ll_fit = ll_fit
                )
        })
        region_loo_results |> rbindlist() |> as_tibble()
    })

    loo_tbl <- loo_results |>
        list_rbind() |>
        as_tibble()
    write_csv(loo_tbl, file = file.path(output_dir,
        paste0(subclass_fileparts[[selected_subclass]], "_loo_results.csv")))
}

subclasses_for_loo_fits |> map(loo_fits_for_subclass)
