library(dplyr)
library(glmnet)
library(progress)
library(parallel)

elastic_rrr <- function(x, y, rank = 2, lambdas = c(), nlambda = 20,
    lambdas_full = c(), alpha = 0.5, max_iter = 100, nlambda_for_path = 100,
    sparsity = "row-wise", tol = 1e-6, verbose = 0) {

    if (length(lambdas) > 0) {
        nlambda <- length(lambdas)
    }

    if (rank >= ncol(y)) {
        stop("Error: Rank must be smaller than the number of features")
    }

    # analytic situation for alpha = 0 case (pure ridge)
    if ((alpha == 0) && (length(lambdas) > 0)) {
        analytic_solution <- function(lambdau) {
            s <- svd(x)
            u <- s$u
            d <- s$d
            v <- s$v

            b <- v %*%
                diag(d / (d^2 + lambdau * nrow(as.matrix(x))),
                    nrow = length(d)) %*%
                t(u) %*% as.matrix(y)

            s <- svd(as.matrix(x) %*% b)
            w <- b %*% s$v[, 1:rank, drop = FALSE]
            v <- s$v[, 1:rank, drop = FALSE]

            pos <- apply(abs(v), MARGIN = 2, FUN = which.max)
            if (length(pos) > 1) {
                flips <- sign(mapply(
                    function(i, j) v[i, j],
                    pos, seq_len(ncol(v))))
                v <- v * t(matrix(rep(flips, nrow(v)), nrow = length(flips)))
                w <- w * t(matrix(rep(flips, nrow(w)), nrow = length(flips)))
            } else {
                flips <- sign(pos)
                v <- v * flips
                w <- w * flips
            }

            return(list(w = w, v = v, lambda = lambdau))
        }

        results <- lapply(lambdas, analytic_solution)
        return(results)
    }

    # Initialize with PLS direction
    s <- svd(t(x) %*% as.matrix(y))
    v <- s$v[, 1:rank]

    # Select lambdas if not pre-specified
    if (length(lambdas) == 0) {
        lambda_selection <- select_lambdas(x, y, rank = rank, nlambda = nlambda,
            alpha = alpha, nlambda_for_path = nlambda_for_path,
            sparsity = sparsity)
        lambdas <- lambda_selection$lambdas
        lambdas_full <- lambda_selection$lambdas_full
    }

    results <- mclapply(lambdas,
        function(l) {
            loss <- vector("numeric", max_iter)

            for (i in seq_len(max_iter)) {
                if (rank == 1) {
                    fit <- glmnet(
                        x = as.matrix(x),
                        y = as.matrix(y) %*% v,
                        alpha = alpha,
                        lambda = lambdas_full,
                        standardize = FALSE,
                        intercept = FALSE
                    )
                    w <- coef(fit, s = l, exact = FALSE)
                } else {
                    if (sparsity == "row-wise") {
                        fit <- glmnet(
                            x = as.matrix(x),
                            y = as.matrix(y) %*% v,
                            family = "mgaussian",
                            alpha = alpha,
                            standardize = FALSE,
                            standardize.response = FALSE,
                            intercept = FALSE
                        )
                        w <- coef(fit, s = l, exact = FALSE)
                    } else {
                        w <- lapply(seq_len(rank),
                            function(j) {
                                fit <- glmnet(
                                    x = as.matrix(x),
                                    y = as.matrix(y) %*% v[, j],
                                    alpha = alpha,
                                    lambda = lambdas_full,
                                    standardize = FALSE,
                                    intercept = FALSE
                                )
                                coef(fit, s = l, exact = FALSE)
                                }
                            )
                    }
                    w <- do.call(cbind, w)
                }

                if (all(w == 0)) {
                    if (verbose > 0) {
                        message("all zeros")
                        print(loss[loss > 0])
                    }
                    v <- v * 0
                    return(list(w = w, v = v, lambda = l))
                }
                # drop intercept row
                w <- w[-1, ]
                a <- t(y) %*% as.matrix(x) %*% w
                s <- svd(a)
                v <- s$u %*% t(s$v)
                pos <- apply(abs(v), MARGIN = 2, FUN = which.max)
                if (length(pos) > 1) {
                    flips <- sign(mapply(
                        function(i, j) v[i, j],
                        pos, seq_len(ncol(v))))
                    v <- v * t(matrix(
                        rep(flips, nrow(v)), nrow = length(flips)))
                    w <- w * t(matrix(
                        rep(flips, nrow(w)), nrow = length(flips)))
                } else {
                    flips <- sign(pos)
                    v <- v * flips
                    w <- w * flips
                }

                loss[i] <- sum((as.matrix(y) -
                    as.matrix(x) %*% w %*% t(v)) ^ 2) /
                    sum(as.matrix(y) ^ 2)
                if ((i > 1) && (abs(loss[i] - loss[i - 1]) < tol)) {
                    if (verbose > 0) {
                        message(paste0("converged in ", i - 1, " iterations"))
                        print(loss[loss > 0])
                    }
                    return(list(w = w, v = v, lambda = l))
                }
                if ((i > 1) && (loss[i] > 2 * loss[i - 1])) {
                    message("Jump in loss value ", loss[i], " ", loss[i - 1])
                    print("old w colsum")
                    print(colSums(old_w))
                    print("new w colsum")
                    print(colSums(w))
                    print("old v")
                    print(old_v)
                    print("new v")
                    print(v)
                }
                old_v <- v
                old_w <- w
            }
            if (verbose > 0) {
                message("Did not converge. Losses:")
                print(loss)
            }
            return(list(w = w, v = v, lambda = l))
        },
        mc.cores = detectCores() / 2)
    return(results)
}


elastic_rrr_cv <- function(x, y, alphas = c(0.2, 0.5, 0.9), lambdas = c(),
    nlambda = NULL, reps = 1, folds = 10, rank = 2, seed = 42,
    sparsity = "row-wise", lambda_relaxed = NULL, preset_foldid = NULL) {
    if (is.null(nlambda)) {
        nlambda <- length(lambdas)
    }
    nalpha <- length(alphas)

    n <- nrow(x)
    r2 <- array(NA, dim = c(reps, folds, nalpha, nlambda))
    r2_relaxed <- array(NA, dim = c(reps, folds, nalpha, nlambda))
    corrs <- array(NA, dim = c(reps, folds, nalpha, nlambda, rank))
    corrs_relaxed <- array(NA, dim = c(reps, folds, nalpha, nlambda, rank))
    nonzero <- array(NA, dim = c(reps, folds, nalpha, nlambda))
    lambdas_used <- array(NA, dim = c(nalpha, nlambda))
    foldid <- array(NA, dim = c(reps, n))

    set.seed(seed)
    for (r in seq_len(reps)) {
        if (is.null(preset_foldid)) {
            this_foldid <- sample(rep(seq(folds), length = n))
        } else {
            this_foldid <- preset_foldid[r, ]
        }
        foldid[r, ] <- this_foldid
        # Determine lambda set for eqch alpha
        message("determining lambdas for alphas")
        lambdas_for_alphas <- lapply(
            alphas,
            select_lambdas,
            x = x,
            y = y,
            rank = rank,
            nlambda = nlambda,
            sparsity = sparsity
        )

        for (i in seq_len(length(lambdas_for_alphas))) {
            lambdas_used[i, ] <- lambdas_for_alphas[[i]]$lambdas
        }

        message("performing cross-validation")
        pb_fold <- progress_bar$new(
            format = "folds [:bar] :current/:total in :elapsed",
            total = folds,
            show_after = 0)
        pb_fold$tick(0)
        for (cvfold in seq_len(folds)) {
            indtest <- which(this_foldid == cvfold)
            indtrain <- seq_len(n)[-indtest]
            x_train <- as.matrix(x[indtrain, , drop = FALSE])
            y_train <- as.matrix(y[indtrain, , drop = FALSE])
            x_test <- as.matrix(x[indtest, , drop = FALSE])
            y_test <- as.matrix(y[indtest, , drop = FALSE])

            # (not implementing the preprocess function)

            # mean centering
            x_mean <- colMeans(x_train)
            x_train <- x_train - x_mean
            x_test <- x_test - x_mean
            y_mean <- colMeans(y_train)
            y_train <- y_train - y_mean
            y_test <- y_test - y_mean
            for (j in seq_len(nalpha)) {
                alpha <- alphas[j]
                results <- elastic_rrr(
                    x_train,
                    y_train,
                    alpha = alpha,
                    lambdas = lambdas_for_alphas[[j]]$lambdas,
                    lambdas_full = lambdas_for_alphas[[j]]$lambdas_full,
                    rank = rank,
                    sparsity = sparsity
                )
                # Loop through results for different lambdas
                i <- 1
                for (res in results) {
                    vx <- as.matrix(res$w)
                    vy <- as.matrix(res$v)
                    lambda <- res$lambda
                    nz <- rowSums(abs(vx)) != 0
                    if (sum(nz) < rank) {
                        i <- i + 1
                        next
                    }

                    if (any(is.na(apply(x_test %*% vx, MARGIN = 2, sd))) ||
                        all(near(apply(x_test %*% vx, MARGIN = 2, sd), 0))) {
                        i <- i + 1
                        next
                    }

                    nonzero[r, cvfold, j, i] <- sum(nz)
                    r2[r, cvfold, j, i] <- 1 -
                        sum((y_test - x_test %*% vx %*% t(vy)) ^ 2) /
                        sum(y_test ^ 2)
                    for (rk in seq_len(rank)) {
                        corrs[r, cvfold, j, i, rk] <- cor(
                            x_test %*% vx[, rk],
                            y_test %*% vy[, rk]
                        )
                    }
                    if (is.null(lambda_relaxed)) {
                        result_relaxed <- elastic_rrr(
                            x_train[, nz],
                            y_train,
                            lambdas = c(lambda),
                            alpha = 0,
                            rank = rank,
                            sparsity = sparsity
                        )
                    } else {
                        result_relaxed <- elastic_rrr(
                            x_train[, nz],
                            y_train,
                            lambdas = c(lambda_relaxed),
                            alpha = 0,
                            rank = rank,
                            sparsity = sparsity
                        )
                    }
                    vx <- as.matrix(vx)
                    vy <- as.matrix(vy)

                    vxr <- result_relaxed[[1]]$w
                    vyr <- result_relaxed[[1]]$v

                    if (sum(nz) >= ncol(vy)) {
                        vx[nz, ] <- vxr
                        vy <- vyr
                    } else {
                        vx[nz, 1:sum(nz)] <- vxr
                        vx[nz, sum(nz):ncol(vx)] <- 0
                        vy[, 1:sum(nz)] <- vyr
                        vy[, sum(nz):ncol(vy)] <- 0
                    }

                    if (all(near(apply(x_test %*% vx, MARGIN = 2, sd), 0))) {
                        i <- i + 1
                        next
                    }
                    r2_relaxed[r, cvfold, j, i] <- 1 -
                        sum((y_test - x_test %*% vx %*% t(vy)) ^ 2) /
                        sum(y_test ^ 2)
                    for (rk in seq_len(rank)) {
                        corrs_relaxed[r, cvfold, j, i, rk] <- cor(
                            x_test %*% vx[, rk],
                            y_test %*% vy[, rk]
                        )
                    }
                    i <- i + 1
                }
            }
            pb_fold$tick()
        }
    }
    return(list(
        nonzero = nonzero,
        r2 = r2,
        r2_relaxed = r2_relaxed,
        corrs = corrs,
        corrs_relaxed = corrs_relaxed,
        lambdas_used = lambdas_used,
        foldid = foldid
    ))
}


select_lambdas <- function(x, y, rank, nlambda, alpha, nlambda_for_path = 100,
    sparsity = "row-wise") {
    nlambda_for_path <- max(nlambda + 2, nlambda_for_path)

    # Initialize with PLS direction
    s <- svd(t(x) %*% as.matrix(y))
    v <- s$v[, 1:rank]

    if (rank == 1) {
        fit <- glmnet(
            x = as.matrix(x),
            y = as.matrix(y) %*% v,
            alpha = alpha,
            nlambda = nlambda_for_path,
            standardize = FALSE,
            intercept = FALSE
        )
        lambda_used <- fit$lambda
    } else {
        if (sparsity == "row-wise") {
            fit <- glmnet(
                x = as.matrix(x),
                y = as.matrix(y) %*% v,
                family = "mgaussian",
                alpha = alpha,
                nlambda = nlambda_for_path,
                standardize = FALSE,
                standardize.response = FALSE,
                intercept = FALSE
            )
            lambda_used <- fit$lambda
        } else {
            lambdas_used_list <- lapply(
                seq_len(rank),
                function(i) {
                    fit <- glmnet(
                        x = as.matrix(x),
                        y = as.matrix(y) %*% v[, i],
                        alpha = alpha,
                        nlambda = nlambda_for_path,
                        standardize = FALSE,
                        intercept = FALSE
                    )
                    fit$lambda
                })
            lambda_used <- lambdas_used_list %>%
                unlist %>%
                unique %>%
                sort(decreasing = TRUE)
        }
    }
    ind <- seq(2, length(lambda_used) - 1, length.out = nlambda) %>% round
    return(list(lambdas = lambda_used[ind], lambdas_full = lambda_used))
}