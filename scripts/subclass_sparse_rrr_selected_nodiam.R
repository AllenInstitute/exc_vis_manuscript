library(arrow)
library(rhdf5)
library(rjson)
library(dplyr)
source("../scripts/sparseRRR.R")


SUBCLASS_INFO <- list(
    list(
        set_of_types = c("L2/3 IT"),
        filepart = "L23-IT"
    ),
    list(
        set_of_types = c("L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"),
        filepart = "L4-L5-IT"
    ),
    list(
        set_of_types = c("L6 IT-1", "L6 IT-2", "L6 IT-3"),
        filepart = "L6-IT"
    ),
    list(
        set_of_types = c("L5/L6 IT Car3"),
        filepart = "L5L6-IT-Car3"
    ),
    list(
        set_of_types = c("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"),
        filepart = "L5-ET"
    ),
    list(
        set_of_types = c("L5 NP"),
        filepart = "L5-NP"
    ),
    list(
        set_of_types = c("L6 CT-1", "L6 CT-2"),
        filepart = "L6-CT"
    ),
    list(
        set_of_types = c("L6b"),
        filepart = "L6b"
    ),
    list(
        set_of_types = c("L5 ET-1 Chrna6"),
        filepart = "L5-ET-Chrna6"
    ),
    list(
        set_of_types = c("L5 ET-2", "L5 ET-3"),
        filepart = "L5-ET-non-Chrna6"
    )
)


preprocess_data <- function(x, log_transform = FALSE, z_score = FALSE) {
    nonzero_mask <- which(abs(colSums(x != 0)) > 0)
    x_process <- x[, nonzero_mask]
    if (log_transform) {
        x_process <- log1p(x_process)
    }
    if (z_score) {
        x_process <- scale(x_process)
    }
    return(x_process)
}


save_to_h5 <- function(result, genes, specimen_ids, h5_filename, filepart, modality,
    r, alpha, requested_lambda) {

    prefix <- paste0(filepart, "/", modality)
    h5createGroup(file = h5_filename, group = prefix)

    # Write attributes
    h5f <- H5Fopen(h5_filename)
    h5g <- H5Gopen(h5f, prefix)
    h5writeAttribute(h5g, attr = r, name = "rank")
    h5writeAttribute(h5g, attr = alpha, name = "alpha")
    h5writeAttribute(h5g, attr = lambda, name = "requested_lambda")
    h5writeAttribute(h5g, attr = result$lambda, name = "lambda")
    H5Gclose(h5g)
    H5Fclose(h5f)

    # Write data sets
    h5write(genes, file = h5_filename, name = paste0(prefix, "/genes"))
    h5write(as.integer(specimen_ids), file = h5_filename,
        name = paste0(prefix, "/specimen_ids"))
    h5write(result$w, file = h5_filename, name = paste0(prefix, "/w"))
    h5write(result$v, file = h5_filename, name = paste0(prefix, "/v"))
}


relaxed_fit <- function(x, y, rank, alpha, lambda, verbose = TRUE) {
    if (verbose) {
        message("Fitting ", ncol(y), " features")
        message("using ", ncol(x), " genes")
        message("using ", nrow(x), " cells")
        message("with rank ", r, ", alpha ", alpha, ", and lambda ", lambda)
    }
    # First fit
    results <- elastic_rrr(
        x,
        y,
        rank = rank,
        alpha = alpha,
        lambdas = c(lambda)
    )

    lambdas <- sapply(results, function(res) {res$lambda})
    lambda_ind <- which.min(abs(lambdas - lambda))
    chosen_result <- results[[lambda_ind]]

    if (verbose) {
        message("solution found with lambda ", lambdas[lambda_ind])
    }

    # Relaxed fit
    nzero <- rowSums(abs(as.matrix(chosen_result$w))) != 0
    relaxed_results <- elastic_rrr(
        x[, nzero],
        y,
        rank = rank,
        alpha = 0,
        lambdas = c(lambdas[lambda_ind])
    )

    # Get the first (and only) value from the list
    relaxed_result <- relaxed_results[[1]]

    relaxed_result <- rebuild_relaxed_matrices(
        as.matrix(chosen_result$w),
        as.matrix(chosen_result$v),
        relaxed_result$w,
        relaxed_result$v,
        nzero
    )
    relaxed_result$lambda <- lambdas[lambda_ind]
    return(relaxed_result)
}


rebuild_relaxed_matrices <- function(vx, vy, vxr, vyr, nz) {
    if (sum(nz) >= ncol(vy)) {
        vx[nz, ] <- vxr
        vy <- vyr
    } else {
        vx[nz, 1:sum(nz)] <- vxr
        vx[nz, sum(nz):ncol(vx)] <- 0
        vy[, 1:sum(nz)] <- vyr
        vy[, sum(nz):ncol(vy)] <- 0
    }
    return(list(w = vx, v = vy))
}

# main ###############

args <- commandArgs(trailingOnly = TRUE)

ps_tx_anno_file <- args[1]
ps_tx_data_file <- args[2]
inf_met_type_file <- args[3]
morph_features_file <- args[4]
select_features_file <- args[5]
chosen_hyperparams_file <- args[6]
ref_pc_dir <- args[7]
output_file <- args[8]


message("loading data")

ps_anno_df <- read_feather(ps_tx_anno_file)
selected_features <- fromJSON(file = select_features_file)
morph_df <- read.csv(morph_features_file, row.names = 1)
inf_met_type_df <- read.csv(inf_met_type_file, row.names = 1)
hyperparams <- fromJSON(file = chosen_hyperparams_file)


if (file.exists(output_file)) {
    file.remove(output_file)
}
h5createFile(output_file)

diam_features_to_drop <- c(
    "apical_dendrite_total_surface_area",
    "basal_dendrite_total_surface_area",
    "apical_dendrite_mean_diameter",
    "basal_dendrite_mean_diameter"
)

for (si in SUBCLASS_INFO) {
    filepart <- si$filepart
    set_of_types <- si$set_of_types
    print(filepart)
    print(set_of_types)

    h5createGroup(file = output_file, group = filepart)

    pc_weights <- read.csv(
        file.path(ref_pc_dir, paste0(filepart, "_tx_pca_weights.csv")),
        row.names = 1)
    hv_genes <- rownames(pc_weights)
    ps_data_df <- read_feather(
        ps_tx_data_file, col_select = append("sample_id", hv_genes))


    message("Morphology")

    specimen_ids <- rownames(inf_met_type_df)[
        inf_met_type_df$inferred_met_type %in% set_of_types]
    morph_spec_ids <- specimen_ids[specimen_ids %in% rownames(morph_df)]
    morph_sample_spec_ids <- ps_anno_df %>%
        filter(spec_id_label %in% morph_spec_ids) %>%
        select(sample_id, spec_id_label) %>%
        arrange(sample_id)
    gene_data <- ps_data_df %>%
        filter(sample_id %in% morph_sample_spec_ids$sample_id) %>%
        arrange(sample_id) %>%
        select(-sample_id)
    gene_data_norm <- preprocess_data(gene_data, log_transform = TRUE)

    morph_data <- morph_df[morph_sample_spec_ids$spec_id_label, ]
    morph_features <- selected_features[[filepart]]$morph

    # Remove diameter-related features
    morph_features <- setdiff(morph_features, diam_features_to_drop)

    morph_data_norm <- morph_data %>%
        select(all_of(morph_features)) %>%
        preprocess_data(z_score = TRUE)

    alpha <- hyperparams[[filepart]]$morph$sparse_rrr$alpha
    lambda <- hyperparams[[filepart]]$morph$sparse_rrr$lambda
    r <- hyperparams[[filepart]]$morph$sparse_rrr$rank

    # First fit
    morph_relaxed_result <- relaxed_fit(
        gene_data_norm,
        morph_data_norm,
        rank = r,
        alpha = alpha,
        lambda = lambda
    )
    genes <- colnames(gene_data_norm)
    print("w")
    print(dim(morph_relaxed_result$w))
    print("v")
    print(dim(morph_relaxed_result$v))

    message("Saving morphology result")
    save_to_h5(morph_relaxed_result, genes, morph_sample_spec_ids$spec_id_label,
        output_file, filepart, "morph", r, alpha, lambda)
}

h5ls(output_file)


