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

run_cv <- function(x, y, alphas = c(0.25, 0.5, 0.75, 1.0),
    nlambda = 20, folds = 10, foldid = NULL) {

    rank_value <- min(3, ncol(y) - 1)
    result <- elastic_rrr_cv(
        x,
        y,
        rank = rank_value,
        reps = 1,
        folds = folds,
        alphas = alphas,
        nlambda = nlambda,
        preset_foldid = foldid
    )

    # find best alpha from last runs to use for ranks
    avg_res <- colMeans(result$r2_relaxed, dims = 2, na.rm = TRUE)
    print(dim(avg_res))
    best_per_alpha <- apply(avg_res, MARGIN = 1, max, na.rm = TRUE)
    print(best_per_alpha)
    best_alpha_ind <- which.max(best_per_alpha)
    print(best_alpha_ind)
    print(alphas[best_alpha_ind])
    message("Using alpha ", alphas[best_alpha_ind], " for ranks")

    alpha_for_ranks <- c(alphas[best_alpha_ind])

    if (is.null(foldid)) {
        foldid <- result$foldid
    }
    alpha_for_ranks <- c(0.02)
    max_n_per_fold = nrow(x) - max(table(foldid))
    max_for_rank = min(ncol(y), nrow(x), max_n_per_fold)
    ranks <- seq(min(11, max_for_rank - 1))
    message("Max rank ", max(ranks))
    rank_results_list <- lapply(
        ranks,
        function(r) {
        	message("rank ", r)
            elastic_rrr_cv(
				x = x,
				y = y,
				rank = r,
				reps = 1,
				folds = folds,
				alphas = alpha_for_ranks,
				nlambda = nlambda,
				preset_foldid = foldid
        )
        }
    )

    return(list(
        results_by_alpha = result,
        alphas = alphas,
        rank_for_alphas = rank_value,
        results_by_rank = rank_results_list,
        ranks = ranks,
        alpha_for_ranks = alpha_for_ranks
    ))

}

save_to_h5 <- function(result, h5_filename, group_name) {
    message("writing results to file")
    h5createGroup(h5_filename, group_name)
    h5createGroup(h5_filename, paste0(group_name, "/", "effect_of_alpha"))
    h5createGroup(h5_filename, paste0(group_name, "/", "effect_of_rank"))

    # Write rank attribute
    h5f <- H5Fopen(h5_filename)
    h5g <- H5Gopen(h5f, paste0(group_name, "/", "effect_of_alpha"))
    h5writeAttribute(h5g, attr = result$rank_for_alphas, name = "rank")
    H5Gclose(h5g)
    H5Fclose(h5f)

    # Write data sets
    h5write(result$alphas, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/alpha"))
    res_alpha <- result$results_by_alpha
    h5write(res_alpha$nonzero, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/nonzero"))
    h5write(res_alpha$r2, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/r2"))
    h5write(res_alpha$r2_relaxed, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/r2_relaxed"))
    h5write(res_alpha$corrs, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/corrs"))
    h5write(res_alpha$corrs_relaxed, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/corrs_relaxed"))
    h5write(res_alpha$lambdas_used, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/lambda"))
    h5write(res_alpha$foldid, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/foldid"))

    # Write alpha attribute
    h5f <- H5Fopen(h5_filename)
    h5g <- H5Gopen(h5f, paste0(group_name, "/", "effect_of_rank"))
    h5writeAttribute(h5g, attr = result$alpha_for_ranks, name = "alpha")
    H5Gclose(h5g)
    H5Fclose(h5f)

    # Write data sets
    h5write(result$ranks, file = h5_filename,
        name = paste0(group_name, "/effect_of_rank/rank"))
    for (i in seq_len(length(result$results_by_rank))) {
        res_rank <- result$results_by_rank[[i]]
        prefix <- paste0(group_name, "/effect_of_rank/rank_", i)
        h5createGroup(file = h5_filename, group = prefix)
        h5write(res_rank$nonzero, file = h5_filename,
            name = paste0(prefix, "/nonzero"))
        h5write(res_rank$r2, file = h5_filename,
            name = paste0(prefix, "/r2"))
        h5write(res_rank$r2_relaxed, file = h5_filename,
            name = paste0(prefix, "/r2_relaxed"))
        h5write(res_rank$corrs, file = h5_filename,
            name = paste0(prefix, "/corrs"))
        h5write(res_rank$corrs_relaxed, file = h5_filename,
            name = paste0(prefix, "/corrs_relaxed"))
        h5write(res_rank$lambdas_used, file = h5_filename,
            name = paste0(prefix, "/lambda"))
        h5write(res_rank$foldid, file = h5_filename,
            name = paste0(prefix, "/foldid"))
    }

    h5closeAll()
}

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


# main

args <- commandArgs(trailingOnly = TRUE)
subclass_index <- strtoi(args[1])

ps_tx_anno_file <- args[2]
ps_tx_data_file <- args[3]
inf_met_type_file <- args[4]
morph_features_file <- args[5]
ephys_features_file <- args[6]
select_features_file <- args[7]
ref_pc_dir <- args[8]
output_dir <- "../derived_data/sparse_rrr_results/"


if (length(args) > 8) {
    add_vgc <- strtoi(args[9])
} else {
    add_vgc <- 0
}

message("loading data")
ps_anno_df <- read_feather(ps_tx_anno_file)

filepart <- SUBCLASS_INFO[[subclass_index]]$filepart
set_of_types <- SUBCLASS_INFO[[subclass_index]]$set_of_types

print(filepart)
print(set_of_types)

pc_weights <- read.csv(
    file.path(ref_pc_dir, paste0(filepart, "_tx_pca_weights.csv")),
    row.names = 1)
hv_genes <- rownames(pc_weights)

if (add_vgc) {
    message("also using voltage-gated ion channel genes")
    print(hv_genes[1:5])
    vgc_file <- "../data/gene_lists/voltage_gated_ion_channel_genes.txt"
    vgc_genes <- read.table(vgc_file) |> pull("V1")
    print(vgc_genes[1:5])
    hv_genes <- c(hv_genes, vgc_genes) |> unique()
    # Get foldid from sRRR run
    srrr_filename <- file.path(output_dir, paste0("sparse_rrr_cv_", filepart, ".h5"))
    ephys_foldid <- h5read(srrr_filename, "ephys/effect_of_alpha/foldid")
    morph_foldid <- h5read(srrr_filename, "morph/effect_of_alpha/foldid")
    print(length(ephys_foldid))
    print(length(morph_foldid))
} else {
    ephys_foldid <- NULL
    morph_foldid <- NULL
}

ps_data_df <- read_feather(
    ps_tx_data_file, col_select = append("sample_id", hv_genes))
inf_met_type_df <- read.csv(inf_met_type_file, row.names = 1)
ephys_df <- read.csv(ephys_features_file, row.names = 1)
morph_df <- read.csv(morph_features_file, row.names = 1)

selected_features <- fromJSON(file = select_features_file)
specimen_ids <- rownames(inf_met_type_df)[
    inf_met_type_df$inferred_met_type %in% set_of_types]
sample_spec_ids <- ps_anno_df |>
    filter(spec_id_label %in% specimen_ids) |>
    select(sample_id, spec_id_label) |>
    arrange(sample_id)

# Set alpha sequence
alphas <- c(0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0)

if (add_vgc) {
    output_file <- file.path(output_dir, paste0("sparse_rrr_cv_", filepart, "_with_vgc.h5"))
} else {
    output_file <- file.path(output_dir, paste0("sparse_rrr_cv_", filepart, ".h5"))
}

message(paste0("Using output file ", output_file))

if (file.exists(output_file)) {
    file.remove(output_file)
}
h5createFile(output_file)

message("CV - genes and ephys")

gene_data <- ps_data_df |>
    filter(sample_id %in% sample_spec_ids$sample_id) |>
    arrange(sample_id) |>
    select(-sample_id)
gene_data_norm <- preprocess_data(gene_data, log_transform = TRUE)

ephys_data <- ephys_df[sample_spec_ids$spec_id_label, ]
ephys_features <- selected_features[[filepart]]$ephys
ephys_features <- paste0("X", ephys_features)
ephys_data_norm <- ephys_data |>
    select(all_of(ephys_features)) |>
    preprocess_data(z_score = TRUE)

message("Fitting ", ncol(ephys_data_norm), " ephys features")
message("Using ", ncol(gene_data_norm), " genes")
message("Using ", nrow(gene_data_norm), " cells")

ephys_result <- run_cv(
    gene_data_norm,
    ephys_data_norm,
    alphas = alphas,
    nlambda = 20,
    folds = 10,
    foldid = ephys_foldid
)

save_to_h5(ephys_result, output_file, "ephys")

morph_spec_ids <- specimen_ids[specimen_ids %in% rownames(morph_df)]
morph_sample_spec_ids <- ps_anno_df |>
    filter(spec_id_label %in% morph_spec_ids)|>
    select(sample_id, spec_id_label) |>
    arrange(sample_id)
gene_data <- ps_data_df |>
    filter(sample_id %in% morph_sample_spec_ids$sample_id) |>
    arrange(sample_id) |>
    select(-sample_id)
gene_data_norm <- preprocess_data(gene_data, log_transform = TRUE)

morph_data <- morph_df[morph_sample_spec_ids$spec_id_label, ]
morph_features <- selected_features[[filepart]]$morph
morph_data_norm <- morph_data |>
    select(all_of(morph_features)) |>
    preprocess_data(z_score = TRUE)

message("Fitting ", ncol(morph_data_norm), " morph features")
message("Using ", ncol(gene_data_norm), " genes")
message("Using ", nrow(gene_data_norm), " cells")

# Adjust CV if number of cells is low
folds <- 10

morph_result <- run_cv(
    gene_data_norm,
    morph_data_norm,
    alphas = alphas,
    nlambda = 20,
    folds = folds,
    foldid = morph_foldid
)

print(morph_result)

save_to_h5(morph_result, output_file, "morph")
print(h5ls(output_file))


