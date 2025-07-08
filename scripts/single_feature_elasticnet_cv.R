library(arrow)
library(rhdf5)
library(rjson)
library(dplyr)
library(glmnet)



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


remove_high_repeat_columns <- function(x) {
		drop_cols <- c()
		for (i in 1:ncol(x)) {
			n_most_repeats = max(table(x[, i]))
			if (n_most_repeats >= 0.9 * nrow(x)) {
				drop_cols <- c(drop_cols, i)
				print(names(x)[[i]])
			}
		}
		if (length(drop_cols) > 0) {
			return(x[, -drop_cols])
		} else {
			return(x)
		}
}

preprocess_data <- function(x, log_transform = FALSE, z_score = FALSE) {
    # check for all zeros
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

require(doMC)
registerDoMC(cores = 5)

alphas <- c(0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0)

args <- commandArgs(trailingOnly = TRUE)
subclass_index <- strtoi(args[1])

ps_tx_anno_file <- args[2]
ps_tx_data_file <- args[3]
inf_met_type_file <- args[4]
morph_features_file <- args[5]
ephys_features_file <- args[6]
ref_pc_dir <- args[7]
output_dir <- "../derived_data/single_feature_fits"


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

ps_data_df <- read_feather(
    ps_tx_data_file, col_select = append("sample_id", hv_genes))
inf_met_type_df <- read.csv(inf_met_type_file, row.names = 1)
ephys_df <- read.csv(ephys_features_file, row.names = 1)
morph_df <- read.csv(morph_features_file, row.names = 1)

specimen_ids <- rownames(inf_met_type_df)[
    inf_met_type_df$inferred_met_type %in% set_of_types]
sample_spec_ids <- ps_anno_df |>
    filter(spec_id_label %in% specimen_ids) |>
    select(sample_id, spec_id_label) |>
    arrange(sample_id)

output_file <- file.path(output_dir, paste0("single_features_elasticnet_cv_", filepart, ".h5"))
message(paste0("Using output file ", output_file))
if (file.exists(output_file)) {
    file.remove(output_file)
}
h5createFile(output_file)

message("Single feature CV - genes and ephys")

gene_data <- ps_data_df |>
    filter(sample_id %in% sample_spec_ids$sample_id) |>
    arrange(sample_id) |>
    select(-sample_id)
gene_data_norm <- preprocess_data(gene_data, log_transform = TRUE)

ephys_data <- ephys_df[sample_spec_ids$spec_id_label, ]
ephys_data_norm <- ephys_data |>
    preprocess_data(z_score = TRUE)

message("Fitting ", ncol(ephys_data_norm), " ephys features")
message("Using ", ncol(gene_data_norm), " genes")
message("Using ", nrow(gene_data_norm), " cells")

h5createGroup(file = output_file, group = "ephys")

foldid = sample(rep(seq(10), length = nrow(gene_data_norm)))

print("Using alphas")
print(alphas)
for (i in 1:ncol(ephys_data_norm)) {
	message(i)
	feature_prefix <- paste0("ephys/", i)
	h5createGroup(file = output_file, group = feature_prefix)
	for (a in alphas) {
		result <- cv.glmnet(as.matrix(gene_data_norm), as.matrix(ephys_data_norm[, i]),
			foldid = foldid, alpha = a, parallel = TRUE)
		prefix <- paste0(feature_prefix, "/", a)
		h5createGroup(file = output_file, group = prefix)
		h5write(result$cvm, file = output_file, name = paste0(prefix, "/mse"))
		h5write(1 - result$cvm / var(ephys_data_norm[, i]),
			file = output_file, name = paste0(prefix, "/r2"))
		h5write(result$nzero, file = output_file, name = paste0(prefix, "/nonzero"))
		h5write(result$lambda, file = output_file, name = paste0(prefix, "/lambdas"))
	}
}


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

morph_data <- morph_df[morph_sample_spec_ids$spec_id_label, ] |>
	remove_high_repeat_columns()
morph_data_norm <- morph_data |>
    preprocess_data(z_score = TRUE)

message("Fitting ", ncol(morph_data_norm), " morph features")
message("Using ", ncol(gene_data_norm), " genes")
message("Using ", nrow(gene_data_norm), " cells")

h5createGroup(file = output_file, group = "morph")
h5write(names(morph_data), file = output_file, name = "morph/morph_features")

foldid = sample(rep(seq(10), length = nrow(gene_data_norm)))

print("Using alphas")
print(alphas)

for (i in 1:ncol(morph_data_norm)) {
	message(names(morph_data)[[i]])
	feature_prefix <- paste0("morph/", i)
	h5createGroup(file = output_file, group = feature_prefix)
	for (a in alphas) {
		result <- cv.glmnet(as.matrix(gene_data_norm), as.matrix(morph_data_norm[, i]),
			foldid = foldid, alpha = a, parallel = TRUE)
		prefix <- paste0(feature_prefix, "/", a)
		h5createGroup(file = output_file, group = prefix)
		h5write(result$cvm, file = output_file, name = paste0(prefix, "/mse"))
		h5write(1 - result$cvm / var(morph_data_norm[, i]),
			file = output_file, name = paste0(prefix, "/r2"))
		h5write(result$nzero, file = output_file, name = paste0(prefix, "/nonzero"))
		h5write(result$lambda, file = output_file, name = paste0(prefix, "/lambdas"))
	}
}

h5closeAll()
