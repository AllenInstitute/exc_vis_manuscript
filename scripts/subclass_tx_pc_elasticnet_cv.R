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


run_cv <- function(x, y, foldid, alphas = c(0.25, 0.5, 0.75, 1.0)) {
    tss <- sum(scale(y, scale = FALSE) ^ 2)

    results <- lapply(
        alphas,
        function(a) {
            cvfit <- cv.glmnet(
                as.matrix(x),
                as.matrix(y),
                family = "mgaussian",
                foldid = foldid,
                alpha = a,
                parallel = TRUE
            )
            r2 <- 1 - (nrow(y) * cvfit$cvm / tss)
            return(list(
                cvfit = cvfit,
                r2 = r2
            ))
        }
    )
    return(list(
        results_by_alpha = results,
        alphas = alphas,
        foldid = foldid
    ))
}


save_to_h5 <- function(result, h5_filename, group_name, specimen_ids) {
    message("writing results to file")
    h5createGroup(h5_filename, group_name)
    h5createGroup(h5_filename, paste0(group_name, "/", "effect_of_alpha"))

    h5write(result$alphas, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/alpha"))
    h5write(result$foldid, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/foldid"))
    h5write(specimen_ids, file = h5_filename,
        name = paste0(group_name, "/effect_of_alpha/specimen_id"))

    counter <- 1
    for (res in result$results_by_alpha) {
        prefix <- paste0(group_name, "/effect_of_alpha/alpha_ind_", counter)

        h5createGroup(file = h5_filename, group = prefix)
        h5write(res$r2, file = h5_filename,
            name = paste0(prefix, "/r2"))
        h5write(res$cvfit$lambda, file = h5_filename,
            name = paste0(prefix, "/lambda"))
        h5write(res$cvfit$nzero, file = h5_filename,
            name = paste0(prefix, "/nonzero"))
        counter <- counter + 1
    }
    h5closeAll()

}

# main
require(doMC)
registerDoMC(cores = 4)

args <- commandArgs(trailingOnly = TRUE)
subclass_index <- strtoi(args[1])

ps_tx_anno_file <- args[2]
ps_tx_data_file <- args[3]
inf_met_type_file <- args[4]
morph_features_file <- args[5]
ephys_features_file <- args[6]
select_features_file <- args[7]
ps_pc_dir <- args[8]
output_dir <- "../derived_data/sparse_rrr_results/"

message("loading data")
ps_anno_df <- read_feather(ps_tx_anno_file)

filepart <- SUBCLASS_INFO[[subclass_index]]$filepart
set_of_types <- SUBCLASS_INFO[[subclass_index]]$set_of_types

print(filepart)
print(set_of_types)

ps_pc_data_df <- read.csv(
    file.path(ps_pc_dir, paste0(filepart, "_ps_transformed_pcs.csv")),
    row.names = 1)
inf_met_type_df <- read.csv(inf_met_type_file, row.names = 1)
ephys_df <- read.csv(ephys_features_file, row.names = 1)
morph_df <- read.csv(morph_features_file, row.names = 1)
selected_features <- fromJSON(file = select_features_file)

specimen_ids <- rownames(inf_met_type_df)[
    inf_met_type_df$inferred_met_type %in% set_of_types]
sample_spec_ids <- ps_anno_df %>%
    filter(spec_id_label %in% specimen_ids) %>%
    select(sample_id, spec_id_label) %>%
    arrange(sample_id)

# Set alpha sequence
alphas <- c(0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0)

output_file <- file.path(output_dir, paste0("tx_pc_elasticnet_cv_", filepart, ".h5"))

if (file.exists(output_file)) {
    file.remove(output_file)
}
h5createFile(output_file)

message("CV - genes and ephys")

tx_pc_data <- ps_pc_data_df[sample_spec_ids$spec_id_label, ]

ephys_data <- ephys_df[sample_spec_ids$spec_id_label, ]
ephys_features <- selected_features[[filepart]]$ephys
ephys_features <- paste0("X", ephys_features)
ephys_data_norm <- ephys_data %>%
    select(all_of(ephys_features)) %>%
    preprocess_data(z_score = TRUE)

# Get foldid from sRRR run
srrr_filename <- file.path(output_dir,
    paste0("sparse_rrr_cv_", filepart, ".h5"))
foldid <- h5read(srrr_filename, "ephys/effect_of_alpha/foldid")[1, , drop = TRUE]

message("Fitting ", ncol(ephys_data_norm), " ephys features")
message("Using ", nrow(tx_pc_data), " cells")

ephys_result <- run_cv(
    tx_pc_data,
    ephys_data_norm,
    alphas = alphas,
    foldid = foldid
)
save_to_h5(ephys_result, output_file, "ephys", sample_spec_ids$spec_id_label)


message("CV - genes and morph")

morph_spec_ids <- specimen_ids[specimen_ids %in% rownames(morph_df)]
morph_sample_spec_ids <- ps_anno_df %>%
    filter(spec_id_label %in% morph_spec_ids) %>%
    select(sample_id, spec_id_label) %>%
    arrange(sample_id)

tx_pc_data <- ps_pc_data_df[morph_sample_spec_ids$spec_id_label, ]

morph_data <- morph_df[morph_sample_spec_ids$spec_id_label, ]
morph_features <- selected_features[[filepart]]$morph
morph_data_norm <- morph_data %>%
    select(all_of(morph_features)) %>%
    preprocess_data(z_score = TRUE)

foldid <- h5read(srrr_filename, "morph/effect_of_alpha/foldid")[1, , drop = TRUE]

message("Fitting ", ncol(morph_data_norm), " morph features")
message("Using ", nrow(tx_pc_data), " cells")

morph_result <- run_cv(
    tx_pc_data,
    morph_data_norm,
    alphas = alphas,
    foldid = foldid
)
save_to_h5(morph_result, output_file, "morph", morph_sample_spec_ids$spec_id_label)
print(h5ls(output_file))