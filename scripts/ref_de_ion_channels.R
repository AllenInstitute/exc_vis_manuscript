library(limma)
library(arrow)
library(tibble)
suppressPackageStartupMessages(library(dplyr))
library(scrattch.hicat)


vgc_genes <- read.table("../data/gene_lists/voltage_gated_ion_channel_genes.txt")
my_de_param <- de_param(
    q.diff.th = 0,
)

args <- commandArgs(trailingOnly = TRUE)

# Load reference data

message("loading reference data")

facs_dir <- args[1]
facs_anno <- arrow::read_feather(file.path(facs_dir, "anno.feather"))
facs_cpm_data <- arrow::read_feather(file.path(facs_dir, "data.feather"),
	col_select = c("sample_id", vgc_genes$V1))
facs_cpm_mat <- facs_cpm_data %>% select(-"sample_id") %>% as.matrix
rownames(facs_cpm_mat) <- facs_cpm_data$sample_id
facs_norm_dat <- log2(facs_cpm_mat + 1)
facs_norm_dat <- t(facs_norm_dat)


# Load Patch-seq data

message("loading patch-seq data")

ps_dir <- args[2]
ephys_df <- read.csv(args[3], row.names = 1)
inf_met_df <- read.csv(args[4], row.names = 1)
ps_anno <- arrow::read_feather(file.path(ps_dir, "anno.feather"))
ps_anno <- ps_anno |>
    filter(Tree_call_label %in% c("Core", "I1", "I2", "I3"), spec_id_label %in% rownames(ephys_df))

ps_cpm_data <- arrow::read_feather(file.path(ps_dir, "data.feather"),
	col_select = c("sample_id", vgc_genes$V1))
ps_cpm_mat <- ps_cpm_data %>% select(-"sample_id") %>% as.matrix
rownames(ps_cpm_mat) <- ps_cpm_data$sample_id
ps_norm_dat <- log2(ps_cpm_mat + 1)
ps_norm_dat <- t(ps_norm_dat)

find_de_ion_channels <- function(ps_norm_dat, facs_norm_dat, ps_anno, facs_anno,
    inf_met_df,
    met_types, ps_met_type_names_for_pairs,
    met_type_t_types, ref_met_type_names_for_pairs,
    my_de_param) {

    ids_with_met_types <- inf_met_df %>%
        rownames_to_column(var = "specimen_id") %>%
        filter(inferred_met_type %in% met_types) %>%
        select(specimen_id)

    ps_sub_anno <- ps_anno %>% filter(spec_id_label %in% ids_with_met_types$specimen_id)
    ps_sub_ids <- ps_sub_anno$spec_id_label
    ps_sub_norm_dat <- ps_norm_dat[, ps_sub_anno$sample_id]

    cl <- inf_met_df[ps_sub_ids, ]$inferred_met_type %>% make.names %>% as.factor
    names(cl) <- ps_sub_anno$sample_id
	print("cl")
	print(head(cl))
	
    ps_de_df <- de_selected_pairs(
        ps_sub_norm_dat[vgc_genes$V1, ],
        cl,
        create_pairs(ps_met_type_names_for_pairs) # use syntactically valid version of names
    )

    ps_de_results <- de_stats_selected_pairs(
        ps_sub_norm_dat[vgc_genes$V1, ],
        cl,
        create_pairs(ps_met_type_names_for_pairs), # use syntactically valid version of names
        de.df = ps_de_df,
        de.param = my_de_param
    )

    low_th <- my_de_param$low.th
    ps_cl_means <- get_cl_means(ps_sub_norm_dat[vgc_genes$V1, ], cl)
    ps_cl_present <- get_cl_means(ps_sub_norm_dat[vgc_genes$V1, ] >= low_th, cl)


    ref_samples <- lapply(
        met_type_t_types,
        function(x) { facs_anno %>% filter(cluster_label %in% x) %>% select(sample_id) }
    )
    cl <- rep(names(ref_samples), times = lapply(ref_samples, nrow))
    names(cl) <- unlist(lapply(ref_samples, function(x) x$sample_id))

    ref_sub_norm_dat <- facs_norm_dat[, names(cl)]

    ref_de_df <- de_selected_pairs(
        ref_sub_norm_dat[vgc_genes$V1, ],
        cl,
        create_pairs(ref_met_type_names_for_pairs), # use syntactically valid version of names
    )

    ref_de_results <- de_stats_selected_pairs(
        ref_sub_norm_dat[vgc_genes$V1, ],
        cl,
        create_pairs(ref_met_type_names_for_pairs), # use syntactically valid version of names
        de.df = ref_de_df,
        de.param = my_de_param
    )

    ref_cl_means <- get_cl_means(ref_sub_norm_dat[vgc_genes$V1, ], cl)
    ref_cl_present <- get_cl_means(ref_sub_norm_dat[vgc_genes$V1, ] >= low_th, cl)

    ps_and_ref_de_genes <- union(ps_de_results[[1]]$genes, ref_de_results[[1]]$genes)

    # Combine the results for output
    ps_de <- ps_de_df[[1]] %>%
        rename_with(~ paste0("ps_de_", .x, recycle0 = TRUE)) %>%
        rownames_to_column(var = "gene") %>%
        mutate(ps_is_de = if_else(gene %in% ps_de_results[[1]]$genes, TRUE, FALSE))
    ref_de <- ref_de_df[[1]] %>%
        rename_with(~ paste0("ref_de_", .x, recycle0 = TRUE)) %>%
        rownames_to_column(var = "gene") %>%
        mutate(ref_is_de = if_else(gene %in% ref_de_results[[1]]$genes, TRUE, FALSE))
    ps_means <- ps_cl_means %>% as_tibble(rownames = NA) %>%
        rename_with(~ paste0("ps_mean_", .x, recycle0 = TRUE)) %>%
        rownames_to_column(var = "gene")
    ref_means <- ref_cl_means %>% as_tibble(rownames = NA) %>%
        rename_with(~ paste0("ref_mean_", .x, recycle0 = TRUE)) %>%
        rownames_to_column(var = "gene")
    ps_present <- ps_cl_present %>% as_tibble(rownames = NA) %>%
        rename_with(~ paste0("ps_present_", .x, recycle0 = TRUE)) %>%
        rownames_to_column(var = "gene")
    ref_present <- ref_cl_present %>% as_tibble(rownames = NA) %>%
        rename_with(~ paste0("ref_present_", .x, recycle0 = TRUE)) %>%
        rownames_to_column(var = "gene")

    combined_output <- ps_de %>%
        inner_join(ref_de, by = "gene", keep = FALSE) %>%
        inner_join(ps_means, by = "gene", keep = FALSE) %>%
        inner_join(ref_means, by = "gene", keep = FALSE) %>%
        inner_join(ps_present, by = "gene", keep = FALSE) %>%
        inner_join(ref_present, by = "gene", keep = FALSE) %>%
        filter(gene %in% ps_and_ref_de_genes)

    return(combined_output)
}


message("analyzing l6 it")

l6_it_met_to_t_types <- list(
	L6.IT.1.2.3 = c(
		"L6 IT VISp Penk Col27a1",
		"L6 IT VISp Penk Fst",
		"L6 IT VISp Col18a1",
		"L6 IT VISp Col23a1 Adamts2"
    ),
    L5.L6.IT.Car3 = c(
    	"L6 IT VISp Car3"
	)
)

l6_it_results <- find_de_ion_channels(
    ps_norm_dat,
    facs_norm_dat,
    ps_anno,
    facs_anno,
    inf_met_df,
    met_types = c("L6 IT-1", "L6 IT-2", "L6 IT-3", "L5/L6 IT Car3"),
    ps_met_type_names_for_pairs = c("L6.IT.2", "L5.L6.IT.Car3"),
    met_type_t_types = l6_it_met_to_t_types,
    ref_met_type_names_for_pairs = c("L6.IT.1.2.3", "L5.L6.IT.Car3"),
    my_de_param
)

write.csv(l6_it_results, "../derived_data/l6_it_de_ion_channel_results.csv")


message("analyzing l5 et")

l5_et_met_to_t_types <- list()
l5_et_met_to_t_types[["L5.ET.1.Chrna6"]] <- c(
    "L5 PT VISp Chrna6"
)
l5_et_met_to_t_types[["L5.ET.2.3"]] <- c(
    "L5 PT VISp C1ql2 Ptgfr",
    "L5 PT VISp C1ql2 Cdh13",
    "L5 PT VISp Krt80",
    "L5 PT VISp Lgr5"
)

l5_et_results <- find_de_ion_channels(
    ps_norm_dat,
    facs_norm_dat,
    ps_anno,
    facs_anno,
    inf_met_df,
    met_types = c("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"),
    ps_met_type_names_for_pairs = c("L5.ET.1.Chrna6", "L5.ET.3"),
    met_type_t_types = l5_et_met_to_t_types,
    ref_met_type_names_for_pairs = c("L5.ET.1.Chrna6", "L5.ET.2.3"),
    my_de_param
)

write.csv(l5_et_results, "../derived_data/l5_et_de_ion_channel_results.csv")
