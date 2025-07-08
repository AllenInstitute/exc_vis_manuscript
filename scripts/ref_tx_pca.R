library(scrattch.hicat)
library(arrow)
library(matrixStats)
library(Matrix)
library(WGCNA)
library(dynamicTreeCut)
library(tibble)
library(dplyr)
library(stats)

# Define the groups to analyze individually (based on MET types & subclasses)
groups_of_ttypes <- list(
    "L23-IT" = c('L2/3 IT VISp Agmat', 'L2/3 IT VISp Rrad', 'L2/3 IT VISp Adamts2'),
	"L4-L5-IT" = c('L4 IT VISp Rspo1', 'L5 IT VISp Hsd11b1 Endou', 'L5 IT VISp Whrn Tox2', 'L5 IT VISp Col6a1 Fezf2', 'L5 IT VISp Batf3', 'L5 IT VISp Col27a1'),
    "L6-IT" = c('L6 IT VISp Col18a1', 'L6 IT VISp Col23a1 Adamts2', 'L6 IT VISp Penk Col27a1', 'L6 IT VISp Penk Fst'),
    "L5L6-IT-Car3" = c('L6 IT VISp Car3'),
    "L5-ET" = c('L5 PT VISp Chrna6', 'L5 PT VISp C1ql2 Cdh13', 'L5 PT VISp C1ql2 Ptgfr', 'L5 PT VISp Krt80', 'L5 PT VISp Lgr5'),
    "L5-ET-Chrna6" = c('L5 PT VISp Chrna6'),
    "L5-ET-non-Chrna6" = c('L5 PT VISp C1ql2 Cdh13', 'L5 PT VISp C1ql2 Ptgfr', 'L5 PT VISp Krt80', 'L5 PT VISp Lgr5'),
    "L5-NP" = c('L5 NP VISp Trhr Met', 'L5 NP VISp Trhr Cpne7'),
    "L6-CT" = c('L6 CT VISp Ctxn3 Brinp3', 'L6 CT VISp Ctxn3 Sla', 'L6 CT VISp Nxph2 Wls', 'L6 CT VISp Krt80 Sla', 'L6 CT VISp Gpr139'),
    "L6b" = c('L6b VISp Mup5', 'L6b Col8a1 Rprm', 'L6b VISp Col8a1 Rxfp1', "L6b VISp Crh", "L6b P2ry12")
)


# Load the reference annotations
args <- commandArgs(trailingOnly = TRUE)
ref_tx_dir <- args[1]

anno <- read_feather(file.path(ref_tx_dir, "anno.feather"))
# subclass_cells <- anno[anno$subclass_label == subclass_label, "sample_id"]$sample_id

# Load the gene matrix
message("loading gene matrix")

# Provides norm.dat object
norm_dat_path <- args[2]
load(norm_dat_path)

# parameters for gene selection
de.param <- de_param(q1.th = 0.5, q.diff.th = 0.7, de.score.th = 150, min.cells = 10)
max_genes <- 4000
vg_padj_th <- 0.5
max_dim <- 20


message('starting tx pc determination')
tx_pcs <- lapply(groups_of_ttypes,
    function (e) {
        # Get the reference cells for the group
        cat(e, "\n")
        group_cells <- anno[anno$cluster_label %in% e, ]$sample_id
        subset_data <- norm.dat[, group_cells]

        # Find the high-variance genes within the group
        select_genes <- rownames(subset_data)[
            which(Matrix::rowSums(subset_data > de.param$low.th) >=
                de.param$min.cells)]

        # Convert to counts for find_vg()
        if (is.matrix(subset_data)) {
            counts <- 2 ^ subset_data - 1
        } else {
            counts <- subset_data
            counts@x <- 2 ^ (counts@x) - 1
        }
        hv_genes <- find_vg(as.matrix(counts[select_genes, ]), plot_file = NULL)

        # Use dispersion criteria for gene selection
        select_genes <- as.character(
            hv_genes[which(hv_genes$loess.padj < vg_padj_th | hv_genes$dispersion > 3), "gene"]
        )
        select_genes <- head(
            select_genes[
                order(hv_genes[select_genes, "loess.padj"],
                    -hv_genes[select_genes, "z"])
            ],
            max_genes
        )

        # Identify tx PCs
        dat <- as.matrix(subset_data[select_genes, ])
        pca <- stats::prcomp(t(dat), tol = 0.01)
        print(summary(pca))
        pca_importance <- summary(pca)$importance
		v <- pca_importance[2,]
		select <- which((v - mean(v)) / sd(v) > 2)
		tmp <- head(select, max_dim)
		if (length(tmp) == 0) {
			return(NULL)
		}
 		rd_dat <- pca$x[, tmp, drop=FALSE]
		rd_wt <- pca$rotation[, tmp, drop=FALSE]
		return(list(rd_dat = rd_dat, rd_wt = rd_wt, center = pca$center, pca = pca))
    }
)

# save results
for (n in names(tx_pcs)) {
    pc_result <- tx_pcs[[n]]
    write.csv(pc_result$rd_dat, file = paste0("../derived_data/ref_tx_pca_results/", n, "_tx_pca_transformed.csv"))
    write.csv(pc_result$rd_wt, file = paste0("../derived_data/ref_tx_pca_results/", n, "_tx_pca_weights.csv"))
	write.csv(pc_result$center, file = paste0("../derived_data/ref_tx_pca_results/", n, "_tx_pca_centers.csv"))
}
