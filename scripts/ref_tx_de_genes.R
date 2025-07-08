library(scrattch.hicat)
library(arrow)
library(matrixStats)
library(Matrix)
library(WGCNA)
library(dynamicTreeCut)
library(tibble)
library(dplyr)
library(stats)

args <- commandArgs(trailingOnly = TRUE)
ref_tx_anno_file <- args[1]
ref_norm_data_file <- args[2]
output_file <- args[3]

ttypes <- c(
	"L2/3 IT VISp Rrad",
	"L2/3 IT VISp Adamts2",
	"L2/3 IT VISp Agmat",
	"L4 IT VISp Rspo1",
	"L5 IT VISp Hsd11b1 Endou",
	"L5 IT VISp Whrn Tox2",
	"L5 IT VISp Batf3",
	"L5 IT VISp Col6a1 Fezf2",
	"L5 IT VISp Col27a1",
	"L6 IT VISp Penk Col27a1",
	"L6 IT VISp Penk Fst",
	"L6 IT VISp Col23a1 Adamts2",
	"L6 IT VISp Col18a1",
	"L6 IT VISp Car3",
	"L5 PT VISp Chrna6",
	"L5 PT VISp Lgr5",
	"L5 PT VISp C1ql2 Ptgfr",
	"L5 PT VISp C1ql2 Cdh13",
	"L5 PT VISp Krt80",
	"L5 NP VISp Trhr Cpne7",
	"L5 NP VISp Trhr Met",
	"L6 CT VISp Nxph2 Wls",
	"L6 CT VISp Gpr139",
	"L6 CT VISp Ctxn3 Brinp3",
	"L6 CT VISp Ctxn3 Sla",
	"L6 CT VISp Krt80 Sla",
	"L6b Col8a1 Rprm",
	"L6b VISp Mup5",
	"L6b VISp Col8a1 Rxfp1",
	"L6b P2ry12",
	"L6b VISp Crh"
)

# Load the reference annotations
anno <- read_feather(ref_tx_anno_file)
# subclass_cells <- anno[anno$subclass_label == subclass_label, "sample_id"]$sample_id

# Load the gene matrix
message("loading gene matrix")

# Provides norm.dat object
load(ref_norm_data_file)

# parameters for gene selection
de.param <- de_param(q1.th = 0.5, q.diff.th = 0.7, de.score.th = 150, min.cells = 10)
max_genes <- 4000
vg_padj_th <- 0.5
max_dim <- 20


message('starting de gene determination')
# Get the reference cells for the specified ttypes
cat(ttypes, "\n")
group_cells <- anno[anno$cluster_label %in% ttypes, ]$sample_id
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

cat(select_genes, "\n")

write.csv(select_genes, file=output_file)

