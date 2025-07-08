library(scrattch.hicat)
library(arrow)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
resp_gene_file <- args[1]
ref_dat_file <- args[2]
ps_dat_file <- args[3]
ref_sc_file <- args[4]
ps_sc_file <- args[5]
output_file <- args[6]

genes <- read.table(resp_gene_file, header=FALSE)$V1
ref_dat <- read_feather(ref_dat_file, col_select = c("sample_id", genes))
ps_dat <- read_feather(ps_dat_file, col_select = c("sample_id", genes))
ref_sc <- read.csv(ref_sc_file)
ps_sc <- read.csv(ps_sc_file)


ref_dat_rownames <- ref_dat$sample_id
ps_dat_rownames <- ps_dat$sample_id

ref_dat %<>% select(-sample_id) %>% as.matrix
rownames(ref_dat) <- ref_dat_rownames

ps_dat %<>% select(-sample_id) %>% as.matrix
rownames(ps_dat) <- ps_dat_rownames


ref_dat <- ref_dat[ref_sc$sample_id, ]
ps_dat <- ps_dat[ps_sc$sample_id, ]

# normalize
ref_dat <- log2(ref_dat + 1)
ps_dat <- log2(ps_dat + 1)

norm_dat <- rbind(ref_dat, ps_dat)

# Labels
cl <- make.names(c(ref_sc$X0, ps_sc$X0))
names(cl) <- rownames(norm_dat)

# create pairs
sort_ref_cl <- sort(unique(ref_sc$X0))
sort_ps_cl <- sort(unique(ps_sc$X0))

pairs <- cbind(make.names(sort_ref_cl), make.names(sort_ps_cl))
row.names(pairs) = paste0(pairs[,1],"_",pairs[,2])

de.param <- de_param(q.diff.th = 0)
res <- de_selected_pairs(t(norm_dat), cl, pairs)
stats <- de_stats_selected_pairs(t(norm_dat), cl, pairs, de.param = de.param)

# find genes consistently modulated across (nearly all) subclasses

consistent_up <- table(unlist(lapply(stats, function(x) x$up.genes)))
consistent_down <- table(unlist(lapply(stats, function(x) x$down.genes)))

min_group = 7
consistent_up <- consistent_up[consistent_up >= min_group]
consistent_down <- consistent_down[consistent_down >= min_group]

up_lfc <- lapply(res, function(x) {
	my_lfc <- x[names(consistent_up), ]$lfc
	names(my_lfc) <- names(consistent_up)
	my_lfc})
up_lfc_df <- as.data.frame(up_lfc)

down_lfc <- lapply(res, function(x) {
	my_lfc <- x[names(consistent_down), ]$lfc
	names(my_lfc) <- names(consistent_down)
	my_lfc})
down_lfc_df <- as.data.frame(down_lfc)

all_lfc_df <- rbind(up_lfc_df, down_lfc_df)

write.csv(all_lfc_df, output_file)