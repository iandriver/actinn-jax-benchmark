#!/usr/bin/env Rscript
# Generic R bridge for R-based annotation methods.
#
# Usage: Rscript run_r_method.R <method> <workdir> <out.csv>
# Reads from <workdir>: ref_counts.mtx, ref_genes.txt, ref_labels.txt,
#                       query_counts.mtx, query_genes.txt  (counts = genes x cells)
# Writes <out.csv>: one column `pred_label`, one row per query cell (query order).
#
# The project R library is passed via R_LIBS_USER by the caller.

suppressMessages({
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; work <- args[2]; out <- args[3]

read_counts <- function(prefix) {
  m <- as(readMM(file.path(work, paste0(prefix, "_counts.mtx"))), "CsparseMatrix")
  genes <- readLines(file.path(work, paste0(prefix, "_genes.txt")))
  rownames(m) <- genes
  colnames(m) <- paste0("cell", seq_len(ncol(m)))
  m
}

ref <- read_counts("ref")
query <- read_counts("query")
ref_labels <- readLines(file.path(work, "ref_labels.txt"))
common <- intersect(rownames(ref), rownames(query))
stopifnot(length(common) >= 500)
ref <- ref[common, , drop = FALSE]
query <- query[common, , drop = FALSE]

pred <- NULL

if (method == "singler") {
  suppressMessages({ library(SingleCellExperiment); library(scuttle); library(SingleR) })
  ref_sce <- logNormCounts(SingleCellExperiment(assays = list(counts = ref)))
  test_sce <- logNormCounts(SingleCellExperiment(assays = list(counts = query)))
  res <- SingleR(test = test_sce, ref = ref_sce, labels = ref_labels,
                 assay.type.test = "logcounts", assay.type.ref = "logcounts")
  pred <- res$labels

} else if (method == "scmap-cluster") {
  suppressMessages({ library(SingleCellExperiment); library(scuttle); library(scmap) })
  ref_sce <- logNormCounts(SingleCellExperiment(assays = list(counts = ref)))
  rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
  colData(ref_sce)$cell_type1 <- ref_labels
  ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE)
  ref_sce <- indexCluster(ref_sce, cluster_col = "cell_type1")
  test_sce <- logNormCounts(SingleCellExperiment(assays = list(counts = query)))
  rowData(test_sce)$feature_symbol <- rownames(test_sce)
  res <- scmapCluster(test_sce, list(ref = metadata(ref_sce)$scmap_cluster_index))
  pred <- as.character(res$combined_labs)

} else if (method == "scpred") {
  suppressMessages({ library(Seurat); library(scPred); library(magrittr) })
  ref_so <- CreateSeuratObject(counts = ref)
  ref_so$cell_type <- ref_labels
  ref_so <- ref_so %>% NormalizeData() %>% FindVariableFeatures() %>%
    ScaleData() %>% RunPCA(npcs = 30)
  ref_so <- getFeatureSpace(ref_so, "cell_type")
  ref_so <- trainModel(ref_so)
  query_so <- CreateSeuratObject(counts = query) %>% NormalizeData()
  query_so <- scPredict(query_so, ref_so)
  pred <- as.character(query_so$scpred_prediction)

} else {
  stop(paste("unknown method:", method))
}

write.csv(data.frame(pred_label = pred), out, row.names = FALSE)
cat("R_METHOD_DONE", method, length(pred), "\n")
