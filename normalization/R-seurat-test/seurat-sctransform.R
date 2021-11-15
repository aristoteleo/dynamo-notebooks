library(Seurat)
library(ggplot2)
library(sctransform)
library(SeuratData)
library(SeuratDisk)
library(Matrix)

# Convert("../data/zebrafish.h5ad", dest = "h5seurat", overwrite = FALSE)
# pbmc <- LoadH5Seurat("../data/zebrafish.h5seurat")
counts <- readMM("../data/counts.mtx")
cell_names = read.csv("../data/cell_names.csv")
rownames(counts) <- cell_names[, "index"]

var_names = read.csv("../data/var_names.csv")
colnames(counts) <- var_names[, "X0"]
counts <- counts[, colSums(counts) != 0]
counts <- t(counts)

pbmc <- CreateSeuratObject(counts = counts)

# h5seurat - never works
# pbmc_data <- Read10X(data.dir = "../data/pbmc3k/filtered_gene_bc_matrices/hg19/")
# pbmc <- CreateSeuratObject(counts = pbmc_data)

# store mitochondrial percentage in object meta data
pbmc <- PercentageFeatureSet(pbmc, pattern = "^MT-", col.name = "percent.mt")

# run sctransform
# # percent nt ver 
# pbmc <- SCTransform(pbmc, vars.to.regress = "percent.mt", verbose = FALSE)
pbmc <- SCTransform(pbmc, method = "glmGamPoi", vars.to.regress = "percent.mt", verbose = FALSE)
pbmc <- RunPCA(pbmc, npcs=50, verbose = FALSE)
write.table(pbmc[["pca"]]@cell.embeddings, "X_pca.csv", sep=",")

pbmc <- RunUMAP(pbmc, dims = 1:50, verbose = FALSE)
pbmc <- FindNeighbors(pbmc, dims = 1:50, verbose = FALSE)
pbmc <- FindClusters(pbmc, verbose = FALSE)
DimPlot(pbmc, label = TRUE) + NoLegend()