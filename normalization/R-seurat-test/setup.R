if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk")
devtools::install_github('satijalab/seurat-data')
BiocManager::install("glmGamPoi")

library(devtools)
# install local version for debug
# install('D:\\seurat')
