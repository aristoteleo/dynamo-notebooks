import dynamo as dyn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dynamo.configuration import DKM, DynamoAdataKeyManager
import warnings
import scvelo as scv
from dynamo.preprocessing import Preprocessor
import pearson_residual_normalization_recipe
import scipy.sparse
warnings.filterwarnings('ignore')

def get_nonzero_np_arr(X: scipy.sparse.csr_matrix):
    return np.array(X[X.nonzero()]).flatten()

def plot_scatter_sparse(m1: scipy.sparse.csr_matrix, m2: scipy.sparse.csr_matrix, title="", **kwargs):
    sns.scatterplot(np.array(m1[m1.nonzero()]).flatten(), np.array(m2[m1.nonzero()]).flatten(), **kwargs).set_title(title)

def pp_benchmark_zebrafish(adata, other_adata):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    plot_scatter_sparse(adata.X, other_adata.X, ax=axes[0][0], title="X comparison")
    plot_scatter_sparse(adata.obsm["X_pca"], other_adata.obsm["X_pca"], ax=axes[0][1], title="X_pca comparison")
    plot_scatter_sparse(adata.layers["X_unspliced"], other_adata.layers["X_unspliced"], ax=axes[1][0], title="X_unspliced comparison")
    plot_scatter_sparse(adata.layers["X_spliced"], other_adata.layers["X_spliced"], ax=axes[1][1], title="X_spliced comparison")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    sns.scatterplot(adata.obs["Size_Factor"], other_adata.obs["Size_Factor"], ax=axes[0][0]).set_title("Size factor comparison")
    sns.scatterplot(adata.obs["unspliced_Size_Factor"], other_adata.obs["unspliced_Size_Factor"], ax=axes[1][0]).set_title("unspliced_Size_Factor comparison")
    sns.scatterplot(adata.obs["spliced_Size_Factor"], other_adata.obs["spliced_Size_Factor"], ax=axes[1][1]).set_title("spliced_Size_Factor comparison")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    sns.histplot(adata.X.data, bins=40, legend=False, ax=axes[0][0]).set_title("preprocessor adata.X")
    sns.histplot(adata.obsm["X_pca"], bins=40, legend=False, ax=axes[0][1]).set_title("preprocessor X_pca")
    sns.histplot(other_adata.X.data, bins=40, legend=False, ax=axes[1][0]).set_title("monocle adata.X")
    sns.histplot(other_adata.obsm["X_pca"], bins=40, legend=False, ax=axes[1][1]).set_title("monocle X_pca")
    plt.show()

    dyn.tl.reduceDimension(adata,basis="pca")
    dyn.tl.dynamics(adata)
    dyn.pl.streamline_plot(adata, color=['Cell_type'], basis='umap', show_legend='on data', show_arrowed_spines=True)
    dyn.pl.phase_portraits(adata, genes=['tfec', 'pnp4a'],  figsize=(6, 4), color='Cell_type')

    dyn.tl.dynamics(other_adata)
    dyn.pl.streamline_plot(other_adata, color=['Cell_type'], basis='umap', show_legend='on data', show_arrowed_spines=True)
    dyn.pl.phase_portraits(other_adata, genes=['tfec', 'pnp4a'],  figsize=(6, 4), color='Cell_type')



def compare_gene_sets(adata, scv_adata, key):
    preprocess_genes = adata.var_names[adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY]]
    scv_genes = scv_adata.var_names[scv_adata.var[DKM.VAR_GENE_HIGHLY_VARIABLE_KEY]]
    assert not set(scv_genes).difference(set(preprocess_genes))
    assert not set(preprocess_genes).difference(set(scv_genes))