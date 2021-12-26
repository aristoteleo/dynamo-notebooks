# %%
dir = "/Users/random/dynamo-notebooks/normalization"
import os
os.chdir(dir)
import dynamo as dyn
import anndata
zebrafish_adata = dyn.sample_data.zebrafish()
adata = dyn.sample_data.zebrafish()


# %%
path = "./data/zebrafish_3d_umap.h5ad"
# dyn.pp.recipe_monocle(adata) 
# dyn.tl.reduceDimension(adata, n_components=3)

# dyn.tl.dynamics(adata)
# dyn.data_io.cleanup(adata)
# adata.write_h5ad(path)
adata = anndata.read_h5ad(path)

# %%
adata

# %%
dyn.dynamo_logger.main_set_level(dyn.dynamo_logger.LoggerManager.DEBUG)
quiver_3d_kwargs = {
    "zorder": 3, 
    "linewidth": 1, 
    "arrow_length_ratio": 10
    }
# dyn.pl.cell_wise_vectors_3d(adata, X=adata.obsm["X_umap"], V=adata.obsm["X_pca"], save_show_or_return="show", qigsize=(10, 30), quiver_3d_kwargs=quiver_3d_kwargs)

# %%

dyn.pl.cell_wise_vectors(adata, basis="umap", projection="3d", vector="X", save_show_or_return="show")
# dyn.pl.umap(adata)


