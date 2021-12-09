from typing import Callable
import dynamo as dyn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dynamo.configuration import DKM
import warnings
warnings.filterwarnings('ignore')
from dynamo.preprocessing import Preprocessor
from anndata import AnnData


def get_clean_organoid_data():
    adata = dyn.sample_data.scEU_seq_organoid()
    adata.obs.time = adata.obs.time.astype('str')
    adata.obs.loc[adata.obs['time'] == 'dmso', 'time'] = -1
    adata.obs['time'] = adata.obs['time'].astype(float)
    adata = adata[adata.obs.time != -1, :]
    adata = adata[adata.obs.exp_type == 'Pulse', :]
    adata.layers['new'], adata.layers['total'] = adata.layers['ul'] + adata.layers['sl'], adata.layers['su'] + adata.layers['sl'] + adata.layers['uu'] + adata.layers['ul']
    del adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
    return adata


def get_clean_rp1_data():
    adata = dyn.sample_data.scEU_seq_rpe1()
    rpe1_kinetics = adata[adata.obs.exp_type=='Pulse', :]
    rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(str)
    rpe1_kinetics.obs.loc[rpe1_kinetics.obs['time'] == 'dmso', 'time'] = -1
    rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(float)
    rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time != -1, :]

    rpe1_kinetics.layers['new'], rpe1_kinetics.layers['total'] = rpe1_kinetics.layers['ul'] + rpe1_kinetics.layers['sl'], rpe1_kinetics.layers['su'] + rpe1_kinetics.layers['sl'] + rpe1_kinetics.layers['uu'] + rpe1_kinetics.layers['ul']

    del rpe1_kinetics.layers['uu'], rpe1_kinetics.layers['ul'], rpe1_kinetics.layers['su'], rpe1_kinetics.layers['sl']
    return rpe1_kinetics


def recipe_benchmark(adata: AnnData, recipe:str, tkey=None):
    preprocessor = Preprocessor()
    preprocessor.preprocess_adata(adata, recipe=recipe, tkey=tkey)


def benchmark_after_pp(adata: AnnData, color: str, file_prefix="pp"):
    dyn.tl.reduceDimension(adata,basis="pca")
    dyn.tl.reduceDimension(adata,basis="pca")
    dyn.pl.umap(adata, color=color, figsize=(10, 10))  # , ax=axes[0])
    plt.savefig("_".join([file_prefix, "umap.png"]))


def benchmark_all_recipes(data_generator: Callable, color, save_dir="./", tkey=None):
    recipes = ["monocle", "seurat", "sctransform", "pearson_residuals"]
    for recipe in recipes:
        adata = data_generator()
        recipe_benchmark(adata, recipe=recipe, tkey=tkey)
        benchmark_after_pp(adata, color=color, file_prefix="pancreas")