import time
from anndata import AnnData
from dynamo.preprocessing import Preprocessor
from typing import Callable
import dynamo as dyn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dynamo.configuration import DKM
import warnings
warnings.filterwarnings('ignore')


def get_clean_organoid_data():
    adata = dyn.sample_data.scEU_seq_organoid()
    adata.obs.time = adata.obs.time.astype('str')
    adata.obs.loc[adata.obs['time'] == 'dmso', 'time'] = -1
    adata.obs['time'] = adata.obs['time'].astype(float)
    adata = adata[adata.obs.time != -1, :]
    adata = adata[adata.obs.exp_type == 'Pulse', :]
    adata.layers['new'], adata.layers['total'] = adata.layers['ul'] + \
        adata.layers['sl'], adata.layers['su'] + \
        adata.layers['sl'] + adata.layers['uu'] + adata.layers['ul']
    del adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
    return adata


def get_clean_rp1_data():
    adata = dyn.sample_data.scEU_seq_rpe1()
    rpe1_kinetics = adata[adata.obs.exp_type == 'Pulse', :]
    rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(str)
    rpe1_kinetics.obs.loc[rpe1_kinetics.obs['time'] == 'dmso', 'time'] = -1
    rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(float)
    rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time != -1, :]

    rpe1_kinetics.layers['new'], rpe1_kinetics.layers['total'] = rpe1_kinetics.layers['ul'] + \
        rpe1_kinetics.layers['sl'], rpe1_kinetics.layers['su'] + \
        rpe1_kinetics.layers['sl'] + \
        rpe1_kinetics.layers['uu'] + rpe1_kinetics.layers['ul']

    del rpe1_kinetics.layers['uu'], rpe1_kinetics.layers['ul'], rpe1_kinetics.layers['su'], rpe1_kinetics.layers['sl']
    return rpe1_kinetics


def get_clean_chromaffin_data():
    adata = dyn.sample_data.chromaffin()
    adata.obs["time"] = 1
    return adata


def get_clean_hgForebrainGlutamatergic_data():
    adata = dyn.sample_data.hgForebrainGlutamatergic()
    adata.obs["time"] = 1
    return adata


def get_clean_bm_data():
    adata = dyn.sample_data.BM()
    adata.obs["time"] = 1
    return adata


def recipe_benchmark(adata: AnnData, recipe: str, tkey=None):
    preprocessor = Preprocessor()
    preprocessor.preprocess_adata(adata, recipe=recipe, tkey=tkey)


def benchmark_umap(adata: AnnData, color: str, file_prefix="pp"):
    dyn.tl.reduceDimension(adata, basis="pca")
    dyn.tl.reduceDimension(adata, basis="pca")
    dyn.pl.umap(adata, color=color, figsize=(10, 10))  # , ax=axes[0])
    plt.savefig("_".join([file_prefix, "umap.png"]))


def benchmark_dynamics(adata: AnnData, color: str, file_prefix="pp", dynamics_kwargs = {}):
    dyn.tl.dynamics(adata, cores=16, **dynamics_kwargs)
    dyn.tl.cell_velocities(adata)
    dyn.pl.streamline_plot(adata, color=color)


def benchmark_all_recipes(data_generator: Callable,
                          color, 
                          recipes=["monocle", "seurat",
                                    "pearson_residuals", "sctransform"],
                            dynamics_kwargs = {}, 
                          save_dir="./", tkey=None, dataset_name="pancreas"):

    do_dynamics_benchmarks = ["monocle", "seurat", "pearson_residuals", "sctransform"]
    recipe2time = {}
    for recipe in recipes:
        adata = data_generator()
        start_time = time.time()
        recipe_benchmark(adata, recipe=recipe, tkey=tkey)
        duration = time.time() - start_time
        recipe2time[recipe] = duration
        benchmark_umap(adata, color=color, file_prefix=dataset_name)
        if recipe in do_dynamics_benchmarks:
            benchmark_dynamics(adata, color=color, dynamics_kwargs=dynamics_kwargs)
    return recipe2time
