from scipy.sparse import issparse
from typing import Optional, Tuple
from anndata import AnnData
import pandas as pd
import numpy as np

from typing import Optional, Dict
from warnings import warn
import dynamo as dyn
from dynamo.dynamo_logger import LoggerManager, main_info
from dynamo.configuration import DKM

main_logger = LoggerManager.main_logger
from dynamo.preprocessing.preprocessor_utils import seurat_get_mean_var
from dynamo.preprocessing.preprocessor_utils import filter_genes_by_outliers, is_nonnegative_integer_arr
from dynamo.preprocessing.utils import pca
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from anndata import AnnData


def _highly_variable_pearson_residuals(
    adata: AnnData,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 1000,
    batch_key: Optional[str] = None,
    chunksize: int = 1000,
    check_values: bool = True,
    layer: Optional[str] = None,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """\
    See `scanpy.experimental.pp.highly_variable_genes`.

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`)
    or updates `.var` with the following fields:

    highly_variable : bool
        boolean indicator of highly-variable genes
    means : float
        means per gene
    variances : float
        variance per gene
    residual_variances : float
        Residual variance per gene. Averaged in the case of multiple batches.
    highly_variable_rank : float
        Rank of the gene according to residual variance, median rank in the case of multiple batches
    highly_variable_nbatches : int
        If `batch_key` given, denotes in how many batches genes are detected as HVG
    highly_variable_intersection : bool
        If `batch_key` given, denotes the genes that are highly variable in all batches
    """

    # view_to_actual(adata)
    X = DKM.select_layer_data(adata, layer)
    _computed_on_prompt_str = layer if layer else "adata.X"

    # Check for raw counts
    if check_values and (is_nonnegative_integer_arr(X) is False):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    # check theta
    if theta <= 0:
        # TODO: would "underdispersion" with negative theta make sense?
        # then only theta=0 were undefined..
        raise ValueError("Pearson residuals require theta > 0")
    # prepare clipping

    if batch_key is None:
        batch_info = np.zeros(adata.shape[0], dtype=int)
    else:
        batch_info = adata.obs[batch_key].values
    n_batches = len(np.unique(batch_info))

    # Get pearson residuals for each batch separately
    residual_gene_vars = []
    for batch in np.unique(batch_info):

        adata_subset = adata[batch_info == batch]

        # Filter out zero genes

        nonzero_genes = filter_genes_by_outliers(adata_subset, min_cell_s=1)
        adata_subset = adata_subset[:, nonzero_genes]

        if layer is not None:
            X_batch = adata_subset.layers[layer]
        else:
            X_batch = adata_subset.X

        # Prepare clipping
        if clip is None:
            n = X_batch.shape[0]
            clip = np.sqrt(n)
        if clip < 0:
            raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

        if sp_sparse.issparse(X_batch):
            sums_genes = np.sum(X_batch, axis=0)
            sums_cells = np.sum(X_batch, axis=1)
            sum_total = np.sum(sums_genes).squeeze()
        else:
            sums_genes = np.sum(X_batch, axis=0, keepdims=True)
            sums_cells = np.sum(X_batch, axis=1, keepdims=True)
            sum_total = np.sum(sums_genes)

        # Compute pearson residuals in chunks
        residual_gene_var = np.empty((X_batch.shape[1]))
        for start in np.arange(0, X_batch.shape[1], chunksize):
            stop = start + chunksize
            mu = np.array(sums_cells @ sums_genes[:, start:stop] / sum_total)
            X_dense = X_batch[:, start:stop].toarray()
            residuals = (X_dense - mu) / np.sqrt(mu + mu ** 2 / theta)
            residuals = np.clip(residuals, a_min=-clip, a_max=clip)
            residual_gene_var[start:stop] = np.var(residuals, axis=0)

        # Add 0 values for genes that were filtered out
        unmasked_residual_gene_var = np.zeros(len(nonzero_genes))
        unmasked_residual_gene_var[nonzero_genes] = residual_gene_var
        residual_gene_vars.append(unmasked_residual_gene_var.reshape(1, -1))

    residual_gene_vars = np.concatenate(residual_gene_vars, axis=0)

    # Get rank per gene within each batch
    # argsort twice gives ranks, small rank means most variable
    ranks_residual_var = np.argsort(np.argsort(-residual_gene_vars, axis=1), axis=1)
    ranks_residual_var = ranks_residual_var.astype(np.float32)
    # count in how many batches a genes was among the n_top_genes
    highly_variable_nbatches = np.sum((ranks_residual_var < n_top_genes).astype(int), axis=0)
    # set non-top genes within each batch to nan
    ranks_residual_var[ranks_residual_var >= n_top_genes] = np.nan
    ranks_masked_array = np.ma.masked_invalid(ranks_residual_var)
    # Median rank across batches, ignoring batches in which gene was not selected
    medianrank_residual_var = np.ma.median(ranks_masked_array, axis=0).filled(np.nan)

    means, variances = seurat_get_mean_var(X)
    df = pd.DataFrame.from_dict(
        dict(
            means=means,
            variances=variances,
            residual_variances=np.mean(residual_gene_vars, axis=0),
            highly_variable_rank=medianrank_residual_var,
            highly_variable_nbatches=highly_variable_nbatches.astype(np.int64),
            highly_variable_intersection=highly_variable_nbatches == n_batches,
        )
    )
    df = df.set_index(adata.var_names)

    # Sort genes by how often they selected as hvg within each batch and
    # break ties with median rank of residual variance across batches
    df.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )

    high_var = np.zeros(df.shape[0])
    high_var[:n_top_genes] = True
    df["highly_variable"] = high_var.astype(bool)
    df = df.loc[adata.var_names, :]

    if inplace:
        adata.uns["hvg"] = {"flavor": "pearson_residuals", "computed_on": _computed_on_prompt_str}
        main_logger.debug(
            "added\n"
            "    'highly_variable', boolean vector (adata.var)\n"
            "    'highly_variable_rank', float vector (adata.var)\n"
            "    'highly_variable_nbatches', int vector (adata.var)\n"
            "    'highly_variable_intersection', boolean vector (adata.var)\n"
            "    'means', float vector (adata.var)\n"
            "    'variances', float vector (adata.var)\n"
            "    'residual_variances', float vector (adata.var)"
        )
        adata.var["means"] = df["means"].values
        adata.var["variances"] = df["variances"].values
        adata.var["residual_variances"] = df["residual_variances"]
        adata.var["highly_variable_rank"] = df["highly_variable_rank"].values
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
            adata.var["highly_variable_intersection"] = df["highly_variable_intersection"].values
        adata.var["highly_variable"] = df["highly_variable"].values

        if subset:
            adata._inplace_subset_var(df["highly_variable"].values)

    else:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches", "highly_variable_intersection"], axis=1)
        if subset:
            df = df.iloc[df.highly_variable.values, :]

        return df


def highly_variable_genes(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: Optional[int] = None,
    batch_key: Optional[str] = None,
    chunksize: int = 1000,
    flavor: str = "pearson_residuals",
    check_values: bool = True,
    layer: Optional[str] = None,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """\
    Annotate highly variable genes using analytic Pearson residuals [Lause21]_.

    For [Lause21]_, Pearson residuals of a negative binomial offset model (with
    overdispersion theta shared across genes) are computed. By default, overdispersion
    theta=100 is used and residuals are clipped to sqrt(n). Finally, genes are ranked
    by residual variance.

    Expects raw count input.


    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    theta
        The negative binomial overdispersion parameter theta for Pearson residuals.
        Higher values correspond to less overdispersion (var = mean + mean^2/theta),
        and `theta=np.Inf` corresponds to a Poisson model.
    clip
        If `flavor='pearson_residuals'`, determines if and how residuals are clipped:

            * If `None`, residuals are clipped to the interval [-sqrt(n), sqrt(n)], \
            where n is the number of cells in the dataset (default behavior).
            * If any scalar c, residuals are clipped to the interval [-c, c]. Set \
            `clip=np.Inf` for no clipping.

    n_top_genes
        Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'` or
        `flavor='pearson_residuals'`.
    batch_key
        If specified, highly-variable genes are selected within each batch separately
        and merged. This simple process avoids the selection of batch-specific genes
        and acts as a lightweight batch correction method. Genes are first sorted by
        how many batches they are a HVG. If `flavor='pearson_residuals'`, ties are
        broken by the median rank (across batches) based on within-batch residual
        variance.
    chunksize
        If `flavor='pearson_residuals'`, this dertermines how many genes are processed at
        once while computing the residual variance. Choosing a smaller value will reduce
        the required memory.
    flavor
        Choose the flavor for identifying highly variable genes. In this experimental
        version, only 'pearson_residuals' is functional.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to
        True. Only used if `flavor='pearson_residuals'`.
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pandas.DataFrame`) or
    updates `.var` with the following fields

    highly_variable : bool
        boolean indicator of highly-variable genes
    means : float
        means per gene
    variances : float
        variance per gene
    residual_variances : float
        For `flavor='pearson_residuals'`, residual variance per gene. Averaged in the
        case of multiple batches.
    highly_variable_rank : float
        For `flavor='pearson_residuals'`, rank of the gene according to residual
        variance, median rank in the case of multiple batches
    highly_variable_nbatches : int
        If `batch_key` given, denotes in how many batches genes are detected as HVG
    highly_variable_intersection : bool
        If `batch_key` given, denotes the genes that are highly variable in all batches

    Notes
    -----
    Experimental version of `sc.pp.highly_variable_genes()`
    """

    main_logger.info("extracting highly variable genes")

    if not isinstance(adata, AnnData):
        raise ValueError(
            "`pp.highly_variable_genes` expects an `AnnData` argument, "
            "pass `inplace=False` if you want to return a `pd.DataFrame`."
        )

    if flavor == "pearson_residuals":
        if n_top_genes is None:
            raise ValueError(
                "`pp.highly_variable_genes` requires the argument `n_top_genes`" " for `flavor='pearson_residuals'`"
            )
        return _highly_variable_pearson_residuals(
            adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            theta=theta,
            clip=clip,
            chunksize=chunksize,
            subset=subset,
            check_values=check_values,
            inplace=inplace,
        )


def _pearson_residuals(X, theta, clip, check_values, copy=False):

    X = X.copy() if copy else X

    # check theta
    if theta <= 0:
        # TODO: would "underdispersion" with negative theta make sense?
        # then only theta=0 were undefined..
        raise ValueError("Pearson residuals require theta > 0")
    # prepare clipping
    if clip is None:
        n = X.shape[0]
        clip = np.sqrt(n)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

    if check_values and not is_nonnegative_integer_arr(X):
        warn(
            "`normalize_pearson_residuals()` expects raw count data, but non-integers were found.",
            UserWarning,
        )

    if issparse(X):
        sums_genes = np.sum(X, axis=0)
        sums_cells = np.sum(X, axis=1)
        sum_total = np.sum(sums_genes).squeeze()
    else:
        sums_genes = np.sum(X, axis=0, keepdims=True)
        sums_cells = np.sum(X, axis=1, keepdims=True)
        sum_total = np.sum(sums_genes)

    mu = np.array(sums_cells @ sums_genes / sum_total)
    diff = np.array(X - mu)
    residuals = diff / np.sqrt(mu + mu ** 2 / theta)

    # clip
    residuals = np.clip(residuals, a_min=-clip, a_max=clip)

    return residuals


def normalize_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    copy: bool = False,
    inplace: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """\
    Applies analytic Pearson residual normalization, based on [Lause21]_.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to sqrt(n) and
    overdispersion `theta=100` is used.
    Expects raw count input.
    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    theta
        The negative binomial overdispersion parameter theta for Pearson residuals.
        Higher values correspond to less overdispersion (var = mean + mean^2/theta),
        and `theta=np.Inf` corresponds to a Poisson model.
    clip
        Determines if and how residuals are clipped:
            * If `None`, residuals are clipped to the interval [-sqrt(n), sqrt(n)], \
            where n is the number of cells in the dataset (default behavior).
            * If any scalar c, residuals are clipped to the interval [-c, c]. Set \
            `clip=np.Inf` for no clipping.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to
        True.
    layer
        Layer to normalize instead of `X`. If `None`, `X` is normalized.
    copy
        Whether to modify copied input object. Not compatible with `inplace=False`.
    inplace
        Whether to update `adata` or return dictionary with normalized copies
        of `adata.X` and `adata.layers`.
    Returns
    -------
    Returns dictionary with Pearson residuals and settings
    or updates `adata` with normalized version of the original
    `adata.X` and `adata.layers`, depending on `inplace`.
    """

    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    # view_to_actual(adata)

    if layer is None:
        layer = DKM.X_LAYER
    X = DKM.select_layer_data(adata, layer=layer)
    computed_on = layer if layer else "adata.X"

    msg = f"computing analytic Pearson residuals on {computed_on}"
    main_logger.info(msg)
    main_logger.log_time()

    residuals = _pearson_residuals(X, theta, clip, check_values, copy=~inplace)
    settings_dict = dict(theta=theta, clip=clip, computed_on=computed_on)

    if inplace:
        main_logger.info("replace layer %s with pearson residuals." % (layer))
        DKM.set_layer_data(adata, layer, residuals)
        adata.uns["pearson_residuals_normalization"] = settings_dict
    else:
        results_dict = dict(X=residuals, **settings_dict)

    main_logger.finish_progress(progress_name="pearson residual normalization")

    if copy:
        return adata
    elif not inplace:
        return results_dict


def normalize_pearson_residuals_pca(
    adata: AnnData,
    theta: float = 100,
    clip: Optional[float] = None,
    n_comps: Optional[int] = 50,
    random_state: Optional[float] = 0,
    kwargs_pca: Optional[dict] = {},
    use_highly_variable: Optional[bool] = None,
    check_values: bool = True,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """\
    Applies analytic Pearson residual normalization and PCA, based on [Lause21]_.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to sqrt(n),
    overdispersion `theta=100` is used, and PCA is run with 50 components.
    Operates on the subset of highly variable genes in `adata.var['highly_variable']`
    by default. Expects raw count input.
    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    theta
        The negative binomial overdispersion parameter theta for Pearson residuals.
        Higher values correspond to less overdispersion (var = mean + mean^2/theta),
        and `theta=np.Inf` corresponds to a Poisson model.
    clip
        Determines if and how residuals are clipped:
            * If `None`, residuals are clipped to the interval [-sqrt(n), sqrt(n)], \
            where n is the number of cells in the dataset (default behavior).
            * If any scalar c, residuals are clipped to the interval [-c, c]. Set \
            `clip=np.Inf` for no clipping.
    n_comps
        Number of principal components to compute for the PCA step.
    random_state
        Change to use different initial states for the optimization of the PCA step.
    kwargs_pca
        Dictionary of further keyword arguments passed on to `scanpy.pp.pca()`.
    use_highly_variable
        Whether to use the gene selection in `adata.var['highly_variable']` to subset
        the data before normalizing (default) or proceed on the full dataset.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to
        True.
    inplace
        Whether to place results in `adata` or return them.
    Returns
    -------
    If `inplace=False`, returns the Pearson residual-based PCA results
    (`adata_pca`).
    If `inplace=True`, updates `adata` with the following fields:
    `.uns['pearson_residuals_normalization']['pearson_residuals_df']`
         The hvg-subset, normalized by Pearson residuals
    `.uns['pearson_residuals_normalization']['theta']`
         The used value of the overdisperion parameter theta
    `.uns['pearson_residuals_normalization']['clip']`
         The used value of the clipping parameter
    `.obsm['X_pca']`
        PCA representation of data after gene selection (if applicable) and Pearson
        residual normalization.
    `.varm['PCs']`
         The principal components containing the loadings. When `inplace=True` and
         `use_highly_variable=True`, this will contain empty rows for the genes not
         selected.
    `.uns['pca']['variance_ratio']`
         Ratio of explained variance.
    `.uns['pca']['variance']`
         Explained variance, equivalent to the eigenvalues of the covariance matrix.
    """

    # check if HVG selection is there if user wants to use it
    if use_highly_variable and "highly_variable" not in adata.var_keys():
        raise ValueError(
            "You passed `use_highly_variable=True`, but no HVG selection was found (`highly_variable` missing in `adata.var_keys()`."
        )

    # default behavior: if there is a HVG selection, we will use it
    if use_highly_variable is None and "highly_variable" in adata.var_keys():
        use_highly_variable = True

    if use_highly_variable:
        adata_sub = adata[:, adata.var["highly_variable"]].copy()
        adata_pca = AnnData(adata_sub.X.copy(), obs=adata_sub.obs[[]], var=adata_sub.var[[]])
    else:
        adata_pca = AnnData(adata.X.copy(), obs=adata.obs[[]], var=adata.var[[]])

    normalize_pearson_residuals(adata_pca, theta=theta, clip=clip, check_values=check_values)
    pca(adata_pca, n_pca_components=n_comps, pca_key="X_pca", **kwargs_pca)

    if inplace:
        norm_settings = adata_pca.uns["pearson_residuals_normalization"]
        norm_dict = dict(**norm_settings, pearson_residuals_df=adata_pca.to_df())
        if use_highly_variable:
            adata.varm["PCs"] = np.zeros(shape=(adata.n_vars, n_comps))
            adata.varm["PCs"][adata.var["highly_variable"]] = adata_pca.varm["PCs"]
        else:
            adata.varm["PCs"] = adata_pca.varm["PCs"]
        adata.uns["pca"] = adata_pca.uns["pca"]
        adata.uns["pearson_residuals_normalization"] = norm_dict
        adata.obsm["X_pca"] = adata_pca.obsm["X_pca"]
        return None
    else:
        return adata_pca


def recipe_pearson_residuals(
    adata: AnnData,
    layer: str = None,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 1000,
    batch_key: Optional[str] = None,
    chunksize: int = 1000,
    n_comps: Optional[int] = 50,
    random_state: Optional[float] = 0,
    kwargs_pca: dict = {},
    check_values: bool = True,
    inplace: bool = True,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """\
    Gene selection and normalization based on [Lause21]_.
    Applies gene selection based on Pearson residuals. On the resulting subset,
    Pearson residual normalization and PCA are performed.
    Expects raw count input.
    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    theta
        The negative binomial overdispersion parameter theta for Pearson residuals.
        Higher values correspond to less overdispersion (var = mean + mean^2/theta),
        and `theta=np.Inf` corresponds to a Poisson model.
    clip
        Determines if and how residuals are clipped:
            * If `None`, residuals are clipped to the interval [-sqrt(n), sqrt(n)], \
            where n is the number of cells in the dataset (default behavior).
            * If any scalar c, residuals are clipped to the interval [-c, c]. Set \
            `clip=np.Inf` for no clipping.
    n_top_genes
        Number of highly-variable genes to keep.
    batch_key
        If specified, highly-variable genes are selected within each batch separately
        and merged. This simple process avoids the selection of batch-specific genes
        and acts as a lightweight batch correction method. Genes are first sorted by
        how many batches they are a HVG. Ties are broken by the median rank (across
        batches) based on within-batch residual variance.
    chunksize
        This dertermines how many genes are processed at once while computing
        the Pearson residual variance. Choosing a smaller value will reduce
        the required memory.
    n_comps
        Number of principal components to compute in the PCA step.
    random_state
        Change to use different initial states for the optimization in the PCA step.
    kwargs_pca
        Dictionary of further keyword arguments passed on to `scanpy.pp.pca()`.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to
        True.
    inplace
        Whether to place results in `adata` or return them.
    Returns
    -------
    If `inplace=False`, separately returns the gene selection results (`hvg`)
    and Pearson residual-based PCA results (`adata_pca`). If `inplace=True`,
    updates `adata` with the following fields for gene selection results:
    `.var['highly_variable']` : bool
        boolean indicator of highly-variable genes.
    `.var['means']` : float
        means per gene.
    `.var['variances']` : float
        variances per gene.
    `.var['residual_variances']` : float
        Pearson residual variance per gene. Averaged in the case of multiple
        batches.
    `.var['highly_variable_rank']` : float
        Rank of the gene according to residual variance, median rank in the
        case of multiple batches.
    `.var['highly_variable_nbatches']` : int
        If batch_key is given, this denotes in how many batches genes are
        detected as HVG.
    `.var['highly_variable_intersection']` : bool
        If batch_key is given, this denotes the genes that are highly variable
        in all batches.
    …and the following fields for Pearson residual-based PCA results and
    normalization settings:
    `.uns['pearson_residuals_normalization']['pearson_residuals_df']`
         The hvg-subset, normalized by Pearson residuals.
    `.uns['pearson_residuals_normalization']['theta']`
         The used value of the overdisperion parameter theta.
    `.uns['pearson_residuals_normalization']['clip']`
         The used value of the clipping parameter.
    `.obsm['X_pca']`
        PCA representation of data after gene selection and Pearson residual
        normalization.
    `.varm['PCs']`
         The principal components containing the loadings. When `inplace=True` this
         will contain empty rows for the genes not selected during HVG selection.
    `.uns['pca']['variance_ratio']`
         Ratio of explained variance.
    `.uns['pca']['variance']`
         Explained variance, equivalent to the eigenvalues of the covariance matrix.
    """
    if layer is None:
        layer = DKM.X_LAYER
    main_info("Gene selection and normalization on layer: " + layer)
    hvg_args = dict(
        flavor="pearson_residuals",
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        theta=theta,
        clip=clip,
        chunksize=chunksize,
        check_values=check_values,
    )

    if inplace:
        highly_variable_genes(adata, **hvg_args, inplace=True)
        # TODO: are these copies needed?
        adata_pca = adata[:, adata.var["highly_variable"]].copy()
    else:
        hvg = highly_variable_genes(adata, **hvg_args, inplace=False)
        # TODO: are these copies needed?
        adata_pca = adata[:, hvg["highly_variable"]].copy()

    normalize_pearson_residuals(adata_pca, layer=layer, theta=theta, clip=clip, check_values=check_values)
    pca(adata_pca, n_pca_components=n_comps, pca_key="X_pca", **kwargs_pca)

    if inplace:
        normalization_param = adata_pca.uns["pearson_residuals_normalization"]
        normalization_dict = dict(**normalization_param, pearson_residuals_df=adata_pca.to_df())

        # adata.uns['pca'] = adata_pca.uns['pca']
        adata.uns["PCs"] = np.zeros(shape=(adata.n_vars, n_comps))
        adata.uns["PCs"][adata.var["highly_variable"]] = adata_pca.uns["PCs"]
        adata.uns["pearson_residuals_normalization"] = normalization_dict
        print(adata_pca)
        adata.obsm["X_pca"] = adata_pca.obsm["X_pca"]
        return None
    else:
        return adata_pca, hvg
