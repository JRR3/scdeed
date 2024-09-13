#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#August 2024
#########################################################
#This is a Python implementation of the R package
#scDEED
#https://www.nature.com/articles/s41467-024-45891-y
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
import os
import seaborn as sb
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scprep
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from typing import Optional
from typing import Union
from multiprocessing import Pool
from scipy.stats import describe
from time import perf_counter as clock
mpl.rcParams["figure.dpi"] = 600
mpl.use("agg")

class scDEED:
    #=================================================
    def __init__(self,
            input: Union[sc.AnnData, str],
            output: Optional[str] = "",
            input_is_matrix_market: Optional[bool] = False,
            n_pcs: Optional[int] = 2,
            use_highly_variable: Optional[bool] = False,
            use_full_matrix: Optional[bool] = False,
            frac_neighbors: Optional[float] = 0.50,
            embedding_type: Optional[str] = "all",
            main_title: Optional[str] = None,
            ):
        """
        The constructor takes the following inputs.

        :param input: Path to input directory or \
                AnnData object.
        :param output: Path to output directory.
        :param input_is_matrix_market: If true, \
                the directory should contain a \
                .mtx file, a barcodes.tsv file \
                and a genes.tsv file.

        :return: a scDEED object.
        :rtype: :obj:`scDEED`

        """

        #We use a directed graph to enforce the parent
        #to child relation.

        if isinstance(input, scDEED):
            pass

        elif isinstance(input, str):
            self.source = os.path.abspath(input)
            if self.source.endswith('.h5ad'):
                self.t0 = clock()
                self.A = sc.read_h5ad(self.source)
                self.tf = clock()
                delta = self.tf - self.t0
                txt = ('Elapsed time for loading: ' +
                        f'{delta:.2f} seconds.')
                print(txt)
            else:
                if input_is_matrix_market:
                    self.A = sc.read_10x_mtx(self.source)
                else:
                    for f in os.listdir(self.source):
                        if f.endswith('.h5ad'):
                            fname = os.path.join(
                                self.source, f)
                            self.t0 = clock()
                            self.A = sc.read_h5ad(fname)
                            self.tf = clock()
                            delta = self.tf - self.t0
                            txt = ('Elapsed time for ' +
                                   'loading: ' +
                                    f'{delta:.2f} seconds.')
                            print(txt)
                            break

        elif isinstance(input, sc.AnnData):
            self.A = input
        else:
            raise ValueError('Unexpected input type.')

        self.is_sparse = sp.issparse(self.A.X)

        #Name for the original expression matrix layer
        self.ori = "original"
        #Name for the permuted expression matrix layer
        self.per = "permuted"

        self.FDT = np.float64

        self.list_of_embeddings = []
        self.embedding_type = embedding_type

        if embedding_type == "all":
            self.list_of_embeddings = ["umap", "tsne"]
        else:
            self.list_of_embeddings = [embedding_type]



        if self.is_sparse:
            #Compute the density of the matrix
            rho = self.A.X.nnz / np.prod(self.A.X.shape)
            #If more than 50% of the matrix is occupied,
            #we generate a dense version of the matrix.

            #sparse_threshold = 0.50
            sparse_threshold = 1
            if use_full_matrix or sparse_threshold < rho:
                self.is_sparse = False
                X = self.A.X.toarray()
                txt = ("Using a dense representation" 
                       " of the count matrix.")
                print(txt)
                txt = ("Values will be converted to" 
                       f" {self.FDT}")
                print(txt)
                X = X.astype(self.FDT)
            else:
                self.is_sparse = True
                #Make sure we use a CSC format.
                X = sp.csc_array(self.A.X,
                                 dtype=self.FDT,
                                 copy=True)


        else:
            #The matrix is dense.
            print("The matrix is dense.")
            self.is_sparse = False
            X = self.A.X.copy()
            txt = ("Values will be converted to" 
                    f" {self.FDT}")
            print(txt)
            X = X.astype(self.FDT)

        if use_highly_variable:

            hv = "highly_variable"
            if hv in self.A.var:
                X = X[:, self.A.var[hv]]
            else:
                sc.pp.highly_variable_genes(
                    self.A,
                    flavor="seurat")
                #Note that when the flavor is seurat,
                #the assumption is that the data have been
                #previously log-normalized.
                X = X[:, self.A.var[hv]]

        n_cells = X.shape[0]

        #We create a copy of the original anndata object
        #to store the permuted version.
        self.B = self.A.copy()

        self.main_title = main_title

        self.A.uns["modality"] = "original"
        self.B.uns["modality"] = "permuted"

        # self.A.layers[self.ori] = X

        #If no output directory is provided,
        #we use the current working directory.
        if output == "":
            output = os.getcwd()
            output = os.path.join(output, "scdeed_outputs")
            print(f"Outputs will be saved in: {output}")

        if not os.path.exists(output):
            os.makedirs(output)

        self.output = os.path.abspath(output)

        self.n_pcs = n_pcs

        n_neighbors = frac_neighbors * n_cells
        self.n_neighbors = int(n_neighbors)

    #=================================
    def permute_expression_matrix(self):
        """
        This function permutes the entries of 
        each column/gene of the original expression
        matrix.

        We use the anndata object B to store anything 
        related to the permuted version.
        """

        X = self.B.X.copy()

        if not sp.issparse(X):
            #Matrix is full.
            #Permute the rows
            X = np.random.permutation(X)

        else:
            #Matrix is sparse.
            #Note that the matrix has the CSC format 
            #by design. This allows us to work directly
            #on the data structure of the matrix.
            n_genes = X.shape[1]
            for k in range(n_genes):
                col = X.getcol(k)
                p_col = np.random.permutation(col.data)
                start = X.indptr[k]
                end = X.indptr[k+1]
                X.data[start:end] = p_col
        
        self.B.X = X

    #=================================
    def compute_spaces_for_anndata(self,
                                   adata: sc.AnnData,
        ):

        sc.pp.pca(adata,
                  n_comps=self.n_pcs,
                  svd_solver="auto",
        )


        #Choose between the different 
        #types of embeddings
        if self.embedding_type == "tsne":
            self.compute_tsne(adata)

        elif self.embedding_type == "umap":
            self.compute_umap(adata)

        elif self.embedding_type == "all":
            self.compute_umap(adata)
            self.compute_tsne(adata)


    #=================================
    def plot_embedding(
            self,
            adata: sc.AnnData,
            descriptor: str,
            color_column: str,
            color_map: Union[str, dict],
            modifier: Optional[str] = "sc",
            set_legend_outside: Optional[bool] = False,
            order: Optional[str] = None,
            fig_format: Optional[str] = "png",
        ):
        """
        Plot the first two columns of the embedding.
        The color_column is the column in the 
        AnnData.obs data frame that contains 
        (a description of) the colors.
        """

        fig, ax = plt.subplots()

        loc = "best"
        if set_legend_outside:
            loc = (1, 0.5)

        mat = adata.obsm[descriptor]
        colors = adata.obs[color_column]

        #Change the order of the points if we want
        #to emphasize certain points in the plot.
        #The points that are located at the end are
        #plotted last.
        if order is None:
            pass
        else:
            indices = np.argsort(adata.obs[order])
            mat = mat[indices,:]
            colors = colors.iloc[indices]

        scprep.plot.scatter2d(
            mat,
            c = colors,
            cmap = color_map,
            legend_loc = loc,
            ticks=False,
            title=self.main_title,
            ax=ax,
        )

        ax.set_aspect("equal")

        modality = adata.uns["modality"]

        fname = f"{descriptor}_{modifier}_{modality}.{fig_format}"
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")

    #=================================
    def compute_tsne(self,
                     adata: sc.AnnData,
                     perplexity: Optional[float] = 30,
        ):

        sc.tl.tsne(adata,
                   perplexity=perplexity,
                   n_pcs=self.n_pcs,
                   use_rep="X_pca",
        )

    #=================================
    def compute_umap(self,
                     adata: sc.AnnData,
                     n_neighbors: Optional[int] = 30,
                     min_distance: Optional[float] = 0.9,
        ):


        neighbors_tag= "umap_neighbors"

        sc.pp.neighbors(adata,
                        n_neighbors = n_neighbors,
                        key_added = neighbors_tag,
                        use_rep="X_pca",
        )

        sc.tl.umap(adata,
                   neighbors_key= neighbors_tag,
        )



    #=================================
    def compute_null_distribution(self):
        mtx = self.A.obsm["X_pca"]
        p_mtx = self.permute_each_column(mtx)

    #=================================
    def plot_spaces_for_anndata(
            self,
            adata: sc.AnnData,
            color_column: Optional[str] = "colors",
            color_map: Optional[str] = "plasma",
    ):
        """
        Plot the pre-embedding space (projected onto R^2) 
        and the embedding space.
        """

        self.plot_embedding(adata,
                            "X_pca",
                            color_column,
                            color_map,
        )

        for embedding_type in self.list_of_embeddings:
            embedding_str = "X_" + embedding_type
            self.plot_embedding(adata,
                                embedding_str,
                                color_column,
                                color_map,
            )

    #=================================
    def compute_proximity_objects_for_anndata(
            self,
            adata: sc.AnnData,
        ):
        """
        This function defines the matrix objects
        that we use to construct the correlations.
        Two types of matrices are produced.
        One is the matrix of neighbors and the 
        second is the matrix of distances.
        The matrix of neighbors is computed using
        the NearestNeighbors function from sklearn.
        The matrix of distances is computed using
        the pairwise function from sklearn.
        """


        #The matrix of neighbors for the PCA space.
        pca_str = f"X_pca"
        pca_nn_obj = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="ball_tree",
            ).fit(adata.obsm[pca_str])

        pca_ngb = f"pca_neighbors"
        adata.obsm[pca_ngb] = pca_nn_obj.kneighbors(
            return_distance=False)

        #The matrix of neighbors for the embedding space.
        for embedding_type in self.list_of_embeddings:

            embedding_str = "X_" + embedding_type
            embd_nn_obj = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm="ball_tree",
                ).fit(adata.obsm[embedding_str])

            embd_sort_dist = (f"{embedding_type}"
                              "_sorted_distances")
            embd_ngb       = f"{embedding_type}_neighbors"
            (adata.obsm[embd_sort_dist],
             adata.obsm[embd_ngb]) = embd_nn_obj.kneighbors()

            embd_dist = f"{embedding_type}_distances"
            adata.obsm[embd_dist] = pairwise_distances(
                adata.obsm[embedding_str], metric="euclidean")

    #=================================
    def correlation_function(self,
                      embedding_dist_vec,
                      embedding_sorted_dist_vec,
                      pre_embedding_order,
                      ):
        """
        This function takes three arguments.
        All the vectors refer to one cell "C".
        (1) The distances between C and the other cells.
        (2) The sorted version of (1).
        (3) The nearest neighbors for C in the
        pre-embedding space.
        """
        a = embedding_dist_vec[pre_embedding_order]
        #b = embedding_dist_vec[embedding_order]
        b = embedding_sorted_dist_vec
        m = np.corrcoef(a,b)

        return m[0,1]

    #=================================
    def compute_correlations_for_anndata(self,
                                         adata: sc.AnnData):

        #embd_ngb_mtx = self.A.obsm[embd_ngb]
        # embd_ngb    = f"embedding_neighbors"

        pca_ngb        = f"pca_neighbors"
        pca_ngb_mtx    = adata.obsm[pca_ngb]

        for embedding_type in self.list_of_embeddings:
            embd_sort_dist = (f"{embedding_type}"
                              "_sorted_distances")
            embd_dist      = f"{embedding_type}_distances"

            embd_sort_dist_mtx = adata.obsm[embd_sort_dist]
            embd_dist_mtx      = adata.obsm[embd_dist]

            Z = zip(embd_dist_mtx,
                    embd_sort_dist_mtx,
                    pca_ngb_mtx)

            with Pool(processes=6) as pool:
                C = pool.starmap(self.correlation_function,
                                Z)

            corr_tag = f"{embedding_type}_correlations"
            adata.obs[corr_tag] = C
        


    #=================================
    def plot_distribution_for_anndata(self,
                                      adata: sc.AnnData,
                                      fig_format: Optional[str] = "png",
        ):
        for embedding_type in self.list_of_embeddings:
            corr_tag = f"{embedding_type}_correlations"
            fig, ax = plt.subplots()
            modality = adata.uns["modality"]
            xlabel = f"Correlation ({modality} cells)"
            scprep.plot.histogram(adata.obs[corr_tag],
                                  xlabel=xlabel,
                                  ylabel="# of cells",
                                  title=self.main_title,
                                  ax = ax,
            )

            fname = f"distribution_{corr_tag}_{modality}.{fig_format}"
            fname = os.path.join(self.output, fname)
            fig.savefig(fname, bbox_inches="tight")

    #=================================
    def determine_percentiles_for_anndata(
            self, adata: sc.AnnData):

        for embedding_type in self.list_of_embeddings:
            corr_tag = f"{embedding_type}_correlations"
            vec = adata.obs[corr_tag]
            pc = np.percentile(vec, [5, 95])
            percentile_tag = f"{embedding_type}_percentile"
            adata.uns[percentile_tag] = pc
            

    #=================================
    def classify_cells(self,
                       adata: sc.AnnData,
                       null_dist_adata: sc.AnnData,
        ):
        """
        The adata object will be classified
        using the null_dist_adata object. 
        The null_dist_adata was generated using a 
        permuted version of the original anndata.

        We have three categories:

        Trustworthy
        Undefined
        Dubious

        """

        color_map = {"Undefined": "gray",
                     "Dubious":"red",
                     "Trustworthy":"blue"}

        order_map = {"Undefined":0,
                     "Trustworthy":1,
                     "Dubious":2,
                     }

        for embedding_type in self.list_of_embeddings:

            corr_tag = f"{embedding_type}_correlations"
            percentile_tag = f"{embedding_type}_percentile"
            null_dist_percentiles = null_dist_adata.uns[
                percentile_tag]
            corr_vec = adata.obs[corr_tag]
            dubious  = corr_vec < null_dist_percentiles[0]
            trustworthy = null_dist_percentiles[1] < corr_vec
            status = f"{embedding_type}_status"
            adata.obs[status] = "Undefined"
            adata.obs.loc[dubious,     status] = "Dubious"
            adata.obs.loc[trustworthy, status] = "Trustworthy"

            embd_tag = f"X_{embedding_type}"


            #Rank the elements using the order_map.
            rank = f"{embedding_type}_rank"
            adata.obs[rank] = adata.obs[status].map(
                order_map)

            self.plot_embedding(adata,
                                descriptor  = embd_tag,
                                color_column= status,
                                color_map   = color_map,
                                modifier    = status,
                                order       = rank,
                                set_legend_outside=True)



    #=================================
    def run(self):

        self.permute_expression_matrix()

        self.compute_spaces_for_anndata(self.A)
        self.compute_spaces_for_anndata(self.B)

        self.plot_spaces_for_anndata(self.A)
        self.plot_spaces_for_anndata(self.B)

        self.compute_proximity_objects_for_anndata(self.A)
        self.compute_proximity_objects_for_anndata(self.B)

        self.compute_correlations_for_anndata(self.A)
        self.compute_correlations_for_anndata(self.B)

        self.plot_distribution_for_anndata(self.A)
        self.plot_distribution_for_anndata(self.B)

        self.determine_percentiles_for_anndata(self.B)
        self.classify_cells(self.A, null_dist_adata = self.B)




