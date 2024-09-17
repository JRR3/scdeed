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
            use_manifold_as_pre: Optional[bool] = False,
            using_1D_ordered_manifold: Optional[bool] = False,
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

        # self.output = os.path.abspath(output)

        self.output = output

        self.n_pcs = n_pcs

        n_neighbors = frac_neighbors * n_cells
        self.n_neighbors = int(n_neighbors)

        self.use_manifold_as_pre = use_manifold_as_pre
        self.using_1D_ordered_manifold = using_1D_ordered_manifold

    #=================================
    def permute_expression_matrix(self,
                                  adata: sc.AnnData):
        """
        This function permutes the entries of 
        each column/gene of the original expression
        matrix.

        We use the anndata object B to store anything 
        related to the permuted version.
        """

        X = adata.X.copy()
        n_genes = X.shape[1]

        if not sp.issparse(X):
            #Matrix is full.
            #Permute the rows

            # X = np.random.permutation(X)
            # Note that if pass the whole matrix
            # to the permutation function, we only
            # get a permutation of row vectors and
            # not of the whole matrix. Hence, we 
            # permute one column at a time.

            for k in range(n_genes):
                X[:,k] = np.random.permutation(X[:,k])

        else:
            #Matrix is sparse.
            #Note that the matrix has the CSC format 
            #by design. This allows us to work directly
            #on the data structure of the matrix.
            for k in range(n_genes):
                col = X.getcol(k)
                p_col = np.random.permutation(col.data)
                start = X.indptr[k]
                end = X.indptr[k+1]
                X.data[start:end] = p_col
        
        adata.X = X

    #=================================
    def compute_spaces_for_anndata(self,
                                   adata: sc.AnnData,
        ):

        sc.pp.pca(adata,
                n_comps=self.n_pcs,
                svd_solver="auto",
        )

        if self.use_manifold_as_pre:
            #Store a reference.
            adata.obsm["X_pre"] = adata.X
        else:
            #We use the PCA space as the
            #pre-embedding space.
            #Store a reference.
            adata.obsm["X_pre"] = adata.obsm["X_pca"]

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
                   use_rep="X_pre",
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
                        use_rep="X_pre",
        )

        sc.tl.umap(adata,
                   neighbors_key= neighbors_tag,
        )



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

        if self.use_manifold_as_pre:
            pass
        else:
            self.plot_embedding(adata,
                                "X_pre",
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
    def compute_manifold_neighbor_matrix(self,
                                         adata: sc.AnnData):
        """
        This function assumes that we have an ordered
        1D manifold. By ordered, we mean that for any
        two distinct points P1, P2, we can establish if 
        P1 < P2, or P2 < P1.
        """
        n_points = adata.X.shape[0]
        manifold_distance_matrix = np.zeros((n_points,n_points))

        # We make the main diagonal equal to infinity
        # to avoid labeling each cell as its nearest neighbor.
        for i in range(n_points):
            manifold_distance_matrix[i,i] = np.inf

        one_step_distances = np.zeros(n_points-1)

        #Compute distances for adjacent points.
        for i in range(n_points-1):
            a = adata.X[i]
            b = adata.X[i+1]
            d = np.linalg.norm(a-b)
            one_step_distances[i] = d

        #Compute distances between two (ordered)
        #points.
        for i in range(n_points-1):
            for j in range(i+1,n_points):
                d = one_step_distances[i:j].sum()
                manifold_distance_matrix[i,j] = d

        # fname = "manifold_distance_matrix.npy"
        # fname = os.path.join(self.output, fname)
        # print(manifold_distance_matrix)
        # np.save(fname, manifold_distance_matrix)

        manifold_distance_matrix += manifold_distance_matrix.T
        # print(manifold_distance_matrix)

        manifold_neighbor_matrix = np.zeros((n_points,
                                             self.n_neighbors),
                                             dtype=int)
        for k, row in enumerate(manifold_distance_matrix):
            indices = np.argsort(row)
            indices = indices[:self.n_neighbors]
            manifold_neighbor_matrix[k] = indices

        # print(manifold_neighbor_matrix)

        txt = "manifold_neighbor_matrix"
        adata.obsm[txt] = manifold_neighbor_matrix


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


        # The matrix of neighbors for the pre-embedding
        # space.
        if self.use_manifold_as_pre:
            pass
        else:
            pre_str = f"X_pre"
            pre_nn_obj = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm="ball_tree",
                ).fit(adata.obsm[pre_str])

            pre_ngb = f"pre_neighbors"
            adata.obsm[pre_ngb] = pre_nn_obj.kneighbors(
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
    def compute_correlations_for_anndata(
            self,
            adata: sc.AnnData,
            pre_ngb_label: Optional[str] = "pre_neighbors",
            special_tag: Optional[str] = "",
        ):

        #embd_ngb_mtx = self.A.obsm[embd_ngb]
        # embd_ngb    = f"embedding_neighbors"

        pre_ngb_mtx    = adata.obsm[pre_ngb_label]

        for embedding_type in self.list_of_embeddings:
            embd_sort_dist = (f"{embedding_type}"
                              "_sorted_distances")
            embd_dist      = f"{embedding_type}_distances"

            embd_sort_dist_mtx = adata.obsm[embd_sort_dist]
            embd_dist_mtx      = adata.obsm[embd_dist]

            Z = zip(embd_dist_mtx,
                    embd_sort_dist_mtx,
                    pre_ngb_mtx)

            with Pool(processes=6) as pool:
                C = pool.starmap(self.correlation_function,
                                Z)

            corr_tag = f"{embedding_type}_correlations"
            corr_tag += special_tag
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
                       special_tag: Optional[str] = "",
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
            corr_tag += special_tag
            percentile_tag = f"{embedding_type}_percentile"
            null_dist_percentiles = null_dist_adata.uns[
                percentile_tag]
            corr_vec = adata.obs[corr_tag]
            #5-th percentile
            dubious  = corr_vec < null_dist_percentiles[0]
            #95-th percentile
            trustworthy = null_dist_percentiles[1] < corr_vec
            dub_state = f"dubiousness_{embedding_type}"
            dub_state += special_tag
            adata.obs[dub_state] = "Undefined"
            adata.obs.loc[dubious,     dub_state] = "Dubious"
            adata.obs.loc[trustworthy, dub_state] = "Trustworthy"

            embd_tag = f"X_{embedding_type}"


            #Rank the elements using the order_map.
            rank = f"{embedding_type}_rank"
            adata.obs[rank] = adata.obs[dub_state].map(
                order_map)

            self.plot_embedding(adata,
                                descriptor  = embd_tag,
                                color_column= dub_state,
                                color_map   = color_map,
                                modifier    = dub_state,
                                order       = rank,
                                set_legend_outside=True)



    #=================================
    def run(self):


        #Originally, B is a copy of A.
        self.permute_expression_matrix(self.B)

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

        if self.using_1D_ordered_manifold:
            self.compute_manifold_neighbor_matrix(self.A)
            self.compute_correlations_for_anndata(
                self.A,
                pre_ngb_label="manifold_neighbor_matrix",
                special_tag = "_manifold"
                )
            self.classify_cells(self.A,
                                null_dist_adata = self.B,
                                special_tag="_manifold")






