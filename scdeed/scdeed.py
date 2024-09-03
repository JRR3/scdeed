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
            frac_neighbors: Optional[float] = 0.25,
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

        self.A.layers[self.ori] = X


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
    def permute_expression_matrix(self,
                                  source_layer: str,
                                  target_layer: str):
        """
        This function permutes the entries of 
        each column/gene.
        """

        X = self.A.layers[source_layer].copy()

        if not self.is_sparse:
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
        
        self.A.layers[target_layer] = X

    #=================================
    def compute_spaces_for_layer(self,
                                 layer: str,
                                 ):
        pca_tag = f"X_pca_{layer}"
        embd_tag= f"X_embd_{layer}"
        self.compute_pca(layer=layer,
                         tag = pca_tag,)
        self.compute_tsne(use_rep=pca_tag,
                          tag = embd_tag,)


    #=================================
    def compute_pca(self,
                    layer: str,
                    tag: str,
                    ):

        self.A.obsm[tag] = sc.pp.pca(
            self.A.layers[layer],
            n_comps=self.n_pcs,
            svd_solver="auto",
        )


    #=================================
    def plot_embedding(self,
                 tag: str,
                 color_column: str,
                 color_map: Union[str, dict],
                 modifier: Optional[str] = "sc",
                 set_legend_outside: Optional[bool] = False,
                 ):
        """
        Plot the first two columns of the embedding.
        """

        fig, ax = plt.subplots()

        scprep.plot.scatter2d(
            self.A.obsm[tag],
            ax=ax,
            c = self.A.obs[color_column],
            cmap = color_map,
            ticks=True,
        )
        # ax.set_aspect("equal")
        if set_legend_outside:
            ax.legend(loc="center left",
                      bbox_to_anchor=(1, 0.5))
        fname = f"{tag}_{modifier}.pdf"
        fname = os.path.join(self.output, fname)
        if set_legend_outside:
            fig.savefig(fname)
        else:
            fig.savefig(fname, bbox_inches="tight")

    #=================================
    def compute_tsne(self,
                     use_rep: str,
                     tag: str,
                     perplexity: Optional[float] = 30):

        sc.tl.tsne(self.A,
                   perplexity=perplexity,
                   n_pcs=self.n_pcs,
                   use_rep=use_rep)

        self.A.obsm[tag] = self.A.obsm["X_tsne"].copy()

    #=================================
    def compute_null_distribution(self):
        mtx = self.A.obsm["X_pca"]
        p_mtx = self.permute_each_column(mtx)

    #=================================
    def plot_spaces_for_layer(
            self,
            layer: str,
            color_column: Optional[str] = "colors",
            color_map: Optional[str] = "plasma",
    ):
        """
        Plot the pre-embedding space (projected onto R^2) 
        and the embedding space.
        """
        pca_tag = f"X_pca_{layer}"
        embd_tag = f"X_embd_{layer}"

        self.plot_embedding(pca_tag, color_column, color_map)
        self.plot_embedding(embd_tag, color_column, color_map)

    #=================================
    def compute_proximity_objects_for_layer(self,
                                            layer: str,
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
        pca_mtx = f"X_pca_{layer}"
        embd_mtx = f"X_embd_{layer}"

        pca_ngb = f"pca_ngb_{layer}"
        embd_ngb = f"embd_ngb_{layer}"
        embd_dist = f"embd_dist_{layer}"
        embd_sort_dist = f"embd_sort_dist_{layer}"

        pca_nn_obj = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="ball_tree",
            ).fit(self.A.obsm[pca_mtx])

        embd_nn_obj = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="ball_tree",
            ).fit(self.A.obsm[embd_mtx])

        self.A.obsm[pca_ngb] = pca_nn_obj.kneighbors(
            return_distance=False)

        (self.A.obsm[embd_sort_dist],
         self.A.obsm[embd_ngb]) = embd_nn_obj.kneighbors()

        self.A.obsm[embd_dist] = pairwise_distances(
            self.A.obsm[embd_mtx], metric="euclidean")

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
    def compute_correlations_for_layer(self,
                                       layer: str):

        pca_ngb = f"pca_ngb_{layer}"
        embd_ngb = f"embd_ngb_{layer}"
        embd_sort_dist = f"embd_sort_dist_{layer}"
        embd_dist = f"embd_dist_{layer}"

        pca_ngb_mtx = self.A.obsm[pca_ngb]
        #embd_ngb_mtx = self.A.obsm[embd_ngb]
        embd_sort_dist_mtx = self.A.obsm[embd_sort_dist]
        embd_dist_mtx = self.A.obsm[embd_dist]

        Z = zip(embd_dist_mtx,
                embd_sort_dist_mtx,
                pca_ngb_mtx)

        with Pool(processes=6) as pool:
            C = pool.starmap(self.correlation_function, Z)
        
        print(describe(C))

        corr_tag = f"corr_{layer}"
        self.A.obs[corr_tag] = C


    #=================================
    def plot_distribution_for_layer(self,
                                       layer: str):
        corr_tag = f"corr_{layer}"
        fig, ax = plt.subplots()
        xlabel = f"Correlation ({layer} cells)"
        scprep.plot.histogram(self.A.obs[corr_tag],
                              xlabel=xlabel,
                              ylabel="# of cells",
                              ax = ax,
                              )

        fname = f"distribution_{layer}.pdf"
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")

    #=================================
    def determine_percentiles(self, layer: str):
        corr_tag = f"corr_{layer}"
        vec = self.A.obs[corr_tag]
        pc = np.percentile(vec, [5, 95])
        return pc

    #=================================
    def classify_cells(self,
                       source_dist: str,
                       percentiles: np.array,
        ):

        corr_tag = f"corr_{source_dist}"
        vec = self.A.obs[corr_tag]
        pc = np.percentile(vec, [5, 95])
        dubious     = vec   < pc[0]
        trustworthy = pc[1] < vec
        status = "status"
        self.A.obs[status] = "Undefined"
        self.A.obs.loc[dubious, status] = "Dubious"
        self.A.obs.loc[trustworthy, status] = "Trustworthy"

        layer = self.ori
        embd_tag = f"X_embd_{layer}"
        color_map = {"Undefined": "gray",
                     "Dubious":"red",
                     "Trustworthy":"blue"}

        self.plot_embedding(embd_tag,
                            status,
                            color_map,
                            set_legend_outside=True)



    #=================================
    def run(self):

        original = self.ori
        permuted = self.per

        self.permute_expression_matrix(original, permuted)


        self.compute_spaces_for_layer(layer=original)
        self.compute_spaces_for_layer(layer=permuted)

        self.plot_spaces_for_layer(layer=original)
        self.plot_spaces_for_layer(layer=permuted)

        self.compute_proximity_objects_for_layer(
            layer=original)
        self.compute_proximity_objects_for_layer(
            layer=permuted)

        self.compute_correlations_for_layer(layer=original)
        self.compute_correlations_for_layer(layer=permuted)

        self.plot_distribution_for_layer(layer=original)
        self.plot_distribution_for_layer(layer=permuted)

        pc = self.determine_percentiles(layer=permuted)
        print(pc)
        self.classify_cells(source_dist=original,
                            percentiles= pc)




