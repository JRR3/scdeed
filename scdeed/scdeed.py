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
import scanpy as sc
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
                    self.convert_mm_from_source_to_anndata()
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

    #=================================
    def compute_pca(self,
                    tag: str,
                    ):
        sc.pp.pca(self.A,
                  n_comps=self.n_pcs,
                  use_highly_variable=False,
                  svd_solver="auto",
        )

        self.A.obsm[tag] = self.A.obsm["X_pca"].copy()

    #=================================
    def plot_embedding(self,
                 tag: str,
                 color_column: str,
                 color_map: Union[str, dict],
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
        ax.set_aspect("equal")
        fname = f"{tag}.pdf"
        fname = os.path.join(self.output, fname)
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
    def run(self):
        self.compute_pca("X_pca_O")
        self.plot_embedding("X_pca_O", "colors", "plasma")
        self.compute_tsne("X_pca_O","X_tsne_O")
        self.plot_embedding("X_tsne_O", "colors", "plasma")



