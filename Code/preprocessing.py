import os, sys
import pickle
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats
from skbio.stats.composition import clr
import seaborn as sns
from tqdm import tqdm, trange
import gc
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt

# RNA data normalization
def rna_normalization(scdata, layers=None):
    scdata_mtx = scdata.X
    
    if layers!=None:
        scdata_mtx = scdata.layers[layers].toarray()
        
    temp_adata = scdata_mtx
    temp_adata = sc.AnnData(temp_adata)
    sc.pp.normalize_total(temp_adata)
    scdata_mtx = np.log(temp_adata.X+1)
    scdata.layers["rna_normalization"] = scdata_mtx
    
    del scdata_mtx, temp_adata
    gc.collect()
    
    print("🫘 %s types of genes are normalized." % (scdata.layers["rna_normalization"].shape[1]))

# Protein data normalization
def protein_normalization(scdata, protein_obsm):
    scdata.obsm["protein_normalization"] = clr(1+scdata.obsm[protein_obsm].values)
    scdata.obsm["protein_normalization"] = (scdata.obsm["protein_normalization"]-np.mean(scdata.obsm["protein_normalization"], axis=0))/np.std(scdata.obsm["protein_normalization"], axis=0)
    scdata.obsm["protein_normalization"] = pd.DataFrame(scdata.obsm["protein_normalization"], index=scdata.obsm["protein_expression"].index, columns=scdata.obsm["protein_expression"].columns)
    
    print("🥚 %s types of proteins are normalized." % (scdata.obsm["protein_normalization"].shape[1]))
