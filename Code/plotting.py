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
from . import tools as tl

# GFPA Visualization of Gene Sets and Proteins
def scatter(scdata, geneset_name, protein_name, xy_alias=[], **kwargs):
    temp = scdata.uns["gfpa"].query("geneset=='%s' &  protein=='%s'" % (geneset_name, protein_name))
    correlation = np.float64(temp["correlation"])
    GFPA_score = np.float64(temp["GFPA_score"])
    p_value = np.float64(temp["p_value"])
    adj_p_value = np.float64(temp["adj_p_value"])
    reliable = np.float64(temp["reliable"])
    
    x = scdata.obsm["geneset"][geneset_name]
    y = scdata.obsm["protein_normalization"][protein_name]
    temp_data = pd.DataFrame({"x":x, "y":y})
    g = sns.lmplot(
        data=temp_data,
        x="x", y="y", **kwargs
    )
    if len(xy_alias)>0:
        g.set_axis_labels(xy_alias[0], xy_alias[1])
    else:
        g.set_axis_labels(geneset_name, protein_name)
    plt.title("GFPA_score:%.4f\ncorrelation:%.4f\nadj_p_value:%.f\np_value:%.4f\nreliable: %s" % (GFPA_score, correlation, adj_p_value, p_value, reliable))
    plt.show()
    plt.close()

# Visualize gene weights
def weight(scdata, target_name, protein_name, model="rf", n=10, **kwargs):
    geneset_weight = tl.geneset2protein_model(scdata, target_name, protein_name, model=model)
    geneset_weight = geneset_weight.sort_values(by='Weight', ascending=False).iloc[:n, :]
    ax = sns.barplot(x="Gene", y="Weight", label="Protein", data=geneset_weight, **kwargs)
    plt.xticks(rotation=90)
    plt.title(protein_name)
    plt.xlabel(target_name)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.show()
    plt.close()

# Visualize the gene set related to the specified protein
def protein(scdata, protein_name, top_n=10, **kwargs):
    data = scdata.uns["gfpa"].query("protein=='%s'" % (protein_name)).sort_values(by='GFPA_score')[::-1][:top_n]
    sns.barplot(x="GFPA_score", y="geneset", data=data, **kwargs)
    plt.title(protein_name)
    plt.xlabel("GFPA_score")
    plt.ylabel("Gene_set")
    plt.show()
    plt.close()

# Visualize proteins associated with a given gene set
def genefunction(scdata, geneset_name, top_n=10, **kwargs):
    data = scdata.uns["gfpa"].query("geneset=='%s'" % (geneset_name)).sort_values(by='GFPA_score')[::-1][:top_n]
    sns.barplot(x="GFPA_score", y="protein", data=data, **kwargs)
    plt.title(geneset_name)
    plt.xlabel("GFPA_score")
    plt.ylabel("Protein")
    plt.show()
    plt.close()  