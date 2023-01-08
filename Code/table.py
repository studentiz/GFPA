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
import statsmodels.stats.multitest
from . import tools as tl

# View the top n GFPA assessment results
def top(scdata, n=20):
    GFPADF = scdata.uns["gfpa"]
    GFPADF = GFPADF.sort_values(by='GFPA_score', ascending=False)
    GFPADF.index = range(GFPADF.shape[0])
    print("🧐 The p value was adjusted by Benjamin/Hochberg independence test.")
    print("🤔 adj_p_value < %s and GFPA_score > %.6f are considered reliable." % (scdata.uns["p_threshold"], scdata.uns["F1_threshold"]))
    return GFPADF.iloc[:n, :]

# tbale gene weights
def weight(scdata, target_name, protein_name, model="rf", color="#71AD47"):
    geneset_weight = tl.geneset2protein_model(scdata, target_name, protein_name, model=model)
    return geneset_weight