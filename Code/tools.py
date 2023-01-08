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

def saveobj2file(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    return("save obj to " + filepath)

def loadobj(filepath):
    tempobj = None
    with open(filepath, 'rb') as f:
        tempobj = pickle.load(f)
    
    return tempobj

# Get the specified cell data from the scanpy object
def celldata(scdata, obs, sp_obs):
    scdata_tempcelltype_index = scdata.obs[obs]==sp_obs
    temp = scdata[scdata_tempcelltype_index]
    print("🍪 %s cells are extracted." % (temp.shape[0]))
    return temp

# Separate out protein expression
def separate_out_protein_expression(scdata, protein_prefix, set_protein_obsm_name="protein_expression", layers=None):
    
    feature_names = scdata.var.index
    feature_names_len = feature_names.shape[0]
    
    protein_index = np.ones(feature_names_len)==0
    gene_index = np.ones(feature_names_len)==1
    
    for i in trange(feature_names_len):
        feature_name = feature_names[i]
        if protein_prefix in feature_name:
            protein_index[i] = True
            gene_index[i] = False

    scdata_with_gene = scdata[:, gene_index]
    
    if layers!=None:
        if type(scdata.layers[layers])==scipy.sparse._csr.csr_matrix or type(scdata.layers[layers])==scipy.sparse._csc.csc_matrix:
            scdata_with_gene.obsm[set_protein_obsm_name] = scdata.layers[layers].toarray()[:, protein_index]
        else:
            scdata_with_gene.obsm[set_protein_obsm_name] = scdata.layers[layers][:, protein_index]
    else:
        if type(scdata.X)==scipy.sparse._csr.csr_matrix or type(scdata.X)==scipy.sparse._csc.csc_matrix:
            scdata_with_gene.obsm[set_protein_obsm_name] = scdata.X.toarray()[:, protein_index]
        else:
            scdata_with_gene.obsm[set_protein_obsm_name] = scdata.X[:, protein_index]

    scdata_with_gene.obsm[set_protein_obsm_name] = pd.DataFrame(scdata_with_gene.obsm[set_protein_obsm_name], index=scdata.obs.index, columns=feature_names[protein_index])
    
    return scdata_with_gene

# Specify gmt file
def geneset(scdata, gmt_filepath):
    
    lines = None
    with open(gmt_filepath, "r") as f:
        lines = f.readlines()
    
    geneset_list = []
    geneset_names = []
    print("🍽 Reading functional gene collection.")
    for line in tqdm(lines):
        templist = str(line).strip().split("\t")
        geneset_name = templist[0]
        geneset_desc = templist[1]
        geneset_set = templist[2:]
        tempdict = {"name":geneset_name, "desc":geneset_desc, "set":geneset_set}
        geneset_names.append(geneset_name)
        geneset_list.append(tempdict)
    
    scdata.uns["geneset_raw"] = geneset_list
    
    print("🥗 %s functional gene collections are extracted." % (len(geneset_list)))

# Calculate GFPA score
def scores(scdata, p_threshold=0.01, corr_threshold=0.5, gfc_strategy="mean", cor_algorithm="spearman", beta=1):
    
    scdata.uns["p_threshold"] = p_threshold
    
    geneset_list = scdata.uns["geneset_raw"]
    
    geneset_count = len(geneset_list)
    protein_type_count = scdata.obsm["protein_normalization"].shape[1]
    
    geneset_pred_protein = np.zeros((scdata.shape[0], len(geneset_list)))
    
    scdata.uns["correlation"] = np.zeros((geneset_count, protein_type_count))
    scdata.uns["p_value"] = np.zeros((geneset_count, protein_type_count))
    
    geneset_names = []
    protein_names = scdata.obsm["protein_normalization"].columns
    
    geneset_exp_mtx = np.zeros((scdata.shape[0], geneset_count))

    scdata_df = pd.DataFrame(scdata.layers["rna_normalization"], index=scdata.obs.index, columns=scdata.var.index) 
    
    print("🙂 GFPA scores are being calculated.")

    if gfc_strategy=="mean":
        gfc_strategy=np.mean
    elif gfc_strategy=="sum":
        gfc_strategy=np.sum
    else:
        raise Exception('Illegal gene function center computing strategy.')
    
    if cor_algorithm=="spearman":
        cor_algorithm=scipy.stats.spearmanr
    elif cor_algorithm=="pearson":
        cor_algorithm=scipy.stats.pearsonr
    elif cor_algorithm=="kendall":
        cor_algorithm=scipy.stats.kendalltau
    else:
        raise Exception('Illegal correlation algorithm.')
        
    for i in trange(geneset_count):
        geneset = geneset_list[i]
        geneset_name = geneset["name"]
        geneset_names.append(geneset_name)
        scdata_df_geneset_index = np.in1d(scdata_df.columns.to_numpy(), geneset["set"])
        scdata_df_geneset_mean = gfc_strategy(scdata_df.iloc[:, scdata_df_geneset_index], axis=1)
        scdata_df_geneset_mean = (scdata_df_geneset_mean-np.mean(scdata_df_geneset_mean))/np.std(scdata_df_geneset_mean)
        geneset_exp_mtx[:, i] = scdata_df_geneset_mean.values
        for j in range(protein_type_count):       
            y = scdata.obsm["protein_normalization"].iloc[:, j].values
            y = (y-np.mean(y))/np.std(y)
            cor, p = cor_algorithm(scdata_df_geneset_mean, y)
            scdata.uns["correlation"][i, j] = cor
            scdata.uns["p_value"][i, j] = p

    scdata.obsm["geneset"] = pd.DataFrame(geneset_exp_mtx, index=scdata.obs.index, columns=geneset_names)
    
    scdata.uns["correlation"] = pd.DataFrame(scdata.uns["correlation"], index=geneset_names, columns=scdata.obsm["protein_normalization"].columns)
    scdata.uns["p_value"] = pd.DataFrame(scdata.uns["p_value"], index=geneset_names, columns=scdata.obsm["protein_normalization"].columns)
    
    scdata.uns["correlation_abs"] = np.abs(scdata.uns["correlation"])
    
    beta = 1
    F1 = (1+np.power(beta, 2))*scdata.uns["correlation_abs"]*(1-scdata.uns["p_value"])/(np.power(beta, 2)*scdata.uns["correlation_abs"]+(1-scdata.uns["p_value"]))
    F1_threshold = (1+np.power(beta, 2))*corr_threshold*(1-p_threshold)/(np.power(beta, 2)*corr_threshold+(1-p_threshold))
    scdata.uns["F1_threshold"] = F1_threshold
    scdata.uns["GFPA_score"] = F1
        
    gc.collect()
    
    print("🍱 Organizing results.")
    
    confidence_mtx = np.zeros(scdata.uns["p_value"].shape)
    scdata.uns["confidence"] = scdata.uns["p_value"]<p_threshold
    
    NP_mtx = np.zeros(scdata.uns["correlation"].shape)
    NP_mtx[scdata.uns["correlation"]>0] = 1
    scdata.uns["NP"] = NP_mtx
    scdata.uns["NP"] = pd.DataFrame(scdata.uns["NP"], index=scdata.uns["confidence"].index, columns=scdata.uns["confidence"].columns)
    
    # scdata.uns["GFPA_score"][scdata.uns["correlation"]<0] = 0
    
    geneset_count = len(geneset_names)
    protein_type_count = scdata.obsm["protein_normalization"].shape[1]
    
    GFPA_score_len = scdata.uns["GFPA_score"].shape[0]*scdata.uns["GFPA_score"].shape[1]

    temp_index = 0
    df_geneset_names = []
    df_protein_names = []
    correlation_list = np.zeros(GFPA_score_len)
    correlation_abs_list = np.zeros(GFPA_score_len)
    NP_list = np.zeros(GFPA_score_len)
    p_list = np.zeros(GFPA_score_len)
    GFPA_score_list = np.zeros(GFPA_score_len)

    for i in tqdm(range(geneset_count)):
        for j in range(protein_type_count):

            df_geneset_names.append(geneset_names[i])
            df_protein_names.append(scdata.obsm["protein_normalization"].columns[j])

            correlation_list[temp_index] = scdata.uns["correlation"].iloc[i, j]
            correlation_abs_list[temp_index] = scdata.uns["correlation_abs"].iloc[i, j]
            NP_list[temp_index] = scdata.uns["NP"].iloc[i, j]
            p_list[temp_index] = scdata.uns["p_value"].iloc[i, j]
            GFPA_score_list[temp_index] = scdata.uns["GFPA_score"].iloc[i, j]

            temp_index += 1
    
    GFPADF = pd.DataFrame({
        "geneset":df_geneset_names, 
        "protein":df_protein_names, 
        "correlation":correlation_list, 
        "p_value":p_list,
        "adj_p_value":p_list,
        "reliable":NP_list,
        "GFPA_score":GFPA_score_list,
    })
        
    scdata.uns["gfpa"] = GFPADF
    
    print("👀 Adjust p value.")
    #fix p value
    # Benjamini/Hochberg for independent tests
    adj = statsmodels.stats.multitest.fdrcorrection(scdata.uns["gfpa"]["p_value"], alpha=p)
    GFPADF["reliable"] = adj[0]
    tempindex = GFPADF["GFPA_score"]<F1_threshold
    temp = GFPADF["reliable"].values.copy()
    temp[tempindex] = False
    GFPADF["reliable"] = temp
    GFPADF["adj_p_value"] = adj[1]
    
    GFPADF = GFPADF.sort_values(by='GFPA_score', ascending=False)
    GFPADF.index = range(GFPADF.shape[0])
    
    GFPADF["adj_GFPA_score"] = GFPADF["GFPA_score"]*GFPADF["reliable"]
    
    scdata.uns["gfpa"] = GFPADF
    
    gc.collect()

# Save GFPA assessment results
def to_csv(scdata, filepath, index=False, sep=","):
    GFPADF = scdata.uns["gfpa"]
    GFPADF.to_csv(filepath,index=index ,sep=",")
    
    del GFPADF
    gc.collect()
    
    print("💾 The GFPA has been saved to %s" % (filepath))

# Gene set transformation into protein model and output weight
def geneset2protein_model(scdata, target_name, protein_name, model="rf"):
    
    geneset_raw = scdata.uns['geneset_raw']
    geneset_typecount = len(geneset_raw)
    
    geneset_set = []
    for i in range(geneset_typecount):
        geneset = geneset_raw[i]
        geneset_name = geneset['name']
        if target_name==geneset_name:
            geneset_set = geneset['set']
            break
    
    geneset_set = np.array(geneset_set)
    protein_vec = scdata.obsm["protein_normalization"][protein_name].values
    
    scdata_df = pd.DataFrame(scdata.layers["rna_normalization"], index=scdata.obs.index, columns=scdata.var.index) 
    scdata_df_geneset_index = np.in1d(scdata_df.columns.to_numpy(), geneset_set)
    geneset_set = geneset_set[np.in1d(geneset_set, scdata_df.columns.to_numpy())]
    geneset_mtx = scdata_df.iloc[:, scdata_df_geneset_index].values
    
    feature_weight = []
    
    if model=="rf":
        reg = RandomForestRegressor(max_depth=2, random_state=0)
        reg.fit(geneset_mtx, protein_vec)
        feature_weight = reg.feature_importances_
    elif model=="linear":
        reg = LinearRegression()
        reg.fit(geneset_mtx, protein_vec)
        feature_weight = reg.coef_
    elif model=="svm":
        reg = SVR(kernel="linear")
        reg.fit(geneset_mtx, protein_vec)
        feature_weight = reg.coef_
    elif model=="xt":
        reg = ExtraTreesRegressor()
        reg.fit(geneset_mtx, protein_vec)
        feature_weight = reg.feature_importances_
    
    feature_weight = feature_weight.reshape(-1)
    
    # feature_sum = np.sum(feature_weight)
    # feature_weight = feature_weight/feature_sum
 
    feature_weight = pd.DataFrame(data={"Gene":geneset_set, "Protein":[protein_name]*geneset_set.shape[0], "Weight":feature_weight}, columns=["Gene", "Protein", "Weight"])

    return feature_weight
