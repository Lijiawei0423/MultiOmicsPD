
import pandas as pd
import numpy as np
from IPython.display import SVG, display
from sknetwork.clustering import Louvain, Leiden, KCenters, PropagationClustering,  get_modularity
from sknetwork.data import from_edge_list
from sknetwork.ranking import PageRank
from statsmodels.stats.multitest import fdrcorrection

group = 'IncidentPDIndividuals'

dpath = '../'
bld_dict = pd.read_csv(dpath + 'Data/BloodData/BloodDict.csv')
mydf_raw = pd.read_csv(dpath + 'Results/NetworkAnalysis/' + group + '/Louvain-CrossOmics/Comb-associations-CrossOmics.csv')

mydf = mydf_raw.loc[mydf_raw.Pval_fdr<0.05]
mydf.reset_index(inplace = True, drop = True)
print(len(mydf))
print(mydf.Omic_Asss.value_counts()/len(mydf))
print(mydf.Omic_Asss.value_counts()/mydf_raw.Omic_Asss.value_counts())

edge_list, weighted_edge_list = [], []

for i in range(len(mydf)):
    edge_list.append((mydf.Omics_code_x.iloc[i], mydf.Omics_code_y.iloc[i]))
    weighted_edge_list.append((mydf.Omics_code_x.iloc[i], mydf.Omics_code_y.iloc[i], abs(mydf.Sp_corr.iloc[i])))

graph = from_edge_list(weighted_edge_list)
adjacency = graph.adjacency
names = graph.names
louvain = Louvain(tol_optimization = 0.01, tol_aggregation = 0.01, shuffle_nodes = False)
louvain_fit = louvain.fit_predict(adjacency)
labels, counts = np.unique(louvain_fit, return_counts=True)
print(labels, counts)
mod_Louvain = get_modularity(adjacency, louvain_fit)
print("modularity Louvain: " + str(mod_Louvain))
lou_cls_df = pd.DataFrame({'Omics_feature':names, 'louvain_cls':louvain_fit})

#image = visualize_graph(adjacency, position, labels=labels)
#image = visualize_graph(adjacency, names=names, display_edge_weight=True, display_node_weight=True)
#SVG(image)

pagerank_all = PageRank()
scores_all = pagerank_all.fit_predict(adjacency)
rank_all_df = pd.DataFrame({'Omics_feature':names, 'PageRank_Score':scores_all})


lou_cls_lst = list(set(lou_cls_df.louvain_cls))
rank_df = pd.DataFrame()

for lou_cls in lou_cls_lst:
    cls_f_lst = lou_cls_df.loc[lou_cls_df.louvain_cls == lou_cls].Omics_feature.tolist()
    select_idx = []
    for i in range(len(mydf)):
        var1, var2 = mydf.Omics_code_x.iloc[i], mydf.Omics_code_y.iloc[i]
        if ((var1 in cls_f_lst) & (var2 in cls_f_lst)):
            select_idx.append(i)
        else:
            pass
    cls_ass_df = mydf.iloc[select_idx, :]
    cls_ass_df.reset_index(inplace = True, drop = True)
    print(len(cls_ass_df))
    weighted_edge_list = []
    for i in range(len(cls_ass_df)):
        weighted_edge_list.append((cls_ass_df.Omics_code_x.iloc[i], cls_ass_df.Omics_code_y.iloc[i], abs(cls_ass_df.Sp_corr.iloc[i])))
    graph = from_edge_list(weighted_edge_list)
    adjacency = graph.adjacency
    names = graph.names
    pagerank = PageRank()
    scores = pagerank.fit_predict(adjacency)
    cls_rank_df = pd.DataFrame({'Omics_feature': names, 'PageRank_Score_cls'+str(lou_cls): scores})
    rank_df = pd.concat([rank_df, cls_rank_df], axis = 0)

gdf = pd.merge(lou_cls_df, rank_all_df, how = 'inner', on = 'Omics_feature')
gdf = pd.merge(gdf, rank_df, how = 'inner', on = 'Omics_feature')
gdf = pd.merge(gdf, bld_dict, how = 'left', left_on = 'Omics_feature', right_on = 'Omics_code')
gdf.sort_values(by = 'louvain_cls', ascending=True, inplace = True)
gdf.to_csv(dpath + 'Results/NetworkAnalysis/' + group + '/Louvain-CrossOmics/Louvain-Rval-Weighted-CrossOmics.csv', index = False)


