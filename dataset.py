import numpy as np
import torch
from torch.utils import data
from gen_mol_graph import *
from configs import Model_config
config = Model_config()

def drug_embedding(id):
    x, edge_attr, edge_index = sdf2graph(id)
    N = x.size(0)
    x = mol_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = mol_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    node = x
    attn_bias = attn_bias
    spatial_pos = spatial_pos
    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    edge_input = torch.from_numpy(edge_input).long()

    return node, attn_bias, spatial_pos, in_degree, out_degree, edge_input


class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]

        drug1_id = self.df.iloc[index]['Drug1']
        drug2_id = self.df.iloc[index]['Drug2']

        d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug_embedding(drug1_id)
        p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input = drug_embedding(drug2_id)
        label = self.labels[index]

        return d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,\
               p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,\
               label

