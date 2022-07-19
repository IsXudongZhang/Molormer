# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen):
    x = x + 1
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen:
        new_x = x.new_zeros([padlen, padlen, xlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_4d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

# dataset：NR --- max_d=130, max_p=3728
def collator(batch, max_d_node=256,
             multi_hop_max_dist=20, spatial_pos_max=20):
    drug1_node, drug1_attn_bias, drug1_spatial_pos, drug1_in_degree, drug1_out_degree, drug1_edge_input = [], [], [], [], [], []
    drug2_node, drug2_attn_bias, drug2_spatial_pos, drug2_in_degree, drug2_out_degree, drug2_edge_input = [], [], [], [], [], []
    labels = []

    for d1_node, d1_attn_bias, d1_spatial_pos, d1_in_degree, d1_out_degree, d1_edge_input, \
    d2_node, d2_attn_bias, d2_spatial_pos, d2_in_degree, d2_out_degree, d2_edge_input, \
        label in batch:
        if d1_node.size(0) <= max_d_node and d2_node.size(0) <= max_d_node:
            drug1_node.append(d1_node)
            d1_attn_bias[1:, 1:][d1_spatial_pos >= spatial_pos_max] = float('-inf')
            drug1_attn_bias.append(d1_attn_bias)
            drug1_spatial_pos.append(d1_spatial_pos)
            drug1_in_degree.append(d1_in_degree)
            drug1_out_degree.append(d1_out_degree)
            drug1_edge_input.append(d1_edge_input[:, :, :multi_hop_max_dist, :])

            drug2_node.append(d2_node)
            d2_attn_bias[1:, 1:][d2_spatial_pos >= spatial_pos_max] = float('-inf')
            drug2_attn_bias.append(d2_attn_bias)
            drug2_spatial_pos.append(d2_spatial_pos)
            drug2_in_degree.append(d2_in_degree)
            drug2_out_degree.append(d2_out_degree)
            drug2_edge_input.append(d2_edge_input[:, :, :multi_hop_max_dist, :])

            labels.append(label)


    #max_d_node_num = max(i.size(0) for i in drug_node)
    #max_p_node_num = max(i.size(0) for i in protein_node)
    max_d1_dist = max(i.size(-2) for i in drug1_edge_input)
    max_d2_dist = max(i.size(-2) for i in drug2_edge_input)
    # node
    drug1_node = torch.cat([pad_2d_unsqueeze(i, max_d_node) for i in drug1_node])
    drug2_node = torch.cat([pad_2d_unsqueeze(i, max_d_node) for i in drug2_node])

    # edge_input
    drug1_edge_input = torch.cat([pad_4d_unsqueeze(
        i, max_d_node, max_d_node, max_d1_dist) for i in drug1_edge_input])

    drug2_edge_input = torch.cat([pad_4d_unsqueeze(
        i, max_d_node, max_d_node, max_d2_dist) for i in drug2_edge_input])

    # attn_bias
    drug1_attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_d_node + 1) for i in drug1_attn_bias])
    drug2_attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_d_node + 1) for i in drug2_attn_bias])

    # spatial_pos
    drug1_spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_d_node)
                        for i in drug1_spatial_pos])
    drug2_spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_d_node)
                        for i in drug2_spatial_pos])

    # in_degree
    drug1_in_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node)
                          for i in drug1_in_degree])
    drug2_in_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node)
                          for i in drug2_in_degree])

    # out_degree
    drug1_out_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node)
                          for i in drug1_out_degree])
    drug2_out_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node)
                          for i in drug2_out_degree])

    # label
    labels = torch.tensor(labels)

    return  drug1_node, drug1_attn_bias, drug1_spatial_pos, drug1_in_degree, drug1_out_degree, drug1_edge_input, \
            drug2_node, drug2_attn_bias, drug2_spatial_pos, drug2_in_degree, drug2_out_degree, drug2_edge_input, \
            labels
