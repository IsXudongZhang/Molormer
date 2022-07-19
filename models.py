from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

torch.manual_seed(1)
np.random.seed(1)


class Molormer(nn.Sequential):
    '''
        Molormer Network with spatial encoder and lightweight attention block
    '''

    def __init__(self, **config):
        super(Molormer, self).__init__()

        self.gpus = torch.cuda.device_count()
        self.n_layers = config['n_layers']
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']
        self.ffn_dim = config['fnn_dim']
        self.flatten_dim = config['flatten_dim']
        self.multi_hop_max_dist = config['multi_hop_max_dist']
        
        # dropout
        self.encoder_dropout = config['encoder_dropout_rate']
        self.attention_dropout = config['attention_dropout_rate']
        self.input_dropout = nn.Dropout(config['input_dropout_rate'])

        # Embeddings
        self.d_node_encoder = nn.Embedding(512*9+1, self.hidden_dim, padding_idx=0)
        self.d_edge_encoder = nn.Embedding(512*3+1, self.num_heads, padding_idx=0)
        self.d_edge_dis_encoder = nn.Embedding(128 * self.num_heads * self.num_heads, 1)
        self.d_spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.d_in_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.d_out_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)

        self.d_encoders = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False),
                                   d_model=self.hidden_dim, n_heads=self.num_heads, mix=False),
                    d_model=self.hidden_dim,
                    d_ff=self.ffn_dim,
                    dropout=0.0,
                ) for l in range(self.n_layers) 
            ],
            [
                Distilling_layer(self.hidden_dim) for _ in range(self.n_layers - 1) 
            ] ,
            norm_layer=torch.nn.LayerNorm(self.hidden_dim)
        )

        self.d_final_ln = nn.LayerNorm(self.hidden_dim)
        self.d_graph_token = nn.Embedding(1, self.hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, config['num_heads'])
        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3)

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, config['num_classes'])

        )

    def forward(self, d1_node, d1_attn_bias, d1_spatial_pos, d1_in_degree, d1_out_degree, d1_edge_input,
                      d2_node, d2_attn_bias, d2_spatial_pos, d2_in_degree, d2_out_degree, d2_edge_input):
        # graph_attn_bias
        drug1_n_graph, drug1_n_node = d1_node.size()[:2]
        drug2_n_graph, drug2_n_node = d2_node.size()[:2]

        drug1_graph_attn_bias = d1_attn_bias.clone()
        drug1_graph_attn_bias = drug1_graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        drug2_graph_attn_bias = d2_attn_bias.clone()
        drug2_graph_attn_bias = drug2_graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        drug1_spatial_pos_bias = self.d_spatial_pos_encoder(d1_spatial_pos).permute(0, 3, 1, 2)
        drug1_graph_attn_bias[:, :, 1:, 1:] = drug1_graph_attn_bias[:,
                                                        :, 1:, 1:] + drug1_spatial_pos_bias

        drug2_spatial_pos_bias = self.d_spatial_pos_encoder(d2_spatial_pos).permute(0, 3, 1, 2)
        drug2_graph_attn_bias[:, :, 1:, 1:] = drug2_graph_attn_bias[:,
                                                        :, 1:, 1:] + drug2_spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        drug1_graph_attn_bias[:, :, 1:, 0] = drug1_graph_attn_bias[:, :, 1:, 0] + t
        drug1_graph_attn_bias[:, :, 0, :] = drug1_graph_attn_bias[:, :, 0, :] + t
        drug2_graph_attn_bias[:, :, 1:, 0] = drug2_graph_attn_bias[:, :, 1:, 0] + t
        drug2_graph_attn_bias[:, :, 0, :] = drug2_graph_attn_bias[:, :, 0, :] + t

        # edge_input
        drug1_spatial_pos = d1_spatial_pos.clone()
        drug1_spatial_pos[drug1_spatial_pos == 0] = 1  # set pad to 1
        # set 1 to 1, x > 1 to x - 1
        drug1_spatial_pos = torch.where(drug1_spatial_pos > 1, drug1_spatial_pos - 1, drug1_spatial_pos)
        if self.multi_hop_max_dist > 0:
            drug1_spatial_pos = drug1_spatial_pos.clamp(0, self.multi_hop_max_dist)
            drug1_edge_input = d1_edge_input[:, :, :, :self.multi_hop_max_dist, :]
        # [n_graph, n_node, n_node, max_dist, n_head]
        drug1_edge_input = self.d_edge_encoder(drug1_edge_input).mean(-2)

        max_dist = drug1_edge_input.size(-2)
        drug1_edge_input_flat = drug1_edge_input.permute(
            3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug1_edge_input_flat = torch.bmm(drug1_edge_input_flat, self.d_edge_dis_encoder.weight.reshape(
            -1, self.num_heads, self.num_heads)[:max_dist, :, :])
        drug1_edge_input = drug1_edge_input_flat.reshape(
            max_dist, drug1_n_graph, drug1_n_node, drug1_n_node, self.num_heads).permute(1, 2, 3, 0, 4)        # 2,43,43,20,8
        drug1_edge_input = (drug1_edge_input.sum(-2) /
                      (drug1_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1, 2)          # 2,8,43,43
###################################################################################################################
        # edge_input
        drug2_spatial_pos = d2_spatial_pos.clone()
        drug2_spatial_pos[drug2_spatial_pos == 0] = 1  # set pad to 1
        # set 1 to 1, x > 1 to x - 1
        drug2_spatial_pos = torch.where(drug2_spatial_pos > 1, drug2_spatial_pos - 1, drug2_spatial_pos)
        if self.multi_hop_max_dist > 0:
            drug2_spatial_pos = drug2_spatial_pos.clamp(0, self.multi_hop_max_dist)
            drug2_edge_input = d2_edge_input[:, :, :, :self.multi_hop_max_dist, :]
        # [n_graph, n_node, n_node, max_dist, n_head]
        drug2_edge_input = self.d_edge_encoder(drug2_edge_input).mean(-2)

        max_dist = drug2_edge_input.size(-2)
        drug2_edge_input_flat = drug2_edge_input.permute(
            3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug2_edge_input_flat = torch.bmm(drug2_edge_input_flat, self.d_edge_dis_encoder.weight.reshape(
            -1, self.num_heads, self.num_heads)[:max_dist, :, :])
        drug2_edge_input = drug2_edge_input_flat.reshape(
            max_dist, drug2_n_graph, drug2_n_node, drug2_n_node, self.num_heads).permute(1, 2, 3, 0, 4)  # 2,43,43,20,8
        drug2_edge_input = (drug2_edge_input.sum(-2) /
                            (drug2_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1, 2)  # 2,8,43,43

        # reset
        drug1_graph_attn_bias[:, :, 1:, 1:] = drug1_graph_attn_bias[:,
                                              :, 1:, 1:] + drug1_edge_input
        drug1_graph_attn_bias = drug1_graph_attn_bias + d1_attn_bias.unsqueeze(1)

        drug2_graph_attn_bias[:, :, 1:, 1:] = drug2_graph_attn_bias[:,
                                              :, 1:, 1:] + drug2_edge_input
        drug2_graph_attn_bias = drug2_graph_attn_bias + d2_attn_bias.unsqueeze(1)

        # node feauture + graph token
        drug1_node_feature = self.d_node_encoder(d1_node).sum(
            dim=-2)  # [n_graph, n_node, n_hidden
        drug1_node_feature = drug1_node_feature + \
                       self.d_in_degree_encoder(d1_in_degree) + \
                       self.d_out_degree_encoder(d1_out_degree)
        drug1_graph_token_feature = self.d_graph_token.weight.unsqueeze(
            0).repeat(drug1_n_graph, 1, 1)
        drug1_graph_node_feature = torch.cat(
            [drug1_graph_token_feature, drug1_node_feature], dim=1)

        drug2_node_feature = self.d_node_encoder(d2_node).sum(
            dim=-2)  # [n_graph, n_node, n_hidden
        drug2_node_feature = drug2_node_feature + \
                             self.d_in_degree_encoder(d2_in_degree) + \
                             self.d_out_degree_encoder(d2_out_degree)
        drug2_graph_token_feature = self.d_graph_token.weight.unsqueeze(
            0).repeat(drug2_n_graph, 1, 1)
        drug2_graph_node_feature = torch.cat(
            [drug2_graph_token_feature, drug2_node_feature], dim=1)

        # transfomrer encoder
        drug1_output = self.input_dropout(drug1_graph_node_feature)
        drug1_output = self.d_encoders(drug1_output, drug1_graph_attn_bias)


        drug2_output = self.input_dropout(drug2_graph_node_feature)
        drug2_output = self.d_encoders(drug2_output, drug2_graph_attn_bias)

        i = torch.cat((drug1_output, drug2_output), dim=1).permute(0, 2, 1)

        i = self.input_dropout(i)
        # print(i.shape)
        i = self.icnn(i)

        # print("i.shape: ", i.shape)
        f = i.view(drug1_n_graph, -1)
        #print(f.shape)
        score = self.decoder(f)
        #score = self.softmax(score)

        return score

    # help classes


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
        attns.append(attn)
    
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Distilling_layer(nn.Module):
    def __init__(self, channel):
        super(Distilling_layer, self).__init__()
    
        self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(channel)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.maxPool(self.activation(self.norm(x))).transpose(1, 2)

        return x
