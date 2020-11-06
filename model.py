import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.FloatTensor)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, graphsage, max_len):
        super(BiRNN, self).__init__()
        self.device = torch.device('cuda')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.graphSage = graphsage
        self.max_len = max_len

        self.timestamp_embedding_W = nn.Parameter(torch.ones(size=(1, 10))).to(self.device)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size

    def forward(self, x):
        # 初始话LSTM的隐层和细胞状态

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 同样考虑向前层和向后层
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        l_list = x[:,:,1]
        bsz = l_list.size(0)
        l_list = l_list.view(l_list.size(0)*l_list.size(1))
        flag = True
        l_emb = None
        for tra_l in l_list:
            if tra_l == -1:
                emb = torch.zeros([1, 128]).to(self.device)
            else:
                # print(tra_l)
                emb = self.graphSage([int(tra_l)])
            if flag:
                l_emb = emb
                flag = False
            else:
                l_emb = torch.cat((l_emb, emb), dim=0)
        l_emb = l_emb.view([bsz, self.max_len, 128])

        t = torch.index_select(x, dim=2, index=torch.tensor([0]).to('cuda')).float().to(self.device)
        t_emb = torch.matmul(t, self.timestamp_embedding_W).to(self.device)
        t_emb = F.normalize(t_emb, p=2, dim=1)
        # l_emb = torch.index_select(x, dim=2, index=torch.tensor([i for i in range(128)]))
        embedding = torch.cat((t_emb, l_emb), 2)

        # 前向传播 LSTM
        out, _ = self.lstm(embedding, (h0, c0))  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)

        # 解码最后一个时刻的隐状态
        out = self.fc(out[:, -1, :])
        return out


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()
        self.input_size = input_size

        # num_node = 300
        # self.input_size = num_node
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        self.raw_features = raw_features

        # self.features_W = nn.Parameter(torch.ones(size=(raw_features.size(1), num_node))).to(self.device)

        self.adj_lists = adj_lists

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else self.input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        # pre_hidden_embs = torch.matmul(self.raw_features, self.features_W).to(self.device)
        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs
        # 输出为节点的embedding
        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    # num_sample = 10
    def _get_unique_neighs_list(self, nodes, num_sample=20):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # self.dc.logger.info('4')

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            # print(mask)
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            # self.dc.logger.info('5')
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats
