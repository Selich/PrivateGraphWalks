import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import prepare_subgraph_data
from utils.dp_utils import dp_sgd_step

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def train(model, disjoint_subgraphs, optimizer):
    model.train()
    total_loss = 0
    
    for sg in disjoint_subgraphs:
        optimizer.zero_grad()

        
        features, adj, labels = prepare_subgraph_data(sg)
        
        output = model(features, adj)
        loss = F.nll_loss(output, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss


def train_with_dp(model, disjoint_subgraphs, optimizer, max_grad_norm, noise_multiplier):
    model.train()
    total_loss = 0
    
    for sg in disjoint_subgraphs:
        assert isinstance(sg, nx.Graph), "sg should be a networkx Graph object"
        features, adj, labels = prepare_subgraph_data(sg)
        loss = dp_sgd_step(model, optimizer, features, adj, labels, max_grad_norm, noise_multiplier)
        total_loss += loss
    
    return total_loss / len(disjoint_subgraphs)