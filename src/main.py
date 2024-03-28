import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from utils.data_utils import load_cora_dataset, split_train_test_subgraphs
from utils.evaluation import evaluate_model
from models.gcn import GCN, train, train_with_dp


def main():
    walk_length = 5 
    train_size = 0.8 
    max_grad_norm = 1.0
    noise_multiplier = 0.5
    
    G_cora = load_cora_dataset(cites_file='../data/cora.cites', content_file='../data/cora.content')
    nfeat = G_cora.nodes[next(iter(G_cora))]['feature'].shape[0]
    nclass = len(set(nx.get_node_attributes(G_cora, 'label').values()))

    train_disjoint_subgraphs, test_disjoint_subgraphs = split_train_test_subgraphs(G_cora, walk_length)

    model_non_dp = GCN(nfeat=nfeat, nhid=16, nclass=nclass)
    model_dp = GCN(nfeat=nfeat, nhid=16, nclass=nclass)

    optimizer_non_dp = torch.optim.Adam(model_non_dp.parameters(), lr=0.01)
    optimizer_dp = torch.optim.Adam(model_dp.parameters(), lr=0.01)


    for sg in train_disjoint_subgraphs:
        train(model_non_dp, sg, optimizer_non_dp)
    
    for sg in train_disjoint_subgraphs:
        train_with_dp(model_dp, sg, optimizer_dp, max_grad_norm, noise_multiplier)

    f1_non_dp = evaluate_model(model_non_dp, test_disjoint_subgraphs)
    f1_dp = evaluate_model(model_dp, test_disjoint_subgraphs)
    
    print(f"Non-DP Model - F1 Micro Score: {f1_non_dp}")
    print(f"DP Model - F1 Micro Score: {f1_dp}")

if __name__ == "__main__":
    main()
