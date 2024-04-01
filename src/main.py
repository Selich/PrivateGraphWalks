import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import torch.nn.functional as F
import networkx as nx
import numpy as np
from utils.arws import adaptive_random_walk_sampling
from utils.data_utils import load_cora_dataset, split_train_test_subgraphs, split_train_test_subgraphs_arws
from utils.evaluation import evaluate_model
from models.gcn import GCN, train, train_with_dp
from utils.random_walk_sampling import generate_disjoint_subgraphs


def main():
    seed_number = 42
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)

    walk_length = 5 
    train_size = 0.8 
    max_grad_norm = 1.0
    noise_multiplier = 0.5
    
    # G_cora = load_cora_dataset(cites_file='../data/webkb/texas.cites', content_file='../data/webkb/texas.content')
    G_cora = load_cora_dataset(cites_file='../data/cora.cites', content_file='../data/cora.content')
    nfeat = G_cora.nodes[next(iter(G_cora))]['feature'].shape[0]
    nclass = len(set(nx.get_node_attributes(G_cora, 'label').values()))

    # disjoint_subgraphs = generate_disjoint_subgraphs(G_cora, walk_length=5)


    train_disjoint_subgraphs, test_disjoint_subgraphs = split_train_test_subgraphs(G_cora, walk_length)

    model_non_dp = GCN(nfeat=nfeat, nhid=16, nclass=nclass)
    model_dp = GCN(nfeat=nfeat, nhid=16, nclass=nclass)

    optimizer_non_dp = torch.optim.Adam(model_non_dp.parameters(), lr=0.01)
    optimizer_dp = torch.optim.Adam(model_dp.parameters(), lr=0.01)

    train(model_non_dp, train_disjoint_subgraphs, optimizer_non_dp)
    train(model_dp, train_disjoint_subgraphs, optimizer_dp)

    f1_non_dp = evaluate_model(model_non_dp, test_disjoint_subgraphs)
    f1_dp = evaluate_model(model_dp, test_disjoint_subgraphs)

    print(f"Non-DP Model - F1 Micro Score: {f1_non_dp}")
    print(f"DP Model - F1 Micro Score: {f1_dp}")

if __name__ == "__main__":
    main()
