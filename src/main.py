import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import torch.nn.functional as F
import networkx as nx
import numpy as np
from utils.arws import adaptive_random_walk_sampling
from utils.data_utils import load_cora_dataset, load_webkb_dataset, split_train_test_subgraphs, split_train_test_subgraphs_arws, split_train_test_subgraphs_drwr
# from utils.drw_with_privacy_budget_allocation import repeated_random_walk
from utils.evaluation import evaluate_model
from models.gcn import GCN, train, train_webKB, train_with_dp, train_with_dp_webKB
from utils.random_walk_sampling import generate_disjoint_subgraphs

# def adaptive_privacy_budget_run(G_cora, model):
#     p = 10 
#     alpha = 0.15 
#     k = 50 
#     lambda_val = 0.5 
#     overlap_threshold = 0.3  

#     significant_clusters = repeated_random_walk(G_cora, walk_length=5, p=p, alpha=alpha, k=k, lambda_val=lambda_val, overlap_threshold=overlap_threshold)
    
#     train_subgraphs_new = [G_cora.subgraph(cluster.nodes) for cluster in significant_clusters]

#     model.apply(reset_weights)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     train(model, train_subgraphs_new, optimizer)

#     return model

def main():
    seed_number = 42
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    max_grad_norm = 0.1
    noise_multiplier = 2
    train_size = 0.8
    walk_length = 5
    R = 2

    # G_cora = load_cora_dataset(cites_file='../data/cora.cites', content_file='../data/cora.content')
    G_cora = load_webkb_dataset(cites_file='../data/webkb/cornell.cites', content_file='../data/webkb/cornell.content')

    # nfeat = G_cora.nodes[next(iter(G_cora))]['feature'].shape[0]
    # nclass = len(set(nx.get_node_attributes(G_cora, 'label').values()))

    train_disjoint_subgraphs, test_disjoint_subgraphs = split_train_test_subgraphs(G_cora, walk_length, train_size)
    train_disjoint_subgraphs_drwr, test_disjoint_subgraphs_drwr = split_train_test_subgraphs_drwr(G_cora, walk_length, train_size, R)

    model_non_dp = GCN(nfeat=1703, nhid=16, nclass=6)
    model_dp = GCN(nfeat=1703, nhid=16, nclass=6)
    model_dp_drwr = GCN(nfeat=1703, nhid=16, nclass=6)

    optimizer_non_dp = torch.optim.Adam(model_non_dp.parameters(), lr=0.01)
    optimizer_dp = torch.optim.Adam(model_dp.parameters(), lr=0.01)

    train(model_non_dp, train_disjoint_subgraphs, optimizer_non_dp)
    train_with_dp(model_dp, train_disjoint_subgraphs, optimizer_dp, max_grad_norm, noise_multiplier)
    train_with_dp(model_dp_drwr, train_disjoint_subgraphs_drwr, optimizer_dp, max_grad_norm, noise_multiplier)

    f1_non_dp = evaluate_model(model_non_dp, test_disjoint_subgraphs)
    f1_dp = evaluate_model(model_dp, test_disjoint_subgraphs)
    f1_dp_drwr = evaluate_model(model_dp_drwr, test_disjoint_subgraphs_drwr)

    print(f"Non-DP Model - F1 Micro Score: {f1_non_dp * 100}")
    print(f"DP Model - F1 Micro Score: {f1_dp * 100}")
    print(f"DP Model - DRW-r - F1 Micro Score: {f1_dp_drwr * 100}")

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

if __name__ == "__main__":
    main()
