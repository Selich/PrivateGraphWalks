import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import pandas as pd

from utils.arws import adaptive_random_walk_sampling
from utils.drw_sampler import drw_sampler
from utils.random_walk_sampling import generate_disjoint_subgraphs

def load_cora_dataset(cites_file='cora.cites', content_file='cora.content'):
    cites = pd.read_csv(cites_file, sep='\t', header=None, names=['target', 'source'])
    edges = list(zip(cites['source'], cites['target']))

    content = pd.read_csv(content_file, sep='\t', header=None)
    content.set_index(0, inplace=True)

    features = content.iloc[:, :-1].values
    labels = pd.get_dummies(content.iloc[:, -1]).values
    node_index = content.index

    G = nx.DiGraph()
    G.add_nodes_from(node_index)
    G.add_edges_from(edges)
    
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['feature'] = features[i]
        G.nodes[node]['label'] = np.argmax(labels[i])
    
    return G

def prepare_data_for_gnn(graph):
    disjoint_subgraphs = generate_disjoint_subgraphs(graph, walk_length=5)
    return disjoint_subgraphs

def split_train_test_subgraphs(full_graph, walk_length, train_size=0.8):
    all_subgraphs = drw_sampler(full_graph, walk_length)

    random.shuffle(all_subgraphs)
    
    split_idx = int(train_size * len(all_subgraphs))
    train_subgraphs = all_subgraphs[:split_idx]
    test_subgraphs = all_subgraphs[split_idx:]
    # This is correct
    

    return train_subgraphs, test_subgraphs
    
def split_train_test_subgraphs_arws(full_graph, walk_length, train_size=0.8):
    all_subgraphs = adaptive_random_walk_sampling(full_graph, walk_length)

    random.shuffle(all_subgraphs)
    
    split_idx = int(train_size * len(all_subgraphs))
    train_subgraphs = all_subgraphs[:split_idx]
    test_subgraphs = all_subgraphs[split_idx:]
    
    return train_subgraphs, test_subgraphs


def prepare_subgraph_data(subgraph):
    adj = nx.to_numpy_array(subgraph) + np.eye(subgraph.number_of_nodes())
    
    features = np.array([subgraph.nodes[n]['feature'] for n in subgraph.nodes()])
    
    labels = np.array([subgraph.nodes[n]['label'] for n in subgraph.nodes()])
    
    return torch.FloatTensor(features), torch.FloatTensor(adj), torch.LongTensor(labels)


def subgraph_to_tensors(subgraph, node_features, node_labels):
    nodes = list(subgraph.nodes())
    
    features = [node_features[node] for node in nodes]
    X = torch.tensor(features, dtype=torch.float)
    
    adj_matrix = nx.to_numpy_matrix(subgraph, nodelist=nodes)
    A = torch.tensor(adj_matrix, dtype=torch.float)
    
    labels = [node_labels[node] for node in nodes]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return X, A, labels_tensor