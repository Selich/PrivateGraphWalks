import numpy as np
import networkx as nx
import heapq
from scipy.stats import laplace
from utils import random_walk_sampling

def drw_with_privacy_budget_allocation(G, walk_length, epsilon_total, alpha):
    epsilon_laplace = alpha * epsilon_total
    epsilon_rr = (1 - alpha) * epsilon_total

    start_vectors = {n: random_walk_sampling(G, n, restart_prob=epsilon_rr) for n in G}

    found_clusters = set()
    D = set()
    W = [] 

    for n, path in start_vectors.items():
        degree_private = np.clip(len(path) + laplace.rvs(scale=1/epsilon_laplace), 0, G.number_of_nodes())

        sampling_probability = degree_private / max(degree_private * (2 * epsilon_rr - 1) + (G.number_of_nodes() - 1) * (1 - epsilon_rr), 1)

        N = {'nodes': path, 'prev_weight': 0, 'weight': 0, 'sampling_probability': sampling_probability}
        heapq.heappush(W, N)
    
    return process_clusters(D, overlap_threshold)

