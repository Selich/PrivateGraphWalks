import random
import numpy as np


def adaptive_random_walk_sampling(G, walk_length):
    start_nodes = list(G.nodes())
    sampled_subgraphs = []
    for start_node in start_nodes:
        current_node = start_node
        visited_nodes = {current_node}
        for _ in range(walk_length):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            visited_nodes.add(next_node)
            current_node = next_node
        subgraph = G.subgraph(visited_nodes)
        sampled_subgraphs.append(subgraph)
    return sampled_subgraphs
