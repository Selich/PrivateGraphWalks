import networkx as nx
import numpy as np
import random

def perform_random_walk(graph, start_node, walk_length):
    walk = [start_node]
    current_node = start_node
    for _ in range(walk_length):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        walk.append(current_node)
    return walk

def generate_disjoint_subgraphs(graph, walk_length):
    all_nodes = set(graph.nodes())
    subgraphs = []
    
    while all_nodes:
        start_node = random.choice(list(all_nodes))
        walk = perform_random_walk(graph, start_node, walk_length)
        subgraph_nodes = set(walk)
        subgraphs.append(graph.subgraph(subgraph_nodes).copy())
        
        all_nodes -= subgraph_nodes
    
    return subgraphs
