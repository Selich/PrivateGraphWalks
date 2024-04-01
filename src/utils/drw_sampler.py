import random
import networkx as nx
import matplotlib.pyplot as plt


def drw_sampler(G, L):
    remaining_nodes = set(G.nodes())
    subgraphs = []
    
    while remaining_nodes:
        subgraph_nodes = []
        start_node = random.sample(remaining_nodes, 1)[0] 

        subgraph_nodes.append(start_node)
        remaining_nodes.remove(start_node)
        walk_length = 0
        
        while walk_length < L and remaining_nodes:
            valid_neighbors = [neighbor for neighbor in G.neighbors(start_node) if neighbor in remaining_nodes]
            if not valid_neighbors:
                break
            next_node = random.sample(valid_neighbors, 1)[0] 
            subgraph_nodes.append(next_node)
            remaining_nodes.remove(next_node)
            start_node = next_node
            walk_length += 1
        
        subgraph_nx = G.subgraph(subgraph_nodes).copy()

        subgraphs.append(subgraph_nx)
    
    return subgraphs


def drw_r_sampler(G, L, R):
    remaining_nodes = set(G.nodes())
    subgraphs = []
    
    while remaining_nodes:
        subgraph = []
        root = random.sample(remaining_nodes, 1)[0]
        subgraph.append(root)
        remaining_nodes.remove(root)
        
        for r in range(R):
            v = root
            l = 0
            while l < L:
                valid_neighbors = [u for u in G.neighbors(v) if u in remaining_nodes]
                if not valid_neighbors:
                    break
                v = random.sample(valid_neighbors, 1)[0]
                subgraph.append(v)
                remaining_nodes.remove(v)
                l += 1
            subgraphs.append(subgraph)
    
    subgraph_nx = [G.subgraph(sg) for sg in subgraphs]
    
    return subgraph_nx