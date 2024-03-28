import random
import networkx as nx

def drw_sampler(G, L):
    remaining_nodes = set(G.nodes())
    subgraphs = []
    
    while remaining_nodes:
        subgraph = []
        v = random.sample(list(remaining_nodes), 1)[0]

        subgraph.append(v)
        remaining_nodes.remove(v)
        l = 0
        
        while l < L and remaining_nodes:
            valid_neighbors = [u for u in G.neighbors(v) if u in remaining_nodes]
            if not valid_neighbors:
                break
            v = random.sample(valid_neighbors, 1)[0]
            subgraph.append(v)
            remaining_nodes.remove(v)
            l += 1
        
        subgraph_nx = G.subgraph(subgraph).copy()
        if not isinstance(subgraph_nx, nx.Graph):
            raise TypeError("Expected the subgraph to be an instance of nx.Graph")
        subgraphs.append(subgraph_nx)
    
    return [G.subgraph(sg) for sg in subgraphs]