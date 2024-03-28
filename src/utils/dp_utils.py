import torch 
import torch.nn.functional as F
import networkx as nx
import numpy as np

def prepare_subgraph_data(subgraph):
    adj = nx.to_numpy_array(subgraph) + np.eye(subgraph.number_of_nodes())
    
    features = np.array([subgraph.nodes[n]['feature'] for n in subgraph.nodes()])
    
    labels = np.array([subgraph.nodes[n]['label'] for n in subgraph.nodes()])
    
    return torch.FloatTensor(features), torch.FloatTensor(adj), torch.LongTensor(labels)

def dp_sgd_step(model, optimizer, features, adj, labels, max_grad_norm, noise_multiplier):
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.nll_loss(output, labels)
    loss.backward()
    
    total_grad_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    clip_coef = max_grad_norm / (total_grad_norm + 1e-6)
    for p in model.parameters():
        p.grad.data.mul_(min(1, clip_coef))
        # Add Gaussian noise to gradients
        p.grad.data.add_(torch.randn_like(p.grad.data) * (max_grad_norm * noise_multiplier))
    
    optimizer.step()
    return loss.item()


