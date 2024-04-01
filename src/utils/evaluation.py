import torch
from sklearn.metrics import f1_score

from utils.data_utils import prepare_subgraph_data

def evaluate_model(model, subgraphs):
    model.eval()
    all_labels = []
    all_predictions = []

    for sg in subgraphs:
        features, adj, labels = prepare_subgraph_data(sg) 
        with torch.no_grad():
            logits = model(features, adj)
            predictions = logits.max(1)[1]
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return f1_score(all_labels, all_predictions, average='micro')
