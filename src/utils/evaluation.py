import torch
from sklearn.metrics import f1_score

def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        predictions = logits.max(1)[1].type_as(labels)
    return f1_score(labels.cpu(), predictions.cpu(), average='micro')