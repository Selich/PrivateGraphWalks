import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_mlp(model, features, labels, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(features)
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Training loss: {loss.item()}')
    return model