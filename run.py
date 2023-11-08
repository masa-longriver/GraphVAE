import torch
import torch.nn.functional as F

def run(config, model, data, optim):
    data = data.to(config['device'])
    model.train()
    optim.zero_grad()
    adj, mu, logvar = model(data.x, data.edge_index)
    loss = model.loss_fn(adj, mu, logvar, data)
    loss.backward()
    optim.step()
    
    return loss.item()
