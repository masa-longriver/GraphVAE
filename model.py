import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, negative_sampling


class Encoder(nn.Module):
    def __init__(self, config, input_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(
            input_dim, config['model']['latent_dim'] * 2, cached=True
        )
        self.conv2 = GCNConv(
            config['model']['latent_dim'] * 2, 
            config['model']['latent_dim'], cached=True
        )
        self.linear_mu = torch.nn.Linear(
            config['model']['latent_dim'], config['model']['latent_dim']
        )
        self.linear_logvar = torch.nn.Linear(
            config['model']['latent_dim'], config['model']['latent_dim']
        )
    
    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        z = self.conv2(z, edge_index)

        mu = self.linear_mu(z)
        logvar = self.linear_logvar(z)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    
    def forward(self, z):
        a = torch.matmul(z, z.t())
        a = torch.sigmoid(a)

        return a


class GraphVAE(nn.Module):
    def __init__(self, config, input_dim):
        super(GraphVAE, self).__init__()
        self.config = config
        self.encoder = Encoder(config, input_dim)
        self.decoder = Decoder()
    
    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        
        else:
            return mu
    
    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)[0]
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)

        return adj, mu, logvar
    
    def loss_fn(self, pred_adj, mu, logvar, data):
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        pos_loss = F.binary_cross_entropy_with_logits(
            pred_adj.view(-1)[
                pos_edge_index[0] * data.num_nodes + pos_edge_index[1]
            ],
            torch.ones(pos_edge_index.size(1), device=self.config['device'])
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            pred_adj.view(-1)[
                neg_edge_index[0] * data.num_nodes + neg_edge_index[1]
            ],
            torch.zeros(neg_edge_index.size(1), device=self.config['device'])
        )
        BCE = pos_loss + neg_loss
        KLD = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
        loss = BCE + KLD

        return loss