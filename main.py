import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub

from config import load_config
from model import GraphVAE
from run import run
from utils import EarlyStopping, Visualize


if __name__ == '__main__':
    config = load_config()

    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    #dataset = Planetoid(root='/tmp/Cora', name='Cora')
    dataset = KarateClub()
    data = dataset[0]

    model = GraphVAE(config, input_dim=dataset.num_features)
    model = model.to(config['device'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optim']['lr'],
        betas=(config['optim']['beta1'], config['optim']['beta2']),
        weight_decay=config['optim']['weight_decay'],
        eps=config['optim']['eps']
    )
    es = EarlyStopping(config)
    visualize = Visualize()

    train_losses = []
    for epoch in range(config['train']['epochs']):
        loss = run(config, model, data, optimizer)
        log_str = f"Epoch: {epoch:>3}, "
        log_str += f"loss: {loss:.4f}, "
        print(log_str, flush=True)
        train_losses.append(loss)

        es_bool = es.check(loss, model, epoch)
        if es_bool:
            break
    
    visualize.save_loss(train_losses)
    visualize.save_result(model, data)