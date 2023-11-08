import datetime as dt
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

def make_dir(path):
    path_list = path.split('/')
    now_path = ""
    for i, dir in enumerate(path_list):
       if i == 0:
          continue
       else:
          now_path += f"/{dir}"
          if not os.path.exists(now_path):
             os.makedirs(now_path)


class EarlyStopping():
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(os.getcwd(), 'log', 'models')
        make_dir(model_path)
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_nm = os.path.join(model_path, f'{now}_model.pt')

        self.best_loss = float('inf')
        self.best_model = None
        self.patience = 0
    
    def save_model(self):
        torch.save(self.best_model.state_dict, self.file_nm)
    
    def check(self, loss, model, epoch):
        bool = False

        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
            self.best_model = model
        
        else:
            self.patience += 1
            if self.patience >= self.config['train']['patience']:
                print("Early Stopping.", flush=True)
                print(f"Best valid loss: {self.best_loss:.4f}", flush=True)
                self.save_model()
                bool = True
        
        if epoch + 1 == self.config['train']['epochs']:
            self.save_model()
        
        return bool

class Visualize():
    def __init__(self):
        self.now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.parent_path = os.path.join(os.getcwd(), 'log/img')
        make_dir(self.parent_path)
    
    def save_loss(self, train_loss):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='train')
        plt.title("losses")
        plt.ylabel('loss')
        plt.xlabel("epoch")
        plt.legend()

        file_nm = os.path.join(self.parent_path, f"{self.now}_loss.png")
        plt.savefig(file_nm)
        plt.close()
    
    def save_result(self, model, data):
        graphs = []
        adj = to_dense_adj(data.edge_index).squeeze(0).cpu().detach().numpy()
        graphs.append(adj)

        model.eval()
        with torch.no_grad():
            _, mu, _ = model(data.x, data.edge_index)
        for i in tqdm(range(4)):
            z = torch.randn_like(mu)
            new_adj = model.decoder(z)
            new_adj = (new_adj > 0.5).float()
            new_adj = new_adj.cpu().detach().numpy()
            np.fill_diagonal(new_adj, 0)
            graphs.append(new_adj)
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
        axes = axes.flatten()
        for i, graph in tqdm(enumerate(graphs)):
            if i >= 4:
                break
            ax = axes[i]
            G = nx.from_numpy_array(graph)
            pos = nx.spring_layout(G, seed=25)
            nx.draw(G, pos, ax=ax, with_labels=False, node_size=20)
            ax.axis('off')
        
        file_nm = os.path.join(self.parent_path, f"{self.now}_result.png")
        plt.tight_layout()
        plt.savefig(file_nm)
        plt.close()