import os.path as osp

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from read_geo import Geo
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GNNExplainer
from torch.nn import Sequential, Linear
import pdb
items = ['edge_index', 'face', 'from_dict', 'is_coalesced', 'is_directed', 'is_undirected', 'keys', 
'norm', 'num_edge_features', 'num_edges', 'num_faces', 'num_features', 
'num_node_features', 'num_nodes', 'pos', 'test_mask', 'to', 'train_mask', 'val_mask', 'x', 'y']
'''
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
print(dir(data))
print(data.x.dtype, data.y.dtype)
'''
dataset = 'geotext'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'geo')
dataset = Geo(path, dataset, transform=None)
data = dataset[0]


#pdb.set_trace()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin = Sequential(Linear(dataset.num_features, 300))
        self.conv1 = GCNConv(300, 100)
        self.conv2 = GCNConv(100, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.lin(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x, edge_index = data.x, data.edge_index

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

explainer = GNNExplainer(model, epochs=200)
node_idx = 9000
node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
plt.show()
