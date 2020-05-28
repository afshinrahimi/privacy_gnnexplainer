import os.path as osp
from math import sqrt
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from read_geo import Geo, get_geo_data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GNNExplainer
from torch.nn import Sequential, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import pdb
import numpy as np
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
A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = get_geo_data(dataset.raw_dir, 'dump.pkl')
U = U_train + U_dev + U_test

def visualize_subgraph(explainer, node_idx, edge_index, edge_mask, y=None, threshold=None, only_topk_edges=None, **kwargs):
    r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
    :attr:`edge_mask`.

    Args:
        node_idx (int): The node id to explain.
        edge_index (LongTensor): The edge indices.
        edge_mask (Tensor): The edge mask.
        y (Tensor, optional): The ground-truth node-prediction labels used
            as node colorings. (default: :obj:`None`)
        threshold (float, optional): Sets a threshold for visualizing
            important edges. If set to :obj:`None`, will visualize all
            edges with transparancy indicating the importance of edges.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments passed to
            :func:`nx.draw`.

    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """

    assert edge_mask.size(0) == edge_index.size(1)

    # Only operate on a k-hop subgraph around `node_idx`.
    subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
        node_idx, explainer.__num_hops__(), edge_index, relabel_nodes=True,
        num_nodes=None, flow=explainer.__flow__())

    edge_mask = edge_mask[hard_edge_mask]
    if only_topk_edges:
        #top_k_nodes = set(edge_index[:, edge_mask.argsort()[-only_topk_edges:]].tolist()[0])
        visible_edges = edge_mask.argsort()[-only_topk_edges:]
        edge_mask = edge_mask[visible_edges]
        edge_index = edge_index[:, visible_edges]
        chosen_nodes = np.unique(edge_index.flatten())
        subset = chosen_nodes
    
    if threshold is not None:
        if threshold < 1:
            edge_mask = (edge_mask >= threshold).to(torch.float)
        else:
            #top N
            edge_mask = np.where(edge_mask >= np.sort(edge_mask)[-threshold], 1, 0)

    if y is None:
        y = torch.zeros(subset.shape[0],
                        device=edge_index.device)
    else:
        y = y[subset].to(torch.float) / y.max().item()
   
    newdata = Data(edge_index=edge_index, att=edge_mask, y=y,
                num_nodes=y.size(0)).to('cpu')
    G = to_networkx(newdata, node_attrs=None, edge_attrs=['att'])
    '''
    #should be done before relabelling
    if only_topk_edges:
        top_k_nodes = set(edge_index[:, edge_mask.argsort()[-only_topk_edges:]].tolist()[0])
        #iter over G and delete anyting not in top_k nodes
        G.remove_nodes_from(G.nodes() - top_k_nodes)
    '''
    
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)
    mapping = {i: U[i] for i in subset.tolist()}
    G = nx.relabel_nodes(G, mapping)

    kwargs['with_labels'] = kwargs.get('with_labels') or True
    kwargs['font_size'] = kwargs.get('font_size') or 3
    kwargs['node_size'] = kwargs.get('node_size') or 20
    kwargs['cmap'] = kwargs.get('cmap') or 'cool'


    pos = nx.spring_layout(G)
    ax = plt.gca()
    
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['att'], 0.1),
                shrinkA=sqrt(kwargs['node_size']) / 2.0,
                shrinkB=sqrt(kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))

    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **kwargs)
    nx.draw_networkx_labels(G, pos, **kwargs)

    return ax, G

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

explainer = GNNExplainer(model, epochs=2)
node_idx = 9000

node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
print("top features:", node_feat_mask.argsort()[-10:])
print("top edges:", edge_index[:, edge_mask.argsort()[-100:]])
only_topk_edges = 100
ax, G = visualize_subgraph(explainer, node_idx, edge_index, edge_mask, y=data.y, threshold=None, only_topk_edges=only_topk_edges)
plt.savefig(f"{node_idx}-{U[node_idx]}-{only_topk_edges}.pdf")
print('done')
