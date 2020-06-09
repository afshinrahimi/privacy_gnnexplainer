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
from haversine import haversine
import logging

def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
        
    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred

def get_distance(user1, user2, userLocation):
    lat1, lon1 = userLocation[user1].split(',')
    lat2, lon2 = userLocation[user2].split(',')
    lat1, lon1 = float(lat1), float(lon1)
    lat2, lon2 = float(lat2), float(lon2)
    distance = haversine((lat1, lon1), (lat2, lon2))
    return distance

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
A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, vocab = get_geo_data(dataset.raw_dir, 'dump.pkl')
U = U_train + U_dev + U_test
locs = np.array([userLocation[u] for u in U])

def visualize_subgraph2(explainer, node_idx, edge_index, edge_mask, y=None, threshold=None, only_topk_edges=None, **kwargs):
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

    #Only operate on a k-hop subgraph around `node_idx`.
    subset, edge_index2, _, hard_edge_mask = k_hop_subgraph(
        node_idx, explainer.__num_hops__(), edge_index, relabel_nodes=True,
        num_nodes=None, flow=explainer.__flow__())
    
    #edge_mask = edge_mask[hard_edge_mask]
    edge_index2 = edge_index[:, hard_edge_mask]
    edge_tuples = set([(a, b) for a, b in zip(edge_index2.numpy()[0], edge_index2.numpy()[1])])
    #edge_index = edge_index[:, edge_mask.argsort()[-only_topk_edges:]]
    #print('subset', subset)
    #print('edge_mask', edge_mask)
    if only_topk_edges:
        #top_k_nodes = set(edge_index[:, edge_mask.argsort()[-only_topk_edges:]].tolist()[0])
        visible_edges = edge_mask.argsort()[-only_topk_edges:]
        to_be_deleted = []
        for i, v in enumerate(visible_edges):
            if (edge_index.numpy()[0, v], edge_index.numpy()[1, v]) not in edge_tuples:

                to_be_deleted.append(i)
            elif edge_index[0, v] == edge_index[1, v]:
                to_be_deleted.append(i)
       
        #visible_edges = np.delete(visible_edges, to_be_deleted) 

        #print(visible_edges)
        edge_mask = edge_mask[visible_edges]
        edge_index = edge_index[:, visible_edges]
        #print(edge_index)
        chosen_nodes = np.unique(edge_index.flatten())

        subset = chosen_nodes
    else:
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
    G = to_networkx(newdata, node_attrs=None, edge_attrs=['att'],  to_undirected=True, remove_self_loops=True)
    '''
    #should be done before relabelling
    if only_topk_edges:
        top_k_nodes = set(edge_index[:, edge_mask.argsort()[-only_topk_edges:]].tolist()[0])
        #iter over G and delete anyting not in top_k nodes
        G.remove_nodes_from(G.nodes() - top_k_nodes)
    '''
    print(edge_index)
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    print(mapping)
    G = nx.relabel_nodes(G, mapping)
    print(G.nodes())
    mapping = {i: U[i].split('_')[1] + '\n' + str(int(all_distances[i])) for i in subset.tolist()}
    G = nx.relabel_nodes(G, mapping)
    print(G.nodes())

    kwargs['with_labels'] = kwargs.get('with_labels') or True
    kwargs['font_size'] = kwargs.get('font_size') or 10
    kwargs['node_size'] = kwargs.get('node_size') or 800
    kwargs['cmap'] = kwargs.get('cmap') or 'cool'


    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    #pos = nx.spectral_layout(G)
    #pos = nx.shell_layout(G)
    ax = plt.gca()
    
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="-",
                alpha=max(data['att'], 0.1),
                shrinkA=sqrt(kwargs['node_size']) / 2.0,
                shrinkB=sqrt(kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))
    #pdb.set_trace()
    #label_mapping = {n: n + '-' + userLocation[n] for n in G.nodes()}
    #G = nx.relabel_nodes(G, mapping)
    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **kwargs)
    nx.draw_networkx_labels(G, pos, **kwargs)

    return ax, G

def visualize_subgraph3(explainer, node_idx, edge_index, edge_mask, y=None,
                        threshold=None, **kwargs):
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

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y = y[subset].to(torch.float) / y.max().item()

    data = Data(edge_index=edge_index, att=edge_mask, y=y,
                num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
    distance_subset = all_distances[subset]
    mapping = {k: i  for k, i in enumerate(subset.tolist())}
    #mapping = {k: str(i) + '\n' + str(int(distance_subset[k])) for k, i in enumerate(subset.tolist())}
    labels = {i: str(i) + '\n' + str(int(all_distances[i])) for i in subset.tolist()}
    G = nx.relabel_nodes(G, mapping)

    kwargs['with_labels'] = kwargs.get('with_labels') or True
    kwargs['font_size'] = kwargs.get('font_size') or 10
    kwargs['node_size'] = kwargs.get('node_size') or 800
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
    nx.draw_networkx_labels(G, pos, labels=labels, **kwargs)

    return ax, G
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = Sequential(Linear(dataset.num_features, 300))
        self.conv1 = GCNConv(300, dataset.num_classes)
        #self.conv2 = GCNConv(300, 300)
        #self.lin2 = Sequential(Linear(300, dataset.num_features))

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        #x = self.conv2(x, edge_index)
        #x = self.lin2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0)
x, edge_index = data.x, data.edge_index
model_path = osp.join(dataset.raw_dir, 'model.pth')
print(f"model path:{model_path}")
if osp.exists(model_path):
    model = torch.load(model_path)

else:
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        torch.save(model, model_path)
log_logists = model(x, edge_index)
y_pred_test = torch.argmax(log_logists, dim=1)[np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test))]
y_pred_test = y_pred_test.detach().numpy()
mean, median, acc, _, _, _ = geo_eval(Y_test, y_pred_test, U_test, classLatMedian, classLonMedian, userLocation)
print(f"mean:{mean} median: {median} acc: {acc}")

explainer = GNNExplainer(model, epochs=200)
node_idx = 9000
print(userLocation[U[node_idx]])

node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
print("top features:\n", node_feat_mask.argsort()[-10:])
print("top edges:\n", edge_index[:, edge_mask.argsort()[-10:]])
only_topk_edges = 20
all_distances = torch.FloatTensor([get_distance(U[node_idx], u, userLocation) for u in U])
#ax, G = visualize_subgraph2(explainer, node_idx, edge_index, edge_mask, y=all_distances, threshold=None, only_topk_edges=only_topk_edges, cmap=plt.cm.cool)
#plt.savefig(f"{node_idx}-{U[node_idx]}-{only_topk_edges}.pdf")
#plt.close()
#print(data.y)

ax, G = visualize_subgraph3(explainer, node_idx, edge_index, edge_mask, y=all_distances, threshold=None, only_topk_edges=only_topk_edges, cmap=plt.cm.cool)
plt.savefig(f"{node_idx}-{U[node_idx]}-{only_topk_edges} - original.pdf")
plt.close()

print('done')
