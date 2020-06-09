import numpy as np
#import cPickle
from torch_geometric.data import Data, InMemoryDataset
import torch
import os.path as osp
import pickle
import gzip
import scipy as sp
import pdb
import os
import logging
from data import DataLoader, dump_obj, load_obj
import networkx as nx
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def load_obj(filename, serializer=pickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin, encoding='latin1')
    return obj



def get_geo_data(raw_dir, name):
    filename = osp.join(raw_dir, name)
    #print(raw_dir, name)
    geo_data = load_obj(filename)
    #A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = geo_data
    return geo_data

def load_geotext(raw_dir, name):
    filename = osp.join(raw_dir, name)
    #print(raw_dir, name)
    #geo_data = load_obj(filename)
    #A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = geo_data
    geo_data = preprocess_data(raw_dir, builddata=False)
    #vocab = None
    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, vocab = geo_data
    A.setdiag(0)
    A[A>0] = 1
    A = A.tocoo()
    edge_index = torch.tensor([A.row, A.col], dtype=torch.long)
    #A is the normalised laplacian matrix as A_hat in Kipf et al. (2016).
    #The X_? and Y_? should be concatenated to be feed to GCN.

    X = sp.sparse.vstack([X_train, X_dev, X_test])
    X = X.todense().astype(np.float32)
    X = torch.from_numpy(X)
    '''
    X = X.tocoo()
    values = X.data
    indices = np.vstack((X.row, X.col))
    X = torch.sparse_coo_tensor(indices = torch.tensor(indices), values = torch.tensor(values), size=X.shape)
    '''

    if len(Y_train.shape) == 1:
        y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        y = np.vstack((Y_train, Y_dev, Y_test))
    #print(A.shape, X.shape, y.shape)
    y = y.astype(np.int64)
    y = torch.from_numpy(y)
    
    #get train/dev/test indices in X, Y, and A.
    train_index = torch.arange(0, X_train.shape[0], dtype=torch.long)
    val_index = torch.arange(X_train.shape[0], X_train.shape[0] + X_dev.shape[0], dtype=torch.long)
    test_index = torch.arange(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0], dtype=torch.long)
    
    #print(val_index, y.size(0))
    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    data = Data(x=X, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, geo_data

def preprocess_data(data_home, builddata=False, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'latin1')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')
    dump_file = os.path.join(data_home, 'dump.pkl')
    if os.path.exists(dump_file) and os.path.exists(vocab_file) and not builddata:
        logging.info('loading data from dumped file...')
        data = load_obj(dump_file)
        vocab = load_obj(vocab_file)
        logging.info('loading data finished!')
        return data, vocab

    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    vocab = dl.vectorizer.vocabulary_
    #logging.info('saving vocab in {}'.format(vocab_file))
    #dump_obj(vocab, vocab_file)
    logging.info('vocab dumped successfully!')
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()    

    dl.get_graph()  
    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')
    
    adj.setdiag(0)
    #selfloop_value = np.asarray(adj.sum(axis=1)).reshape(-1,)
    selfloop_value = 1
    adj.setdiag(selfloop_value)
    n,m = adj.shape
    diags = adj.sum(axis=1).flatten()
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0/sp.sqrt(diags)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    A = D_pow_neghalf * adj * D_pow_neghalf
    A = A.astype(dtype)
    logging.info('adjacency matrix created.')

    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}
    
    
    
    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    
    data = (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, vocab)
    if not builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')

    return data

class Geo(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name, split="public", num_train_per_class=20,
                 num_val=500, num_test=1000, transform=None,
                 pre_transform=None):
        self.name = name

        self.data_geo = None
        super(Geo, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero().view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero().view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = ['dump.pkl']
        return names

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data, _ = load_geotext(self.raw_dir, 'dump.pkl')
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)