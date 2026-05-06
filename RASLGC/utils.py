import numpy as np
from sklearn import cluster
from sklearn import metrics
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.data import Data
import math
import pandas as pd
def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def enhance_sim_matrix(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = 0.5 * (L + L.T)
    L = L/L.max()
    return L


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    L = enhance_sim_matrix(C, K, d, alpha)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    #print("Fitting Spectral Clustering...", flush=True)
    spectral.fit(L)
    #print("Predicting Spectr Clustering...", flush=True)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def err_rate(gt_s, s):
    y_pred = best_map(gt_s,s)
    #err_x = np.sum(gt_s[:] != c_x[:])
    #missrate = err_x.astype(float) / (gt_s.shape[0])
    acc = metrics.accuracy_score(gt_s, y_pred)
    nmi = metrics.normalized_mutual_info_score(gt_s, y_pred)
    f1_macro = metrics.f1_score(gt_s, y_pred, average='macro')
    return [acc, nmi, f1_macro]


def get_dataset(path, name):
    assert name in ['Cora', 'PubMed', 'PubMed', 'physics']

    if name == 'physics':
        return Coauthor(path, name)
    else:
        return Planetoid(path, name)


def load_wiki(wiki_path):
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]

    #replace with actual data from Wiki
    features = 0*torch.FloatTensor(2405, 4973)
    adj = 0*torch.LongTensor(2,17981)
    labels = 0*torch.LongTensor(2405)

    print("Loading Wiki dataset")
    with open('wiki/graph0_0.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1
    # print("processed")
    # print(adj)

    with open(wiki_path+'/wiki/tfidf.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            u = int(temp_list[0])
            v = int(temp_list[1])
            features[u,v] = float(temp_list[2])
            i+=1
    #print("processed")
    #print(features[:20, :20])

    with open(wiki_path+'/wiki/group.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            node = int(temp_list[0])
            label = int(temp_list[1])
            #labels 12 and 14 are missing in data. Rename 18 and 19 to 12 and 14
            if label == 18:
                label = 12
            if label == 19:
                label = 14
            labels[node] = label-1
            i+=1
    #print("processed")
    print(labels)
    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features


def load_TTC():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('TTC/TTC_normalized.csv').values
    label = pd.read_csv('TTC/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,3600*6)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('TTC/cosgraph.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[1,i] = int(temp_list[0])
            adj[0,i] = int(temp_list[1])
            i+=1


    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_TUANDROMD():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('TUANDROMD/label.csv').values
    label = pd.read_csv('./TUANDROMD/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,26784)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('TUANDROMD/graph0_0.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1


    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_REJAFADA():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('REJAFADA/REJAFADA_normalized.csv').values
    label = pd.read_csv('./REJAFADA/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,1996)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('REJAFADA/graph0.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1


    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_CNAE():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('CNAE/CNAE_normalized.csv').values
    label = pd.read_csv('./CNAE/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,1080)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('CNAE/graph0.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1


    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_Cardiotocography():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('Cardiotocography/Cardiotocography_normalized.csv').values
    label = pd.read_csv('Cardiotocography/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,2126)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('Cardiotocography/graph0.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1


    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_spambase():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('spambase/spambase_normalized.csv').values
    label = pd.read_csv('./spambase/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,4600)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('spambase/graph0.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1


    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_NIST():
    #load a dummy dataset to return the data in the same format as
    #those available in pytorch geometric
    path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    dataset = get_dataset(path, "Cora")
    data = dataset[0]
    content = pd.read_csv('NIST/NIST_normalized.csv').values
    label = pd.read_csv('./NIST/data.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')
    adj = 0*torch.LongTensor(2,56200)
    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('NIST/10graph.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0,i] = int(temp_list[0])
            adj[1,i] = int(temp_list[1])
            i+=1


    data.x = features
    data.y = labels
    data.edge_index = adj
    num_features = features.shape[1]
    return data, num_features

def load_Cora(i):
    # path = osp.join(osp.expanduser('~'), 'datasets', "Cora")
    # dataset = get_dataset(path, "Cora")
    # data = dataset[0]

    content = pd.read_csv('Cora/data.csv').values
    label = pd.read_csv('Cora/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,10556)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('Cora/graph_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features


def load_CiteSeer(i):
    # path = osp.join(osp.expanduser('~'), 'datasets', "CiteSeer")
    # dataset = get_dataset(path, "CiteSeer")
    # data = dataset[0]

    content = pd.read_csv('CiteSeer/data.csv').values
    label = pd.read_csv('CiteSeer/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,9104)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('CiteSeer/graph_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features



def load_Wiki(i):
    # path = osp.join(osp.expanduser('~'), 'datasets', "Wiki")
    # dataset = get_dataset(path, "Wiki")
    # data = dataset[0]

    content = pd.read_csv('Wiki/data.csv').values
    label = pd.read_csv('Wiki/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,17981)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('Wiki/graph_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features

def load_acm(i):
    # path = osp.join(osp.expanduser('~'), 'datasets', "acm")
    # dataset = get_dataset(path, "acm")
    # data = dataset[0]

    content = pd.read_csv('acm/data.csv').values
    label = pd.read_csv('acm/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,26256)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('acm/graph0_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features

def load_PubMed():
    # path = osp.join(osp.expanduser('~'), 'datasets', "PubMed")
    # dataset = get_dataset(path, "PubMed")
    # data = dataset[0]

    content = pd.read_csv('PubMed/data.csv').values
    label = pd.read_csv('PubMed/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,88648)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('PubMed/graph.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features

def load_dblp(i):
    # path = osp.join(osp.expanduser('~'), 'datasets', "dblp")
    # dataset = get_dataset(path, "dblp")
    # data = dataset[0]

    content = pd.read_csv('dblp/data.csv').values
    label = pd.read_csv('dblp/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,7056)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('dblp/graph_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features


def load_amac(i):
    # path = osp.join(osp.expanduser('~'), 'datasets', " amac")
    # dataset = get_dataset(path, " amac")
    # data = dataset[0]

    content = pd.read_csv('amac/data.csv').values
    label = pd.read_csv('amac/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,491722)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('amac/graph0_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features

def load_CACS():
    content = pd.read_csv('CACS/data.csv').values
    label = pd.read_csv('CACS/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,163788)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('CACS/graph.txt', 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features



def load_physics(i):
    content = pd.read_csv('physics/data.csv').values
    label = pd.read_csv('physics/label.csv').values.flatten()
    features = torch.tensor(content, dtype=torch.float32).to(device='cuda')

    adj = 0 * torch.LongTensor(2,495924)




    labels = torch.tensor(label, dtype=torch.int).to(device='cuda')

    print("Loading  dataset")
    with open('physics/graph0_{}.txt'.format(i), 'r') as f:
        i = 0
        for line in f:
            temp_list = line.split()
            adj[0, i] = int(temp_list[0])
            adj[1, i] = int(temp_list[1])
            i += 1

    # data.x = features
    # data.y = labels
    # data.edge_index = adj
    data = Data(x=features, edge_index=adj, y=labels)
    num_features = features.shape[1]
    return data, num_features