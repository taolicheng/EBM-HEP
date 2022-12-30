
import os
import numpy as np
import h5py

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F

import uproot_methods
from utils import jet_e, jet_pt, jet_mass
from numpy import inf

class data():
    def __init__(self, n_train=10000, input_dim=160, scale=False):
        self.n_train = n_train
        self.input_dim = input_dim
        self.scale = scale
        self.val_fn = os.environ['CLFAD_DIR'] + 'qcd_pt600_preprocessed.h5'
        self.test1_fn = os.environ['CLFAD_DIR'] + 'top_pt600_preprocessed.h5'
        self.test2_fn = os.environ['VAE_DIR'] + 'h3_m174_h80_01_preprocessed.h5'
                
        
    def load(self):
        train_set, scaler = load_attn_train(n_train=self.n_train, input_dim=self.input_dim, scale=self.scale)           
        val_set = load_attn_test(scaler, self.val_fn, input_dim=self.input_dim, n_test=10000, scale=self.scale)
        test1_set = load_attn_test(scaler, self.test1_fn, input_dim=self.input_dim, n_test=10000, scale=self.scale)
        test2_set = load_attn_test(scaler, self.test2_fn, input_dim=self.input_dim, n_test=10000, scale=self.scale)
        
        return train_set, val_set, test1_set, test2_set
    
    def obs(self):                             
        train_jets, test_jets = self.load()

        train_jets = jet_from_ptetaphi(train_jets, scaled=self.scale)
        test_jets = jet_from_ptetaphi(test_jets, scaled=self.scale)

        ms_qcd = list(map(jet_mass, train_jets))
        ms_top = list(map(jet_mass, test_jets))
        return ms_qcd, ms_top

def load_attn_train(n_train=None, input_dim=160, scale=False, topref=False):
    
    if topref:
        f = h5py.File(os.environ['TOPREF_DIR']+'train_preprocessed.h5', "r")
        X_train = np.array(f['table'])
        y_train = np.array(f['labels'])
        qcd_train = X_train[y_train==0]
    else:    
        f = h5py.File(os.environ["VAE_DIR"] +"qcd_preprocessed.h5", "r")
        qcd_train = f["constituents" if "constituents" in f.keys() else "table"]
    
    if n_train:
        qcd_train = qcd_train[:n_train, :input_dim]
    else:
        qcd_train = qcd_train[:, :input_dim]
    
    X = qcd_train
    e_j = np.array(list(map(jet_e, X))).reshape(-1,1)
    pt_j = np.array(list(map(jet_pt, X))).reshape(-1,1)
    
    X = X.reshape(len(X), -1, 4)
    e = X[:,:,0]
    px = X[:,:,1]
    py = X[:,:,2]
    pz = X[:,:,3]
   
    v = {}
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, e)

    e = np.log(e)
    pt = np.log(p4.pt)
    eta = p4.eta
    phi = p4.phi
    pt[pt == -inf] = 0.0
    e[e == -inf] = 0.0
    eta = np.nan_to_num(eta)
    
    e = e.reshape(len(e), -1, 1)
    pt = pt.reshape(len(pt), -1, 1)
    eta = eta.reshape(len(eta), -1, 1)
    phi = phi.reshape(len(phi), -1, 1)
    
    X = np.concatenate((pt, eta, phi), -1)
    X = X.reshape(len(X), -1)
    
    if scale:
        scaler = RobustScaler().fit(X)
        X = scaler.transform(X) 
    else:
        scaler = None
    return X, scaler

def load_attn_val(scaler, n_val=10000, input_dim=160, scale=False, pt_scaling=False, pt_refine=True, m_window=False, topref=False):
    '''
    construct validation set for OOD detection.
    different from training data, validation set has sample lables.
    TODO: readjust n_val to match the final number of events
    '''
    from sklearn.utils import shuffle
    from utils import jet_pt, jet_mass
    
    if topref:
        f = h5py.File(os.environ['TOPREF_DIR']+'val_preprocessed.h5', "r")
        val_X = np.array(f['table'])
        val_y = np.array(f['labels'])
        val_X = val_X[-n_val:, :input_dim]
        val_y = val_y[-n_val:]
    else:
        f1 = h5py.File(os.environ["VAE_DIR"] +"qcd_preprocessed.h5", "r")

        qcd_val = f1["constituents" if "constituents" in f1.keys() else "table"]

        qcd_val = np.array(qcd_val)

        if pt_refine:
            from utils import jet_pt, jet_mass
            pts = []
            for j in qcd_val:
                pts.append(jet_pt(j))
            pts = np.array(pts)
            qcd_val = qcd_val[(pts>550) & (pts<=650)]

        qcd_val = qcd_val[-n_val:, :input_dim]

        f = h5py.File(os.environ["VAE_DIR"] +"top_preprocessed.h5", 'r')

        for key in ['table', 'constituents', 'jet1']:
            if key in f.keys():
                w_test=f[key]
                if key == "jet1":
                    labels=f["labels"]
                    labels=np.array(labels)

        w_test = np.array(w_test)

        if pt_refine:
            from utils import jet_pt, jet_mass
            pts = []
            for j in w_test:
                pts.append(jet_pt(j))
            pts = np.array(pts)
            w_test = w_test[(pts>550) & (pts<=650)]
        if m_window:
            ms=[]
            for j in w_test:
                ms.append(jet_mass(j))
            ms=np.array(ms)
            w_test=w_test[(ms>150)&(ms<=200)]

        if pt_scaling:
             for i in range(len(w_test)):
                pt=jet_pt(w_test[i])
                w_test[i]=w_test[i]/pt

        w_test = w_test[-n_val:, :input_dim]           

        val_X = np.concatenate((qcd_val, w_test))
        val_y = np.concatenate((np.zeros(len(qcd_val)), np.ones(len(w_test))))

        val_X, val_y = shuffle(val_X, val_y)
        f1.close()
    
    X = val_X
    e_j = np.array(list(map(jet_e, X))).reshape(-1,1)
    pt_j = np.array(list(map(jet_pt, X))).reshape(-1,1)
    X = X.reshape(len(X), -1, 4)
    e = X[:,:,0]
    px = X[:,:,1]
    py = X[:,:,2]
    pz = X[:,:,3]
   
    v = {}
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, e)

    e = np.log(e)
    pt = np.log(p4.pt)
    eta = p4.eta
    phi = p4.phi
    pt[pt == -inf] = 0.0
    e[e == -inf] = 0.0
    eta = np.nan_to_num(eta)

    e = e.reshape(len(e), -1, 1)
    pt = pt.reshape(len(pt), -1, 1)
    eta = eta.reshape(len(eta), -1, 1)
    phi = phi.reshape(len(phi), -1, 1)
    
    X = np.concatenate((pt, eta, phi), -1)
    X = X.reshape(len(X), -1)
    
    if scale:
        val_X = scaler.transform(X)
    val_X = X
    f.close()
    return val_X, val_y

def load_attn_test(scaler, fn, input_dim=160, n_test=10000, scale=False, pt_scaling=False, pt_refine=True, m_window=False):

    f = h5py.File(fn, 'r')

    for key in ['table', 'constituents', 'jet1']:
        if key in f.keys():
            w_test=f[key]
            if key == "jet1":
                labels=f["labels"]
                labels=np.array(labels)
                
    w_test = np.array(w_test)
    
    if pt_refine:
        from utils import jet_pt, jet_mass
        pts=[]
        for j in w_test:
            pts.append(jet_pt(j))
        pts=np.array(pts)
        w_test=w_test[(pts>550)&(pts<=650)]
    
    if m_window:
        ms=[]
        for j in w_test:
            ms.append(jet_mass(j))
        ms=np.array(ms)
        w_test=w_test[(ms>150)&(ms<=200)]

    w_test = w_test[:n_test,:input_dim]    

    if pt_scaling:
         for i in range(len(w_test)):
            pt=jet_pt(w_test[i])
            w_test[i]=w_test[i]/pt

    X = w_test
    e_j = np.array(list(map(jet_e, X))).reshape(-1,1)
    pt_j = np.array(list(map(jet_pt, X))).reshape(-1,1)
    X = X.reshape(len(X), -1, 4)
    e = X[:,:,0]
    px = X[:,:,1]
    py = X[:,:,2]
    pz = X[:,:,3]
   
    v = {}
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, e)

    e = np.log(e)
    pt = np.log(p4.pt)
    eta = p4.eta
    phi = p4.phi
    pt[pt == -inf] = 0.0
    e[e == -inf] = 0.0
    eta = np.nan_to_num(eta)

    e = e.reshape(len(e), -1, 1)
    pt = pt.reshape(len(pt), -1, 1)
    eta = eta.reshape(len(eta), -1, 1)
    phi = phi.reshape(len(phi), -1, 1)
    
    X = np.concatenate((pt, eta, phi), -1)
    X = X.reshape(len(X), -1)
    
    if scale:
        X = scaler.transform(X)
    
    f.close()
    return X

def load_clf_train(n_train=None, input_dim=80, ova=None):
    '''
    ova: 1 - QCD/others; 2 - W/others; 3 - Top/others
    '''
    
    def load_data(n_train_pclass=350000, input_dim=160, ova=None):
        from sklearn.utils import shuffle

        f = h5py.File(os.environ["CLFAD_DIR"] + 'qcd_pt600_preprocessed.h5', 'r')
        qcd = np.array(f['constituents'])
        f.close()
        f = h5py.File(os.environ["CLFAD_DIR"] + 'w_pt600_preprocessed.h5', 'r')
        w = np.array(f['constituents'])
        f.close()
        f = h5py.File(os.environ["CLFAD_DIR"] + 'top_pt600_preprocessed.h5', 'r')
        top = np.array(f['constituents'])
        f.close()
        
        X = np.concatenate((qcd[:n_train_pclass, :input_dim], w[:n_train_pclass, :input_dim], top[:n_train_pclass, :input_dim]), axis=0)
        #m = np.concatenate((qcd_obs[:n_train_pclass], w_obs[:n_train_pclass], top_obs[:n_train_pclass]))
        if ova:
            y = np.concatenate(((1 - (ova == 1))*np.ones(n_train_pclass), (1 - (ova == 2))*np.ones(n_train_pclass), (1 - (ova == 3))*np.ones(n_train_pclass)))
        else:
            labels_2 = np.empty(n_train_pclass)
            labels_2.fill(2)
            y = np.concatenate((np.zeros(n_train_pclass),np.ones(n_train_pclass), labels_2))

        X, y = shuffle(X, y)

        #y = F.one_hot(torch.tensor(y).to(torch.int64), num_classes=3) # commented out due to torch.nn.CrossEntropyLoss()

        return X, y
    
    
    X, y = load_data(n_train // 3, input_dim, ova)
    e_j = np.array(list(map(jet_e, X))).reshape(-1,1)
    pt_j = np.array(list(map(jet_pt, X))).reshape(-1,1)
    
    X = X.reshape(len(X), -1, 4)
    e = X[:,:,0]
    px = X[:,:,1]
    py = X[:,:,2]
    pz = X[:,:,3]
   
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, e)

    e = np.log(e)
    pt = np.log(p4.pt)
    eta = p4.eta
    phi = p4.phi
    pt[pt == -inf] = 0.0
    e[e == -inf] = 0.0
    eta = np.nan_to_num(eta)
    
    e = e.reshape(len(e), -1, 1)
    pt = pt.reshape(len(pt), -1, 1)
    eta = eta.reshape(len(eta), -1, 1)
    phi = phi.reshape(len(phi), -1, 1)
    
    X = np.concatenate((pt, eta, phi), -1)
    X = X.reshape(len(X), -1)
        
    return X, y