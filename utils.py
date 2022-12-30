
import os
from pathlib import Path
import random
import h5py
import numpy as np
from numpy import inf
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
import uproot_methods
    
def jet_from_ptetaphi(X, scaled=False):
    from sklearn.preprocessing import RobustScaler
    
    def load_attn_train(n_train=None, input_dim=160, scale=False):
    
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
    
    if scaled:
        input_dim = X.shape[1] // 3 * 4
        _, scaler = load_attn_train(n_train=10000, input_dim=input_dim, scale=True)
        X = scaler.inverse_transform(X)
        
    X = np.reshape(X, (len(X), -1, 3))
    log_pt = X[:,:,0]
    eta = X[:,:,1]
    phi = X[:,:,2]
    pt = np.exp(log_pt)
    m = np.zeros_like(pt)
    
    p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, m)
    
    e = p4.energy
    px = p4.x
    py = p4.y
    pz = p4.z
    
    e = e.reshape(len(e), -1, 1)
    px = px.reshape(len(px), -1, 1)
    py = py.reshape(len(py), -1, 1)
    pz = pz.reshape(len(pz), -1, 1)
    
    X = np.concatenate((e, px, py, pz), -1)
    X = X.reshape(len(X), -1)    
    return X

def jet_e(jet):
    E_j = 0.0
    Px_j = 0.0
    Py_j = 0.0
    Pz_j = 0.0

    jet = np.reshape(jet, (-1, 4))

    E_j, _, _, _ = np.sum(jet, axis=0)

    return E_j

def jet_mass(jet):
    E_j=0
    Px_j=0
    Py_j=0
    Pz_j=0

    jet = np.reshape(jet, (-1, 4))
    
    E_j, Px_j, Py_j, Pz_j = np.sum(jet, axis=0)
    
    if E_j**2 > (Px_j**2 + Py_j**2 + Pz_j**2):
            m = np.sqrt(E_j**2 - (Px_j**2 + Py_j**2 + Pz_j**2))
    else:
            m = 0

    return m

def jet_pt(jet):
    Px_j=0
    Py_j=0

    jet = np.reshape(jet, (-1, 4))
    n_consti = len(jet)

    for i in range(n_consti):
            Px_j += jet[i, 1]
            Py_j += jet[i ,2]
            
    pt = np.sqrt(Px_j**2 + Py_j**2)
    return pt  

def jet_girth(jet): ##### to be modified
    jet = copy.deepcopy(jet)
    eta_j=jet["eta"] # just using pseudo-rapidity here
    phi_j=jet["phi"]
    pt_j=jet["pt"]
    m_j=jet["mass"]

    j=LorentzVector()
    j.set_pt_eta_phi_m(pt_j, eta_j, phi_j, m_j)
    rap_j =  j.Rapidity() # jet rapidity here
    constituents = jet["content"][jet["tree"][:, 0] == -1]


    g = 0
    for i in range(len(constituents)):
        v = LorentzVector(constituents[i])
        e=v.E()
        pz=v.Pz()
        pt=v.Pt()
        eta = 0.5 * (np.log(e + pz) - np.log(e - pz)) # using rapidity here
        phi=v.phi()
        delta_eta=eta-rap_j
        delta_phi=phi-phi_j
        if (delta_phi)>np.pi:
            delta_phi -= 2*np.pi
        elif (delta_phi)<-np.pi:
            delta_phi += 2*np.pi
        dr=np.sqrt(delta_eta**2 + delta_phi**2)
        g += pt * dr

    g /= pt_j
    return g

def plot_jet_image(jets, ax, cmap="Blues"):
    '''
    Inputs: [n, l]
    n: number of jets
    l: four-vectors of jet constituents
    Four-vectors: (E, Px, Py, Pz)
    Outputs: average jet images on (eta, phi) plane
    '''
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    #plt.rcParams["figure.figsize"] = (6,6)
    
    a=[]

    for i in range(len(jets)):
        constituents=jets[i].reshape(-1,4)
        jet=constituents.sum(axis=0)
        #v=LorentzVector(jet[1], jet[2], jet[3], jet[0])
        #pt_j=v.Pt()
        pt_j=np.sqrt(jet[1]**2+jet[2]**2)
        for c in constituents:
            if c[0]<1e-10:
                continue
            eta=0.5*np.log((c[0]+c[3])/(c[0]-c[3]))
            phi=np.arctan2(c[2], c[1])
            pt=np.sqrt(c[1]**2+c[2]**2)
            #v=LorentzVector(c[1], c[2], c[3], c[0])
            #a.append(np.array([v.eta(), v.phi(), v.Pt()/pt_j]))
            a.append(np.array([eta, phi, pt/pt_j]))

    a=np.vstack(a)

    ax.hist2d(a[:, 0], a[:, 1], range=[(-1.0, 1.0), (-1.0,1.0)], 
               weights=a[:, 2],
               bins=50, cmap=cmap, norm=LogNorm())
    
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
                
class LitProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch:
            print()
        super().on_train_epoch_start(trainer, pl_module)                              

    def get_metrics(self, trainer, pl_module, **kwargs):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items    
    
class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, interval, **kwargs):
        super().__init__()
        self.interval = interval

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if pl_module.global_step % self.interval == 0:
            assert self.dirpath is not None
            #current = Path(self.dirpath) / f"{pl_module.global_step // self.interval}-{pl_module.global_step}.ckpt"
            current = Path(self.dirpath) / f"e{pl_module.global_step // self.interval}.ckpt"
            prev = Path(self.dirpath) / f"{pl_module.global_step - self.interval}.ckpt"
            trainer.save_checkpoint(current)
            #prev.unlink()                 