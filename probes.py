"""
probes.py

Defines the various Machine Learning probes used to classify truthful vs. deceptive 
statements based on the extracted feature vectors.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import CONFIG
from utils import print_info, _ensure_dir, _atomic_save_torch, _save_scaler_object, checkpoint_basename, rotate_checkpoints_dir

class PyTorchLogisticProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.scaler = StandardScaler()
        self.loss_history = []

    def fit(self, X, y, epochs=None, batch_size=None, lr=None, weight_decay=0.0, checkpoint_dir=None, checkpoint_prefix="logistic", model_name=None, dataset_name=None, resume_from=None, save_every=1):
        epochs = epochs if epochs is not None else CONFIG.EPOCHS
        batch_size = batch_size if batch_size is not None else CONFIG.BATCH_SIZE
        lr = lr if lr is not None else CONFIG.LEARNING_RATE
        if len(X) == 0: return
        self.loss_history = []

        checkpoint_dir = checkpoint_dir or CONFIG.CHECKPOINT_DIR
        _ensure_dir(checkpoint_dir)
        Xs = self.scaler.fit_transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=CONFIG.DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=CONFIG.DEVICE).unsqueeze(1)
        
        self.to(CONFIG.DEVICE)
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCEWithLogitsLoss()

        if resume_from:
            try:
                ck = torch.load(resume_from, map_location="cpu")
                if "model_state" in ck: self.load_state_dict(ck["model_state"])
            except Exception: pass

        idxs = np.arange(Xt.shape[0])
        for ep in range(epochs):
            np.random.shuffle(idxs)
            ep_loss = 0.0
            batch_count = 0
            p = tqdm(range(0, Xt.shape[0], batch_size), desc=f"{checkpoint_prefix} ep{ep+1}/{epochs}", leave=False)
            for start in p:
                b_idx = torch.tensor(idxs[start:start+batch_size], dtype=torch.long, device=CONFIG.DEVICE)
                opt.zero_grad()
                loss = crit(self.linear(Xt[b_idx]), yt[b_idx])
                loss.backward(); opt.step()
                ep_loss += float(loss.item())
                batch_count += 1

            if batch_count > 0:
                self.loss_history.append(ep_loss / batch_count)

            if save_every and ((ep + 1) % save_every == 0 or ep == epochs - 1):
                ck_name = checkpoint_basename(checkpoint_prefix, model_name=model_name, dataset_name=dataset_name, epoch=ep+1)
                ck_path = os.path.join(checkpoint_dir, ck_name + ".pth")
                _atomic_save_torch({"epoch": ep, "model_state": self.state_dict()}, ck_path)

    def predict_score(self, X):
        if X is None or len(X) == 0: return np.array([])
        Xs = self.scaler.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=CONFIG.DEVICE)
        self.eval()
        with torch.no_grad(): 
            logits = self.linear(Xt).cpu().numpy().flatten()
            
        # CLIPPED to avoid RuntimeWarning: overflow encountered in exp
        logits = np.clip(logits, -50, 50) 
        return 1.0 / (1.0 + np.exp(-logits))

class MassMeanProbe:
    def __init__(self):
        self.direction = None
        self.mean_vec = None
        
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        lies, honest = X[y == 1], X[y == 0]
        if len(lies) == 0 or len(honest) == 0: return
        mean_lie, mean_honest = np.mean(lies, axis=0), np.mean(honest, axis=0)
        self.direction = (mean_lie - mean_honest).astype(np.float32)
        self.mean_vec = ((mean_lie + mean_honest) / 2.0).astype(np.float32)
        
    def predict_score(self, X):
        X = np.asarray(X, dtype=np.float32)
        # Calculate projection along mass mean direction and clip to prevent overflow
        logits = np.clip((X - self.mean_vec) @ self.direction, -50, 50)
        return 1.0 / (1.0 + np.exp(-logits))

class INLPProbe:
    """Iterative Null-space Projection. Learns a subspace where the dataset classes become indistinguishable."""
    def __init__(self, max_iters: int = 20, clf_C: float = 1.0, tol_auc: float = 0.52):
        self.max_iters = max_iters
        self.clf_C = clf_C
        self.tol_auc = tol_auc
        self.projection = None

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        n, d = X.shape
        P = np.eye(d, dtype=np.float64)
        directions = []
        
        print_info(f"[INLP] Starting INLP fit for max {self.max_iters} iterations on {n}x{d} data.")
        pbar = tqdm(range(self.max_iters), desc="INLP Iterations", leave=False)
        for it in pbar:
            Xp = X @ P
            clf = LogisticRegression(C=self.clf_C, solver="lbfgs", max_iter=1000).fit(Xp, y)
            probs = clf.predict_proba(Xp)[:,1]
            auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
            pbar.set_postfix({"AUC": f"{auc:.4f}"})
            
            if auc <= self.tol_auc: break
            
            directions.append(clf.coef_.reshape(-1))
            u, s, vt = np.linalg.svd(np.vstack(directions), full_matrices=False)
            rank = np.sum(s > 1e-12)
            
            if rank >= d: break
            
            N = vt[rank:].T
            P = np.eye(d) - (N @ N.T)
            
        self.projection = P
        if len(directions) > 0:
            self.final_clf = LogisticRegression(C=self.clf_C, solver="lbfgs", max_iter=1000).fit(X @ P, y)
        else: self.final_clf = None

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.final_clf is None: return np.zeros(X.shape[0])
        return self.final_clf.predict_proba(X @ self.projection)[:,1]

class TruncatedPolynomialProbe(nn.Module):
    def __init__(self, input_dim, rank=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.U = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        
    def forward(self, x):
        lin = self.linear(x)
        ux = torch.matmul(x, self.U.t()); vx = torch.matmul(x, self.V.t())
        quad = torch.sum(ux * vx, dim=1, keepdim=True)
        return lin + quad

class TruncatedPolynomialProbeWrapper:
    def __init__(self, input_dim, rank=8):
        self.model = TruncatedPolynomialProbe(input_dim, rank=rank)
        self.scaler = StandardScaler()
        self.loss_history = []
    
    def fit(self, X, y, epochs=None, batch_size=None, lr=None, weight_decay=0.0, clip_grad_norm=None, checkpoint_dir=None, checkpoint_prefix="tpc", model_name=None, dataset_name=None, resume_from=None, save_every=1):
        epochs = epochs if epochs is not None else CONFIG.EPOCHS
        batch_size = batch_size if batch_size is not None else CONFIG.BATCH_SIZE
        lr = lr if lr is not None else CONFIG.LEARNING_RATE
        if len(X) == 0: return
        self.loss_history = []
        
        checkpoint_dir = checkpoint_dir or CONFIG.CHECKPOINT_DIR
        _ensure_dir(checkpoint_dir)
        
        Xs = self.scaler.fit_transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=CONFIG.DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=CONFIG.DEVICE).unsqueeze(1)
        
        self.model.to(CONFIG.DEVICE)
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCEWithLogitsLoss()

        idxs = np.arange(Xt.shape[0])
        for ep in range(epochs):
            np.random.shuffle(idxs)
            ep_loss = 0.0
            batch_count = 0
            p = tqdm(range(0, Xt.shape[0], batch_size), desc=f"TPC ep{ep+1}/{epochs}", leave=False)
            for start in p:
                b_idx = torch.tensor(idxs[start:start+batch_size], dtype=torch.long, device=CONFIG.DEVICE)
                opt.zero_grad()
                loss = crit(self.model(Xt[b_idx]), yt[b_idx])
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                opt.step()
                ep_loss += float(loss.item())
                batch_count += 1

            if batch_count > 0:
                self.loss_history.append(ep_loss / batch_count)

            if save_every and ((ep + 1) % save_every == 0 or ep == epochs - 1):
                ck_name = checkpoint_basename(checkpoint_prefix, model_name=model_name, dataset_name=dataset_name, epoch=ep+1)
                _atomic_save_torch({"epoch": ep, "model_state": self.model.state_dict()}, os.path.join(checkpoint_dir, ck_name + ".pth"))

    def predict_score(self, X):
        if X is None or len(X) == 0: return np.array([])
        Xs = self.scaler.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=CONFIG.DEVICE)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xt).cpu().numpy().flatten()
        logits = np.clip(logits, -50, 50) 
        return 1.0 / (1.0 + np.exp(-logits))

class TruthUniversal2DProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.project = nn.Linear(input_dim, 2, bias=False)
        self.classifier = nn.Linear(2,1)
    def forward(self, x):
        z = self.project(x)
        return self.classifier(z)

class TruthUniversal2DProbeWrapper:
    def __init__(self, input_dim):
        self.model = TruthUniversal2DProbe(input_dim)
        self.scaler = StandardScaler()
        self.loss_history = []
    
    def fit(self, X, y, epochs=None, batch_size=None, lr=None, weight_decay=0.0, ortho_penalty_alpha=0.0, checkpoint_dir=None, checkpoint_prefix="truth2d", model_name=None, dataset_name=None, resume_from=None, save_every=1):
        epochs = epochs if epochs is not None else CONFIG.EPOCHS
        batch_size = batch_size if batch_size is not None else CONFIG.BATCH_SIZE
        lr = lr if lr is not None else CONFIG.LEARNING_RATE
        if len(X) == 0: return
        self.loss_history = []
        
        checkpoint_dir = checkpoint_dir or CONFIG.CHECKPOINT_DIR
        _ensure_dir(checkpoint_dir)

        Xs = self.scaler.fit_transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=CONFIG.DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=CONFIG.DEVICE).unsqueeze(1)
        
        self.model.to(CONFIG.DEVICE)
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCEWithLogitsLoss()

        idxs = np.arange(Xt.shape[0])
        for ep in range(epochs):
            np.random.shuffle(idxs)
            ep_loss = 0.0
            batch_count = 0
            p = tqdm(range(0, Xt.shape[0], batch_size), desc=f"Truth2D ep{ep+1}/{epochs}", leave=False)
            for start in p:
                b_idx = torch.tensor(idxs[start:start+batch_size], dtype=torch.long, device=CONFIG.DEVICE)
                opt.zero_grad()
                loss = crit(self.model(Xt[b_idx]), yt[b_idx])
                
                # Enforce orthogonality in the 2D projection
                if ortho_penalty_alpha and ortho_penalty_alpha > 0.0:
                    W = self.model.project.weight
                    ortho = torch.norm(W @ W.t() - torch.eye(W.size(0), device=W.device))
                    loss = loss + ortho_penalty_alpha * ortho
                    
                loss.backward(); opt.step()
                ep_loss += float(loss.item())
                batch_count += 1

            if batch_count > 0:
                self.loss_history.append(ep_loss / batch_count)

            if save_every and ((ep + 1) % save_every == 0 or ep == epochs - 1):
                ck_name = checkpoint_basename(checkpoint_prefix, model_name=model_name, dataset_name=dataset_name, epoch=ep+1)
                _atomic_save_torch({"epoch": ep, "model_state": self.model.state_dict()}, os.path.join(checkpoint_dir, ck_name + ".pth"))

    def predict_score(self, X):
        if X is None or len(X) == 0: return np.array([])
        Xs = self.scaler.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=CONFIG.DEVICE)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xt).cpu().numpy().flatten()
        logits = np.clip(logits, -50, 50)
        return 1.0 / (1.0 + np.exp(-logits))