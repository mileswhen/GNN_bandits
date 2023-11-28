# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch_geometric
import torch_geometric
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.data import Data
# misc
from typing import List, Callable, Tuple
from functorch import make_functional_with_buffers, jacrev



class GNN(torch.nn.Module):
    def __init__(self, width: int = 2048):
        '''GNN model w/ 2 ReLU layers.
        NOTE: no conv layers because input features pre-aggregated w/ GINConv.
        ---
        width: width of hidden layers (should be large in principle)
        '''
        super().__init__()
        self.f1 = nn.Linear(5, width, bias=False)
        self.f2 = nn.Linear(width, 1, bias=False)
        
        # rescaling factor (using c=1) for NTK
        self.cm = torch.sqrt(torch.tensor(1/width))
        
        # init normal
        self.width = width
        for p in self.parameters():
            nn.init.normal_(p, std=1)
        
        # functional for NTK computation
        fnet, self.params, self.bufs = make_functional_with_buffers(self)
        self.jacobian = jacrev(fnet)
    
    def reinit(self, std=1) -> None:
        # init normal
        for p in self.parameters():
            nn.init.normal_(p, std=std)
        
    def forward(self, hbar: torch.Tensor) -> torch.Tensor:
        '''Forward pass.
        ---
        STEPS
        1. Precomputed node features hbar
        2. Hidden layers w/ NTK rescaling
        3. Global mean pooling
        ---
        returns graph representation
        '''
        x = self.cm * F.relu(self.f1(hbar))
        x = self.f2(x)
        gx = global_mean_pool(x, batch=None)
        
        return gx

    def NTK(self, G1: torch.Tensor, G2: torch.Tensor) -> torch.Tensor:
        '''Computes empirical finite-width NTK between two graphs w/ jacobian
        ---
        G1: aggregate node features hbar of shape (N, D)
        G2: aggregate node features hbar of shape (N, D)
        ---
        returns scalar kernel value, k(G1, G2) = inner_prod(g(G1), g(G2))
        '''
        with torch.no_grad():
            # Compute J(G1)
            J1 = self.jacobian(self.params, self.bufs, G1)
            J1 = [j.flatten(2) for j in J1]

            # Compute J(G2)
            J2 = self.jacobian(self.params, self.bufs, G2)
            J2 = [j.flatten(2) for j in J2]

            # Compute J(G1) @ J(G2).T
            # d := dimension of output f_gnn(G)
            # K is a (d,d) matrix-valued kernel if d != 1
            k = torch.stack(
                [Ja @ Jb.mT for Ja, Jb in zip(J1, J2)], dim=0
            ).mean(dim=0)
        
        # return a scalar
        return k.squeeze()
    
    def batchNTK(self, 
            batch1: torch.Tensor, 
            batch2: torch.Tensor,
            diag: bool = False
        ) -> torch.Tensor:
        '''Computes empirical NTK matrix between a batch of N graphs. 
        NOTE: unvectorized
        ---
        batch1: (B, N, D) batched node feature matrix
        batch2: (B, N, D) batched node feature matrix
        diag: whether to only compute the diagonal elements
            (returns a flat tensor)
        ---
        returns (N, N) kernel matrix
        '''
        N = batch1.size(0)
        
        if diag:
            K = torch.zeros(N)
            for i in range(N):
                K[i] = self.NTK(batch1[i], batch2[i])
        else:
            M = batch2.size(0)
            K = torch.zeros(size=(N,M))
            for i in range(N):
                for j in range(M):
                    K[i][j] = self.NTK(batch1[i], batch2[j])
                
        return K
    
    def jac0(self, hbar: torch.Tensor) -> torch.Tensor:
        '''Computes a flattened jacobian w/ params from init.
        '''
        with torch.no_grad():
            G = self.jacobian(self.params, self.bufs, hbar)
            G = torch.hstack([g.flatten(2).squeeze() for g in G])
        
        return G
