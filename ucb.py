import torch
from gnn import GNN
from typing import Callable


class GNNUCB:
    def __init__(
            self,
            model: GNN,
            lambd: float = .0025, 
            beta: float = .1,
            m: int = 512,
            T: int = 100
        ):
        '''Util class for computing + updating the acquisition func.
        ---
        PARAMS
        mode: GNN model w/ a lazy jacobian method
        lambd: regularization param
        beta: uncertainty scaling factor
        m: hidden layer width
        '''
        # computes jacobian at weight init
        self.jac0: Callable = model.jac0
        
        # constants
        self.lambd = lambd
        self.beta = beta
        self.m = m
        
        # init exploration bonus matrix
        num_params = sum([p.numel() for p in model.parameters()])
        self.K = lambd * torch.ones(num_params)
        self.K_inv = torch.tensor(1.) / self.K
        
    def update_K(self, hbar: torch.Tensor) -> None:
        '''Updates the exploration bonus matrix.
        ---
        PARAMS
        hbar: aggr node feature matrix of graph, shape (N, D)
        '''
        g = self.jac0(hbar)
        self.K += torch.diag(torch.outer(g, g)/self.m)
        self.K_inv = torch.tensor(1.) / self.K

    def __call__(self, mu: torch.Tensor, hbar: torch.Tensor) -> torch.Tensor:
        '''GNN-UCB acquisition function.
        ---
        PARAMS
        hbar: precomputed aggr node feature matrix of graph Gi
        model: GNN model w/ batchNTK() method
        ---
        returns optimistic outcome of choosing graph Gi
        '''
        g = self.jac0(hbar)
        sigma = self.beta * torch.sqrt(torch.dot(g,  self.K_inv * g)/self.m)
        return mu + sigma
