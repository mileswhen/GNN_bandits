import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.data import Data
from typing import List


class ENGraph:
    def __init__(self, p, n):
        '''Container for erdos-renyi graphs.
        ---
        p: probability of edge
        n: num nodes
        '''
        assert 0 < p < 1
        self.p = p
        self.N = n
        
        # init random graph
        edges = erdos_renyi_graph(n, p, directed=False)
        x = torch.normal(0, 1, size=(n, 5))
        self.y = None
        
        # graph object
        graph = Data(x=x, edge_index=edges)
        self.graph = graph
        
        # precomp. aggregated node features
        self.conv = GINConv(nn.Identity())
        self.hbar = F.normalize(self.conv(x, edges))
        
    def __repr__(self):
        return f"ENGraph (p={self.p}, N={self.N})"


class GraphData(Dataset):
    def __init__(self, graph_domain: List[ENGraph]):
        '''PyTorch Dataset container.
        ---
        graph_domain: list of ENgraph objects from a specific graph domain
        '''
        # TODO: assert belongs to same domain
        self.domain = graph_domain
        self.train = set()
        
    def __len__(self):
        return len(self.train)

    def train_data(self) -> torch.Tensor:
        '''Returns a batch of graph node feature matrices.
        '''
        data = torch.concat(
            [g.hbar.unsqueeze(0) for g in self.train]
        )
        return data
    
    def train_targets(self) -> torch.Tensor:
        targets = torch.concat(
            [g.y.unsqueeze(0) for g in self.train]
        ).unsqueeze(1)
        return targets


class Dotdict(dict):
    '''Acess dictionary keys with dot notation, recursively.
    '''
    def __getattr__(self, key): 
        return self.get(key)
    
    def __setattr__(self, key, val):
        self[key] = val

    def __delattr(self, key):
        self.__delitem__(key)

    def __init__(self, dct):
        for key, val in dct.items():
            if hasattr(val, "keys"): 
                val = Dotdict(val)
            self[key] = val

