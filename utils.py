import torch
from torch.utils.data import Dataset
from typing import List


class ENGraph:
    '''Container for erdos-renyi graphs'''
    def __init__(self, p, n):
        assert 0 < p < 1
        self.p = p
        self.N = n
        
        # init random graph
        edges = erdos_renyi_graph(n, p, directed=False)
        x = torch.normal(0, 1, size=(n, 5))
        
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
        '''
        PARAMS
        graph_domain: list of ENgraph objects from a specific graph domain
        '''
        # TODO: assert belongs to same domain
        self.domain = graph_domain
        self.train = []
        
    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.train[idx]
    
    def train_data(self) -> torch.Tensor:
        '''returns a batch of graph node feature matrices'''
        data = torch.concat(
            [g.hbar.unsqueeze(0) for g in self.train]
        )
        return data
    
    def train_targets(self) -> torch.Tensor:
        targets = torch.concat(
            [g.y.unsqueeze(0) for g in self.train]
        ).unsqueeze(1)
        return targets