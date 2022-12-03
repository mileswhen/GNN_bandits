import gpytorch
from typing import List, Callable, Tuple


class GNTK(gpytorch.kernels.Kernel):
    '''Empirical GNTK for gpytorch'''
    def __init__(self, NTK: Callable, x_shape: Tuple[int,int]):
        super(GNTK, self).__init__()
        is_stationary = False
        self.NTK = NTK
        self.x_shape = x_shape
    
    def forward(
            self, h1: torch.Tensor, h2: torch.Tensor, **params):
        '''
        M:=batchsize, N:=num_nodes, D:=dimension of node features
        h1: batch tensor with shape (M, N*D) 
        h2: batch tensor with shape (M, N*D)
        '''
        h1 = h1.reshape(-1, *self.x_shape)
        h2 = h2.reshape(-1, *self.x_shape)
        return self.NTK(h1, h2)


class GP_reward(gpytorch.models.ExactGP):
    def __init__(self, 
            train_x, train_y, likelihood, NTK: Callable,
            x_shape: Tuple[int,int]
        ):
        '''
        train_x: batched node feature matrix, shape (M, N, D)
        train_y: labels, shape (M)
        '''
        super(GP_reward, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = GNTK(NTK, x_shape)

    def forward(self, x):
        # reshape x for mean_module (expects nxd)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar) 