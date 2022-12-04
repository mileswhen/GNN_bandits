import tomli
import pickle
import torch
import collections
import argparse
from utils import GraphData, Dotdict, ENGraph
from gnn import GNN
from ucb import GNNUCB


def explore(data: GraphData, model: GNN, ucb: GNNUCB, t: int) -> torch.Tensor:
    '''
    data: graph domain object 
    model: GNN model object
    ucb: acquisition object
    t: timestep
    ---
    returns reward of selected graph and adds it to train data
    '''
    # evaluation
    model.eval()

    # get max_G UCB(G)
    best = (-torch.tensor(float('inf')), 0)
    for g_id, graph in enumerate(data.domain):
        mu = model(graph.hbar).squeeze()
        u = ucb(mu, graph.hbar)
        if u > best[0]:
            best = (u, g_id)

    # add to sample
    best_graph = data.domain[best[1]]
    data.train.add(best_graph)

    # update uncertainty bound with new graph
    ucb.update_K(best_graph.hbar)

    return best_graph.y


def train(
        optimizer,
        criterion,
        loss_0: torch.Tensor,
        data: GraphData,
        model: GNN,
        log: bool = False
    ) -> dict:
    '''Training until stopping criterion is reached'''
    loss = loss_0
    last_loss = loss_0
    model.reinit()
    model.train()
    while loss.item() >= loss_0.item():
        # get train data and labels
        train_data = data.train_data()
        train_targets = data.train_targets()

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute loss
        outputs = model(train_data)
        loss = criterion(outputs, train_targets)

        # check a simplified stopping criterion
        if (loss.item() - last_loss.item()) < 0.02:
            break

        # step
        loss.backward()
        optimizer.step()
        if log: print(".",end="")

    return model.state_dict()


def main(conf: Dotdict):
    # init GNN
    model = GNN(width=conf.width)
    model.eval()

    # init training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    criterion = torch.nn.MSELoss()
    loss_0 = torch.tensor(conf.l0)

    # load data
    graphs = pickle.load(open(conf.dataset,"rb"))
    data = GraphData(graphs[(0.25, 10)])
    y_max = torch.max(torch.tensor([g.y for g in data.domain]))

    # init acquisition function
    ucb = GNNUCB(model)

    # store regrets at each timestep
    regrets = []

    # randomly explore domain to pretrain
    print("random exploration (40 steps)")
    for t in range(conf.warmup):
        sample_idx = torch.randint(len(data.domain)-1, size=(1,1))
        graph = data.domain[sample_idx]
        data.train.add(graph)
        regrets.append(y_max - graph.y)    

    # "exploration and exploitation"
    print("explore and exploit (80 steps)",end="")
    for t in range(conf.warmup, conf.max_T):
        # explore (pick graphb from domain using prev model)
        y = explore(data, model, ucb, t)
        regrets.append(y_max - y)

        # exploit (train GNN)
        if t%5==0:
            print(f"\nt={t}, regret={y_max-y:.3f} ",end="")
            params_t = train(optimizer, criterion, loss_0, data, model, True)
        else:
            params_t = train(optimizer, criterion, loss_0, data, model, False)

        if t==conf.max_T:
            torch.save(params_t, conf.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # args
    parser.add_argument(
        "--config",
        default="configs/bandit.toml",
        help="bandit opt config"
    )
    args = parser.parse_args()

    config = Dotdict(tomli.load(open(args.config, "rb")))
    main(config)

    
