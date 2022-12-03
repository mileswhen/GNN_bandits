# Graph Neural Network Bandits

This repo contains experiments that attempt to reproduce and understand the results from [Graph Neural Network Bandits](https://arxiv.org/pdf/2207.06456.pdf).

## Notebook experiments
See the `GNN_bandits` notebook for the following content.

1. **GNN model definition** — first experiment with a vanilla GNN in `pytorch_geometric` to compute graph representations.
2. **Synthetic graph domains** — sample random graphs to construct graph domains as described in paper.
3. **Compute empirical G-NTK** — use jacobian contraction to compute the empirical Graph-NTK matrix for a subset of a graph domain ${\mathcal{G}_{p,N}}$.
4. **Gaussian process reward** —  Use a GP to learn a smooth reward function $f: \mathcal{G} \rightarrow \mathbb{R}$ and update the synthetic dataset $\{\mathcal{G}_i, f(\mathcal{G}_i)\}$.
5. **GNN-UCB** — contextual bandits using the Graph-NTK and neural upper confidence bound (NeuralUCB) as the acquisition function.

