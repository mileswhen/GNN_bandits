# Graph Neural Network Bandits
`December 3, 2022`

This repo contains experiments that attempt to reproduce and understand the results from [Graph Neural Network Bandits](https://arxiv.org/pdf/2207.06456.pdf). See the notebook [here](https://mileswhen.com/posts/gnn_bandits/).

Disclaimer: this is for personal experimentation and understanding only.

## Repo structure

* `./`
  * `gnn.py` defines GNN model and methods for computing the empirical G-NTK. Note, assumes node features have been pre-aggregated (why there's no conv layer).
  * `ucb.py` defines acquisition function
  * `gp.py` defines classes for computing gaussian process using G-NTK
  * `bandit.py` perform the GNN-UCB algorithm as described in the paper
  * `utils.py` helper data classes and misc.
* `notebooks/`
  * `GNN_bandits.ipynb` experimental notebook with visualizations
* `configs/`
  * `bandit.toml` config for `bandit.py`
* `data/`
  * `graphs.pkl` precomputed synthetic dataset of graph domains, node features are pre-aggregated.

## Installation and basic usage

```sh
conda create -n env python=3.8
pip install -r requirements
```

Then run

```sh
python3 bandit.py
```

## Notebook experiments
See the `GNN_bandits.ipynb` notebook for the following content:
1. **GNN model definition** — first experiment with a vanilla GNN in `pytorch_geometric` to compute graph representations.
2. **Synthetic graph domains** — sample random graphs to construct graph domains as described in paper.
3. **Compute empirical G-NTK** — use jacobian contraction to compute the empirical Graph-NTK matrix for a subset of a graph domain ${\mathcal{G}\_{p,N}}$.
4. **Gaussian process reward** —  Use a GP to learn a smooth reward function $f: \mathcal{G} \rightarrow \mathbb{R}$ and update the synthetic dataset $\{G\_i, f(G\_i)\}$.
5. **GNN-UCB** — kernelized bandits using the Graph-NTK and neural upper confidence bound (NeuralUCB) as the acquisition function.

## Background

GNN-UCB is an acquisition function for bayesian optimization with GNNs.

The authors propose to use the neural tangent kernel (Graph-NTK) to balance exploitation, i.e. training of GNN, and exploration of graphs $(G\_i, y\_i)$. In the lazy (overparameterized) regime, neural networks are essentially gaussian processes, which allows one to quantify the upper uncertainty bound (UCB) of a GNN with the NTK. For simplicity and understanding we will only implement a variant of NeuralUCB, GNN-UCB.

* $\text{UCB}(G; \mu, \sigma) = \mu(G) + \beta\_t\sigma(G)$ is the acquisition function
* $\mu \triangleq f\_\text{GNN}$ is straightforward, now how to quantify $\sigma$?
* recall GP posteriors admit a closed form for $k' = k(x^\*, x^\*) - k^\top\_{X,x^\*}(K\_{XX}+\sigma^2I)^{-1}k\_{X,x^\*}$

Brief overview of kernelized bandits and Linear-UCB for understanding. Please refer to *section 4.* on regret analysis in the [NeuralUCB paper](https://arxiv.org/pdf/1911.04462.pdf), the original [LinUCB paper](https://arxiv.org/pdf/1003.0146.pdf), or a [lecture on linear bandits](https://sites.cs.ucsb.edu/~yuxiangw/classes/RLCourse-2021Spring/Lectures/scribe\_linear\_bandit.pdf) for in-depth background.

* $y\_t = f(\mathbf{x}\_t) + \epsilon\_t$. One observes the reward $y\_t$ from a function with some noise. In this case just linear regression: $f(\mathbf{x}\_t) = \mathbf{w}^\top \mathbf{x}$.
* Assuming a ridge loss, we have the closed form estimate for the model: $\hat{\mathbf{w}}\_t = (X\_tX\_t^\top + \lambda I)^{-1}X\_t^\top y$. Notice that $K = X\_tX\_t^\top$ is the linear kernel matrix.
* The cumulative regret is $R\_T = \Sigma^T\_{t=1} f^\*(\mathbf{x}) - f(\mathbf{x}\_t)$, this should be bounded for the algorithm to converge.
* Omitting the proof, the error of predictions are bounded by the (scaled) standard deviation of the expected reward: $|f^\* - f| \leq \beta\_t \sqrt{\mathbf{x}\_t^\top (\lambda I + K)^{-1} \mathbf{x}\_t}$
* This uncertainty bound can then be used in the vanilla UCB acquisition function $\text{UCB}(\mathbf{x}\_t) = f(x\_t) + \beta\_t \sqrt{\mathbf{\mathbf{x}}\_t^\top (\lambda I + K)^{-1} \mathbf{x}\_t}$

For GNN-UCB:
* Can understand jacobian $\mathbf{g}(G)$ as a feature map $\phi(G)$. 
* Here $K = \mathbf{gg}^\top /{m}$, is the gram matrix normalized by the layer width $m$

$$\text{GNN-UCB}(\mathcal{G\_t}) = f\_\text{GNN}(G\_t;\theta\_{t-1}) + \beta\_t \sqrt{\mathbf{g}(G\_t;\theta\_{t-1})^\top(\lambda I+K)^{-1}\mathbf{g}(G\_t;\theta\_{t-1})/m}$$

### Practical implementation

Using modified instructions specified in *Appendix D.2, D.3*:

* rounds $T = 175$
* invert $\hat{K} =(\lambda I + \mathbf{gg}^\top/m)$ by approximating it with a diagonal matrix $\text{diag}(\hat{K})$ 
* We store $\hat{K}$ as a flat tensor. Since $\hat{K}$ is a diag, we approximate $\mathbf{g}^\top \hat{K} \mathbf{g}$ with $\mathbf{g}^\top \text{diag}(\hat{K}) \odot \mathbf{g}$
* "explore" for $T_0$ steps with random samples of $G_i$ to pre-train
* for the subsequent $T_1$ steps train but re-init model parameters at each step
* use $\beta = 0.3$ and $\lambda=0.0025$

GNN training:
* use $m=2048$, $L=2$ with gaussian weight init for the GNN
* use `MSELoss`, no L2-reg since no weight decay: $m\lambda \|\theta - \theta^{(0)}\|\_2^2$
* use `Adam` optimizer (`lr=0.001`) instead of `SGD` (not theoretically correct but practical)
* train network for gradient steps $J\_t = \min J$ such that $\mathcal{L}(\theta\_{t-1})\leq 10^{-4}$ or the relative change in loss is less than $10^{-3}$

