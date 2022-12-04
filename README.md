# Graph Neural Network Bandits

This repo contains experiments that attempt to reproduce and understand the results from [Graph Neural Network Bandits](https://arxiv.org/pdf/2207.06456.pdf).

## Repo structure

* `gnn.py` defines GNN model and methods for computing the empirical G-NTK
* `gp.py` defines classes for computing gaussian process posterior using G-NTK
* `bandit.py` perform the GNN-UCB algorithm as described in the paper
* `utils.py` helper data classes and misc.

## Notebook experiments
See the `GNN_bandits.ipynb` notebook for the following content:

1. **GNN model definition** — first experiment with a vanilla GNN in `pytorch_geometric` to compute graph representations.
2. **Synthetic graph domains** — sample random graphs to construct graph domains as described in paper.
3. **Compute empirical G-NTK** — use jacobian contraction to compute the empirical Graph-NTK matrix for a subset of a graph domain ${\mathcal{G}_{p,N}}$.
4. **Gaussian process reward** —  Use a GP to learn a smooth reward function $f: \mathcal{G} \rightarrow \mathbb{R}$ and update the synthetic dataset $\{\mathcal{G}_i, f(\mathcal{G}_i)\}$.
5. **GNN-UCB** — contextual bandits using the Graph-NTK and neural upper confidence bound (NeuralUCB) as the acquisition function.

## GNN-UCB

Contextual bandits on graphs, using a GNN to approximate some unknown reward function.

### Background
The authors propose to use the Graph-NTK to balance exploitation, i.e. training of GNN, and exploration of arms, i.e. acquiring new samples $(\mathcal{G}_i, y_i)$. In the lazy (overparameterized) regime, neural networks are essentially gaussian processes, which allows one to quantify the upper uncertainty bound (UCB) of a GNN with the NTK. For simplicity and understanding we will only implement a variant of NeuralUCB, GNN-UCB.

* $\text{UCB}(\mathcal{G}; \mu, \sigma) = \mu(\mathcal{G}) + \beta_t\sigma(\mathcal{G})$ is the acquisition function
* $\mu \triangleq f_\text{GNN}$ is straightforward, now how to quantify $\sigma$?
* recall GP posteriors admit a closed form for $k(x^*, x^*) = k^\top_{X,x^*}(K_{XX}+\sigma^2I)^{-1}k_{X,x^*}$

Brief overview of contextual bandits and Linear-UCB for understanding. Please refer to *section 4.* on regret analysis in the [NeuralUCB paper](https://arxiv.org/pdf/1911.04462.pdf), the original [LinUCB paper](https://arxiv.org/pdf/1003.0146.pdf), or a [lecture on linear bandits](https://sites.cs.ucsb.edu/~yuxiangw/classes/RLCourse-2021Spring/Lectures/scribe_linear_bandit.pdf) for in-depth background.

* $y_t = f(\mathbf{x}_t) + \epsilon_t$. One observes the reward $y_t$ from a function with some noise. In this case just linear regression: $f(\mathbf{x}_t) = \mathbf{w}^\top \mathbf{x}$. Here $\mathbf{x}$ is called the "context".
* Assuming a ridge loss, we have the closed form estimate for the model: $\hat{\mathbf{w}}_t = (X_tX_t^\top + \lambda I)^{-1}X_t^\top y$. Notice that $K = X_tX_t^\top$ is the kernel matrix.
* The cumulative regret is $R_T = \sum^T_{t=1} |f^*(x_t) - f(x_t)|$, this should be bounded for the algorithm to converge.
* Omitting the proof, the error of predictions are bounded by the (scaled) standard deviation of the expected reward: $|f^* - f| \leq \beta_t \sqrt{\mathbf{x}_t^\top (\lambda I + K)^{-1} \mathbf{x}_t}$
* This uncertainty bound can then be used in the vanilla UCB acquisition function $\text{UCB}(x_t) = f(x_t) + \beta_t \sqrt{\mathbf{x}_t^\top (\lambda I + K)^{-1} \mathbf{x}_t}$

For GNN-UCB:
* Can understand $\mathbf{g}(\mathcal{G})$ as a basis function $\phi(\mathcal{G})$
* Here $K = GG^\top /{m}$, is the gram matrix normalized by the layer width $m$
* Analogous to LinUCB, the uncertainty can be understood as enforcing a bound on the cumulative regret $R_T$.

$$\text{GNN-UCB}(\mathcal{G_t}) = f_\text{GNN}(\mathcal{G_t};\theta_{t-1}) + \beta_t \sqrt{\mathbf{g}(\mathcal{G_t};\theta_{t-1})^\top(\lambda I+K)^{-1}\mathbf{g}(\mathcal{G_t};\theta_{t-1})/m}$$

### Practical implementation

Using modified instructions specified in *Appendix D.2, D.3*.

* define rounds $T = 100$
* invert $\hat{K} =(\lambda I + GG^\top)$ by first approximating it with a diagonal matrix $diag(\hat{K})$ 
* We store $\hat{K}$ as a flat tensor. Since $\hat{K}$ is a diag, we approximate $g^\top \hat{K} g$ with $g^\top diag(\hat{K}) \odot g$
* "explore" for $T_0$ steps with random samples of $G_i$ to pre-train
* for the subsequent $T_1$ steps train but re-init model parameters at each step
* for any remaining steps train only every 20 steps
* use $\beta = 0.0002$ and $\lambda=0.0025$ found from gridsearch (section D.3)

GNN training:
* use $m=2048$, $L=2$ with gaussian weight init for the GNN
* use `MSELoss`: $\mathcal{L}(\theta) = \frac{1}{t}\sum_{i<t} (f_\text{GNN}(\mathcal{G}_i;\theta_t) - y_i)^2$, no L2-reg since we don't use weight decay: $m\lambda \|\theta - \theta^{(0)}\|_2^2$
* use `Adam` optimizer (`lr=0.001`) instead of `SGD` (not theoretically correct but practical)
* train network for gradient steps $J_t = \min J$ such that $\mathcal{L}(\theta_{t-1})\leq 10^{-4}$ or the relative change in loss is less than $10^{-3}$
