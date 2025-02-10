# Stochastic Optimisation Project 2025
by Timur Kharisov, Redouane Yagouti and William Ford.

### Problem description

In this project we solve logistic regression with L2 regularisation for the MNIST dataset. The data $x \in \mathbb{R}^{784} $ are 60,000 $28 \times 28$ greyscale images of digits from $0$-$9$, of which 12,000 were used for validation, and a further 10,000 further for testing. These are labelled as binary unit vectors $y\mathbb{R}^{10}$ with $y_i = 1$ if the image represents the digit $i$, else $y_j =0$.


The model parameters are a weights $W \in \R^{784 \times 10}$ (we use no biases), and logistic cost for a pair $(x, y)$
$$
f_{W} : \mathbb{R}^{784} \to \mathbb{R}
$$
$$
(x, y) \mapsto \log\left( \sum_{j=1}^{9} \exp(Wx)\right) - \langle y, Wx\rangle_{\mathbb{R}^{10}}
$$
Assuming the law of $(x, y) \sim \mathcal{D}$ is unknown the objective is
$$
F_\alpha(W) = \mathbb{E}_{(x, y) \sim \mathcal{D}}[f_{W}(x, y)] + \frac{\alpha}{2} \lVert W \rVert^2
$$
where $\alpha$ is a regularisation hyperparameter which we would like to choose optimally. We would like to solve
$$
\min_{W} F_\alpha(W).
$$
By the law of large numbers, we approximate the expectation with the empirical mean over the training data and solve the problem
$$
\underset{W}{\operatorname{Argmin}} \; \hat{F}_\alpha(W) := \frac{1}{48000} \sum_{i=1}^{48000} f_{W}(x_i^{\text{train}}, y_i^{\text{train}}) + \frac{\alpha}{2} \lVert W \rVert^2
$$
This amounts to a convex optimisation problem in $\mathbb{R}^{7840}$, which is further strongly $\alpha$-convex from the regularisation term.

Searching for the optimal $\alpha$, we consider an accuracy of average success of parameters $W$;
$$
L(W) = \frac{1}{12000} \sum_{i=1}^{12000} l(W, x_i^{\text{valid}}, y_i^{\text{valid}})
$$
where
$$
l(x, y) = \begin{cases} 1 & Wx \text{ and } y \text{ have the same largest coordinate}\\
0 & \text{ otherwise}. \end{cases}
$$
We then want to identify
$$
\underset{\alpha}{\operatorname{Argmax}}\left\{ L(W_\alpha) : W_\alpha = \underset{W}{\operatorname{Argmin}} \; \hat{F}_\alpha(W)\right\}.
$$

### Implementation


We Implemented X different optimisation algorithms: Stochastic (Batch) Gradient Descent (SGD), Adam, [Shampoo](https://arxiv.org/abs/1802.09568) and (Stochastic) Coordinate Descent


### Results

