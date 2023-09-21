# [Probabilistic Artificial Intelligence](https://las.inf.ethz.ch/pai-f20)


## Bacis Recap of Probability
### **Bayes' Rule** 
- computes <span style="color:blue">posterior</span>:  $P(X|Y) = \frac{P(X)P(Y|X)}{\sum_{X=x}P(X=x)P(Y|X=x)}$
- <span style="color:red">Prior</span>($P(X)$) + <span style="color:red">Likelihood</span>($P(Y|X)$) $\Longrightarrow$ <span style="color:red">Posterior</span>($P(X|Y)$)
- <span style="color:blue">Evidence</span> $P(Y) = \sum_{X=x}P(X=x)P(Y|X=x)$

### Conditional Independence: $X\perp Y|Z$

### Two Rule for Joint Distribution
- Sum rule(Marginalization): $P(X_i)=\sum\limits_{\substack{x_1,...,x_{i-1},x_{i+1},...,x_n}} P(x_1,...,x_{i-1},X_i,x_{i+1},...,x_n)$
- Product rule(Chain rule): $P(X_1,...,X_n)=P(X_1)P(X_2|X_1)...P(X_n|X_1,...,X_{n-1})$

### Multivariate Gaussian: $\mathbf{X}\ \sim\ \mathcal{N}(\boldsymbol\mu,\, \boldsymbol\Sigma)$
- Multiples of Gaussians are Gaussian
- Sums of Gaussians are Gaussian

## Bayesian Learning (Get Posterior of $\theta$ and do prediction)

### **Bayesian Linear Regression(BLR)**: $y = \mathbf{w}^T \mathbf{x} + \mathbf{\epsilon}$, $ \mathbf{\epsilon} \ \sim\ \mathcal{N}(0, \sigma_n^2 \boldsymbol I)$

#### Recall: Ridge Regression
- Regularized optimization problem: $\min _{\mathbf{w}} \sum_{i=1}^{n}\left(y_{i}-\mathbf{w}^{T} \mathbf{x}_{i}\right)^{2}+\lambda\|\mathbf{w}\| 2$
- Analytical solution: $\hat{\mathbf{w}}=\left(\mathbf{X}^{T} \mathbf{X}+\lambda \mathbf{I}\right)^{-1} \mathbf{X}^{T} \mathbf{y}$

#### Ridge Regression as Bayesian Inference
- Ridge regression = MAP estimation: $\arg \min _{\mathbf{w}} \sum_{i=1}^{n}\left(y_{i}-\mathbf{w}^{T} \mathbf{x}_{i}\right)^{2}+\lambda\|\mathbf{w}\|_{2}^{2} \equiv \arg \max _{\mathbf{w}} P(\mathbf{w}) \prod_{i} P\left(y_{i} \mid \mathbf{x}_{i}, \mathbf{w}\right)$
- $\lambda = {\sigma_n^2}/{\sigma_p^2}$, where $\sigma_p^2$ stands for prior
- <span style="color:blue">Prior acts as Regulazior</span>

#### Posterior Distributions in BLR
##### Prior: $p(\mathbf{w})=\mathcal{N}(0, \sigma_p^2 \mathbf{I})$
##### Likelihood: $p\left(y \mid \mathbf{x}, \mathbf{w}, \sigma_{n}\right)=\mathcal{N}\left(y ; \mathbf{w}^{T} \mathbf{x}, \sigma_{n}^{2}\right)$
##### Posterior: $p(\mathbf{w} \mid \mathbf{X}, \mathbf{y})=\mathcal{N}(\mathbf{w} ; \bar{\mu}, \bar{\Sigma})$
- $\bar{\mu}=\left(\mathbf{X}^{T} \mathbf{X}+(\sigma_{n}^{2}/\sigma_p^2) \mathbf{I}\right)^{-1} \mathbf{X}^{T} \mathbf{y}$
- $\bar{\Sigma}=\left(\sigma_{n}^{-2} \mathbf{X}^{T} \mathbf{X}+\sigma_p^{-2}\mathbf{I}\right)^{-1}$

#### Making Predictions in BLR(Using properties of multivariate Gaussian)
##### $p\left(y^{*} \mid \mathbf{X}, \mathbf{y}, \mathbf{x}^{*}\right) = \int p(y^{*}|x^{*},\mathbf{w})p(\mathbf{w}|\mathbf{X}, \mathbf{y})d\mathbf{w}$
- <span style="color:red">Likelihood</span>($p(y^{*}|x^{*},\mathbf{w})$) + <span style="color:red">Posterior</span>($p(\mathbf{w}|\mathbf{X}, \mathbf{y})$) $\Longrightarrow$ <span style="color:red">Predictions</span>($p\left(y^{*} \mid \mathbf{X}, \mathbf{y}, \mathbf{x}^{*}\right)$)
##### $p\left(f^{*} \mid \mathbf{X}, \mathbf{y}, \mathbf{x}^{*}\right)=\mathcal{N}\left(\bar{\mu}^{T} \mathbf{x}^{*}, \mathbf{x}^{* T} \bar{\Sigma} \mathbf{x}^{*}\right)$, where $\mathbf{x}^*$ is test point and $f^{*}=\mathbf{w}^{T} \mathbf{x}^{*}$
##### $p\left(y^{*} \mid \mathbf{X}, \mathbf{y}, \mathbf{x}^{*}\right)=\mathcal{N}\left(\bar{\mu}^{T} \mathbf{x}^{*}, \mathbf{x}^{* T} \bar{\Sigma} \mathbf{x}^{*}+\sigma_{n}^{2}\right)$
##### $\mathbf{x}^{* T} \bar{\Sigma} \mathbf{x}^{*}$: Uncertainty about $f^*$(<span style="color:blue">epistemic</span>)
##### $\sigma_{n}^{2}$: Noise / uncertainty about $y^*$ given $f^*$(<span style="color:blue">aleatoric</span>)
##### Epistemic uncertainty: Uncertainty about the model due to the lack of data
##### Aleatoric uncertainty: Irreducible noise

#### Ridge Regression vs Bayesian Regression
- $p\left(y^{*} \mid \mathbf{x}^{*}, \mathbf{x}_{1: n}, y_{1: n}\right)=\int p\left(y^{*} \mid \mathbf{x}^{*}, \mathbf{w}\right) p\left(\mathbf{w} \mid \mathbf{x}_{1: n}, y_{1: n}\right) d \mathbf{w}$ $\approx \int p\left(y^{*} \mid \mathbf{x}^{*}, \mathbf{w}\right) \delta_{\hat{\mathbf{w}}}(\mathbf{w}) d \mathbf{w}=p\left(y^{*} \mid \mathbf{x}^{*}, \hat{\mathbf{w}}\right)$, put all weight on the mode
- $\hat{\mathbf{w}} \equiv \mathbf{w}_{MAP} = \arg \max _{\mathbf{w}} p\left(\mathbf{w} \mid \mathbf{x}_{1: n}, y_{1: n}\right)$

#### Recursive Bayesian Updates
- <span style="color:red">“Today’s posterior is tomorrow’s prior”</span>
- $p^{j+1}(\theta) = p(\theta|y_{1:j+1}) = \frac{1}{Z}p(\theta|y_{1:j})p(y_{j+1}|\theta,y_{1:j})$
- $p(\theta|y_{1:j}) = p^{j}(\theta)$,   $p(y_{j+1}|\theta,y_{1:j}) = p_{j+1}(y_{j+1}|\theta)$

### **Kalman Filters**

#### Models
##### Variables 
- $\mathbf{x}_1,...,\mathbf{x}_T$: Location of object being tracked
- $\mathbf{y}_1,...,\mathbf{y}_T$: Observations
- $P(\mathbf{x}_1)$: Prior belief
##### $P(\mathbf{x}_{t+1} | \mathbf{x}_t)$ <span style="color:blue">Motion(Transition) model</span>: $\mathbf{x}_{t+1}=\mathbf{F} \mathbf{x}_{t}+\varepsilon_{t}$ where $\varepsilon_{t} \in \mathcal{N}\left(0, \Sigma_{x}\right)$
##### $P(\mathbf{y}_{t} | \mathbf{x}_t)$ <span style="color:blue">Sensor(Observation) model</span>: $\mathbf{y}_{t}=\mathbf{H x}_{t}+\eta_{t}$ where $\eta_{t} \in \mathcal{N}\left(0, \Sigma_{y}\right)$

#### Bayesian Filtering
#####  <span style="color:blue">Conditioning(Belief)</span>: $P(\mathbf{x}_t|\mathbf{y}_{1:t}) = \frac{1}{Z} P(\mathbf{x}_{t} | \mathbf{y}_{1:t-1})P(\mathbf{y}_{t} | \mathbf{x}_t)$
- Conditioning is composed of Prediction and Observation
#####  <span style="color:blue">Prediction</span>: $\int P(\mathbf{x}_{t+1} | \mathbf{x}_t)P(\mathbf{x}_t|\mathbf{y}_{1:t})d\mathbf{x}_t$
- Prediction is composed of Motion and Conditioning
##### <span style="color:red">Given Motion, Observation and Prior Belief, one can calculate Prediction and Conditioning iterable(Belief)</span>

#### General Kalman Update(Gaussian)
- Transition(Motion) model: $ P\left(\mathbf{x}_{t+1} | \mathbf{x}_{t}\right) =\mathcal{N}\left(\mathbf{x}_{t+1} ; \mathbf{F} \mathbf{x}_{t}, \Sigma_{x}\right) $
- Sensor model: $ P\left(\mathbf{y}_{t} | \mathbf{x}_{t}\right) =\mathcal{N}\left(\mathbf{y}_{t} ; \mathbf{H} \mathbf{x}_{t}, \Sigma_{y}\right)$
##### Kalman Update: 
- $\mu_{t+1}=\mathbf{F} \mu_{t}+\mathbf{K}_{t+1}\left(\mathbf{y}_{t+1}-\mathbf{H} \mathbf{F} \mu_{t}\right)$
  $\boldsymbol{\Sigma}_{t+1}=\left(\mathbf{I}-\mathbf{K}_{k+1} \mathbf{H}\right)\left(\mathbf{F} \boldsymbol{\Sigma}_{t} \mathbf{F}^{\top}+\boldsymbol{\Sigma}_{x}\right)$
##### Kalman Gain:
- $\mathbf{K}_{t+1}=\left(\mathbf{F} \Sigma_{t} \mathbf{F}^{T}+\Sigma_{x}\right) \mathbf{H}^{T}\left(\mathbf{H}\left(\mathbf{F} \Sigma_{t} \mathbf{F}^{T}+\Sigma_{x}\right) \mathbf{H}^{T}+\Sigma_{y}\right)^{-1}$
- Can compute $\boldsymbol{\Sigma}_{t}$ and $\mathbf{K}_{t}$ offline (they do not depend on the variables $\mathbf{x}_t$ and $\mathbf{y}_t$)

#### Example: 1D Random Walk 

#### BLR vs Kalman Filtering
- Can view Bayesian linear regression as a form of a Kalman filter!

### Non-linear Functions $\rightarrow$ Kernelized Bayesian Linear Regression: **Gaussian Processes**

#### Kernel Trick
##### Reasons
- Applying linear method (like BLR) on nonlinearly transformed data.
  However, computational cost increases with dimensionality of the feature space!
##### $\mathbf{x}_i^T \mathbf{x}_j \Longrightarrow k(\mathbf{x}_i,\mathbf{x}_j)$
##### Weight vs Function Space View
- Gaussian prior on the weights: $p(\mathbf{w})=\mathcal{N}(0, \sigma_p^2 \mathbf{I})$
- A set(m) of n-dimension inputs $\mathbf{X}$
- $f = \mathbf{X} \mathbf{w}$: Multiples of Gaussians are Gaussian
   $f \sim \mathcal{N}(0, \sigma_p^2 \mathbf{X}\mathbf{X}^T)$
- Kernelize: $f \sim \mathcal{N}(0, \sigma_p^2 \mathbf{K})$

#### Gaussian Process(GP) Definition
##### An (infinite) set of random variables, indexed by some set X i.e., there exists functions
- <span style="color:blue">mean function</span> $\mu: \mathbf{X} \rightarrow \mathbb{R} $ 
- <span style="color:blue">covariance (kernel) function</span> $k: X \times X \rightarrow \mathbb{R}$
##### GP Marginals
- For specific input $\mathbf{x}'$, $p(f(\mathbf{x}'))=G P\left(f ; \mu(\mathbf{x}^{\prime}), k(\mathbf{x}^{\prime},\mathbf{x}^{\prime})\right)=\mathcal{N}(f(\mathbf{x}') ; \mu(\mathbf{x}') , k(\mathbf{x}', \mathbf{x}'))$
##### Kernels
###### Properties
- Symmetric
- Positive Definite
###### Species
- Linear kernel: $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T\mathbf{x}'$
- Linear kernel with features: $k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T\phi(\mathbf{x}')$, where $\Phi(\mathbf{x})$ can be polynomial, sine, etc
- Squared exponential (aka RBF, Gaussian) kernel: $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\exp \left(-|| \mathbf{x}-\mathbf{x}^{\prime}||_{2}^{2} / h^{2}\right)$
- Exponential kernel: $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\exp \left(-|| \mathbf{x}-\mathbf{x}^{\prime}||_{2} / h\right)$
- Matérn kernel: $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2 \nu}\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|_{2}}{\rho}\right)^{\nu} K_{\nu}\left(\frac{\sqrt{2 \nu}\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|_{2}}{\rho}\right)$
###### Composition Rules 
- $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)+k_{2}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$
- $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) k_{2}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$
- $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=c k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ for $c>0$
- $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=f\left(k_{1}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)$

#### Making Predictions with GPs
##### Suppose $p(f)=G P(f ; \mu, k)$ (prior by selecting $\mu$ and $k$)
##### Observe $\mathbf{y}_{i}=f\left(\mathbf{x}_{i}\right)+\epsilon_{i} \quad A=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{m}\right\}$
##### $p\left(f \mid \mathbf{x}_{1}, \ldots, \mathbf{x}_{m}, \mathbf{y}_{1}, \ldots, \mathbf{y}_{m}\right)=G P\left(f ; \mu^{\prime}, k^{\prime}\right)$
- $\begin{aligned} \mu^{\prime}(\mathbf{x}) &=\mu(\mathbf{x})+\mathbf{k}_{x, A}\left(\mathbf{K}_{A A}+\sigma^{2} \mathbf{I}\right)^{-1}\left(\mathbf{y}_{A}-\mu_{A}\right) \\ k^{\prime}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) &=k(\mathbf{x}, \mathbf{x'})-\mathbf{k}_{x, A}\left(\mathbf{K}_{A A}+\sigma^{2} \mathbf{I}\right)^{-1} \mathbf{k}_{x^{\prime}, A}^{T} \end{aligned}$
##### Properties
- Closed form formulas for prediction
- Posterior covariance $\mathbf{k}'$ does not depend on $\mathbf{y_A}$

#### Sample from a GP

#### Model Selection for GPs(A set of kernels determined by kernel parameters $\theta$)
##### $\log p(\mathbf{y} \mid X, \theta)=-\frac{1}{2} \mathbf{y}^{T} \mathbf{K}_{y}^{-1} \mathbf{y}-\frac{1}{2} \log \left|\mathbf{K}_{y}\right|-\frac{n}{2} \log 2 \pi$
##### $\hat{\theta} = \arg\max_\theta p(\mathbf{y} \mid X, \theta)$
##### Can get converged $\theta$ using gradient descent, treat $p(\mathbf{y} \mid X, \theta)$ as loss function 
##### Or: select model using $p(\mathbf{y} \mid X, \theta) = \int p(\mathbf{y} \mid X, f) p(f \mid \theta) df$

#### Computational Issues
##### $\mu^{\prime}(\mathbf{x})$ and $k^{\prime}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ require calculating $\left(\mathbf{K}_{A A}+\sigma^{2} \mathbf{I}\right)^{-1}$
##### $|A|=n$ variables $\rightarrow \Theta\left(n^{3}\right)$

#### Fast GP Methods
##### Exploiting parallelism (GPU computations)
##### Local GP methods: Covariance functions that decay with distance of points (e.g., RBF, Matern, kernels) lend themselves to local computations
##### Kernel function approximations (RFFs, QFFs, …)
##### <span style="color:blue">Inducing point methods</span> (SoR, FITC, VFE etc.)
- “Summarize” data via function values of f at a set u of m inducing points
  $p\left(f^{*}, f\right)=\int p\left(\mathbf{f}^{*}, \mathbf{f}, \mathbf{u}\right) d \mathbf{u}=\int p\left(\mathbf{f}^{*}, \mathbf{f} \mid \mathbf{u}\right) p(\mathbf{u}) d \mathbf{u}$
- Key idea: Approximate by
  $p\left(\mathbf{f}^{*}, \mathbf{f}\right) \approx q\left(\mathbf{f}^{*}, \mathbf{f}\right)=\int q\left(\mathbf{f}^{*} \mid \mathbf{u}\right) q(\mathbf{f} \mid \mathbf{u}) p(\mathbf{u}) d \mathbf{u}$
- $K_{u,u}$ has much smaller size
- Need to ensure u is representative of the data and where predictions are made

### Bayesian learning more generally
#### Posterior and Prediction are not in closed-form(intractable) $\longrightarrow$ need approximations

#### Approximate Inference
##### $p(\theta \mid y)=\frac{1}{Z} p(\theta, y) \approx q(\theta \mid \lambda)$
##### Laplace Approximation
- $q(\theta) =\mathcal{N}\left(\theta ; \hat{\theta}, \Lambda^{-1}\right)$
  posterior mode $ \hat{\theta} =\arg \max _{\theta} p(\theta \mid y)$
  $ \Lambda =-\nabla \nabla \log p(\hat{\theta} \mid y)$
- Issue: This can lead to poor (e.g., overconfident) approximations for multi-mode curvature

##### Example: Bayesian logistic regression
- Likelihood: $P(y \mid \mathbf{x}, \mathbf{w})=\operatorname{Ber}\left(y ; \sigma\left(\mathbf{w}^{T} \mathbf{x}\right)\right)$

##### Making Predictions
- Given Gaussian approximation of posterior: $p\left(\mathbf{w} \mid \mathbf{x}_{1: n}, y_{1: n}\right) \approx q(\mathbf{w})=\mathcal{N}\left(\mathbf{w} ; \hat{\mathbf{w}} ; \Lambda^{-1}\right)$
- Predictive distribution for test point $\mathbf{x}^{*}$:
  $p\left(y^{*} \mid \mathbf{x}^{*}, \mathbf{x}_{1: n}, y_{1: n}\right) \approx \int \sigma\left(y^{*} \mathbf{w}^{T} \mathbf{x}\right) \mathcal{N}\left(\mathbf{w} ; \hat{\mathbf{w}}, \Lambda^{-1}\right) d \mathbf{w} =\int \sigma\left(y^{*} f\right) \mathcal{N}\left(f ; \hat{\mathbf{w}}^{T} \mathbf{x}^{*}, \mathbf{x}^{* T} \Lambda^{-1} \mathbf{x}^{*}\right) d f$

#### **Variational Inference** (see [this blog](https://yuanzhi-zhu.github.io/2023/05/25/Bayesian-Posterior-Sampling/)): $q^{*} \in \arg \min _{q \in \mathcal{Q}} K L(q \| p)$
##### Variational family: 
- Gaussian distributions
- Gaussians with diagonal covariance $\mathcal{Q}=\{q(\theta)=\mathcal{N}(\theta ; \mu, \operatorname{diag}([\sigma]))\}$
- ….

#### **KL-Divergence**: $K L(q \| p)=\int q(\theta) \log \frac{q(\theta)}{p(\theta)} d \theta$
##### Properties
- Non-negative
- Zero if and only if $p$ & $q$ agree almost everywhere
- Not generally symmetric
##### Example: KL Divergence between Gaussians
##### Minimizing KL Divergence
- $\arg \min _{q} K L(q \| p)=\arg \min _{q} \int q(\theta) \log \frac{q(\theta)}{\frac{1}{Z} p(\theta, y)} d \theta =\arg \max _{q} \{\mathbb{E}_{\theta \sim q(\theta)}[\log p(\theta, y)]+H(q)\}=\arg \max _{q} \{\mathbb{E}_{\theta \sim q(\theta)}[\log p(y \mid \theta)]-K L(q \| p(\cdot))\}$
##### Maximizing Lower Bound on Evidence: “ELBO” (Evidence Lower BOund) 
- $\log p(y)=\log \int p(y \mid \theta) p(\theta) d \theta \geq \mathbb{E}_{\theta \sim q}\left[\log \left(p(y \mid \theta) \frac{p(\theta)}{q(\theta)}\right)\right] d \theta =\mathbb{E}_{\theta \sim q}[\log p(y \mid \theta)]-K L(q \| p(\cdot))$
##### Inference as Optimization
- Task: Infer the most close $q = \arg \min _{q \in \mathcal{Q}} K L(q \| p(\cdot \mid y))$
- <span style="color:blue">“ELBO” (Evidence lower bound)</span> $L(q) = \mathbb{E}_{\theta \sim q(\theta)}[\log p(\theta, y)]+H(q)$ as loss function
- Gradient of the ELBO $\longrightarrow$ Optimization
##### <span style="color:blue">Reparameterization Trick</span> for Optimization
- $q(\theta \mid \lambda)=\phi(\epsilon)\left|\nabla_{\epsilon} g(\epsilon ; \lambda)\right|^{-1} \Longrightarrow \mathbb{E}_{\theta \sim q_{\lambda}}[f(\theta)]=\mathbb{E}_{\epsilon \sim \phi}[f(g(\epsilon ; \lambda))]$
- Exchange Expectation and Gradient: $\nabla_{\lambda} \mathbb{E}_{\theta \sim q_{\lambda}}[f(\theta)] {=} \mathbb{E}_{\epsilon \sim \phi}\left[\nabla_{\lambda} f(g(\epsilon ; \lambda))\right]$
- Example: Reparametrizing the ELBO for B. Log. Reg


#### **Markov-Chain Monte Carlo(MCMC)** (see [this blog](https://yuanzhi-zhu.github.io/2023/06/26/Introduction-to-MCMC/))
##### Key Idea: $\mathbb{E}_{\theta \sim p(\cdot \mid x_{1:n},y_{1:n})}[f(\theta)] \approx \frac{1}{N} \sum_{i=1}^{N} f\left(\theta^{i}\right)$, where $f\left(\theta\right) = p(y^* \mid x^*,\theta)$
##### Given Unnormalized Distribution: $P(x) = \frac{1}{Z}Q(x)$
- Q(X) efficient to evaluate, but normalizer Z intractable
- <span style="color:red">MCMC Method: design a Markov chain($P(x | x')$) with stationary distribution $\pi(\mathbf{x})=\frac{1}{Z} Q(\mathbf{x}) = P(x)$</span>
##### Markov Chains
- Prior $P(X1)$ and Transition probabilities $P(X_{t+1} | X_t
)$ independent of $t$
- Ergodic Markov Chains: there exists a finite $t$ such that every state can be reached from every state in exactly $t$ steps
- Stationary Distributions: $\lim\limits_{N \rightarrow \infty}  P\left(X_{N}=x\right)=\pi(x)$, which is independent of $P(X_1)$

##### <span style="color:blue">Metropolis Hastings</span>
- <span style="color:blue">Detailed Balance Equation</span>: $Q(\mathbf{x}) P\left(\mathbf{x}^{\prime} \mid \mathbf{x}\right)=Q\left(\mathbf{x}^{\prime}\right) P\left(\mathbf{x} \mid \mathbf{x}^{\prime}\right)$
- Proposal distribution(Transition probability) $R(X' | X)$
- Acceptance distribution: 
  With probability $\alpha=\min \left\{1, \frac{Q\left(x^{\prime}\right) R\left(x \mid x^{\prime}\right)}{Q(x) R\left(x^{\prime} \mid x\right)}\right\}$ set $X_{t+1} = x'$
  With probability $1-\alpha$, set $X_{t+1} = x$
##### Gibbs Sampling 
###### Random Order
###### Practical Variant
- Start with initial assignment $x^{(0)}$ to all variables
- Fix observed variables $X_B$ to their observed value $x_B$
- For $t = 1$ to $\infty$ do
    Set $x^{(t)} = x^{(t-1)}$
    For each variable $X_i$(except those in $\mathbf{B}$)
      Set $v_i$ = values of all $x^{(t)}$ except $x_i$
      Sample $x^{(t)}_i$ from $P(X_i | v_i)$

##### Computing Expectations via MCMC
- To let the Markov chain "burn in", ignore the first samples, and approximate
- $\mathbb{E}\left[f(\mathbf{X}) \mid \mathbf{x}_{B}\right] \approx \frac{1}{T-t_{0}} \sum_{\tau=t_{0}+1}^{T} f\left(\mathbf{X}^{(\tau)}\right)$

##### MCMC for Continuous RVs: $p(\mathbf{x})=\frac{1}{Z} \exp (-f(\mathbf{x}))$
- Metropolis Hastings with Gaussian Proposals: $R\left(x^{\prime} \mid x\right)=\mathcal{N}\left(x^{\prime} ; x ; \tau I\right)$
- Improved proposals(MALA): $R\left(x^{\prime} \mid x\right)=\mathcal{N}\left(x^{\prime} ; x-\tau \nabla f(x) ; 2 \tau I\right)$
- Stochastic Gradient Langevin Dynamics(SGLD): $\theta \sim \exp \left(\log p(\theta)+\sum_{i=1}^{n} \log p\left(y_{i} \mid x_{i}, \theta\right)\right)$
  Loss function: $L(\theta) = \log p(\theta)+\sum_{i=1}^{n} \log p\left(y_{i} \mid x_{i}, \theta\right)$

##### Outlook: Hamiltonian Monte Carlo (HMC)



### **Bayesian Deep Learning**
#### (Deep) Artificial Neural Networks
- $f(\mathbf{x} ; \mathbf{w})=\varphi\left(\mathbf{W}_{1} \varphi\left(\mathbf{W}_{2}\left(\ldots \varphi\left(\mathbf{W}_{\ell} \mathbf{x}\right)\right)\right)\right)$


#### Activation Functions
- Tanh
- Relu
- Sigmoid

#### Bayesian Neural Networks
- Gaussian prior: $p(\theta)=\mathcal{N}\left(\theta ; 0, \sigma_{p}^{2} I\right)$
- Likelihood: $p(y \mid \mathbf{x}, \theta)=\mathcal{N}\left(y ; f(\mathbf{x}, \theta), \sigma^{2}\right)$

##### <span style="color:blue">Heteroscedastic Noise</span>
- More complex likelihood: $p(y \mid \mathbf{x}, \theta)=\mathcal{N}\left(y ; f_{1}(\mathbf{x}, \theta), \exp \left(f_{2}(\mathbf{x}, \theta)\right)\right)$


#### MAP Estimates for Bayesian NNs
- $\hat{\theta}=\arg \min _{\theta}\{-\log p(\theta)-\sum_{i=1}^{n} \log p\left(y_{i} \mid \mathbf{x}_{i}, \theta\right)\}$
$=\arg \min _{\theta}\{-\lambda\|\theta\|_{2}^{2}+\sum_{i=1}^{n} \frac{1}{2 \sigma\left(\mathbf{x}_{i} ; \theta\right)^{2}}\left\|y_{i}-\mu\left(\mathbf{x}_{i} ; \theta\right)\right\|^{2}+\frac{1}{2} \log \sigma\left(\mathbf{x}_{i} ; \theta\right)^{2}\}$
- Stochastic Gradient Descent

#### Bayesian Learning with Neural Nets:  Integrals are Intractable
##### Approximate Inference for BNNs
- Variational Inference for BNN(aka Bayes by Backprop)
- Markov-Chain Monte Carlo for BNN(SGLD,Stochastic Gradient Hamiltonian Monte Carlo,etc)
- Dropout as Variational Inference
- Probabilistic Ensembles of NNs
##### Only need to be able to compute gradients, which can be done using automatic differentiation 





## Bayesian Learning: use of the uncertainty for deciding which data to collect

### **Active Learning**(Query points whose observation provides most useful information about the unknown function)

#### Optimizing <span style="color:blue">Mutual Information</span>
##### Set $D$ of points to observe $f$ at
##### Find $S \subseteq D$ maximizing information gain:
##### $F(S):=H(f)-H\left(f \mid y_{S}\right)=I\left(f ; y_{S}\right)=\frac{1}{2} \log \left|I+\sigma^{-2} K_{S}\right|$
##### <span style="color:blue">Greedy Algorithm</span>
- For $S_t = \{x_1,...,x_t\}$
- $x_{t+1}=\underset{{x \in D}}{\arg \max }  F\left(S_{t} \cup\{x\}\right)=\underset{{x \in D}}{\arg \max }  \sigma_{x \mid S_{t}}^{2}$

##### Submodularity of Mutual Information

##### Heteroscedastic case
-  most uncertain outcomes are not necessarily most informative
- $x_{t+1} \in \arg \max _{x} \frac{\sigma_{f}^{2}(x)}{\sigma_{n}^{2}(x)}$

#### Active learning for classification
##### Maximize entropy of the predicted label
##### $x_{t+1} \in \arg \max _{x} H\left(Y \mid x, x_{1: t}, y_{1: t}\right)$
##### Informative Sampling for Classification (BALD)


### **Bayesian Optimization**
#### Task
- Given: Set of possible inputs $D$; noisy black-box access to unknown function $f \in \mathcal{F}, \quad f: D \rightarrow \mathbb{R}$
- Task: Adaptively choose inputs $x_1,...,x_T$ from $D$. After each selection, observe $y_{t}=f\left(x_{t}\right)+\varepsilon_{t}$
- Cumulative regret: $R_{T}=\sum_{t=1}^{T}\left(\max _{x} f(x)-f\left(x_{t}\right)\right)$
- $R_T/T\rightarrow 0$ $\Rightarrow$ Sublinear $\Rightarrow$ $\max _{t} f\left(x_{t}\right) \rightarrow f\left(x^{*}\right)$
#### Optimistic Bayesian Optimization with GPs
##### Upper confidence sampling(GP-UCB)
###### <span style="color:blue">Acquisition Function(AF)</span> $x_{t}=\arg \max _{x \in D} \mu_{t-1}(x)+\beta_{t} \sigma_{t-1}(x)$
###### Information capacity of GPs

##### Alternative Acquisition Functions
###### Expected Improvement(EI)
###### Probability of Improvement(PI)
###### Information Directed Sampling
###### Thompson Sampling
- Each iteration $t$, Thompson sampling draws $\tilde{f} \sim P\left(f \mid x_{1: t}, y_{1: t}\right)$
- Then selects $x_{t+1} \in \arg \max _{x \in D} \tilde{f}(x)$





## **Reinforce Learning**

### **Markov Decision Processes(MDP)**
#### MDP model
##### A set of states $X = \{1,...,n\}$
##### A set of actions $A = \{1,...,m\}$
##### Transition probabilities $P\left(x^{\prime} \mid x, a\right)=\operatorname{Prob}\left(\right.$ Next state $=x^{\prime} \mid$ Action $a$ in state $\left.x\right)$
##### A reward function $r(x, a)$ or $r(x, a, x')$
##### Assume $r$ and $P$ are known

#### Planning in MDPs
##### Policy
- Deterministic Policy $\pi: X\rightarrow A$
- Randomized Policy $\pi: X\rightarrow P(A)$
##### Induces a Markov chain with transition probabilities $P\left(X_{t+1}=x^{\prime} \mid X_{t}=x\right)=P\left(x^{\prime} \mid x, \pi(x)\right)$
##### Expected value $J(\pi) = \mathbb{E}[r(X_0,\pi(X_0))+\gamma r(X_1,\pi(X_1)) + \gamma^2 r(X_2,\pi(X_2)) + ...]$, where $\gamma \in [0,1)$ is the discount factor
##### Value function(given policy) $V^{\pi}(x)=J\left(\pi \mid X_{0}=x\right)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(X_{t}, \pi\left(X_{t}\right)\right) \mid X_{0}=x\right]$
###### Recursion: $V^{\pi}(x)=r(x, \pi(x))+\gamma \sum_{x^{\prime}} P\left(x^{\prime} \mid x, \pi(x)\right) V^{\pi}\left(x^{\prime}\right)$
###### Fixed point iteration $V^{\pi} = r^{\pi} + \gamma T^{\pi} V^{\pi}$
- $V^{\pi}_i = V^{\pi}(x_i)$
- $r^{\pi}_i = r^{\pi}(x_i,\pi(x_i))$
- $T^{\pi}_{i,j} = P\left(x_j \mid x_i, \pi(x_i)\right)$
##### <span style="color:red">Value Functions and (Greedy) Policies</span>
###### Every value function induces a policy
###### Every (greedy) policy induces a value function

#### Ways for solving MDPs

##### <span style="color:blue">Policy Iteration</span>
###### Start with an arbitrary (e.g., random) policy $\pi$
###### Until converged do
- Compute value function $V^{\pi}(x)$
- Compute greedy policy $\pi_G$ w.r.t. $V^{\pi}$
- Set $\pi \leftarrow \pi_G$

##### <span style="color:blue">Value Iteration</span>
###### Initialize $V_0(x) = max_a r(x, a)$
###### For $t = 1$ to $\infty$
- For each $x$, $a$, let $Q_{t}(x, a)=r(x, a)+\gamma \sum_{x^{\prime}} P\left(x^{\prime} \mid x, a\right) V_{t-1}\left(x^{\prime}\right)$
- For each $x$ let $V_{t}(x)=\max _{a} Q_{t}(x, a)$
- Break if $\left\|V_{t}-V_{t-1}\right\|_{\infty}=\max _{x}\left|V_{t}(x)-V_{t-1}(x)\right| \leq \varepsilon$
###### Then choose greedy policy w.r.t. $V_t$

##### Tradeoffs: Value vs Policy Iteration

##### Linear Programming



#### POMDP = Belief-state MDP

##### Key Idea: POMDP as MDP with enlarged state space
##### States: Beliefs over states for original POMDP $\mathcal{B}=\Delta(\{1, \ldots, n\})=\left\{b:\{1, \ldots, n\} \rightarrow[0,1], \sum_{x} b(x)=1\right\}$
##### Actions: Same as original MDP
##### Transition model: 
###### Stochastic observation: $P\left(Y_{t+1}=y \mid b_{t}, a_{t}\right)=\sum_{x, x^{\prime}} b_{t}(x) P\left(x^{\prime} \mid x, a_{t}\right) P\left(y \mid x^{\prime}\right)$
###### State update (Bayesian filtering!) Given $b_t, a_t, y_{t+1}$: $b_{t+1}\left(x^{\prime}\right)=\frac{1}{Z} \sum_{x} b_{t}(x) P\left(X_{t+1}=x^{\prime} \mid X_{t}=x, a_{t}\right) P\left(y_{t+1} \mid x^{\prime}\right)$
##### Reward function: $r\left(b_{t}, a_{t}\right)=\sum_{x} b_{t}(x) r\left(x, a_{t}\right)$


##### Solving POMDPs
###### For finite horizon T, set of reachable belief states is finite (but exponential in T)
###### Can calculate optimal action using dynamic programming
###### Approximate Solutions to POMDPs(Key idea: most belief states never reached)
- Discretize the belief space by sampling
- Point based methods
- May want to apply dimensionality reduction
- Policy Gradient Methods



### Reinforcement Learning(RL) 
#### Key Idea: RL = Planning in unknown MDPs(Trannsitions and Rewards unknown)
#### <span style="color:blue">Exploration—Exploitation Dilemma</span>
##### Always pick a random action(Exploration)
- Will eventually* correctly estimate all probabilities and rewards :)
- May do extremely poorly in terms of rewards :(
##### Always pick the best action according to current knowledge(Exploitation)
- Quickly get some reward :)
- Can get stuck in suboptimal action :(


### Off-policy vs on-policy RL
#### On-policy RL
- Agent has full control over which actions to pick
- Can choose how to trade exploration and exploitation
#### Off-policy RL
- Agent has no control over actions, only gets observational data (e.g., demonstrations, data collected by applying a different policy, …)


### Two basic approaches to RL

#### Model-based RL
##### Learn the MDP and optimize policy based on estimated MDP 
###### Data set: $\tau = (x_0,a_0,r_0,x_1,a_1,r_1,...,x_{T-1},a_{T-1},r_{T-1},x_T)$
###### Estimate transitions $P\left(X_{t+1} \mid X_{t}, A\right) \approx \frac{\operatorname{Count}\left(X_{t+1}, X_{t}, A\right)}{\operatorname{Count}\left(X_{t}, A\right)}$
###### Estimate rewards $r(x, a) \approx \frac{1}{N_{x, a}} \sum_{t: X_{t}=x, A_{t}=a} R_{t}$

##### $\epsilon_t$ Greedy
###### With probability $\epsilon_t$: Pick random action
###### With probability $(1-\epsilon_t)$: Pick best action

##### $R_{max}$ Algorithm
###### <span style="color:red">Optimism in the face of uncertainty</span>
###### Input: Starting state $x_0$, discount factor $\gamma$
###### Initialization: 
- Add fairy tale state $x^*$ to MDP
- Set $r(x, a) = R_{max}$ for all states $x$ and actions $a$
- Set $P(x^* | x, a) = 1$ for all states $x$ and actions $a$
- Choose optimal policy for $r$ and $P$ 
###### Repeat:
- Execute policy $\pi$
- For each visited state action pair $x$, $a$, update $r(x, a)$
- Estimate transition probabilities $P(x' | x, a)$
- If observed “enough” transitions/rewards, recompute policy $\pi$ according to current model $P$ and $r$

###### **Theorem**: Every T timesteps, w.h.p.(with high probability), Rmax either
- Obtains near-optimal reward, or
- Visits at least one unknown state-action pair

###### Hoeffding Bound: How many samples do we need to accurately estimate $P(x' | x, a)$ or $r(x,a)$ given the desired precision

##### Model-based Deep RL
###### Receding-horizon / Model-predictive control(MPC)(Planning with a known deterministic model)
- Key idea: Plan over a finite horizon H, carry out first action, then replan
- workflow
  - At each iteration $t$, observe $x_t$
  - Optimize performance over horizon $H$
    $\max _{a_{t: t+H-1}} \sum_{\tau=t: t+H-1} \gamma^{\tau-t} r_{\tau}\left(x_{\tau}, a_{\tau}\right)$ s.t. $x_{\tau+1}=f\left(x_{\tau}, a_{\tau}\right)$
  - Carry out action $a_t$, then replan
- Solving the optimization problem
  - At each iteration, need to solve
    - $\max _{a_{t: t+H-1}} \sum_{\tau=t: t+H-1} \gamma^{\tau-t} r_{\tau}\left(x_{\tau}, a_{\tau}\right)$ s.t. $x_{\tau+1}=f\left(x_{\tau}, a_{\tau}\right)$
  - For deterministic models f, $x_{\tau}$ is determined by $a_{t:t-1}$ via
    - $x_{\tau}:=x_{\tau}\left(a_{t: \tau-1}\right):=f\left(f\left(\ldots\left(f\left(x_{t}, a_{t}\right), a_{t+1}\right), \ldots\right), a_{\tau-1}\right)$
  - Thus, at step $t$, need to maximize
    $J_{H}\left(a_{t: t+H-1}\right):=\sum_{\tau=t: t+H-1} \gamma^{\tau-t} r_{\tau}\left(x_{\tau}\left(a_{t: \tau-1}\right), a_{\tau}\right)$
    - For continuous actions, differentiable rewards and differentiable dynamics, cananalytically compute gradients (backpropagation through time)
    - Random shooting methods
      - Generate $m$ sets of random samples $a_{t:t+H}^{(i)}$
        E.g., from a Gaussian distribution,cross-entropy method, ...
      - Pick the sequence $a_{t:t+H}^{(i^*)}$ that optimizes $i^{*}=\arg \max _{i \in\{1 . . m\}} J_{H}\left(a_{t: t+H-1}^{(i)}\right)$
    - Using a value estimate
      - Suppose we have access to (an estimate of) the value function $V$. Then we can consider $J_{H}\left(a_{t: t+H-1}\right):=\sum_{\tau=t: t+H-1} \gamma^{\tau-t} r_{\tau}\left(x_{\tau}\left(a_{t: \tau-1}\right), a_{\tau}\right)+\gamma^{H} V\left(x_{t+H}\right)$
      - For $H=1$, $a_{t}=\arg \max _{a} J_{H}(a)$ is simply the greedy policy w.r.t. $V$
    - Can also optimize using gradient-based or global optimization (shooting) methods
    - Can obtain value estimates using off-policy estimation (as discussed earlier)
###### MPC for stochastic transition models
- workflow
  - At each iteration $t$, observe $x_t$
  - Optimize performance over horizon $H$
    $\max _{a_{t: t+H-1}} \mathbb{E}_{x_{t+1: t+H}}\left[\sum_{\tau=t: t+H-1} \gamma^{\tau-t} r_{\tau}+\gamma^{H} V\left(x_{t+H}\right) \mid a_{t: t+H-1}\right]$
  - Carry out action $a_t$, then replan
- Optimizing expected performance
  - For probabilistic transition models via MPC, need to optimize $J_{H}\left(a_{t: t+H-1}\right):=\mathbb{E}_{x_{t+1: t+H}}\left[\sum_{\tau=t: t+H-1} \gamma^{\tau-t} r_{\tau}+\gamma^{H} V\left(x_{t+H}\right) \mid a_{t: t+H-1}\right]$
- Monte-Carlo trajectory sampling
  - Suppose the transition model is reparametrizable
- Using parametrized policies
###### Unknown Dynamics($f$ and $r$ are unknown)
- Start with initial policy $\pi$
  Iterate for several episodes
  - Roll out policy $\pi$ to collect data 
  - Learn a model for $f$, $r$ (and $Q$) from the collected data
  - Plan a new policy $\pi$ based on the estimated models
- How can we learn $f$ and $r$
  - Key insight: due to the Markovian structure of the MDP, observed transitions and rewards are (conditionally) independent
  - If we don’t know the dynamics & reward, can estimate them off-policy with standard supervised learning techniques from a replay buffer (data set) $D=\left\{\left(x_{i}, a_{i}, r_{i}, x_{i+1}\right)_{i}\right\}$
- Learning dynamics models $f$
  - For continuous state spaces, learning $f$ and $r$ is basically a regression problem
  - Each experience $\left(x_{i}, a_{i}, r_{i}, x_{i+1}\right)$ provides a labeled data point $(z,y)$ with $z:= (x, a)$ as input and $ y:= x'$ rsp. $r$ as label
- Learning probabilistic dynamics models for $x_{t+1} \sim f\left(x_{t}, a_{t} ; \theta\right)$
  - Learning with MAP estimation
  - Bayesian learning of dynamics models
  - Greedy exploitation for model-based RL
  - Thompson sampling









#### Model-free RL
##### Estimate the value function directly
###### Temporal Difference (TD)-Learning
- Follow policy $\pi$ to obtain a transition $(x,a,r,x’)$
- Update value estimate using bootstrapping $V(x) \leftarrow\left(1-\alpha_{t}\right) V(x)+\alpha_{t}\left(r+\gamma V\left(x^{\prime}\right)\right)$, where $\alpha_t$ is the learning rate
- TD-Learning as SGD:
  - Squared loss $\ell_{2}\left(\theta ; x, x^{\prime}, r\right)=\frac{1}{2}\left(V(x ; \theta)-r-\gamma V\left(x^{\prime} ; \theta_{\text {old }}\right)\right)^{2}$
  - Parameters are entries in value vector
  - Bootstrapping means to use “old” value estimates as labels (a.k.a. targets)

###### Q-Learning
- Estimate $Q^{*}(x, a)=r(x, a)+\gamma \sum_{x^{\prime}} P\left(x^{\prime} \mid x, a\right) V^{*}\left(x^{\prime}\right)$;  $V^{*}(x)=\max _{a} Q^{*}(x, a)$

- Suppose we: 
  - Have initial estimate of $Q(x, a)$
  - observe transition $x$, $a$, $x'$ with reward $r$
  - $Q(x, a) \leftarrow\left(1-\alpha_{t}\right) Q(x, a)+\alpha_{t}\left(r+\gamma \max _{a^{\prime}} Q\left(x^{\prime}, a^{\prime}\right)\right)$


##### **Policy Gradients** Methods

###### Objective: maximize $J(\theta)=\mathbb{E}_{x_{0: T}, a_{0: T} \sim \pi_{\theta}} \sum_{t=0}^{T} \gamma^{t} r\left(x_{t}, a_{t}\right)=\mathbb{E}_{\tau \sim \pi_{\theta}} r(\tau)$


###### Obtaining Policy Gradient
- $\nabla J(\theta)=\nabla \mathbb{E}_{\tau \sim \pi_{\theta}} r(\tau)=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[r(\tau) \nabla \log \pi_{\theta}(\tau)\right]$

###### Exploiting the MDP structure $r(\tau)=\sum_{t=0}^{T} \gamma^{t} r\left(s_{t}, a_{t}\right)$
- $\pi_{\theta}(\tau)=P\left(x_{0}\right) \prod_{t=0}^{T} \pi\left(a_{t} \mid x_{t} ; \theta\right) P\left(x_{t+1} \mid x_{t}, a_{t}\right)$
- $\mathbb{E}_{\tau \sim \pi_{\theta}}\left[r(\tau) \nabla \log \pi_{\theta}(\tau)\right]=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[r(\tau) \sum_{t=0}^{T} \nabla \log \pi\left(a_{t} \mid x_{t} ; \theta\right)\right]$

###### Reducing Variance(Baseline)

-  $\mathbb{E}_{\tau \sim \pi_{\theta}}\left[r(\tau) \nabla \log \pi_{\theta}(\tau)\right]=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[(r(\tau)-b) \nabla \log \pi_{\theta}(\tau)\right]$, where $b$ is the baseline

- State-dependent baselines $\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} r(\tau) \nabla \log \pi\left(a_{t} \mid x_{t} ; \theta\right)\right]=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T}\left(r(\tau)-b\left(\tau_{0: t-1}\right)\right) \nabla \log \pi\left(a_{t} \mid x_{t} ; \theta\right)\right]$
  - For example，$b\left(\tau_{0: t-1}\right)=\sum_{t^{\prime}=0}^{t-1} \gamma^{t^{\prime}} r_{t^{\prime}}$
  - $\nabla J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \gamma^{t} G_{t} \nabla \log \pi\left(a_{t} \mid x_{t} ; \theta\right)\right]$
  - where $G_{t}=\sum_{t^{\prime}=t}^{T} \gamma^{t^{\prime}-t} r_{t^{\prime}}$ is the reward to go following action $\alpha_t$

- REINFORCE
- Further variance reduction

###### On-policy policy gradient methods
- REINFORCE
- Actor Critic methods
- TRPO

###### Off-policy policy gradient methods
- DDPG
- TD3
- SAC

##### **Actor-Critic** Methods
###### Actor-Critic Algorithm
- New Concept: Advantage Function
  - Advantage of playing action $a$ in state $x$: $A^{\pi}(x, a)=Q^{\pi}(x, a)-V^{\pi}(x)$

- Can use value function estimates in conjunction with policy gradient methods: 
  - $\nabla J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{\infty} \gamma^{t} Q\left(x_{t}, a_{t} ; \theta_{Q}\right) \nabla \log \pi\left(a_{t} \mid x_{t} ; \theta\right)\right] =: \mathbb{E}_{(x, a) \sim \pi_{\theta}}\left[Q\left(x, a ; \theta_{Q}\right) \nabla \log \pi(a \mid x ; \theta)\right]$
  - where $\rho(x)=\sum_{t=0}^{\infty} \gamma^{t} p\left(x_{t}=x\right)$

- Allows application in the online (non-episodic) setting

- At time $t$, upon observing $a$ transition $(x,a,r,x')$, update: 
  - $\theta_{\pi} \leftarrow \theta_{\pi}-\eta_{t} Q\left(x, a ; \theta_{Q}\right) \nabla \log \pi\left(a \mid x ; \theta_{\pi}\right)$
    $\theta_{Q} \leftarrow \theta_{Q}-\eta_{t}\left(Q\left(x, a ; \theta_{Q}\right)-r-\gamma Q\left(x^{\prime}, \pi\left(x^{\prime}, \theta_{\pi}\right) ; \theta_{Q}\right)\right) \nabla Q\left(x, a ; \theta_{Q}\right)$


###### A2C Algorithm: Variance reduction via baselines
- $\theta_{\pi} \leftarrow \theta_{\pi}+\eta_{t}\left[Q\left(x, a ; \theta_{Q}\right)-V\left(x ; \theta_{V}\right)\right] \nabla \log \pi\left(a \mid x ; \theta_{\pi}\right)$
- Advantage Function Estimate: $ \left[Q\left(x, a ; \theta_{Q}\right)-V\left(x ; \theta_{V}\right)\right] $


###### Another approach to policy gradients: replace the exact maximum by a parametrized policy
- $L\left(\theta_{Q}\right)=\sum_{\left(x, a, r, x^{\prime}\right) \in D}\left(r+\gamma Q\left(x^{\prime}, \pi\left(x^{\prime} ; \theta_{\pi}\right) ; \theta_{Q}^{\text {old }}\right)-Q\left(x, a ; \theta_{Q}\right)\right)^{2}$
- Updating policy parameters
- Computing gradients

###### Deep Deterministic Policy Gradients (DDPG)
- Dealing with randomized policies
- Reparametrization gradients







### Reinforcement Learning via Function Approximation(**Parameterization**)

#### Parametric Value Function Approximation
##### To scale to large state spaces, learn an approximation of (action) value function $V(x;\theta)$ or $Q(x,a;\theta)$
##### Examples
###### (Deep) Neural Networks $\rightarrow$ Deep RL
###### Gradients for Q-learning with Function Approximation
- Example: Linear Function Approximation $Q(x, a ; \theta)=\theta^{T} \phi(x, a)$, where $\phi(x,a)$ are a set of (hand-designed) features
- After observing transition (x,a,r,x’)
  update via gradient of $\left.\ell_{2}\left(\theta ; x, a, r, x^{\prime}\right)=\frac{1}{2}\left(Q(x, a ; \theta)-r-\gamma \max _{a^{\prime}} Q\left(x^{\prime}, a^{\prime} ; \theta_{o l d}\right)\right]\right)^{2}$

###### Neural Fitted Q-iteration / DQN
###### Double DQN


#### Policy Search Methods(Dealing with large action sets)

##### Learning a Parameterized Policy $\pi(x) = \pi(x;\theta)$

##### For episodic tasks (i.e., can "reset" "agent"), can compute expected reward $J(\theta)$ by "rollouts" 

##### Find optimal parameters through global optimization
- $\theta^{*}=\arg \max _{\theta} J(\theta)$
