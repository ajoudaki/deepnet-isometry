# Isometry of activations and normalization 

This repository provides the code for paper: [On the impact of activation and normalization in obtaining isometric embeddings at initialization
](https://arxiv.org/abs/2305.18399)

Main concepts introduced in the paper are isometry and non-linearity strength: 

- **Isometry:**  Given PSD matrix $M$, isometry is defined as the ratio of geometric-mean, to arithmetic-mean of its eigenvalues: 
$\mathcal{I}(M) = \frac{\det(M)^{1/n} }{\frac1n Tr(M) }$
- **Non-linearity strength $\beta_0$:** Given activation $\sigma$ and its [Hermite coefficients](https://en.wikipedia.org/wiki/Hermite_polynomials) $c_0, c_1, \dots$, non-linearity strength $\beta_0$ is defined as $\beta_0 = \frac{c_1^2}{\Sigma_{k=1}^{\infty}c_k^2}.$

See the paper for more elaborate discussion of these concepts. 

**Structure**:
- `validations.ipynb`: The validations of theories about activations and normalization layers
- `training.ipynb`: the empirical results on predicting training SGD speed using non-linearity strength $\beta_0$
- `isometry_transformers.ipynb`: investigating isometry of different layers in pre-trained transformers, including GPT2 and BERT 
- `util.py`: some utility functions
- `requirements.txt`: the python packages necessary for running the notebooks, can be installed by `pip install -r requirements.txt`



