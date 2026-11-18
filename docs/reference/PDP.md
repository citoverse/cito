# Partial Dependence Plot (PDP)

Calculates the Partial Dependency Plot for one feature, either numeric
or categorical. Returns it as a plot.

## Usage

``` r
PDP(
  model,
  variable = NULL,
  data = NULL,
  ice = FALSE,
  resolution.ice = 20,
  plot = TRUE,
  parallel = FALSE,
  ...
)

# S3 method for class 'citodnn'
PDP(
  model,
  variable = NULL,
  data = NULL,
  ice = FALSE,
  resolution.ice = 20,
  plot = TRUE,
  parallel = FALSE,
  ...
)

# S3 method for class 'citodnnBootstrap'
PDP(
  model,
  variable = NULL,
  data = NULL,
  ice = FALSE,
  resolution.ice = 20,
  plot = TRUE,
  parallel = FALSE,
  ...
)
```

## Arguments

- model:

  a model created by [`dnn`](dnn.md)

- variable:

  variable (as a string) for which the PDP should be computed. If none
  is supplied, it is computed for all variables.

- data:

  new data on which the PDP should be computed. If NULL, the PDP is
  computed on the training data.

- ice:

  if TRUE, the Individual Conditional Expectation (ICE) curves are also
  shown

- resolution.ice:

  resolution (number of grid points) at which the ICE curves are
  computed

- plot:

  whether to plot the PDP

- parallel:

  whether to parallelize over the bootstrap models

- ...:

  arguments passed to [`predict`](https://rdrr.io/r/stats/predict.html)

## Value

A list of plots made with 'ggplot2', one for each selected variable.

## Description

Performs a partial dependence plot (PDP) estimation to analyze the
relationship between a selected feature and the model's predictions.

The PDP function estimates the partial function \\\hat{f}\_S\\:

\\\hat{f}\_S(x_S)=\frac{1}{n}\sum\_{i=1}^n\hat{f}(x_S,x^{(i)}\_{C})\\

using a Monte Carlo estimation, i.e. it computes the average prediction
over the data while the selected feature is held fixed at a given value
and the remaining features are kept at their observed values.

For categorical features, every observation is set to each level of the
feature in turn, the average prediction per level is calculated, and the
result is shown as a bar plot.

If `ice = TRUE`, the Individual Conditional Expectation (ICE) curves are
also shown, with the PDP highlighted in yellow. Each ICE curve
illustrates how the prediction for a single observation changes as the
feature varies. ICE curves are computed on a value grid rather than at
every observed feature value, and are not available for categorical
features.

## See also

[`ALE`](ALE.md)

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris)

PDP(nn.fit, variable = "Petal.Length")
}
#> Loss at epoch 1: 3.493913, lr: 0.01000

#> Loss at epoch 2: 0.165550, lr: 0.01000
#> Loss at epoch 3: 0.148354, lr: 0.01000
#> Loss at epoch 4: 0.267113, lr: 0.01000
#> Loss at epoch 5: 0.277618, lr: 0.01000
#> Loss at epoch 6: 0.200132, lr: 0.01000
#> Loss at epoch 7: 0.203337, lr: 0.01000
#> Loss at epoch 8: 0.168046, lr: 0.01000
#> Loss at epoch 9: 0.160961, lr: 0.01000
#> Loss at epoch 10: 0.142151, lr: 0.01000
#> Loss at epoch 11: 0.142992, lr: 0.01000
#> Loss at epoch 12: 0.137267, lr: 0.01000
#> Loss at epoch 13: 0.132910, lr: 0.01000
#> Loss at epoch 14: 0.169810, lr: 0.01000
#> Loss at epoch 15: 0.186857, lr: 0.01000
#> Loss at epoch 16: 0.357855, lr: 0.01000
#> Loss at epoch 17: 0.139162, lr: 0.01000
#> Loss at epoch 18: 0.158268, lr: 0.01000
#> Loss at epoch 19: 0.178185, lr: 0.01000
#> Loss at epoch 20: 0.252760, lr: 0.01000
#> Loss at epoch 21: 0.154684, lr: 0.01000
#> Loss at epoch 22: 0.117982, lr: 0.01000
#> Loss at epoch 23: 0.335422, lr: 0.01000
#> Loss at epoch 24: 0.274726, lr: 0.01000
#> Loss at epoch 25: 0.119752, lr: 0.01000
#> Loss at epoch 26: 0.129957, lr: 0.01000
#> Loss at epoch 27: 0.297912, lr: 0.01000
#> Loss at epoch 28: 0.181735, lr: 0.01000
#> Loss at epoch 29: 0.174297, lr: 0.01000
#> Loss at epoch 30: 0.136931, lr: 0.01000
#> Loss at epoch 31: 0.128013, lr: 0.01000
#> Loss at epoch 32: 0.111283, lr: 0.01000
#> Loss at epoch 33: 0.140515, lr: 0.01000
#> Loss at epoch 34: 0.111067, lr: 0.01000
#> Loss at epoch 35: 0.120084, lr: 0.01000
#> Loss at epoch 36: 0.195256, lr: 0.01000
#> Loss at epoch 37: 0.225211, lr: 0.01000
#> Loss at epoch 38: 0.139601, lr: 0.01000
#> Loss at epoch 39: 0.144723, lr: 0.01000
#> Loss at epoch 40: 0.142679, lr: 0.01000
#> Loss at epoch 41: 0.152094, lr: 0.01000
#> Loss at epoch 42: 0.167447, lr: 0.01000
#> Loss at epoch 43: 0.166415, lr: 0.01000
#> Loss at epoch 44: 0.110415, lr: 0.01000
#> Loss at epoch 45: 0.125671, lr: 0.01000
#> Loss at epoch 46: 0.151080, lr: 0.01000
#> Loss at epoch 47: 0.142388, lr: 0.01000
#> Loss at epoch 48: 0.171630, lr: 0.01000
#> Loss at epoch 49: 0.208987, lr: 0.01000
#> Loss at epoch 50: 0.197812, lr: 0.01000
#> Loss at epoch 51: 0.116922, lr: 0.01000
#> Loss at epoch 52: 0.175013, lr: 0.01000
#> Loss at epoch 53: 0.130506, lr: 0.01000
#> Loss at epoch 54: 0.145471, lr: 0.01000
#> Loss at epoch 55: 0.139491, lr: 0.01000
#> Loss at epoch 56: 0.115283, lr: 0.01000
#> Loss at epoch 57: 0.157710, lr: 0.01000
#> Loss at epoch 58: 0.171651, lr: 0.01000
#> Loss at epoch 59: 0.159542, lr: 0.01000
#> Loss at epoch 60: 0.123145, lr: 0.01000
#> Loss at epoch 61: 0.121279, lr: 0.01000
#> Loss at epoch 62: 0.171066, lr: 0.01000
#> Loss at epoch 63: 0.260192, lr: 0.01000
#> Loss at epoch 64: 0.111687, lr: 0.01000
#> Loss at epoch 65: 0.164341, lr: 0.01000
#> Loss at epoch 66: 0.126802, lr: 0.01000
#> Loss at epoch 67: 0.209207, lr: 0.01000
#> Loss at epoch 68: 0.204702, lr: 0.01000
#> Loss at epoch 69: 0.181967, lr: 0.01000
#> Loss at epoch 70: 0.191331, lr: 0.01000
#> Loss at epoch 71: 0.148004, lr: 0.01000
#> Loss at epoch 72: 0.190106, lr: 0.01000
#> Loss at epoch 73: 0.108358, lr: 0.01000
#> Loss at epoch 74: 0.131573, lr: 0.01000
#> Loss at epoch 75: 0.127027, lr: 0.01000
#> Loss at epoch 76: 0.165078, lr: 0.01000
#> Loss at epoch 77: 0.118841, lr: 0.01000
#> Loss at epoch 78: 0.121878, lr: 0.01000
#> Loss at epoch 79: 0.128868, lr: 0.01000
#> Loss at epoch 80: 0.133490, lr: 0.01000
#> Loss at epoch 81: 0.134318, lr: 0.01000
#> Loss at epoch 82: 0.107645, lr: 0.01000
#> Loss at epoch 83: 0.203882, lr: 0.01000
#> Loss at epoch 84: 0.134695, lr: 0.01000
#> Loss at epoch 85: 0.160067, lr: 0.01000
#> Loss at epoch 86: 0.267251, lr: 0.01000
#> Loss at epoch 87: 0.149095, lr: 0.01000
#> Loss at epoch 88: 0.123882, lr: 0.01000
#> Loss at epoch 89: 0.121558, lr: 0.01000
#> Loss at epoch 90: 0.123816, lr: 0.01000
#> Loss at epoch 91: 0.104826, lr: 0.01000
#> Loss at epoch 92: 0.125039, lr: 0.01000
#> Loss at epoch 93: 0.166805, lr: 0.01000
#> Loss at epoch 94: 0.114725, lr: 0.01000
#> Loss at epoch 95: 0.129813, lr: 0.01000
#> Loss at epoch 96: 0.114272, lr: 0.01000
#> Loss at epoch 97: 0.163984, lr: 0.01000
#> Loss at epoch 98: 0.106345, lr: 0.01000
#> Loss at epoch 99: 0.107780, lr: 0.01000
#> Loss at epoch 100: 0.135817, lr: 0.01000

# }
```
