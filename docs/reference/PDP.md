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
#> Loss at epoch 1: 4.606122, lr: 0.01000

#> Loss at epoch 2: 1.861340, lr: 0.01000
#> Loss at epoch 3: 0.275380, lr: 0.01000
#> Loss at epoch 4: 0.165700, lr: 0.01000
#> Loss at epoch 5: 0.324601, lr: 0.01000
#> Loss at epoch 6: 0.411893, lr: 0.01000
#> Loss at epoch 7: 0.212315, lr: 0.01000
#> Loss at epoch 8: 0.246592, lr: 0.01000
#> Loss at epoch 9: 0.207658, lr: 0.01000
#> Loss at epoch 10: 0.318335, lr: 0.01000
#> Loss at epoch 11: 0.206189, lr: 0.01000
#> Loss at epoch 12: 0.256269, lr: 0.01000
#> Loss at epoch 13: 0.209588, lr: 0.01000
#> Loss at epoch 14: 0.134961, lr: 0.01000
#> Loss at epoch 15: 0.130333, lr: 0.01000
#> Loss at epoch 16: 0.156714, lr: 0.01000
#> Loss at epoch 17: 0.143964, lr: 0.01000
#> Loss at epoch 18: 0.175631, lr: 0.01000
#> Loss at epoch 19: 0.179973, lr: 0.01000
#> Loss at epoch 20: 0.256741, lr: 0.01000
#> Loss at epoch 21: 0.170175, lr: 0.01000
#> Loss at epoch 22: 0.150789, lr: 0.01000
#> Loss at epoch 23: 0.169185, lr: 0.01000
#> Loss at epoch 24: 0.193105, lr: 0.01000
#> Loss at epoch 25: 0.197702, lr: 0.01000
#> Loss at epoch 26: 0.153822, lr: 0.01000
#> Loss at epoch 27: 0.145460, lr: 0.01000
#> Loss at epoch 28: 0.140340, lr: 0.01000
#> Loss at epoch 29: 0.183538, lr: 0.01000
#> Loss at epoch 30: 0.162210, lr: 0.01000
#> Loss at epoch 31: 0.130685, lr: 0.01000
#> Loss at epoch 32: 0.153436, lr: 0.01000
#> Loss at epoch 33: 0.142338, lr: 0.01000
#> Loss at epoch 34: 0.151305, lr: 0.01000
#> Loss at epoch 35: 0.118266, lr: 0.01000
#> Loss at epoch 36: 0.124101, lr: 0.01000
#> Loss at epoch 37: 0.153747, lr: 0.01000
#> Loss at epoch 38: 0.177563, lr: 0.01000
#> Loss at epoch 39: 0.131300, lr: 0.01000
#> Loss at epoch 40: 0.130929, lr: 0.01000
#> Loss at epoch 41: 0.227478, lr: 0.01000
#> Loss at epoch 42: 0.167781, lr: 0.01000
#> Loss at epoch 43: 0.153333, lr: 0.01000
#> Loss at epoch 44: 0.120087, lr: 0.01000
#> Loss at epoch 45: 0.114331, lr: 0.01000
#> Loss at epoch 46: 0.221238, lr: 0.01000
#> Loss at epoch 47: 0.147749, lr: 0.01000
#> Loss at epoch 48: 0.272247, lr: 0.01000
#> Loss at epoch 49: 0.151783, lr: 0.01000
#> Loss at epoch 50: 0.172041, lr: 0.01000
#> Loss at epoch 51: 0.106329, lr: 0.01000
#> Loss at epoch 52: 0.163754, lr: 0.01000
#> Loss at epoch 53: 0.159017, lr: 0.01000
#> Loss at epoch 54: 0.316706, lr: 0.01000
#> Loss at epoch 55: 0.252798, lr: 0.01000
#> Loss at epoch 56: 0.210757, lr: 0.01000
#> Loss at epoch 57: 0.345870, lr: 0.01000
#> Loss at epoch 58: 0.295486, lr: 0.01000
#> Loss at epoch 59: 0.132946, lr: 0.01000
#> Loss at epoch 60: 0.108467, lr: 0.01000
#> Loss at epoch 61: 0.109636, lr: 0.01000
#> Loss at epoch 62: 0.122107, lr: 0.01000
#> Loss at epoch 63: 0.156824, lr: 0.01000
#> Loss at epoch 64: 0.134927, lr: 0.01000
#> Loss at epoch 65: 0.097858, lr: 0.01000
#> Loss at epoch 66: 0.113777, lr: 0.01000
#> Loss at epoch 67: 0.107187, lr: 0.01000
#> Loss at epoch 68: 0.325509, lr: 0.01000
#> Loss at epoch 69: 0.311214, lr: 0.01000
#> Loss at epoch 70: 0.149914, lr: 0.01000
#> Loss at epoch 71: 0.218414, lr: 0.01000
#> Loss at epoch 72: 0.157143, lr: 0.01000
#> Loss at epoch 73: 0.123207, lr: 0.01000
#> Loss at epoch 74: 0.106913, lr: 0.01000
#> Loss at epoch 75: 0.144194, lr: 0.01000
#> Loss at epoch 76: 0.109224, lr: 0.01000
#> Loss at epoch 77: 0.132769, lr: 0.01000
#> Loss at epoch 78: 0.117631, lr: 0.01000
#> Loss at epoch 79: 0.150344, lr: 0.01000
#> Loss at epoch 80: 0.125002, lr: 0.01000
#> Loss at epoch 81: 0.211028, lr: 0.01000
#> Loss at epoch 82: 0.137392, lr: 0.01000
#> Loss at epoch 83: 0.106693, lr: 0.01000
#> Loss at epoch 84: 0.098619, lr: 0.01000
#> Loss at epoch 85: 0.104559, lr: 0.01000
#> Loss at epoch 86: 0.102997, lr: 0.01000
#> Loss at epoch 87: 0.116201, lr: 0.01000
#> Loss at epoch 88: 0.116986, lr: 0.01000
#> Loss at epoch 89: 0.226089, lr: 0.01000
#> Loss at epoch 90: 0.169897, lr: 0.01000
#> Loss at epoch 91: 0.169624, lr: 0.01000
#> Loss at epoch 92: 0.146927, lr: 0.01000
#> Loss at epoch 93: 0.115330, lr: 0.01000
#> Loss at epoch 94: 0.113338, lr: 0.01000
#> Loss at epoch 95: 0.137184, lr: 0.01000
#> Loss at epoch 96: 0.156633, lr: 0.01000
#> Loss at epoch 97: 0.136654, lr: 0.01000
#> Loss at epoch 98: 0.120192, lr: 0.01000
#> Loss at epoch 99: 0.110320, lr: 0.01000
#> Loss at epoch 100: 0.101586, lr: 0.01000

# }
```
