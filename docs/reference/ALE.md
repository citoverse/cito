# Accumulated Local Effect Plot (ALE)

Performs an ALE for one or more features.

## Usage

``` r
ALE(
  model,
  variable = NULL,
  data = NULL,
  type = "response",
  analytical = FALSE,
  center = FALSE,
  K = 10,
  ALE_type = c("equidistant", "quantile"),
  plot = TRUE,
  parallel = FALSE,
  ...
)

# S3 method for class 'citodnn'
ALE(
  model,
  variable = NULL,
  data = NULL,
  type = "response",
  analytical = TRUE,
  center = FALSE,
  K = 10,
  ALE_type = c("quantile", "equidistant"),
  plot = TRUE,
  parallel = FALSE,
  ...
)

# S3 method for class 'citodnnBootstrap'
ALE(
  model,
  variable = NULL,
  data = NULL,
  type = "response",
  analytical = TRUE,
  center = FALSE,
  K = 10,
  ALE_type = c("quantile", "equidistant"),
  plot = TRUE,
  parallel = FALSE,
  ...
)
```

## Arguments

- model:

  a model created by [`dnn`](dnn.md)

- variable:

  variable (as a string) for which the ALE should be computed. If none
  is supplied, it is computed for all variables.

- data:

  data on which the ALE is computed; if NULL, the training data is used

- type:

  ALE on which scale response or link, default is response

- analytical:

  Analytical ALE based on conditional effects or not

- center:

  center ALE or not (only available for analytical ALE)

- K:

  number of neighborhoods the original feature space is divided into

- ALE_type:

  method by which the feature bins (neighborhoods) are created

- plot:

  plot ALE or not

- parallel:

  parallelize over bootstrap models or not

- ...:

  arguments passed to [`predict`](https://rdrr.io/r/stats/predict.html)

## Value

A list of plots made with 'ggplot2' consisting of an individual plot for
each defined variable.

## Explanation

Accumulated Local Effect plots (ALE) quantify how the predictions change
when the features change. They are similar to partial dependency plots
but are more robust to feature collinearity.

## Mathematical details

If the defined variable is a numeric feature, the ALE is performed.
Here, the non centered effect for feature j with k equally distant
neighborhoods is defined as:

\\
\hat{\tilde{f}}\_{j,ALE}(x)=\sum\_{k=1}^{k_j(x)}\frac{1}{n_j(k)}\sum\_{i:x\_{j}^{(i)}\in{}N_j(k)}\left\[\hat{f}(z\_{k,j},x^{(i)}\_{\setminus{}j})-\hat{f}(z\_{k-1,j},x^{(i)}\_{\setminus{}j})\right\]\\

Where \\N_j(k)\\ is the k-th neighborhood and \\n_j(k)\\ is the number
of observations in the k-th neighborhood.

The last part of the equation,
\\\left\[\hat{f}(z\_{k,j},x^{(i)}\_{\setminus{}j})-\hat{f}(z\_{k-1,j},x^{(i)}\_{\setminus{}j})\right\]\\
represents the difference in model prediction when the value of feature
j is exchanged with the upper and lower border of the current
neighborhood.

## See also

[`PDP`](PDP.md)

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris)

ALE(nn.fit, variable = "Petal.Length")
}
#> Loss at epoch 1: 4.133062, lr: 0.01000

#> Loss at epoch 2: 0.198660, lr: 0.01000
#> Loss at epoch 3: 0.308893, lr: 0.01000
#> Loss at epoch 4: 0.492745, lr: 0.01000
#> Loss at epoch 5: 0.221361, lr: 0.01000
#> Loss at epoch 6: 0.141474, lr: 0.01000
#> Loss at epoch 7: 0.156454, lr: 0.01000
#> Loss at epoch 8: 0.357028, lr: 0.01000
#> Loss at epoch 9: 0.142769, lr: 0.01000
#> Loss at epoch 10: 0.231149, lr: 0.01000
#> Loss at epoch 11: 0.231651, lr: 0.01000
#> Loss at epoch 12: 0.179644, lr: 0.01000
#> Loss at epoch 13: 0.279894, lr: 0.01000
#> Loss at epoch 14: 0.256342, lr: 0.01000
#> Loss at epoch 15: 0.309539, lr: 0.01000
#> Loss at epoch 16: 0.219900, lr: 0.01000
#> Loss at epoch 17: 0.163498, lr: 0.01000
#> Loss at epoch 18: 0.242580, lr: 0.01000
#> Loss at epoch 19: 0.131889, lr: 0.01000
#> Loss at epoch 20: 0.220039, lr: 0.01000
#> Loss at epoch 21: 0.196040, lr: 0.01000
#> Loss at epoch 22: 0.186878, lr: 0.01000
#> Loss at epoch 23: 0.140709, lr: 0.01000
#> Loss at epoch 24: 0.190116, lr: 0.01000
#> Loss at epoch 25: 0.298376, lr: 0.01000
#> Loss at epoch 26: 0.185756, lr: 0.01000
#> Loss at epoch 27: 0.224469, lr: 0.01000
#> Loss at epoch 28: 0.155850, lr: 0.01000
#> Loss at epoch 29: 0.168168, lr: 0.01000
#> Loss at epoch 30: 0.202955, lr: 0.01000
#> Loss at epoch 31: 0.263840, lr: 0.01000
#> Loss at epoch 32: 0.162559, lr: 0.01000
#> Loss at epoch 33: 0.136287, lr: 0.01000
#> Loss at epoch 34: 0.130841, lr: 0.01000
#> Loss at epoch 35: 0.156277, lr: 0.01000
#> Loss at epoch 36: 0.130158, lr: 0.01000
#> Loss at epoch 37: 0.140786, lr: 0.01000
#> Loss at epoch 38: 0.137379, lr: 0.01000
#> Loss at epoch 39: 0.154045, lr: 0.01000
#> Loss at epoch 40: 0.122024, lr: 0.01000
#> Loss at epoch 41: 0.153013, lr: 0.01000
#> Loss at epoch 42: 0.123950, lr: 0.01000
#> Loss at epoch 43: 0.129806, lr: 0.01000
#> Loss at epoch 44: 0.148276, lr: 0.01000
#> Loss at epoch 45: 0.130305, lr: 0.01000
#> Loss at epoch 46: 0.161422, lr: 0.01000
#> Loss at epoch 47: 0.136417, lr: 0.01000
#> Loss at epoch 48: 0.110883, lr: 0.01000
#> Loss at epoch 49: 0.191575, lr: 0.01000
#> Loss at epoch 50: 0.267317, lr: 0.01000
#> Loss at epoch 51: 0.123554, lr: 0.01000
#> Loss at epoch 52: 0.164400, lr: 0.01000
#> Loss at epoch 53: 0.229359, lr: 0.01000
#> Loss at epoch 54: 0.253268, lr: 0.01000
#> Loss at epoch 55: 0.174824, lr: 0.01000
#> Loss at epoch 56: 0.163012, lr: 0.01000
#> Loss at epoch 57: 0.159088, lr: 0.01000
#> Loss at epoch 58: 0.122461, lr: 0.01000
#> Loss at epoch 59: 0.143644, lr: 0.01000
#> Loss at epoch 60: 0.234793, lr: 0.01000
#> Loss at epoch 61: 0.155528, lr: 0.01000
#> Loss at epoch 62: 0.126108, lr: 0.01000
#> Loss at epoch 63: 0.177455, lr: 0.01000
#> Loss at epoch 64: 0.176817, lr: 0.01000
#> Loss at epoch 65: 0.115860, lr: 0.01000
#> Loss at epoch 66: 0.115514, lr: 0.01000
#> Loss at epoch 67: 0.112353, lr: 0.01000
#> Loss at epoch 68: 0.118895, lr: 0.01000
#> Loss at epoch 69: 0.139058, lr: 0.01000
#> Loss at epoch 70: 0.112961, lr: 0.01000
#> Loss at epoch 71: 0.155569, lr: 0.01000
#> Loss at epoch 72: 0.133239, lr: 0.01000
#> Loss at epoch 73: 0.127252, lr: 0.01000
#> Loss at epoch 74: 0.112130, lr: 0.01000
#> Loss at epoch 75: 0.160378, lr: 0.01000
#> Loss at epoch 76: 0.375632, lr: 0.01000
#> Loss at epoch 77: 0.144231, lr: 0.01000
#> Loss at epoch 78: 0.117092, lr: 0.01000
#> Loss at epoch 79: 0.128590, lr: 0.01000
#> Loss at epoch 80: 0.136595, lr: 0.01000
#> Loss at epoch 81: 0.119222, lr: 0.01000
#> Loss at epoch 82: 0.151231, lr: 0.01000
#> Loss at epoch 83: 0.157426, lr: 0.01000
#> Loss at epoch 84: 0.169059, lr: 0.01000
#> Loss at epoch 85: 0.112998, lr: 0.01000
#> Loss at epoch 86: 0.148308, lr: 0.01000
#> Loss at epoch 87: 0.155876, lr: 0.01000
#> Loss at epoch 88: 0.142132, lr: 0.01000
#> Loss at epoch 89: 0.133715, lr: 0.01000
#> Loss at epoch 90: 0.131114, lr: 0.01000
#> Loss at epoch 91: 0.126465, lr: 0.01000
#> Loss at epoch 92: 0.203596, lr: 0.01000
#> Loss at epoch 93: 0.203023, lr: 0.01000
#> Loss at epoch 94: 0.125553, lr: 0.01000
#> Loss at epoch 95: 0.143504, lr: 0.01000
#> Loss at epoch 96: 0.123426, lr: 0.01000
#> Loss at epoch 97: 0.115125, lr: 0.01000
#> Loss at epoch 98: 0.159435, lr: 0.01000
#> Loss at epoch 99: 0.095720, lr: 0.01000
#> Loss at epoch 100: 0.102692, lr: 0.01000

# }
```
