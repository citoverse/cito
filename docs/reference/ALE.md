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
#> Loss at epoch 1: 4.914242, lr: 0.01000

#> Loss at epoch 2: 0.176130, lr: 0.01000
#> Loss at epoch 3: 0.201305, lr: 0.01000
#> Loss at epoch 4: 0.156102, lr: 0.01000
#> Loss at epoch 5: 0.205768, lr: 0.01000
#> Loss at epoch 6: 0.302277, lr: 0.01000
#> Loss at epoch 7: 0.202699, lr: 0.01000
#> Loss at epoch 8: 0.216685, lr: 0.01000
#> Loss at epoch 9: 0.293162, lr: 0.01000
#> Loss at epoch 10: 0.185508, lr: 0.01000
#> Loss at epoch 11: 0.133667, lr: 0.01000
#> Loss at epoch 12: 0.229732, lr: 0.01000
#> Loss at epoch 13: 0.180576, lr: 0.01000
#> Loss at epoch 14: 0.152205, lr: 0.01000
#> Loss at epoch 15: 0.141884, lr: 0.01000
#> Loss at epoch 16: 0.127801, lr: 0.01000
#> Loss at epoch 17: 0.128789, lr: 0.01000
#> Loss at epoch 18: 0.152421, lr: 0.01000
#> Loss at epoch 19: 0.167167, lr: 0.01000
#> Loss at epoch 20: 0.387693, lr: 0.01000
#> Loss at epoch 21: 0.147846, lr: 0.01000
#> Loss at epoch 22: 0.127138, lr: 0.01000
#> Loss at epoch 23: 0.146197, lr: 0.01000
#> Loss at epoch 24: 0.149135, lr: 0.01000
#> Loss at epoch 25: 0.145470, lr: 0.01000
#> Loss at epoch 26: 0.157540, lr: 0.01000
#> Loss at epoch 27: 0.152055, lr: 0.01000
#> Loss at epoch 28: 0.222894, lr: 0.01000
#> Loss at epoch 29: 0.252881, lr: 0.01000
#> Loss at epoch 30: 0.253556, lr: 0.01000
#> Loss at epoch 31: 0.354123, lr: 0.01000
#> Loss at epoch 32: 0.132018, lr: 0.01000
#> Loss at epoch 33: 0.123437, lr: 0.01000
#> Loss at epoch 34: 0.183068, lr: 0.01000
#> Loss at epoch 35: 0.127581, lr: 0.01000
#> Loss at epoch 36: 0.145880, lr: 0.01000
#> Loss at epoch 37: 0.128653, lr: 0.01000
#> Loss at epoch 38: 0.164206, lr: 0.01000
#> Loss at epoch 39: 0.165303, lr: 0.01000
#> Loss at epoch 40: 0.182125, lr: 0.01000
#> Loss at epoch 41: 0.113952, lr: 0.01000
#> Loss at epoch 42: 0.181321, lr: 0.01000
#> Loss at epoch 43: 0.144857, lr: 0.01000
#> Loss at epoch 44: 0.230883, lr: 0.01000
#> Loss at epoch 45: 0.198376, lr: 0.01000
#> Loss at epoch 46: 0.173881, lr: 0.01000
#> Loss at epoch 47: 0.133808, lr: 0.01000
#> Loss at epoch 48: 0.174305, lr: 0.01000
#> Loss at epoch 49: 0.122228, lr: 0.01000
#> Loss at epoch 50: 0.126567, lr: 0.01000
#> Loss at epoch 51: 0.128183, lr: 0.01000
#> Loss at epoch 52: 0.250601, lr: 0.01000
#> Loss at epoch 53: 0.238612, lr: 0.01000
#> Loss at epoch 54: 0.133065, lr: 0.01000
#> Loss at epoch 55: 0.143712, lr: 0.01000
#> Loss at epoch 56: 0.200197, lr: 0.01000
#> Loss at epoch 57: 0.204981, lr: 0.01000
#> Loss at epoch 58: 0.119559, lr: 0.01000
#> Loss at epoch 59: 0.189971, lr: 0.01000
#> Loss at epoch 60: 0.162161, lr: 0.01000
#> Loss at epoch 61: 0.124953, lr: 0.01000
#> Loss at epoch 62: 0.104866, lr: 0.01000
#> Loss at epoch 63: 0.132726, lr: 0.01000
#> Loss at epoch 64: 0.120438, lr: 0.01000
#> Loss at epoch 65: 0.131755, lr: 0.01000
#> Loss at epoch 66: 0.147324, lr: 0.01000
#> Loss at epoch 67: 0.152975, lr: 0.01000
#> Loss at epoch 68: 0.108295, lr: 0.01000
#> Loss at epoch 69: 0.233508, lr: 0.01000
#> Loss at epoch 70: 0.165323, lr: 0.01000
#> Loss at epoch 71: 0.216021, lr: 0.01000
#> Loss at epoch 72: 0.246804, lr: 0.01000
#> Loss at epoch 73: 0.165989, lr: 0.01000
#> Loss at epoch 74: 0.137194, lr: 0.01000
#> Loss at epoch 75: 0.137962, lr: 0.01000
#> Loss at epoch 76: 0.192267, lr: 0.01000
#> Loss at epoch 77: 0.213936, lr: 0.01000
#> Loss at epoch 78: 0.104222, lr: 0.01000
#> Loss at epoch 79: 0.123245, lr: 0.01000
#> Loss at epoch 80: 0.104446, lr: 0.01000
#> Loss at epoch 81: 0.236492, lr: 0.01000
#> Loss at epoch 82: 0.283942, lr: 0.01000
#> Loss at epoch 83: 0.145503, lr: 0.01000
#> Loss at epoch 84: 0.153396, lr: 0.01000
#> Loss at epoch 85: 0.231004, lr: 0.01000
#> Loss at epoch 86: 0.115683, lr: 0.01000
#> Loss at epoch 87: 0.108684, lr: 0.01000
#> Loss at epoch 88: 0.126434, lr: 0.01000
#> Loss at epoch 89: 0.138856, lr: 0.01000
#> Loss at epoch 90: 0.145540, lr: 0.01000
#> Loss at epoch 91: 0.129825, lr: 0.01000
#> Loss at epoch 92: 0.118418, lr: 0.01000
#> Loss at epoch 93: 0.127346, lr: 0.01000
#> Loss at epoch 94: 0.146294, lr: 0.01000
#> Loss at epoch 95: 0.177933, lr: 0.01000
#> Loss at epoch 96: 0.225690, lr: 0.01000
#> Loss at epoch 97: 0.112504, lr: 0.01000
#> Loss at epoch 98: 0.135326, lr: 0.01000
#> Loss at epoch 99: 0.132041, lr: 0.01000
#> Loss at epoch 100: 0.264817, lr: 0.01000

# }
```
