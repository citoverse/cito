# Calculate average conditional effects

Average conditional effects calculate the local derivatives for each
observation for each feature. They are similar to marginal effects. And
the average of these conditional effects is an approximation of linear
effects (see Pichler and Hartig, 2023 for more details). You can use
this function to either calculate main effects (on the diagonal, take a
look at the example) or interaction effects (off-diagonals) between
features.

To obtain uncertainties for these effects, enable the bootstrapping
option in the `dnn(..)` function (see example).

## Usage

``` r
conditionalEffects(
  object,
  method = c("analytical", "finite"),
  subsample = FALSE,
  interactions = FALSE,
  epsilon = 0.1,
  device = c("cpu", "cuda", "mps"),
  indices = NULL,
  data = NULL,
  batchsize = NULL,
  type = "response",
  return_vars = FALSE,
  ...
)

# S3 method for class 'citodnn'
conditionalEffects(
  object,
  method = c("analytical", "finite"),
  subsample = FALSE,
  interactions = FALSE,
  epsilon = 0.1,
  device = c("cpu", "cuda", "mps"),
  indices = NULL,
  data = NULL,
  batchsize = NULL,
  type = "response",
  return_vars = FALSE,
  ...
)

# S3 method for class 'citodnnBootstrap'
conditionalEffects(
  object,
  method = c("analytical", "finite"),
  subsample = FALSE,
  interactions = FALSE,
  epsilon = 0.1,
  device = c("cpu", "cuda", "mps"),
  indices = NULL,
  data = NULL,
  batchsize = NULL,
  type = "response",
  return_vars = FALSE,
  ...
)
```

## Arguments

- object:

  object of class `citodnn`

- method:

  Calculate the conditional effects analytically or via the finite
  difference

- subsample:

  subsample data to decrease computational runtime, must be either FALSE
  or in the range of `[0,1]`

- interactions:

  calculate interactions or not (computationally expensive)

- epsilon:

  difference used to calculate derivatives

- device:

  which device

- indices:

  of variables for which the ACE are calculated

- data:

  data which is used to calculate the ACE

- batchsize:

  batchsize

- type:

  ACE on which scale (response or link)

- return_vars:

  return variances of variables (internally required)

- ...:

  additional arguments that are passed to the predict function

## Value

an S3 object of class `"conditionalEffects"` is returned. The list
consists of the following attributes:

- result:

  3-dimensional array with the raw results

- mean:

  Matrix, average conditional effects

- abs:

  Matrix, summed absolute conditional effects

- sd:

  Matrix, standard deviation of the conditional effects

## References

Scholbeck, C. A., Casalicchio, G., Molnar, C., Bischl, B., & Heumann, C.
(2022). Marginal effects for non-linear prediction functions. arXiv
preprint arXiv:2201.08837.

Pichler, M., & Hartig, F. (2023). Can predictive models be used for
causal inference?. arXiv preprint arXiv:2306.10551.

## Author

Maximilian Pichler

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# Build and train  Network
nn.fit = dnn(Sepal.Length~., data = datasets::iris)

# Calculate average conditional effects
ACE = conditionalEffects(nn.fit)

## Main effects (categorical features are not supported)
ACE

## With interaction effects:
ACE = conditionalEffects(nn.fit, interactions = TRUE)
## The off diagonal elements are the interaction effects
ACE[[1]]$mean
## ACE is a list, elements correspond to the number of response classes
## Sepal.length == 1 Response so we have only one
## list element in the ACE object

# Re-train NN with bootstrapping to obtain standard errors
nn.fit = dnn(Sepal.Length~., data = datasets::iris, bootstrap = 30L)
## The summary method calculates also the conditional effects, and if
## bootstrapping was used, it will also report standard errors and p-values:
summary(nn.fit)


}
#> Loss at epoch 1: 3.495571, lr: 0.01000

#> Loss at epoch 2: 0.403073, lr: 0.01000
#> Loss at epoch 3: 0.227997, lr: 0.01000
#> Loss at epoch 4: 0.325119, lr: 0.01000
#> Loss at epoch 5: 0.234093, lr: 0.01000
#> Loss at epoch 6: 0.338094, lr: 0.01000
#> Loss at epoch 7: 0.337047, lr: 0.01000
#> Loss at epoch 8: 0.181394, lr: 0.01000
#> Loss at epoch 9: 0.144518, lr: 0.01000
#> Loss at epoch 10: 0.163092, lr: 0.01000
#> Loss at epoch 11: 0.278988, lr: 0.01000
#> Loss at epoch 12: 0.177131, lr: 0.01000
#> Loss at epoch 13: 0.187383, lr: 0.01000
#> Loss at epoch 14: 0.179560, lr: 0.01000
#> Loss at epoch 15: 0.134141, lr: 0.01000
#> Loss at epoch 16: 0.160394, lr: 0.01000
#> Loss at epoch 17: 0.184941, lr: 0.01000
#> Loss at epoch 18: 0.136792, lr: 0.01000
#> Loss at epoch 19: 0.191172, lr: 0.01000
#> Loss at epoch 20: 0.198693, lr: 0.01000
#> Loss at epoch 21: 0.204919, lr: 0.01000
#> Loss at epoch 22: 0.132417, lr: 0.01000
#> Loss at epoch 23: 0.164022, lr: 0.01000
#> Loss at epoch 24: 0.433782, lr: 0.01000
#> Loss at epoch 25: 0.162285, lr: 0.01000
#> Loss at epoch 26: 0.138401, lr: 0.01000
#> Loss at epoch 27: 0.142592, lr: 0.01000
#> Loss at epoch 28: 0.196304, lr: 0.01000
#> Loss at epoch 29: 0.150156, lr: 0.01000
#> Loss at epoch 30: 0.130699, lr: 0.01000
#> Loss at epoch 31: 0.155975, lr: 0.01000
#> Loss at epoch 32: 0.130619, lr: 0.01000
#> Loss at epoch 33: 0.513178, lr: 0.01000
#> Loss at epoch 34: 0.260187, lr: 0.01000
#> Loss at epoch 35: 0.175384, lr: 0.01000
#> Loss at epoch 36: 0.152960, lr: 0.01000
#> Loss at epoch 37: 0.161398, lr: 0.01000
#> Loss at epoch 38: 0.193205, lr: 0.01000
#> Loss at epoch 39: 0.164674, lr: 0.01000
#> Loss at epoch 40: 0.146532, lr: 0.01000
#> Loss at epoch 41: 0.127841, lr: 0.01000
#> Loss at epoch 42: 0.114656, lr: 0.01000
#> Loss at epoch 43: 0.115106, lr: 0.01000
#> Loss at epoch 44: 0.129272, lr: 0.01000
#> Loss at epoch 45: 0.129459, lr: 0.01000
#> Loss at epoch 46: 0.131002, lr: 0.01000
#> Loss at epoch 47: 0.156157, lr: 0.01000
#> Loss at epoch 48: 0.118366, lr: 0.01000
#> Loss at epoch 49: 0.187427, lr: 0.01000
#> Loss at epoch 50: 0.152946, lr: 0.01000
#> Loss at epoch 51: 0.134873, lr: 0.01000
#> Loss at epoch 52: 0.174101, lr: 0.01000
#> Loss at epoch 53: 0.130667, lr: 0.01000
#> Loss at epoch 54: 0.141658, lr: 0.01000
#> Loss at epoch 55: 0.122140, lr: 0.01000
#> Loss at epoch 56: 0.118248, lr: 0.01000
#> Loss at epoch 57: 0.129116, lr: 0.01000
#> Loss at epoch 58: 0.386791, lr: 0.01000
#> Loss at epoch 59: 0.259534, lr: 0.01000
#> Loss at epoch 60: 0.124725, lr: 0.01000
#> Loss at epoch 61: 0.157113, lr: 0.01000
#> Loss at epoch 62: 0.170495, lr: 0.01000
#> Loss at epoch 63: 0.127937, lr: 0.01000
#> Loss at epoch 64: 0.147003, lr: 0.01000
#> Loss at epoch 65: 0.236262, lr: 0.01000
#> Loss at epoch 66: 0.118800, lr: 0.01000
#> Loss at epoch 67: 0.176087, lr: 0.01000
#> Loss at epoch 68: 0.184887, lr: 0.01000
#> Loss at epoch 69: 0.130517, lr: 0.01000
#> Loss at epoch 70: 0.111122, lr: 0.01000
#> Loss at epoch 71: 0.117378, lr: 0.01000
#> Loss at epoch 72: 0.109894, lr: 0.01000
#> Loss at epoch 73: 0.112309, lr: 0.01000
#> Loss at epoch 74: 0.139595, lr: 0.01000
#> Loss at epoch 75: 0.128516, lr: 0.01000
#> Loss at epoch 76: 0.140758, lr: 0.01000
#> Loss at epoch 77: 0.137276, lr: 0.01000
#> Loss at epoch 78: 0.135476, lr: 0.01000
#> Loss at epoch 79: 0.156879, lr: 0.01000
#> Loss at epoch 80: 0.186887, lr: 0.01000
#> Loss at epoch 81: 0.223893, lr: 0.01000
#> Loss at epoch 82: 0.136169, lr: 0.01000
#> Loss at epoch 83: 0.139724, lr: 0.01000
#> Loss at epoch 84: 0.163311, lr: 0.01000
#> Loss at epoch 85: 0.119046, lr: 0.01000
#> Loss at epoch 86: 0.122599, lr: 0.01000
#> Loss at epoch 87: 0.124426, lr: 0.01000
#> Loss at epoch 88: 0.280331, lr: 0.01000
#> Loss at epoch 89: 0.119590, lr: 0.01000
#> Loss at epoch 90: 0.119572, lr: 0.01000
#> Loss at epoch 91: 0.110078, lr: 0.01000
#> Loss at epoch 92: 0.147239, lr: 0.01000
#> Loss at epoch 93: 0.167505, lr: 0.01000
#> Loss at epoch 94: 0.138169, lr: 0.01000
#> Loss at epoch 95: 0.162219, lr: 0.01000
#> Loss at epoch 96: 0.114526, lr: 0.01000
#> Loss at epoch 97: 0.104417, lr: 0.01000
#> Loss at epoch 98: 0.118581, lr: 0.01000
#> Loss at epoch 99: 0.177616, lr: 0.01000
#> Loss at epoch 100: 0.105170, lr: 0.01000
#> Summary of Deep Neural Network Model
#> 
#> 
#> ── Feature Importance 
#>  
#>                  Importance Std.Err Z value Pr(>|z|)    
#> Sepal.Width ->       0.0490  0.0140    3.51  0.00044 ***
#> Petal.Length ->      1.3857  0.3293    4.21  2.6e-05 ***
#> Petal.Width ->       0.0508  0.0464    1.09  0.27401    
#> ---
#> Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#> 
#> 
#> 
#> ── Average Conditional Effects 
#>                      ACE Std.Err Z value Pr(>|z|)    
#> Sepal.Width ->    0.4945  0.0702    7.04  1.9e-12 ***
#> Petal.Length ->   0.6543  0.0766    8.54  < 2e-16 ***
#> Petal.Width ->   -0.2580  0.1337   -1.93    0.054 .  
#> ---
#> Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# }
```
