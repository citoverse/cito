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
#> Loss at epoch 1: 8.240370, lr: 0.01000

#> Loss at epoch 2: 0.161552, lr: 0.01000
#> Loss at epoch 3: 0.165531, lr: 0.01000
#> Loss at epoch 4: 0.152306, lr: 0.01000
#> Loss at epoch 5: 0.159184, lr: 0.01000
#> Loss at epoch 6: 0.140654, lr: 0.01000
#> Loss at epoch 7: 0.140591, lr: 0.01000
#> Loss at epoch 8: 0.189794, lr: 0.01000
#> Loss at epoch 9: 0.155306, lr: 0.01000
#> Loss at epoch 10: 0.134698, lr: 0.01000
#> Loss at epoch 11: 0.139754, lr: 0.01000
#> Loss at epoch 12: 0.146963, lr: 0.01000
#> Loss at epoch 13: 0.155633, lr: 0.01000
#> Loss at epoch 14: 0.141824, lr: 0.01000
#> Loss at epoch 15: 0.175305, lr: 0.01000
#> Loss at epoch 16: 0.182765, lr: 0.01000
#> Loss at epoch 17: 0.141169, lr: 0.01000
#> Loss at epoch 18: 0.132948, lr: 0.01000
#> Loss at epoch 19: 0.144895, lr: 0.01000
#> Loss at epoch 20: 0.166582, lr: 0.01000
#> Loss at epoch 21: 0.154567, lr: 0.01000
#> Loss at epoch 22: 0.175165, lr: 0.01000
#> Loss at epoch 23: 0.122142, lr: 0.01000
#> Loss at epoch 24: 0.144362, lr: 0.01000
#> Loss at epoch 25: 0.182364, lr: 0.01000
#> Loss at epoch 26: 0.127609, lr: 0.01000
#> Loss at epoch 27: 0.142924, lr: 0.01000
#> Loss at epoch 28: 0.131521, lr: 0.01000
#> Loss at epoch 29: 0.151307, lr: 0.01000
#> Loss at epoch 30: 0.119252, lr: 0.01000
#> Loss at epoch 31: 0.174257, lr: 0.01000
#> Loss at epoch 32: 0.138618, lr: 0.01000
#> Loss at epoch 33: 0.143725, lr: 0.01000
#> Loss at epoch 34: 0.217856, lr: 0.01000
#> Loss at epoch 35: 0.138308, lr: 0.01000
#> Loss at epoch 36: 0.184583, lr: 0.01000
#> Loss at epoch 37: 0.124516, lr: 0.01000
#> Loss at epoch 38: 0.173207, lr: 0.01000
#> Loss at epoch 39: 0.148287, lr: 0.01000
#> Loss at epoch 40: 0.164703, lr: 0.01000
#> Loss at epoch 41: 0.116158, lr: 0.01000
#> Loss at epoch 42: 0.131927, lr: 0.01000
#> Loss at epoch 43: 0.112757, lr: 0.01000
#> Loss at epoch 44: 0.130076, lr: 0.01000
#> Loss at epoch 45: 0.151813, lr: 0.01000
#> Loss at epoch 46: 0.114471, lr: 0.01000
#> Loss at epoch 47: 0.134920, lr: 0.01000
#> Loss at epoch 48: 0.110376, lr: 0.01000
#> Loss at epoch 49: 0.110678, lr: 0.01000
#> Loss at epoch 50: 0.112920, lr: 0.01000
#> Loss at epoch 51: 0.130650, lr: 0.01000
#> Loss at epoch 52: 0.150889, lr: 0.01000
#> Loss at epoch 53: 0.138823, lr: 0.01000
#> Loss at epoch 54: 0.170651, lr: 0.01000
#> Loss at epoch 55: 0.109749, lr: 0.01000
#> Loss at epoch 56: 0.159096, lr: 0.01000
#> Loss at epoch 57: 0.134429, lr: 0.01000
#> Loss at epoch 58: 0.154904, lr: 0.01000
#> Loss at epoch 59: 0.127312, lr: 0.01000
#> Loss at epoch 60: 0.129628, lr: 0.01000
#> Loss at epoch 61: 0.145166, lr: 0.01000
#> Loss at epoch 62: 0.172856, lr: 0.01000
#> Loss at epoch 63: 0.117008, lr: 0.01000
#> Loss at epoch 64: 0.104005, lr: 0.01000
#> Loss at epoch 65: 0.121906, lr: 0.01000
#> Loss at epoch 66: 0.154362, lr: 0.01000
#> Loss at epoch 67: 0.166254, lr: 0.01000
#> Loss at epoch 68: 0.258008, lr: 0.01000
#> Loss at epoch 69: 0.132822, lr: 0.01000
#> Loss at epoch 70: 0.104390, lr: 0.01000
#> Loss at epoch 71: 0.144974, lr: 0.01000
#> Loss at epoch 72: 0.125812, lr: 0.01000
#> Loss at epoch 73: 0.132556, lr: 0.01000
#> Loss at epoch 74: 0.106472, lr: 0.01000
#> Loss at epoch 75: 0.156821, lr: 0.01000
#> Loss at epoch 76: 0.298801, lr: 0.01000
#> Loss at epoch 77: 0.131193, lr: 0.01000
#> Loss at epoch 78: 0.117236, lr: 0.01000
#> Loss at epoch 79: 0.140375, lr: 0.01000
#> Loss at epoch 80: 0.104779, lr: 0.01000
#> Loss at epoch 81: 0.148468, lr: 0.01000
#> Loss at epoch 82: 0.102939, lr: 0.01000
#> Loss at epoch 83: 0.240818, lr: 0.01000
#> Loss at epoch 84: 0.186349, lr: 0.01000
#> Loss at epoch 85: 0.148747, lr: 0.01000
#> Loss at epoch 86: 0.242672, lr: 0.01000
#> Loss at epoch 87: 0.102463, lr: 0.01000
#> Loss at epoch 88: 0.119630, lr: 0.01000
#> Loss at epoch 89: 0.116775, lr: 0.01000
#> Loss at epoch 90: 0.110028, lr: 0.01000
#> Loss at epoch 91: 0.108358, lr: 0.01000
#> Loss at epoch 92: 0.098841, lr: 0.01000
#> Loss at epoch 93: 0.110693, lr: 0.01000
#> Loss at epoch 94: 0.151225, lr: 0.01000
#> Loss at epoch 95: 0.127100, lr: 0.01000
#> Loss at epoch 96: 0.121911, lr: 0.01000
#> Loss at epoch 97: 0.096995, lr: 0.01000
#> Loss at epoch 98: 0.178196, lr: 0.01000
#> Loss at epoch 99: 0.127936, lr: 0.01000
#> Loss at epoch 100: 0.120012, lr: 0.01000
#> Summary of Deep Neural Network Model
#> 
#> 
#> ── Feature Importance 
#>  
#>                  Importance Std.Err Z value Pr(>|z|)    
#> Sepal.Width ->       0.0487  0.0125    3.88   0.0001 ***
#> Petal.Length ->      1.3237  0.3067    4.32  1.6e-05 ***
#> Petal.Width ->       0.0533  0.0432    1.23   0.2172    
#> ---
#> Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#> 
#> 
#> 
#> ── Average Conditional Effects 
#>                      ACE Std.Err Z value Pr(>|z|)    
#> Sepal.Width ->    0.4951  0.0604    8.20  2.5e-16 ***
#> Petal.Length ->   0.6407  0.0728    8.81  < 2e-16 ***
#> Petal.Width ->   -0.2650  0.1387   -1.91    0.056 .  
#> ---
#> Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# }
```
