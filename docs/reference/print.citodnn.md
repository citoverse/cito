# Print class citodnn

Print class citodnn

## Usage

``` r
# S3 method for class 'citodnn'
print(x, ...)

# S3 method for class 'citodnnBootstrap'
print(x, ...)
```

## Arguments

- x:

  a model created by [`dnn`](dnn.md)

- ...:

  additional arguments

## Value

original object x gets returned

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)
validation_set<- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])

# Structure of Neural Network
print(nn.fit)
}
#> Loss at epoch 1: 3.737705, lr: 0.01000

#> Loss at epoch 2: 0.218649, lr: 0.01000
#> Loss at epoch 3: 0.210346, lr: 0.01000
#> Loss at epoch 4: 0.217299, lr: 0.01000
#> Loss at epoch 5: 0.154313, lr: 0.01000
#> Loss at epoch 6: 0.241306, lr: 0.01000
#> Loss at epoch 7: 0.355227, lr: 0.01000
#> Loss at epoch 8: 0.178036, lr: 0.01000
#> Loss at epoch 9: 0.314221, lr: 0.01000
#> Loss at epoch 10: 0.169176, lr: 0.01000
#> Loss at epoch 11: 0.367349, lr: 0.01000
#> Loss at epoch 12: 0.211596, lr: 0.01000
#> Loss at epoch 13: 0.249384, lr: 0.01000
#> Loss at epoch 14: 0.180685, lr: 0.01000
#> Loss at epoch 15: 0.245464, lr: 0.01000
#> Loss at epoch 16: 0.215767, lr: 0.01000
#> Loss at epoch 17: 0.240539, lr: 0.01000
#> Loss at epoch 18: 0.211490, lr: 0.01000
#> Loss at epoch 19: 0.297613, lr: 0.01000
#> Loss at epoch 20: 0.231728, lr: 0.01000
#> Loss at epoch 21: 0.171675, lr: 0.01000
#> Loss at epoch 22: 0.139948, lr: 0.01000
#> Loss at epoch 23: 0.215202, lr: 0.01000
#> Loss at epoch 24: 0.139519, lr: 0.01000
#> Loss at epoch 25: 0.217700, lr: 0.01000
#> Loss at epoch 26: 0.127412, lr: 0.01000
#> Loss at epoch 27: 0.211010, lr: 0.01000
#> Loss at epoch 28: 0.331747, lr: 0.01000
#> Loss at epoch 29: 0.178273, lr: 0.01000
#> Loss at epoch 30: 0.232226, lr: 0.01000
#> Loss at epoch 31: 0.153916, lr: 0.01000
#> Loss at epoch 32: 0.220207, lr: 0.01000
#> Loss at epoch 33: 0.213160, lr: 0.01000
#> Loss at epoch 34: 0.174416, lr: 0.01000
#> Loss at epoch 35: 0.169368, lr: 0.01000
#> Loss at epoch 36: 0.146585, lr: 0.01000
#> Loss at epoch 37: 0.160818, lr: 0.01000
#> Loss at epoch 38: 0.161755, lr: 0.01000
#> Loss at epoch 39: 0.125707, lr: 0.01000
#> Loss at epoch 40: 0.134351, lr: 0.01000
#> Loss at epoch 41: 0.136901, lr: 0.01000
#> Loss at epoch 42: 0.122469, lr: 0.01000
#> Loss at epoch 43: 0.142511, lr: 0.01000
#> Loss at epoch 44: 0.242963, lr: 0.01000
#> Loss at epoch 45: 0.109623, lr: 0.01000
#> Loss at epoch 46: 0.205159, lr: 0.01000
#> Loss at epoch 47: 0.224710, lr: 0.01000
#> Loss at epoch 48: 0.195654, lr: 0.01000
#> Loss at epoch 49: 0.135476, lr: 0.01000
#> Loss at epoch 50: 0.127950, lr: 0.01000
#> Loss at epoch 51: 0.137908, lr: 0.01000
#> Loss at epoch 52: 0.106593, lr: 0.01000
#> Loss at epoch 53: 0.124437, lr: 0.01000
#> Loss at epoch 54: 0.172451, lr: 0.01000
#> Loss at epoch 55: 0.125250, lr: 0.01000
#> Loss at epoch 56: 0.142408, lr: 0.01000
#> Loss at epoch 57: 0.242427, lr: 0.01000
#> Loss at epoch 58: 0.204139, lr: 0.01000
#> Loss at epoch 59: 0.177707, lr: 0.01000
#> Loss at epoch 60: 0.200892, lr: 0.01000
#> Loss at epoch 61: 0.157913, lr: 0.01000
#> Loss at epoch 62: 0.192958, lr: 0.01000
#> Loss at epoch 63: 0.256160, lr: 0.01000
#> Loss at epoch 64: 0.141715, lr: 0.01000
#> Loss at epoch 65: 0.198260, lr: 0.01000
#> Loss at epoch 66: 0.214597, lr: 0.01000
#> Loss at epoch 67: 0.159912, lr: 0.01000
#> Loss at epoch 68: 0.164198, lr: 0.01000
#> Loss at epoch 69: 0.193808, lr: 0.01000
#> Loss at epoch 70: 0.149910, lr: 0.01000
#> Loss at epoch 71: 0.175848, lr: 0.01000
#> Loss at epoch 72: 0.234285, lr: 0.01000
#> Loss at epoch 73: 0.131777, lr: 0.01000
#> Loss at epoch 74: 0.116496, lr: 0.01000
#> Loss at epoch 75: 0.158338, lr: 0.01000
#> Loss at epoch 76: 0.141701, lr: 0.01000
#> Loss at epoch 77: 0.120723, lr: 0.01000
#> Loss at epoch 78: 0.136505, lr: 0.01000
#> Loss at epoch 79: 0.166402, lr: 0.01000
#> Loss at epoch 80: 0.136074, lr: 0.01000
#> Loss at epoch 81: 0.185172, lr: 0.01000
#> Loss at epoch 82: 0.114109, lr: 0.01000
#> Loss at epoch 83: 0.183417, lr: 0.01000
#> Loss at epoch 84: 0.120123, lr: 0.01000
#> Loss at epoch 85: 0.181159, lr: 0.01000
#> Loss at epoch 86: 0.121450, lr: 0.01000
#> Loss at epoch 87: 0.159683, lr: 0.01000
#> Loss at epoch 88: 0.120387, lr: 0.01000
#> Loss at epoch 89: 0.226890, lr: 0.01000
#> Loss at epoch 90: 0.157515, lr: 0.01000
#> Loss at epoch 91: 0.237093, lr: 0.01000
#> Loss at epoch 92: 0.145901, lr: 0.01000
#> Loss at epoch 93: 0.128487, lr: 0.01000
#> Loss at epoch 94: 0.116143, lr: 0.01000
#> Loss at epoch 95: 0.136838, lr: 0.01000
#> Loss at epoch 96: 0.108723, lr: 0.01000
#> Loss at epoch 97: 0.139561, lr: 0.01000
#> Loss at epoch 98: 0.268605, lr: 0.01000
#> Loss at epoch 99: 0.186004, lr: 0.01000
#> Loss at epoch 100: 0.144506, lr: 0.01000
#> dnn(formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width + 
#>     Species, data = datasets::iris[-validation_set, ])
#> An `nn_module` containing 2,901 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • l_1: <nn_linear> #300 parameters
#> • l_2: <nn_selu> #0 parameters
#> • l_3: <nn_linear> #2,550 parameters
#> • l_4: <nn_selu> #0 parameters
#> • l_5: <nn_linear> #51 parameters
# }
```
