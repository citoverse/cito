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
#> Loss at epoch 1: 3.866420, lr: 0.01000

#> Loss at epoch 2: 0.234258, lr: 0.01000
#> Loss at epoch 3: 0.370444, lr: 0.01000
#> Loss at epoch 4: 0.198731, lr: 0.01000
#> Loss at epoch 5: 0.334092, lr: 0.01000
#> Loss at epoch 6: 0.195311, lr: 0.01000
#> Loss at epoch 7: 0.201337, lr: 0.01000
#> Loss at epoch 8: 0.235897, lr: 0.01000
#> Loss at epoch 9: 0.173819, lr: 0.01000
#> Loss at epoch 10: 0.132383, lr: 0.01000
#> Loss at epoch 11: 0.189515, lr: 0.01000
#> Loss at epoch 12: 0.135725, lr: 0.01000
#> Loss at epoch 13: 0.291763, lr: 0.01000
#> Loss at epoch 14: 0.147263, lr: 0.01000
#> Loss at epoch 15: 0.275358, lr: 0.01000
#> Loss at epoch 16: 0.148327, lr: 0.01000
#> Loss at epoch 17: 0.164146, lr: 0.01000
#> Loss at epoch 18: 0.159690, lr: 0.01000
#> Loss at epoch 19: 0.173855, lr: 0.01000
#> Loss at epoch 20: 0.186329, lr: 0.01000
#> Loss at epoch 21: 0.199985, lr: 0.01000
#> Loss at epoch 22: 0.137265, lr: 0.01000
#> Loss at epoch 23: 0.174208, lr: 0.01000
#> Loss at epoch 24: 0.142945, lr: 0.01000
#> Loss at epoch 25: 0.153674, lr: 0.01000
#> Loss at epoch 26: 0.185146, lr: 0.01000
#> Loss at epoch 27: 0.167225, lr: 0.01000
#> Loss at epoch 28: 0.228078, lr: 0.01000
#> Loss at epoch 29: 0.189768, lr: 0.01000
#> Loss at epoch 30: 0.178881, lr: 0.01000
#> Loss at epoch 31: 0.136810, lr: 0.01000
#> Loss at epoch 32: 0.145890, lr: 0.01000
#> Loss at epoch 33: 0.146021, lr: 0.01000
#> Loss at epoch 34: 0.296884, lr: 0.01000
#> Loss at epoch 35: 0.135385, lr: 0.01000
#> Loss at epoch 36: 0.162278, lr: 0.01000
#> Loss at epoch 37: 0.126019, lr: 0.01000
#> Loss at epoch 38: 0.152524, lr: 0.01000
#> Loss at epoch 39: 0.168497, lr: 0.01000
#> Loss at epoch 40: 0.245844, lr: 0.01000
#> Loss at epoch 41: 0.155806, lr: 0.01000
#> Loss at epoch 42: 0.110159, lr: 0.01000
#> Loss at epoch 43: 0.113476, lr: 0.01000
#> Loss at epoch 44: 0.185212, lr: 0.01000
#> Loss at epoch 45: 0.132524, lr: 0.01000
#> Loss at epoch 46: 0.143656, lr: 0.01000
#> Loss at epoch 47: 0.142250, lr: 0.01000
#> Loss at epoch 48: 0.222144, lr: 0.01000
#> Loss at epoch 49: 0.137827, lr: 0.01000
#> Loss at epoch 50: 0.159686, lr: 0.01000
#> Loss at epoch 51: 0.195173, lr: 0.01000
#> Loss at epoch 52: 0.129429, lr: 0.01000
#> Loss at epoch 53: 0.201530, lr: 0.01000
#> Loss at epoch 54: 0.170445, lr: 0.01000
#> Loss at epoch 55: 0.279729, lr: 0.01000
#> Loss at epoch 56: 0.157466, lr: 0.01000
#> Loss at epoch 57: 0.167470, lr: 0.01000
#> Loss at epoch 58: 0.164595, lr: 0.01000
#> Loss at epoch 59: 0.113692, lr: 0.01000
#> Loss at epoch 60: 0.140160, lr: 0.01000
#> Loss at epoch 61: 0.176282, lr: 0.01000
#> Loss at epoch 62: 0.216001, lr: 0.01000
#> Loss at epoch 63: 0.149162, lr: 0.01000
#> Loss at epoch 64: 0.148194, lr: 0.01000
#> Loss at epoch 65: 0.130076, lr: 0.01000
#> Loss at epoch 66: 0.117276, lr: 0.01000
#> Loss at epoch 67: 0.147498, lr: 0.01000
#> Loss at epoch 68: 0.123835, lr: 0.01000
#> Loss at epoch 69: 0.163302, lr: 0.01000
#> Loss at epoch 70: 0.211621, lr: 0.01000
#> Loss at epoch 71: 0.185477, lr: 0.01000
#> Loss at epoch 72: 0.175082, lr: 0.01000
#> Loss at epoch 73: 0.201842, lr: 0.01000
#> Loss at epoch 74: 0.226234, lr: 0.01000
#> Loss at epoch 75: 0.252262, lr: 0.01000
#> Loss at epoch 76: 0.140377, lr: 0.01000
#> Loss at epoch 77: 0.131470, lr: 0.01000
#> Loss at epoch 78: 0.113801, lr: 0.01000
#> Loss at epoch 79: 0.135757, lr: 0.01000
#> Loss at epoch 80: 0.164330, lr: 0.01000
#> Loss at epoch 81: 0.156980, lr: 0.01000
#> Loss at epoch 82: 0.154314, lr: 0.01000
#> Loss at epoch 83: 0.103820, lr: 0.01000
#> Loss at epoch 84: 0.142576, lr: 0.01000
#> Loss at epoch 85: 0.226980, lr: 0.01000
#> Loss at epoch 86: 0.191171, lr: 0.01000
#> Loss at epoch 87: 0.113399, lr: 0.01000
#> Loss at epoch 88: 0.121433, lr: 0.01000
#> Loss at epoch 89: 0.158810, lr: 0.01000
#> Loss at epoch 90: 0.156285, lr: 0.01000
#> Loss at epoch 91: 0.135744, lr: 0.01000
#> Loss at epoch 92: 0.121272, lr: 0.01000
#> Loss at epoch 93: 0.201970, lr: 0.01000
#> Loss at epoch 94: 0.158616, lr: 0.01000
#> Loss at epoch 95: 0.133237, lr: 0.01000
#> Loss at epoch 96: 0.264144, lr: 0.01000
#> Loss at epoch 97: 0.164422, lr: 0.01000
#> Loss at epoch 98: 0.113050, lr: 0.01000
#> Loss at epoch 99: 0.157334, lr: 0.01000
#> Loss at epoch 100: 0.249578, lr: 0.01000
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
