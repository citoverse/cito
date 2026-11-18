# Visualize training of Neural Network

After training a model with cito, this function helps to analyze the
training process and decide on best performing model. Creates a 'plotly'
figure which allows to zoom in and out on training graph

## Usage

``` r
analyze_training(object)
```

## Arguments

- object:

  a model created by [`dnn`](dnn.md), [`cnn`](cnn.md) or [`mmn`](mmn.md)

## Value

a 'plotly' figure

## Details

The baseline loss is the most important reference. If the model was not
able to achieve a better (lower) loss than the baseline (which is the
loss for a intercept only model), the model probably did not converge.
Possible reasons include an improper learning rate, too few epochs, or
too much regularization. See the [`?dnn`](dnn.md) help or the
[`vignette("B-Training_neural_networks")`](../articles/B-Training_neural_networks.md).

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)
set.seed(222)
validation_set<- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,],validation = 0.1)

# show zoomable plot of training and validation losses
analyze_training(nn.fit)

# Use model on validation set
predictions <- predict(nn.fit, iris[validation_set,])

# Scatterplot
plot(iris[validation_set,]$Sepal.Length,predictions)
}
#> Loss at epoch 1: training: 7.704, validation: 0.232, lr: 0.01000

#> Loss at epoch 2: training: 0.202, validation: 0.244, lr: 0.01000
#> Loss at epoch 3: training: 0.231, validation: 0.272, lr: 0.01000
#> Loss at epoch 4: training: 0.215, validation: 0.160, lr: 0.01000
#> Loss at epoch 5: training: 0.219, validation: 0.153, lr: 0.01000
#> Loss at epoch 6: training: 0.162, validation: 0.231, lr: 0.01000
#> Loss at epoch 7: training: 0.158, validation: 0.232, lr: 0.01000
#> Loss at epoch 8: training: 0.199, validation: 0.245, lr: 0.01000
#> Loss at epoch 9: training: 0.175, validation: 0.517, lr: 0.01000
#> Loss at epoch 10: training: 0.217, validation: 0.145, lr: 0.01000
#> Loss at epoch 11: training: 0.185, validation: 0.134, lr: 0.01000
#> Loss at epoch 12: training: 0.152, validation: 0.367, lr: 0.01000
#> Loss at epoch 13: training: 0.185, validation: 0.137, lr: 0.01000
#> Loss at epoch 14: training: 0.321, validation: 0.671, lr: 0.01000
#> Loss at epoch 15: training: 0.308, validation: 0.489, lr: 0.01000
#> Loss at epoch 16: training: 0.194, validation: 0.455, lr: 0.01000
#> Loss at epoch 17: training: 0.257, validation: 0.198, lr: 0.01000
#> Loss at epoch 18: training: 0.144, validation: 0.119, lr: 0.01000
#> Loss at epoch 19: training: 0.196, validation: 0.151, lr: 0.01000
#> Loss at epoch 20: training: 0.129, validation: 0.246, lr: 0.01000
#> Loss at epoch 21: training: 0.130, validation: 0.120, lr: 0.01000
#> Loss at epoch 22: training: 0.127, validation: 0.157, lr: 0.01000
#> Loss at epoch 23: training: 0.202, validation: 0.111, lr: 0.01000
#> Loss at epoch 24: training: 0.212, validation: 0.563, lr: 0.01000
#> Loss at epoch 25: training: 0.155, validation: 0.126, lr: 0.01000
#> Loss at epoch 26: training: 0.180, validation: 0.139, lr: 0.01000
#> Loss at epoch 27: training: 0.138, validation: 0.396, lr: 0.01000
#> Loss at epoch 28: training: 0.145, validation: 0.236, lr: 0.01000
#> Loss at epoch 29: training: 0.145, validation: 0.367, lr: 0.01000
#> Loss at epoch 30: training: 0.145, validation: 0.153, lr: 0.01000
#> Loss at epoch 31: training: 0.155, validation: 0.119, lr: 0.01000
#> Loss at epoch 32: training: 0.208, validation: 0.213, lr: 0.01000
#> Loss at epoch 33: training: 0.195, validation: 0.112, lr: 0.01000
#> Loss at epoch 34: training: 0.201, validation: 0.250, lr: 0.01000
#> Loss at epoch 35: training: 0.217, validation: 0.235, lr: 0.01000
#> Loss at epoch 36: training: 0.166, validation: 0.385, lr: 0.01000
#> Loss at epoch 37: training: 0.166, validation: 0.184, lr: 0.01000
#> Loss at epoch 38: training: 0.131, validation: 0.099, lr: 0.01000
#> Loss at epoch 39: training: 0.228, validation: 0.255, lr: 0.01000
#> Loss at epoch 40: training: 0.138, validation: 0.094, lr: 0.01000
#> Loss at epoch 41: training: 0.133, validation: 0.115, lr: 0.01000
#> Loss at epoch 42: training: 0.195, validation: 0.179, lr: 0.01000
#> Loss at epoch 43: training: 0.132, validation: 0.123, lr: 0.01000
#> Loss at epoch 44: training: 0.176, validation: 0.193, lr: 0.01000
#> Loss at epoch 45: training: 0.157, validation: 0.172, lr: 0.01000
#> Loss at epoch 46: training: 0.156, validation: 0.197, lr: 0.01000
#> Loss at epoch 47: training: 0.184, validation: 0.123, lr: 0.01000
#> Loss at epoch 48: training: 0.122, validation: 0.131, lr: 0.01000
#> Loss at epoch 49: training: 0.207, validation: 0.103, lr: 0.01000
#> Loss at epoch 50: training: 0.118, validation: 0.455, lr: 0.01000
#> Loss at epoch 51: training: 0.152, validation: 0.413, lr: 0.01000
#> Loss at epoch 52: training: 0.151, validation: 0.317, lr: 0.01000
#> Loss at epoch 53: training: 0.146, validation: 0.087, lr: 0.01000
#> Loss at epoch 54: training: 0.267, validation: 0.269, lr: 0.01000
#> Loss at epoch 55: training: 0.210, validation: 0.119, lr: 0.01000
#> Loss at epoch 56: training: 0.153, validation: 0.090, lr: 0.01000
#> Loss at epoch 57: training: 0.121, validation: 0.204, lr: 0.01000
#> Loss at epoch 58: training: 0.123, validation: 0.537, lr: 0.01000
#> Loss at epoch 59: training: 0.167, validation: 0.304, lr: 0.01000
#> Loss at epoch 60: training: 0.197, validation: 0.202, lr: 0.01000
#> Loss at epoch 61: training: 0.182, validation: 0.085, lr: 0.01000
#> Loss at epoch 62: training: 0.227, validation: 0.088, lr: 0.01000
#> Loss at epoch 63: training: 0.158, validation: 0.083, lr: 0.01000
#> Loss at epoch 64: training: 0.143, validation: 0.089, lr: 0.01000
#> Loss at epoch 65: training: 0.136, validation: 0.108, lr: 0.01000
#> Loss at epoch 66: training: 0.115, validation: 0.111, lr: 0.01000
#> Loss at epoch 67: training: 0.209, validation: 0.545, lr: 0.01000
#> Loss at epoch 68: training: 0.135, validation: 0.311, lr: 0.01000
#> Loss at epoch 69: training: 0.367, validation: 0.087, lr: 0.01000
#> Loss at epoch 70: training: 0.126, validation: 0.093, lr: 0.01000
#> Loss at epoch 71: training: 0.167, validation: 0.081, lr: 0.01000
#> Loss at epoch 72: training: 0.142, validation: 0.152, lr: 0.01000
#> Loss at epoch 73: training: 0.132, validation: 0.135, lr: 0.01000
#> Loss at epoch 74: training: 0.191, validation: 0.431, lr: 0.01000
#> Loss at epoch 75: training: 0.308, validation: 0.081, lr: 0.01000
#> Loss at epoch 76: training: 0.135, validation: 0.294, lr: 0.01000
#> Loss at epoch 77: training: 0.192, validation: 0.677, lr: 0.01000
#> Loss at epoch 78: training: 0.222, validation: 0.084, lr: 0.01000
#> Loss at epoch 79: training: 0.135, validation: 0.134, lr: 0.01000
#> Loss at epoch 80: training: 0.134, validation: 0.104, lr: 0.01000
#> Loss at epoch 81: training: 0.112, validation: 0.270, lr: 0.01000
#> Loss at epoch 82: training: 0.127, validation: 0.103, lr: 0.01000
#> Loss at epoch 83: training: 0.109, validation: 0.078, lr: 0.01000
#> Loss at epoch 84: training: 0.130, validation: 0.154, lr: 0.01000
#> Loss at epoch 85: training: 0.169, validation: 0.116, lr: 0.01000
#> Loss at epoch 86: training: 0.114, validation: 0.192, lr: 0.01000
#> Loss at epoch 87: training: 0.173, validation: 0.147, lr: 0.01000
#> Loss at epoch 88: training: 0.120, validation: 0.181, lr: 0.01000
#> Loss at epoch 89: training: 0.147, validation: 0.155, lr: 0.01000
#> Loss at epoch 90: training: 0.141, validation: 0.071, lr: 0.01000
#> Loss at epoch 91: training: 0.134, validation: 0.071, lr: 0.01000
#> Loss at epoch 92: training: 0.135, validation: 0.073, lr: 0.01000
#> Loss at epoch 93: training: 0.118, validation: 0.088, lr: 0.01000
#> Loss at epoch 94: training: 0.140, validation: 0.301, lr: 0.01000
#> Loss at epoch 95: training: 0.164, validation: 0.179, lr: 0.01000
#> Loss at epoch 96: training: 0.120, validation: 0.091, lr: 0.01000
#> Loss at epoch 97: training: 0.167, validation: 0.466, lr: 0.01000
#> Loss at epoch 98: training: 0.191, validation: 0.075, lr: 0.01000
#> Loss at epoch 99: training: 0.135, validation: 0.067, lr: 0.01000
#> Loss at epoch 100: training: 0.139, validation: 0.142, lr: 0.01000

# }
```
