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
#> Loss at epoch 1: training: 4.139, validation: 0.145, lr: 0.01000

#> Loss at epoch 2: training: 0.490, validation: 0.257, lr: 0.01000
#> Loss at epoch 3: training: 0.190, validation: 0.185, lr: 0.01000
#> Loss at epoch 4: training: 0.450, validation: 1.223, lr: 0.01000
#> Loss at epoch 5: training: 0.264, validation: 0.812, lr: 0.01000
#> Loss at epoch 6: training: 0.250, validation: 0.168, lr: 0.01000
#> Loss at epoch 7: training: 0.200, validation: 0.152, lr: 0.01000
#> Loss at epoch 8: training: 0.201, validation: 0.347, lr: 0.01000
#> Loss at epoch 9: training: 0.302, validation: 0.134, lr: 0.01000
#> Loss at epoch 10: training: 0.236, validation: 0.132, lr: 0.01000
#> Loss at epoch 11: training: 0.146, validation: 0.609, lr: 0.01000
#> Loss at epoch 12: training: 0.218, validation: 0.920, lr: 0.01000
#> Loss at epoch 13: training: 0.266, validation: 0.175, lr: 0.01000
#> Loss at epoch 14: training: 0.191, validation: 0.132, lr: 0.01000
#> Loss at epoch 15: training: 0.162, validation: 0.505, lr: 0.01000
#> Loss at epoch 16: training: 0.317, validation: 0.972, lr: 0.01000
#> Loss at epoch 17: training: 0.291, validation: 0.276, lr: 0.01000
#> Loss at epoch 18: training: 0.234, validation: 0.163, lr: 0.01000
#> Loss at epoch 19: training: 0.322, validation: 0.282, lr: 0.01000
#> Loss at epoch 20: training: 0.214, validation: 0.125, lr: 0.01000
#> Loss at epoch 21: training: 0.190, validation: 0.419, lr: 0.01000
#> Loss at epoch 22: training: 0.196, validation: 0.290, lr: 0.01000
#> Loss at epoch 23: training: 0.159, validation: 0.141, lr: 0.01000
#> Loss at epoch 24: training: 0.214, validation: 0.123, lr: 0.01000
#> Loss at epoch 25: training: 0.307, validation: 0.201, lr: 0.01000
#> Loss at epoch 26: training: 0.152, validation: 0.169, lr: 0.01000
#> Loss at epoch 27: training: 0.151, validation: 0.122, lr: 0.01000
#> Loss at epoch 28: training: 0.187, validation: 0.133, lr: 0.01000
#> Loss at epoch 29: training: 0.146, validation: 0.314, lr: 0.01000
#> Loss at epoch 30: training: 0.247, validation: 0.213, lr: 0.01000
#> Loss at epoch 31: training: 0.152, validation: 0.117, lr: 0.01000
#> Loss at epoch 32: training: 0.239, validation: 0.238, lr: 0.01000
#> Loss at epoch 33: training: 0.210, validation: 0.147, lr: 0.01000
#> Loss at epoch 34: training: 0.326, validation: 0.287, lr: 0.01000
#> Loss at epoch 35: training: 0.168, validation: 0.170, lr: 0.01000
#> Loss at epoch 36: training: 0.152, validation: 0.129, lr: 0.01000
#> Loss at epoch 37: training: 0.171, validation: 0.423, lr: 0.01000
#> Loss at epoch 38: training: 0.203, validation: 0.145, lr: 0.01000
#> Loss at epoch 39: training: 0.188, validation: 1.070, lr: 0.01000
#> Loss at epoch 40: training: 0.217, validation: 0.325, lr: 0.01000
#> Loss at epoch 41: training: 0.123, validation: 0.125, lr: 0.01000
#> Loss at epoch 42: training: 0.165, validation: 0.799, lr: 0.01000
#> Loss at epoch 43: training: 0.226, validation: 0.112, lr: 0.01000
#> Loss at epoch 44: training: 0.219, validation: 0.593, lr: 0.01000
#> Loss at epoch 45: training: 0.197, validation: 0.104, lr: 0.01000
#> Loss at epoch 46: training: 0.155, validation: 0.293, lr: 0.01000
#> Loss at epoch 47: training: 0.158, validation: 0.222, lr: 0.01000
#> Loss at epoch 48: training: 0.146, validation: 0.103, lr: 0.01000
#> Loss at epoch 49: training: 0.180, validation: 0.118, lr: 0.01000
#> Loss at epoch 50: training: 0.141, validation: 0.271, lr: 0.01000
#> Loss at epoch 51: training: 0.171, validation: 0.201, lr: 0.01000
#> Loss at epoch 52: training: 0.186, validation: 0.243, lr: 0.01000
#> Loss at epoch 53: training: 0.160, validation: 0.119, lr: 0.01000
#> Loss at epoch 54: training: 0.148, validation: 0.162, lr: 0.01000
#> Loss at epoch 55: training: 0.328, validation: 0.402, lr: 0.01000
#> Loss at epoch 56: training: 0.272, validation: 0.123, lr: 0.01000
#> Loss at epoch 57: training: 0.150, validation: 0.244, lr: 0.01000
#> Loss at epoch 58: training: 0.140, validation: 0.349, lr: 0.01000
#> Loss at epoch 59: training: 0.304, validation: 0.140, lr: 0.01000
#> Loss at epoch 60: training: 0.187, validation: 0.513, lr: 0.01000
#> Loss at epoch 61: training: 0.145, validation: 0.312, lr: 0.01000
#> Loss at epoch 62: training: 0.138, validation: 0.136, lr: 0.01000
#> Loss at epoch 63: training: 0.156, validation: 0.123, lr: 0.01000
#> Loss at epoch 64: training: 0.146, validation: 0.250, lr: 0.01000
#> Loss at epoch 65: training: 0.189, validation: 0.135, lr: 0.01000
#> Loss at epoch 66: training: 0.181, validation: 0.222, lr: 0.01000
#> Loss at epoch 67: training: 0.125, validation: 0.227, lr: 0.01000
#> Loss at epoch 68: training: 0.191, validation: 0.237, lr: 0.01000
#> Loss at epoch 69: training: 0.179, validation: 0.112, lr: 0.01000
#> Loss at epoch 70: training: 0.237, validation: 0.219, lr: 0.01000
#> Loss at epoch 71: training: 0.141, validation: 0.394, lr: 0.01000
#> Loss at epoch 72: training: 0.163, validation: 0.219, lr: 0.01000
#> Loss at epoch 73: training: 0.138, validation: 0.099, lr: 0.01000
#> Loss at epoch 74: training: 0.181, validation: 0.218, lr: 0.01000
#> Loss at epoch 75: training: 0.129, validation: 0.112, lr: 0.01000
#> Loss at epoch 76: training: 0.159, validation: 0.151, lr: 0.01000
#> Loss at epoch 77: training: 0.218, validation: 0.225, lr: 0.01000
#> Loss at epoch 78: training: 0.331, validation: 0.133, lr: 0.01000
#> Loss at epoch 79: training: 0.169, validation: 0.092, lr: 0.01000
#> Loss at epoch 80: training: 0.127, validation: 0.401, lr: 0.01000
#> Loss at epoch 81: training: 0.132, validation: 0.168, lr: 0.01000
#> Loss at epoch 82: training: 0.243, validation: 0.101, lr: 0.01000
#> Loss at epoch 83: training: 0.141, validation: 0.093, lr: 0.01000
#> Loss at epoch 84: training: 0.167, validation: 0.134, lr: 0.01000
#> Loss at epoch 85: training: 0.159, validation: 0.119, lr: 0.01000
#> Loss at epoch 86: training: 0.146, validation: 0.342, lr: 0.01000
#> Loss at epoch 87: training: 0.206, validation: 0.109, lr: 0.01000
#> Loss at epoch 88: training: 0.153, validation: 0.332, lr: 0.01000
#> Loss at epoch 89: training: 0.131, validation: 0.114, lr: 0.01000
#> Loss at epoch 90: training: 0.158, validation: 0.676, lr: 0.01000
#> Loss at epoch 91: training: 0.169, validation: 0.293, lr: 0.01000
#> Loss at epoch 92: training: 0.194, validation: 0.445, lr: 0.01000
#> Loss at epoch 93: training: 0.185, validation: 0.144, lr: 0.01000
#> Loss at epoch 94: training: 0.124, validation: 0.120, lr: 0.01000
#> Loss at epoch 95: training: 0.131, validation: 0.088, lr: 0.01000
#> Loss at epoch 96: training: 0.154, validation: 0.289, lr: 0.01000
#> Loss at epoch 97: training: 0.123, validation: 0.086, lr: 0.01000
#> Loss at epoch 98: training: 0.123, validation: 0.164, lr: 0.01000
#> Loss at epoch 99: training: 0.119, validation: 0.161, lr: 0.01000
#> Loss at epoch 100: training: 0.174, validation: 0.184, lr: 0.01000

# }
```
