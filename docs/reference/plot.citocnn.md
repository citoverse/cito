# Plot a fitted CNN model

This function plots the architecture of a Convolutional Neural Network
(CNN) model created using the [`cnn`](cnn.md) function.

## Usage

``` r
# S3 method for class 'citocnn'
plot(x, ...)
```

## Arguments

- x:

  A model created by [`cnn`](cnn.md).

- ...:

  Additional arguments (currently not used).

## Value

The original model object `x`, returned invisibly.

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)

device <- ifelse(torch::cuda_is_available(), "cuda", "cpu")

## Data
shapes <- cito:::simulate_shapes(320, 28)
X <- shapes$data
Y <- shapes$labels

## Architecture
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
cnn.fit <- cnn(X, Y, architecture, loss = "cross-entropy", epochs = 50,
               validation = 0.1, lr = 0.05, device = device)

## Structure of Neural Network
plot(cnn.fit)
}
#> Loss at epoch 1: training: 0.699, validation: 0.706, lr: 0.05000

#> Loss at epoch 2: training: 0.696, validation: 0.702, lr: 0.05000
#> Loss at epoch 3: training: 0.694, validation: 0.700, lr: 0.05000
#> Loss at epoch 4: training: 0.693, validation: 0.698, lr: 0.05000
#> Loss at epoch 5: training: 0.693, validation: 0.695, lr: 0.05000
#> Loss at epoch 6: training: 0.691, validation: 0.693, lr: 0.05000
#> Loss at epoch 7: training: 0.690, validation: 0.692, lr: 0.05000
#> Loss at epoch 8: training: 0.691, validation: 0.691, lr: 0.05000
#> Loss at epoch 9: training: 0.689, validation: 0.689, lr: 0.05000
#> Loss at epoch 10: training: 0.688, validation: 0.688, lr: 0.05000
#> Loss at epoch 11: training: 0.688, validation: 0.687, lr: 0.05000
#> Loss at epoch 12: training: 0.686, validation: 0.686, lr: 0.05000
#> Loss at epoch 13: training: 0.686, validation: 0.684, lr: 0.05000
#> Loss at epoch 14: training: 0.685, validation: 0.684, lr: 0.05000
#> Loss at epoch 15: training: 0.684, validation: 0.683, lr: 0.05000
#> Loss at epoch 16: training: 0.682, validation: 0.680, lr: 0.05000
#> Loss at epoch 17: training: 0.681, validation: 0.678, lr: 0.05000
#> Loss at epoch 18: training: 0.679, validation: 0.677, lr: 0.05000
#> Loss at epoch 19: training: 0.677, validation: 0.673, lr: 0.05000
#> Loss at epoch 20: training: 0.674, validation: 0.670, lr: 0.05000
#> Loss at epoch 21: training: 0.671, validation: 0.666, lr: 0.05000
#> Loss at epoch 22: training: 0.668, validation: 0.667, lr: 0.05000
#> Loss at epoch 23: training: 0.663, validation: 0.656, lr: 0.05000
#> Loss at epoch 24: training: 0.658, validation: 0.659, lr: 0.05000
#> Loss at epoch 25: training: 0.655, validation: 0.658, lr: 0.05000
#> Loss at epoch 26: training: 0.646, validation: 0.634, lr: 0.05000
#> Loss at epoch 27: training: 0.638, validation: 0.625, lr: 0.05000
#> Loss at epoch 28: training: 0.630, validation: 0.625, lr: 0.05000
#> Loss at epoch 29: training: 0.620, validation: 0.605, lr: 0.05000
#> Loss at epoch 30: training: 0.598, validation: 0.584, lr: 0.05000
#> Loss at epoch 31: training: 0.594, validation: 0.557, lr: 0.05000
#> Loss at epoch 32: training: 0.614, validation: 0.564, lr: 0.05000
#> Loss at epoch 33: training: 0.564, validation: 0.511, lr: 0.05000
#> Loss at epoch 34: training: 0.507, validation: 0.518, lr: 0.05000
#> Loss at epoch 35: training: 0.553, validation: 0.504, lr: 0.05000
#> Loss at epoch 36: training: 0.425, validation: 0.551, lr: 0.05000
#> Loss at epoch 37: training: 0.611, validation: 0.486, lr: 0.05000
#> Loss at epoch 38: training: 0.501, validation: 0.398, lr: 0.05000
#> Loss at epoch 39: training: 0.430, validation: 0.346, lr: 0.05000
#> Loss at epoch 40: training: 0.541, validation: 0.516, lr: 0.05000
#> Loss at epoch 41: training: 0.443, validation: 0.370, lr: 0.05000
#> Loss at epoch 42: training: 0.360, validation: 0.332, lr: 0.05000
#> Loss at epoch 43: training: 0.297, validation: 0.200, lr: 0.05000
#> Loss at epoch 44: training: 0.175, validation: 0.132, lr: 0.05000
#> Loss at epoch 45: training: 0.112, validation: 0.099, lr: 0.05000
#> Loss at epoch 46: training: 0.157, validation: 0.084, lr: 0.05000
#> Loss at epoch 47: training: 0.082, validation: 0.057, lr: 0.05000
#> Loss at epoch 48: training: 0.049, validation: 0.037, lr: 0.05000
#> Loss at epoch 49: training: 0.037, validation: 0.031, lr: 0.05000
#> Loss at epoch 50: training: 0.026, validation: 0.022, lr: 0.05000

# }
```
