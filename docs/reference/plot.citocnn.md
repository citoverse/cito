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
#> Loss at epoch 1: training: 0.710, validation: 0.686, lr: 0.05000

#> Loss at epoch 2: training: 0.704, validation: 0.686, lr: 0.05000
#> Loss at epoch 3: training: 0.700, validation: 0.686, lr: 0.05000
#> Loss at epoch 4: training: 0.698, validation: 0.687, lr: 0.05000
#> Loss at epoch 5: training: 0.696, validation: 0.687, lr: 0.05000
#> Loss at epoch 6: training: 0.695, validation: 0.688, lr: 0.05000
#> Loss at epoch 7: training: 0.694, validation: 0.689, lr: 0.05000
#> Loss at epoch 8: training: 0.694, validation: 0.690, lr: 0.05000
#> Loss at epoch 9: training: 0.694, validation: 0.690, lr: 0.05000
#> Loss at epoch 10: training: 0.693, validation: 0.691, lr: 0.05000
#> Loss at epoch 11: training: 0.693, validation: 0.691, lr: 0.05000
#> Loss at epoch 12: training: 0.694, validation: 0.692, lr: 0.05000
#> Loss at epoch 13: training: 0.693, validation: 0.691, lr: 0.05000
#> Loss at epoch 14: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 15: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 16: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 17: training: 0.693, validation: 0.693, lr: 0.05000
#> Loss at epoch 18: training: 0.693, validation: 0.693, lr: 0.05000
#> Loss at epoch 19: training: 0.693, validation: 0.693, lr: 0.05000
#> Loss at epoch 20: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 21: training: 0.693, validation: 0.693, lr: 0.05000
#> Loss at epoch 22: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 23: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 24: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 25: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 26: training: 0.693, validation: 0.693, lr: 0.05000
#> Loss at epoch 27: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 28: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 29: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 30: training: 0.692, validation: 0.692, lr: 0.05000
#> Loss at epoch 31: training: 0.692, validation: 0.692, lr: 0.05000
#> Loss at epoch 32: training: 0.693, validation: 0.692, lr: 0.05000
#> Loss at epoch 33: training: 0.692, validation: 0.692, lr: 0.05000
#> Loss at epoch 34: training: 0.692, validation: 0.691, lr: 0.05000
#> Loss at epoch 35: training: 0.692, validation: 0.691, lr: 0.05000
#> Loss at epoch 36: training: 0.692, validation: 0.691, lr: 0.05000
#> Loss at epoch 37: training: 0.692, validation: 0.691, lr: 0.05000
#> Loss at epoch 38: training: 0.692, validation: 0.691, lr: 0.05000
#> Loss at epoch 39: training: 0.691, validation: 0.691, lr: 0.05000
#> Loss at epoch 40: training: 0.691, validation: 0.691, lr: 0.05000
#> Loss at epoch 41: training: 0.691, validation: 0.690, lr: 0.05000
#> Loss at epoch 42: training: 0.691, validation: 0.690, lr: 0.05000
#> Loss at epoch 43: training: 0.691, validation: 0.690, lr: 0.05000
#> Loss at epoch 44: training: 0.691, validation: 0.690, lr: 0.05000
#> Loss at epoch 45: training: 0.690, validation: 0.689, lr: 0.05000
#> Loss at epoch 46: training: 0.690, validation: 0.689, lr: 0.05000
#> Loss at epoch 47: training: 0.690, validation: 0.689, lr: 0.05000
#> Loss at epoch 48: training: 0.690, validation: 0.688, lr: 0.05000
#> Loss at epoch 49: training: 0.689, validation: 0.688, lr: 0.05000
#> Loss at epoch 50: training: 0.689, validation: 0.687, lr: 0.05000

# }
```
