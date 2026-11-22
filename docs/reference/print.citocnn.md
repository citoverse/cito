# Print a fitted CNN model

This function prints the architecture of a Convolutional Neural Network
(CNN) model created using the [`cnn`](cnn.md) function.

## Usage

``` r
# S3 method for class 'citocnn'
print(x, ...)
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
cnn.fit <- cnn(X, Y, architecture, loss = "cross-entropy",
               epochs = 50, validation = 0.1, lr = 0.05, device=device)

# Structure of Neural Network
print(cnn.fit)
}
#> Loss at epoch 1: training: 0.708, validation: 0.721, lr: 0.05000

#> Loss at epoch 2: training: 0.701, validation: 0.713, lr: 0.05000
#> Loss at epoch 3: training: 0.698, validation: 0.707, lr: 0.05000
#> Loss at epoch 4: training: 0.696, validation: 0.703, lr: 0.05000
#> Loss at epoch 5: training: 0.695, validation: 0.700, lr: 0.05000
#> Loss at epoch 6: training: 0.694, validation: 0.698, lr: 0.05000
#> Loss at epoch 7: training: 0.693, validation: 0.697, lr: 0.05000
#> Loss at epoch 8: training: 0.693, validation: 0.695, lr: 0.05000
#> Loss at epoch 9: training: 0.692, validation: 0.694, lr: 0.05000
#> Loss at epoch 10: training: 0.692, validation: 0.693, lr: 0.05000
#> Loss at epoch 11: training: 0.692, validation: 0.692, lr: 0.05000
#> Loss at epoch 12: training: 0.692, validation: 0.691, lr: 0.05000
#> Loss at epoch 13: training: 0.691, validation: 0.691, lr: 0.05000
#> Loss at epoch 14: training: 0.691, validation: 0.691, lr: 0.05000
#> Loss at epoch 15: training: 0.690, validation: 0.690, lr: 0.05000
#> Loss at epoch 16: training: 0.690, validation: 0.690, lr: 0.05000
#> Loss at epoch 17: training: 0.690, validation: 0.689, lr: 0.05000
#> Loss at epoch 18: training: 0.689, validation: 0.688, lr: 0.05000
#> Loss at epoch 19: training: 0.688, validation: 0.688, lr: 0.05000
#> Loss at epoch 20: training: 0.688, validation: 0.687, lr: 0.05000
#> Loss at epoch 21: training: 0.687, validation: 0.686, lr: 0.05000
#> Loss at epoch 22: training: 0.686, validation: 0.685, lr: 0.05000
#> Loss at epoch 23: training: 0.685, validation: 0.684, lr: 0.05000
#> Loss at epoch 24: training: 0.684, validation: 0.683, lr: 0.05000
#> Loss at epoch 25: training: 0.684, validation: 0.681, lr: 0.05000
#> Loss at epoch 26: training: 0.683, validation: 0.680, lr: 0.05000
#> Loss at epoch 27: training: 0.681, validation: 0.678, lr: 0.05000
#> Loss at epoch 28: training: 0.679, validation: 0.675, lr: 0.05000
#> Loss at epoch 29: training: 0.677, validation: 0.673, lr: 0.05000
#> Loss at epoch 30: training: 0.675, validation: 0.670, lr: 0.05000
#> Loss at epoch 31: training: 0.673, validation: 0.668, lr: 0.05000
#> Loss at epoch 32: training: 0.669, validation: 0.663, lr: 0.05000
#> Loss at epoch 33: training: 0.667, validation: 0.662, lr: 0.05000
#> Loss at epoch 34: training: 0.664, validation: 0.655, lr: 0.05000
#> Loss at epoch 35: training: 0.659, validation: 0.650, lr: 0.05000
#> Loss at epoch 36: training: 0.656, validation: 0.646, lr: 0.05000
#> Loss at epoch 37: training: 0.654, validation: 0.640, lr: 0.05000
#> Loss at epoch 38: training: 0.648, validation: 0.639, lr: 0.05000
#> Loss at epoch 39: training: 0.644, validation: 0.632, lr: 0.05000
#> Loss at epoch 40: training: 0.639, validation: 0.626, lr: 0.05000
#> Loss at epoch 41: training: 0.633, validation: 0.625, lr: 0.05000
#> Loss at epoch 42: training: 0.628, validation: 0.613, lr: 0.05000
#> Loss at epoch 43: training: 0.622, validation: 0.604, lr: 0.05000
#> Loss at epoch 44: training: 0.619, validation: 0.607, lr: 0.05000
#> Loss at epoch 45: training: 0.612, validation: 0.600, lr: 0.05000
#> Loss at epoch 46: training: 0.604, validation: 0.592, lr: 0.05000
#> Loss at epoch 47: training: 0.598, validation: 0.579, lr: 0.05000
#> Loss at epoch 48: training: 0.595, validation: 0.578, lr: 0.05000
#> Loss at epoch 49: training: 0.583, validation: 0.566, lr: 0.05000
#> Loss at epoch 50: training: 0.585, validation: 0.586, lr: 0.05000
#> cnn(X = X, Y = Y, architecture = architecture, loss = "cross-entropy", 
#>     lr = 0.05, validation = 0.1, epochs = 50, device = device)
#> -------------------------------------------------------------------------------
#> Convolution|Input: 1x28x28
#>            |Output: 5x26x26
#>            |Kernel: 3x3 (stride=1x1, padding=0x0, dilation=1x1)
#>            |Bias: TRUE
#>            |Activation: relu
#> -------------------------------------------------------------------------------
#> MaxPool    |Input: 5x26x26
#>            |Output: 5x13x13
#>            |Kernel: 2x2 (stride=2x2, padding=0x0, dilation=1x1)
#> -------------------------------------------------------------------------------
#> Convolution|Input: 5x13x13
#>            |Output: 5x11x11
#>            |Kernel: 3x3 (stride=1x1, padding=0x0, dilation=1x1)
#>            |Bias: TRUE
#>            |Activation: relu
#> -------------------------------------------------------------------------------
#> MaxPool    |Input: 5x11x11
#>            |Output: 5x5x5
#>            |Kernel: 2x2 (stride=2x2, padding=0x0, dilation=1x1)
#> -------------------------------------------------------------------------------
#> Linear     |Input: 125
#>            |Output: 10
#>            |Bias: TRUE
#>            |Activation: relu
#> -------------------------------------------------------------------------------
#> Linear     |Input: 10
#>            |Output: 2
#>            |Bias: TRUE
#>            |Activation: Depends on loss
#> -------------------------------------------------------------------------------
# }
```
