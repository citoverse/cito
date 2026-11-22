# Create a CNN Architecture

This function constructs a `citoarchitecture` object that defines the
architecture of a Convolutional Neural Network (CNN). The
`citoarchitecture` object can be used by the [`cnn`](cnn.md) function to
specify the structure of the network, including layer types, parameters,
and default values.

## Usage

``` r
create_architecture(
  ...,
  default_n_neurons = 10,
  default_n_kernels = 10,
  default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
  default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
  default_padding = list(conv = 0, maxPool = 0, avgPool = 0),
  default_dilation = list(conv = 1, maxPool = 1),
  default_bias = list(conv = TRUE, linear = TRUE),
  default_activation = list(conv = "relu", linear = "relu"),
  default_normalization = list(conv = FALSE, linear = FALSE),
  default_dropout = list(conv = 0, linear = 0)
)
```

## Arguments

- ...:

  Objects of class `citolayer` created by [`linear`](linear.md),
  [`conv`](conv.md), [`maxPool`](maxPool.md), [`avgPool`](avgPool.md),
  or [`transfer`](transfer.md). These layers define the architecture of
  the CNN.

- default_n_neurons:

  (integer) Default number of neurons in a linear layer. Default is 10.

- default_n_kernels:

  (integer) Default number of kernels in a convolutional layer. Default
  is 10.

- default_kernel_size:

  (integer or tuple) Default size of kernels in convolutional and
  pooling layers. Can be a single integer or a tuple if sizes differ
  across dimensions. Default is
  `list(conv = 3, maxPool = 2, avgPool = 2)`.

- default_stride:

  (integer or tuple) Default stride of kernels in convolutional and
  pooling layers. Can be a single integer, a tuple if strides differ
  across dimensions, or `NULL` to use the kernel size. Default is
  `list(conv = 1, maxPool = NULL, avgPool = NULL)`.

- default_padding:

  (integer or tuple) Default zero-padding added to both sides of the
  input. Can be a single integer or a tuple if padding differs across
  dimensions. Default is `list(conv = 0, maxPool = 0, avgPool = 0)`.

- default_dilation:

  (integer or tuple) Default dilation of kernels in convolutional and
  max pooling layers. Can be a single integer or a tuple if dilation
  differs across dimensions. Default is `list(conv = 1, maxPool = 1)`.

- default_bias:

  (boolean) Default value indicating if a learnable bias should be added
  to neurons of linear layers and kernels of convolutional layers.
  Default is `list(conv = TRUE, linear = TRUE)`.

- default_activation:

  (character) Default activation function used after linear and
  convolutional layers. Supported activation functions include "relu",
  "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu",
  "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
  "tanhshrink", "softshrink", "hardshrink", "log_sigmoid". Default is
  `list(conv = "relu", linear = "relu")`.

- default_normalization:

  (boolean) Default value indicating if batch normalization should be
  applied after linear and convolutional layers. Default is
  `list(conv = FALSE, linear = FALSE)`.

- default_dropout:

  (numeric) Default dropout rate for linear and convolutional layers.
  Set to 0 for no dropout. Default is `list(conv = 0.0, linear = 0.0)`.

## Value

An S3 object of class `"citoarchitecture"` that encapsulates the
architecture of the CNN.

## Details

This function creates a `citoarchitecture` object that outlines the
CNN's architecture based on the provided layers and default parameters.
The final architecture consists of layers in the order they are
provided. Any unspecified parameters in the `citolayer` objects are
filled with the provided default values for their respective layer
types. Defaults can be specified for each layer type individually or for
all layers at once.

## See also

[`cnn`](cnn.md), [`linear`](linear.md), [`conv`](conv.md),
[`maxPool`](maxPool.md), [`avgPool`](avgPool.md),
[`transfer`](transfer.md),
[`print.citoarchitecture`](print.citoarchitecture.md),
[`plot.citoarchitecture`](plot.citoarchitecture.md)

## Author

Armin Schenk

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# Convolutional layers with different n_kernels and kernel_sizes
c1 <- conv(n_kernels = 8, kernel_size = 5)
c2 <- conv(n_kernels = 16, kernel_size = 3)

# Linear layer
l <- linear(n_neurons = 100)

# MaxPooling layer
mP <- maxPool(kernel_size = 2)

# Create the architecture by using the created layers
# Change the defaults with which the not assigned layer parameters will be filled e.g.
# change default dropout to different values for linear and convolutional layer
# only change the default normalization for linear layers
# change default activation of both linear and convolutional layers to 'selu'
architecture <- create_architecture(c1, c1, mP, c2, c2, mP, l,
                                    default_dropout = list(linear=0.6, conv=0.4),
                                    default_normalization = list(linear=TRUE),
                                    default_activation = "selu")

# See how the finished CNN would look like for specific input and output shapes
print(architecture, c(3,128,128), 10)

# To use predefined architectures  use the transfer() layer
alexnet <- transfer("alexnet")

# No other linear layers are used after the transfer layer:
# The cnn() function will replace the linear classifier of the
# alexnet architecture with a single linear output layer
architecture <- create_architecture(alexnet)
print(architecture, c(3,128,128), 10)

# Some linear layers are used after the transfer layer:
# The cnn() function will replace the linear classifier of the alexnet architecture
# with the specified linear layers + an output layer that matches the output dimensions
architecture <- create_architecture(alexnet, linear(300), linear(100))
print(architecture, c(3,128,128), 10)
}
#> -------------------------------------------------------------------------------
#> Convolution|Input: 3x128x128
#>            |Output: 8x124x124
#>            |Kernel: 5x5 (stride=1x1, padding=0x0, dilation=1x1)
#>            |Bias: TRUE
#>            |Activation: selu
#>            |Dropout: rate=0.4
#> -------------------------------------------------------------------------------
#> Convolution|Input: 8x124x124
#>            |Output: 8x120x120
#>            |Kernel: 5x5 (stride=1x1, padding=0x0, dilation=1x1)
#>            |Bias: TRUE
#>            |Activation: selu
#>            |Dropout: rate=0.4
#> -------------------------------------------------------------------------------
#> MaxPool    |Input: 8x120x120
#>            |Output: 8x60x60
#>            |Kernel: 2x2 (stride=2x2, padding=0x0, dilation=1x1)
#> -------------------------------------------------------------------------------
#> Convolution|Input: 8x60x60
#>            |Output: 16x58x58
#>            |Kernel: 3x3 (stride=1x1, padding=0x0, dilation=1x1)
#>            |Bias: TRUE
#>            |Activation: selu
#>            |Dropout: rate=0.4
#> -------------------------------------------------------------------------------
#> Convolution|Input: 16x58x58
#>            |Output: 16x56x56
#>            |Kernel: 3x3 (stride=1x1, padding=0x0, dilation=1x1)
#>            |Bias: TRUE
#>            |Activation: selu
#>            |Dropout: rate=0.4
#> -------------------------------------------------------------------------------
#> MaxPool    |Input: 16x56x56
#>            |Output: 16x28x28
#>            |Kernel: 2x2 (stride=2x2, padding=0x0, dilation=1x1)
#> -------------------------------------------------------------------------------
#> Linear     |Input: 12544
#>            |Output: 100
#>            |Bias: TRUE
#>            |Batch normalization
#>            |Activation: selu
#>            |Dropout: rate=0.6
#> -------------------------------------------------------------------------------
#> Linear     |Input: 100
#>            |Output: 10
#>            |Bias: TRUE
#>            |Activation: Depends on loss
#> -------------------------------------------------------------------------------
#> -------------------------------------------------------------------------------
#> Transfer   |Input: 3x128x128
#>            |Output: 256x6x6
#>            |Network: alexnet
#>            |Pretrained: TRUE
#>            |Weights frozen: TRUE
#> -------------------------------------------------------------------------------
#> Linear     |Input: 9216
#>            |Output: 10
#>            |Bias: TRUE
#>            |Activation: Depends on loss
#> -------------------------------------------------------------------------------
#> -------------------------------------------------------------------------------
#> Transfer   |Input: 3x128x128
#>            |Output: 256x6x6
#>            |Network: alexnet
#>            |Pretrained: TRUE
#>            |Weights frozen: TRUE
#> -------------------------------------------------------------------------------
#> Linear     |Input: 9216
#>            |Output: 300
#>            |Bias: TRUE
#>            |Activation: relu
#> -------------------------------------------------------------------------------
#> Linear     |Input: 300
#>            |Output: 100
#>            |Bias: TRUE
#>            |Activation: relu
#> -------------------------------------------------------------------------------
#> Linear     |Input: 100
#>            |Output: 10
#>            |Bias: TRUE
#>            |Activation: Depends on loss
#> -------------------------------------------------------------------------------
# }
```
