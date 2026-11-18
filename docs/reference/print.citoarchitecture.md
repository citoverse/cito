# Print method for citoarchitecture objects

This method provides a visual representation of the network architecture
defined by an object of class `citoarchitecture`, including information
about each layer's configuration. It helps in understanding the
structure of the architecture defined by
[`create_architecture`](create_architecture.md).

## Usage

``` r
# S3 method for class 'citoarchitecture'
print(x, input_shape, output_shape = NULL, ...)
```

## Arguments

- x:

  An object of class `citoarchitecture`, created by
  [`create_architecture`](create_architecture.md).

- input_shape:

  A numeric vector specifying the dimensions of a single sample (e.g.,
  `c(3, 28, 28)` for an RGB image with height and width of 28 pixels).
  This argument is required for a detailed output.

- output_shape:

  An integer specifying the number of nodes in the output layer. If
  `NULL`, no output layer is printed.

- ...:

  Additional arguments (currently not used).

## Value

The original `citoarchitecture` object, returned invisibly.

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

c1 <- conv(n_kernels = 8, kernel_size = 5)
c2 <- conv(n_kernels = 16, kernel_size = 3)
l <- linear(n_neurons = 100)
mP <- maxPool(kernel_size = 2)
architecture <- create_architecture(c1, c1, mP, c2, c2, mP, l,
                                    default_dropout = list(linear=0.6, conv=0.4),
                                    default_normalization = list(linear=TRUE),
                                    default_activation = "selu")

# See how the finished CNN would look like for specific input and output shapes
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
# }
```
