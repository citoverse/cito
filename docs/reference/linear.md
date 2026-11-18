# Create a Linear Layer for a CNN Architecture

This function creates a `linear` layer object of class `citolayer` for
use in constructing a Convolutional Neural Network (CNN) architecture.
The resulting layer object can be passed to the
[`create_architecture`](create_architecture.md) function to define the
structure of the network.

## Usage

``` r
linear(
  n_neurons = NULL,
  bias = NULL,
  activation = NULL,
  normalization = NULL,
  dropout = NULL
)
```

## Arguments

- n_neurons:

  (integer) The number of hidden neurons in this layer.

- bias:

  (boolean) If `TRUE`, a learnable bias is added to the neurons of this
  layer.

- activation:

  (character) The activation function applied after this layer.
  Supported activation functions include "relu", "leaky_relu", "tanh",
  "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6",
  "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink",
  "hardshrink", "log_sigmoid".

- normalization:

  (boolean) If `TRUE`, batch normalization is applied after this layer.

- dropout:

  (numeric) The dropout rate for this layer. Set to 0 to disable
  dropout.

## Value

An S3 object of class `"linear" "citolayer"`, representing a linear
layer in the CNN architecture.

## Details

This function creates a `linear` layer object, which is used to define a
linear layer in a CNN architecture. Parameters not specified (and thus
set to `NULL`) will be filled with default values provided to the
[`create_architecture`](create_architecture.md) function.

## See also

[`create_architecture`](create_architecture.md)

## Author

Armin Schenk

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# A linear layer where all available parameters are assigned
# No value will be overwritten by 'create_architecture()'
layer1 <- linear(100, TRUE, "relu", FALSE, 0.5)

# A linear layer where only the activation function is assigned
# n_neurons, bias, normalization and dropout are filled with the defaults
# passed to the 'create_architecture()' function
layer2 <- linear(activation="selu")
}
# }
```
