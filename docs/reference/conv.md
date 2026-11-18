# Create a Convolutional Layer for a CNN Architecture

This function creates a `conv` layer object of class `citolayer` for use
in constructing a Convolutional Neural Network (CNN) architecture. The
resulting layer object can be passed to the
[`create_architecture`](create_architecture.md) function to define the
structure of the network.

## Usage

``` r
conv(
  n_kernels = NULL,
  kernel_size = NULL,
  stride = NULL,
  padding = NULL,
  dilation = NULL,
  bias = NULL,
  activation = NULL,
  normalization = NULL,
  dropout = NULL
)
```

## Arguments

- n_kernels:

  (integer) The number of kernels (or filters) in this layer.

- kernel_size:

  (integer or tuple) The size of the kernels in this layer. Use a tuple
  if the kernel size is different in each dimension.

- stride:

  (integer or tuple) The stride of the kernels in this layer. If `NULL`,
  the stride is set to the kernel size. Use a tuple if the stride is
  different in each dimension.

- padding:

  (integer or tuple) The amount of zero-padding added to the input on
  both sides. Use a tuple if the padding is different in each dimension.

- dilation:

  (integer or tuple) The dilation of the kernels in this layer. Use a
  tuple if the dilation is different in each dimension.

- bias:

  (boolean) If `TRUE`, a learnable bias is added to the kernels of this
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

An S3 object of class `"conv" "citolayer"`, representing a convolutional
layer in the CNN architecture.

## Details

This function creates a `conv` layer object, which is used to define a
convolutional layer in a CNN architecture. Parameters that are not
specified (and thus set to `NULL`) will be filled with default values
provided to the [`create_architecture`](create_architecture.md)
function.

## See also

[`create_architecture`](create_architecture.md)

## Author

Armin Schenk

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# A convolutional layer where all available parameters are assigned
# No value will be overwritten by 'create_architecture()'
layer1 <- conv(10, 3, 1, 0, 1, TRUE, "relu", FALSE, 0.5)

# A convolutional layer where only the activation function is assigned
# n_kernels, kernel_size, stride, padding, dilation, bias,
# normalization and dropout are filled with the defaults
# passed to the 'create_architecture()' function
layer2 <- conv(activation="selu")
}
# }
```
