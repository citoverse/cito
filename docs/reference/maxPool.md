# Create a Maximum Pooling Layer for a CNN Architecture

This function creates a `maxPool` layer object of class `citolayer` for
use in constructing a Convolutional Neural Network (CNN) architecture.
The resulting layer object can be passed to the
[`create_architecture`](create_architecture.md) function to define the
structure of the network.

## Usage

``` r
maxPool(kernel_size = NULL, stride = NULL, padding = NULL, dilation = NULL)
```

## Arguments

- kernel_size:

  (integer or tuple) The size of the kernel in this layer. Use a tuple
  if the kernel size varies across dimensions.

- stride:

  (integer or tuple) The stride of the kernel in this layer. If `NULL`,
  the stride is set to the kernel size. Use a tuple if the stride
  differs across dimensions.

- padding:

  (integer or tuple) The amount of zero-padding added to the input on
  both sides. Use a tuple if the padding differs across dimensions.

- dilation:

  (integer or tuple) The dilation of the kernel in this layer. Use a
  tuple if the dilation varies across dimensions.

## Value

An S3 object of class `"maxPool" "citolayer"`, representing a maximum
pooling layer in the CNN architecture.

## Details

This function creates a `maxPool` layer object, which represents a
maximum pooling layer in a CNN architecture. Parameters not specified
(and thus set to `NULL`) will be filled with default values provided to
the [`create_architecture`](create_architecture.md) function.

## See also

[`create_architecture`](create_architecture.md)

## Author

Armin Schenk

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# A maximum pooling layer where all available parameters are assigned
# No value will be overwritten by 'create_architecture()'
layer1 <- maxPool(3, 1, 0, 1)

# A maximum pooling layer where only the kernel size is assigned
# stride, padding and dilation are filled with the defaults
# passed to the 'create_architecture()' function
layer2 <- maxPool(kernel_size=4)
}
# }
```
