# Include a Pretrained Model in a CNN Architecture

This function creates a `transfer` layer object of class `citolayer` for
use in constructing a Convolutional Neural Network (CNN) architecture.
The resulting layer object allows the use of pretrained models available
in the 'torchvision' package within cito.

## Usage

``` r
transfer(
  name = c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152",
    "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11",
    "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    "wide_resnet101_2", "wide_resnet50_2"),
  pretrained = TRUE,
  freeze = TRUE,
  rgb = TRUE
)
```

## Arguments

- name:

  (character) The name of the pretrained model. Available options
  include: "alexnet", "inception_v3", "mobilenet_v2", "resnet101",
  "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d",
  "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16",
  "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2",
  "wide_resnet50_2".

- pretrained:

  (boolean) If `TRUE`, the model uses its pretrained weights. If
  `FALSE`, random weights are initialized.

- freeze:

  (boolean) If `TRUE`, the weights of the pretrained model (except the
  "classifier" part at the end) are not updated during training. This
  setting only applies if `pretrained = TRUE`.

- rgb:

  (boolean) If `FALSE`, the pretrained weights of the first
  convolutional layer are averaged across the channel dimension. This is
  useful if your data has 3 channels but isn't an RGB image. This
  setting only applies if `pretrained = TRUE`.

## Value

An S3 object of class `"transfer" "citolayer"`, representing a
pretrained model of the `torchvision` package in the CNN architecture.

## Details

This function creates a `transfer` layer object, which represents a
pretrained model of the `torchvision` package with the linear
"classifier" part removed. This allows the pretrained features of the
model to be utilized while enabling customization of the classifier.
When using this function with
[`create_architecture`](create_architecture.md), only linear layers can
be added after the `transfer` layer. These linear layers define the
"classifier" part of the network. If no linear layers are provided
following the `transfer` layer, the default classifier will consist of a
single output layer.

Additionally, the `pretrained` argument specifies whether to use the
pretrained weights or initialize the model with random weights. If
`freeze` is set to `TRUE`, only the weights of the final linear layers
(the "classifier") are updated during training, while the rest of the
pretrained model remains unchanged. Note that `freeze` has no effect
unless `pretrained` is set to `TRUE`.

If your data has three channels but is not an RGB image set `rgb` to
`FALSE` to average the pretrained weights of the first convolutional
layer, so that each channel is treated equally. This is also done if
your data has more or less channels than 3.

## See also

[`create_architecture`](create_architecture.md)

## Author

Armin Schenk

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# Creates a "transfer" "citolayer" object that later tells the cnn() function that
# the alexnet architecture and its pretrained weights should be used, but none
# of the weights are frozen
alexnet <- transfer(name="alexnet", pretrained=TRUE, freeze=FALSE)

# Creates a "transfer" "citolayer" object that later tells the cnn() function that
# the resnet18 architecture and its pretrained weights should be used.
# Also all weights except from the linear layer at the end are frozen (and
# therefore not changed during training)
resnet18 <- transfer(name="resnet18", pretrained=TRUE, freeze=TRUE)
}
# }
```
