# Retrieve parameters of a fitted MMN model

This function returns the list of parameters (weights and biases) and
buffers (e.g. running mean and variance of batch normalization layers)
currently in use by the neural network model created using the
[`mmn`](mmn.md) function.

## Usage

``` r
# S3 method for class 'citommn'
coef(object, ...)
```

## Arguments

- object:

  A model created by [`mmn`](mmn.md).

- ...:

  Additional arguments (currently not used).

## Value

A list with two components:

- `net_parameters`: A list of the model's weights and biases for the
  currently used model epoch.

- `net_buffers`: A list of buffers (e.g., running statistics) for the
  currently used model epoch.

- `loss_parameters`: A list of the loss function's parameters for the
  currently used model epoch.
