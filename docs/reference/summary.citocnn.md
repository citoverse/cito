# Summarize a fitted CNN model

This function provides a summary of a Convolutional Neural Network (CNN)
model created using the [`cnn`](cnn.md) function. It currently
replicates the output of the [`print.citocnn`](print.citocnn.md) method.

## Usage

``` r
# S3 method for class 'citocnn'
summary(object, ...)
```

## Arguments

- object:

  A model created by [`cnn`](cnn.md).

- ...:

  Additional arguments (currently not used).

## Value

The original model object `object`, returned invisibly.
