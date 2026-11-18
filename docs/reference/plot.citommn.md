# Plot a fitted MMN model

Visualizes either the permutation importance of the data streams or the
architecture of a Multi-Modal Neural Network.

## Usage

``` r
# S3 method for class 'citommn'
plot(
  x,
  plot_type = c("importance", "arch"),
  n_permute = 10L,
  device = NULL,
  ...
)
```

## Arguments

- x:

  a model created by [`mmn`](mmn.md)

- plot_type:

  `"importance"` draws the permutation importance of each data stream,
  `"arch"` the architecture: one row per data stream with its submodel,
  the concatenation, the fusion network and the output.
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) on a single
  [`create_architecture`](create_architecture.md) object still draws
  that submodel's layers in full.

- n_permute:

  number of permutations per data stream, only used for `"importance"`

- device:

  device on which to evaluate, defaults to the device the model was
  trained on

- ...:

  passed to [`barplot`](https://rdrr.io/r/graphics/barplot.html), only
  used for `"importance"`

## Value

invisibly, the importance table for `"importance"` and `NULL` otherwise
