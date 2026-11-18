# Summarize a fitted MMN model

Reports the permutation importance of each data stream: one stream is
shuffled across the batch while the others are left intact, and the
increase in loss is what that stream contributed.

## Usage

``` r
# S3 method for class 'citommn'
summary(object, n_permute = 10L, device = NULL, ...)

# S3 method for class 'summary.citommn'
print(x, ...)
```

## Arguments

- object:

  A model created by [`mmn`](mmn.md).

- n_permute:

  Number of permutations per data stream. Default is 10.

- device:

  Device on which to evaluate. Defaults to the device the model was
  trained on.

- ...:

  Additional arguments (currently not used).

- x:

  a summary of a fitted MMN

## Value

An object of class `"summary.citommn"` holding one row per data stream
with the increase in loss, its standard deviation across permutations,
and the share of the total.

## Details

This is the stream-level analogue of permutation importance for a DNN,
and is the interpretive tool a multi-modal model admits, since
per-feature effects are not comparable across modalities of different
kinds. The standard deviation across permutations is reported beside the
estimate, because a stream that contributes little can come out negative
by chance.
