# Tune hyperparameter

Control hyperparameter tuning

## Usage

``` r
tune(
  lower = NULL,
  upper = NULL,
  fixed = NULL,
  additional = NULL,
  values = NULL
)
```

## Arguments

- lower:

  numeric, numeric vector, character, lower boundaries of tuning space

- upper:

  numeric, numeric vector, character, upper boundaries of tuning space

- fixed:

  character, used for multi-dimensional hyperparameters such as hidden,
  which dimensions should be fixed

- additional:

  numeric, additional control parameter which sets the value of the
  fixed argument

- values:

  custom values from which hyperparameters are sampled, must be a matrix
  for hidden layers (first column == nodes, second column == number of
  layers)
