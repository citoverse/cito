# Config hyperparameter tuning

Config hyperparameter tuning

## Usage

``` r
config_tuning(
  CV = 5,
  steps = 10,
  parallel = FALSE,
  NGPU = 1,
  cancel = TRUE,
  bootstrap_final = NULL,
  bootstrap_parallel = FALSE,
  blocking = NULL,
  return_models = FALSE
)
```

## Arguments

- CV:

  numeric, specifies k-folded cross validation

- steps:

  numeric, number of random tuning steps

- parallel:

  numeric, number of parallel cores (tuning steps are parallelized)

- NGPU:

  numeric, set if more than one GPU is available, tuning will be
  parallelized over CPU cores and GPUs, only works for NCPU \> 1

- cancel:

  CV/tuning for specific hyperparameter set if model cannot reduce loss
  below baseline after burnin or returns NA loss

- bootstrap_final:

  bootstrap final model, if all models should be boostrapped it must be
  set globally via the bootstrap argument in the [`dnn()`](dnn.md)
  function

- bootstrap_parallel:

  should the bootstrapping be parallelized or not

- blocking:

  blocking variable, must be a factor

- return_models:

  return individual models

## Value

A list with the tuning settings, passed to the `tuning` argument of
[`dnn`](dnn.md).

## Details

Note that hyperparameter tuning can be expensive. We have implemented an
option to parallelize hyperparameter tuning, including parallelization
over one or more GPUs (the hyperparameter evaluation is parallelized,
not the CV). This can be especially useful for small models. For
example, if you have 4 GPUs, 20 CPU cores, and 20 steps (random samples
from the random search), you could run \`dnn(..., device="cuda",lr =
tune(), batchsize=tune(), tuning=config_tuning(parallel=20, NGPU=4)',
which will distribute 20 model fits across 4 GPUs, so that each GPU will
process 5 models (in parallel).
