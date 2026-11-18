# Creation of customized learning rate scheduler objects

Helps create custom learning rate schedulers for [`dnn`](dnn.md).

## Usage

``` r
config_lr_scheduler(
  type = c("lambda", "multiplicative", "reduce_on_plateau", "one_cycle", "step"),
  verbose = FALSE,
  ...
)
```

## Arguments

- type:

  String defining which type of scheduler should be used. See Details.

- verbose:

  If TRUE, additional information about scheduler will be printed to
  console.

- ...:

  additional arguments to be passed to scheduler. See Details.

## Value

object of class cito_lr_scheduler to give to [`dnn`](dnn.md)

## Details

different learning rate scheduler need different variables, these
functions will tell you which variables can be set:

- lambda:
  [`lr_lambda`](https://torch.mlverse.org/docs/reference/lr_lambda.html)

- multiplicative:
  [`lr_multiplicative`](https://torch.mlverse.org/docs/reference/lr_multiplicative.html)

- reduce_on_plateau:
  [`lr_reduce_on_plateau`](https://torch.mlverse.org/docs/reference/lr_reduce_on_plateau.html)

- one_cycle:
  [`lr_one_cycle`](https://torch.mlverse.org/docs/reference/lr_one_cycle.html)

- step:
  [`lr_step`](https://torch.mlverse.org/docs/reference/lr_step.html)

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# create learning rate scheduler object
scheduler <- config_lr_scheduler(type = "step",
                        step_size = 30,
                        gamma = 0.15,
                        verbose = TRUE)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris, lr_scheduler = scheduler)

}
#> Learning rate Scheduler step
#> step_size: [30]
#> gamma: [0.15] 
#> last_epoch: [-1] 
#> Loss at epoch 1: 2.940436, lr: 0.01000

#> Loss at epoch 2: 0.201631, lr: 0.01000
#> Loss at epoch 3: 0.213944, lr: 0.01000
#> Loss at epoch 4: 0.264422, lr: 0.01000
#> Loss at epoch 5: 0.230633, lr: 0.01000
#> Loss at epoch 6: 0.265682, lr: 0.01000
#> Loss at epoch 7: 0.166983, lr: 0.01000
#> Loss at epoch 8: 0.163049, lr: 0.01000
#> Loss at epoch 9: 0.138647, lr: 0.01000
#> Loss at epoch 10: 0.150287, lr: 0.01000
#> Loss at epoch 11: 0.221042, lr: 0.01000
#> Loss at epoch 12: 0.169148, lr: 0.01000
#> Loss at epoch 13: 0.149148, lr: 0.01000
#> Loss at epoch 14: 0.134916, lr: 0.01000
#> Loss at epoch 15: 0.139639, lr: 0.01000
#> Loss at epoch 16: 0.159920, lr: 0.01000
#> Loss at epoch 17: 0.142356, lr: 0.01000
#> Loss at epoch 18: 0.128160, lr: 0.01000
#> Loss at epoch 19: 0.141422, lr: 0.01000
#> Loss at epoch 20: 0.178323, lr: 0.01000
#> Loss at epoch 21: 0.281369, lr: 0.01000
#> Loss at epoch 22: 0.135427, lr: 0.01000
#> Loss at epoch 23: 0.183387, lr: 0.01000
#> Loss at epoch 24: 0.152871, lr: 0.01000
#> Loss at epoch 25: 0.135944, lr: 0.01000
#> Loss at epoch 26: 0.166093, lr: 0.01000
#> Loss at epoch 27: 0.172522, lr: 0.01000
#> Loss at epoch 28: 0.322305, lr: 0.01000
#> Loss at epoch 29: 0.319153, lr: 0.01000
#> Loss at epoch 30: 0.146458, lr: 0.00150
#> Loss at epoch 31: 0.123754, lr: 0.00150
#> Loss at epoch 32: 0.121260, lr: 0.00150
#> Loss at epoch 33: 0.119630, lr: 0.00150
#> Loss at epoch 34: 0.120490, lr: 0.00150
#> Loss at epoch 35: 0.119286, lr: 0.00150
#> Loss at epoch 36: 0.119678, lr: 0.00150
#> Loss at epoch 37: 0.120862, lr: 0.00150
#> Loss at epoch 38: 0.119194, lr: 0.00150
#> Loss at epoch 39: 0.119210, lr: 0.00150
#> Loss at epoch 40: 0.121300, lr: 0.00150
#> Loss at epoch 41: 0.117858, lr: 0.00150
#> Loss at epoch 42: 0.118506, lr: 0.00150
#> Loss at epoch 43: 0.117125, lr: 0.00150
#> Loss at epoch 44: 0.118367, lr: 0.00150
#> Loss at epoch 45: 0.117296, lr: 0.00150
#> Loss at epoch 46: 0.117385, lr: 0.00150
#> Loss at epoch 47: 0.117293, lr: 0.00150
#> Loss at epoch 48: 0.117791, lr: 0.00150
#> Loss at epoch 49: 0.117772, lr: 0.00150
#> Loss at epoch 50: 0.118958, lr: 0.00150
#> Loss at epoch 51: 0.117071, lr: 0.00150
#> Loss at epoch 52: 0.118016, lr: 0.00150
#> Loss at epoch 53: 0.117237, lr: 0.00150
#> Loss at epoch 54: 0.118471, lr: 0.00150
#> Loss at epoch 55: 0.117485, lr: 0.00150
#> Loss at epoch 56: 0.115518, lr: 0.00150
#> Loss at epoch 57: 0.115262, lr: 0.00150
#> Loss at epoch 58: 0.116068, lr: 0.00150
#> Loss at epoch 59: 0.116858, lr: 0.00150
#> Loss at epoch 60: 0.116840, lr: 0.00022
#> Loss at epoch 61: 0.114647, lr: 0.00022
#> Loss at epoch 62: 0.114480, lr: 0.00022
#> Loss at epoch 63: 0.114480, lr: 0.00022
#> Loss at epoch 64: 0.114803, lr: 0.00022
#> Loss at epoch 65: 0.114717, lr: 0.00022
#> Loss at epoch 66: 0.114774, lr: 0.00022
#> Loss at epoch 67: 0.114734, lr: 0.00022
#> Loss at epoch 68: 0.114645, lr: 0.00022
#> Loss at epoch 69: 0.114611, lr: 0.00022
#> Loss at epoch 70: 0.114535, lr: 0.00022
#> Loss at epoch 71: 0.114489, lr: 0.00022
#> Loss at epoch 72: 0.114253, lr: 0.00022
#> Loss at epoch 73: 0.114483, lr: 0.00022
#> Loss at epoch 74: 0.114663, lr: 0.00022
#> Loss at epoch 75: 0.114457, lr: 0.00022
#> Loss at epoch 76: 0.114511, lr: 0.00022
#> Loss at epoch 77: 0.114223, lr: 0.00022
#> Loss at epoch 78: 0.114277, lr: 0.00022
#> Loss at epoch 79: 0.114388, lr: 0.00022
#> Loss at epoch 80: 0.114425, lr: 0.00022
#> Loss at epoch 81: 0.114293, lr: 0.00022
#> Loss at epoch 82: 0.114324, lr: 0.00022
#> Loss at epoch 83: 0.114444, lr: 0.00022
#> Loss at epoch 84: 0.114231, lr: 0.00022
#> Loss at epoch 85: 0.114102, lr: 0.00022
#> Loss at epoch 86: 0.114213, lr: 0.00022
#> Loss at epoch 87: 0.114102, lr: 0.00022
#> Loss at epoch 88: 0.114284, lr: 0.00022
#> Loss at epoch 89: 0.114198, lr: 0.00022
#> Loss at epoch 90: 0.113974, lr: 0.00003
#> Loss at epoch 91: 0.113833, lr: 0.00003
#> Loss at epoch 92: 0.113825, lr: 0.00003
#> Loss at epoch 93: 0.113835, lr: 0.00003
#> Loss at epoch 94: 0.113864, lr: 0.00003
#> Loss at epoch 95: 0.113823, lr: 0.00003
#> Loss at epoch 96: 0.113790, lr: 0.00003
#> Loss at epoch 97: 0.113815, lr: 0.00003
#> Loss at epoch 98: 0.113830, lr: 0.00003
#> Loss at epoch 99: 0.113818, lr: 0.00003
#> Loss at epoch 100: 0.113787, lr: 0.00003
# }
```
