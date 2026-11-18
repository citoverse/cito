# Creation of customized optimizer objects

Helps you create custom optimizer for [`dnn`](dnn.md). It is recommended
to set learning rate in [`dnn`](dnn.md).

## Usage

``` r
config_optimizer(
  type = c("adam", "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "ignite_adam"),
  verbose = FALSE,
  ...
)
```

## Arguments

- type:

  character string defining which optimizer should be used. See Details.

- verbose:

  If TRUE, additional information about scheduler will be printed to
  console

- ...:

  additional arguments to be passed to optimizer. See Details.

## Value

object of class cito_optim to give to [`dnn`](dnn.md)

## Details

different optimizer need different variables, this function will tell
you how the variables are set. For more information see the
corresponding functions:

- adam:
  [`optim_adam`](https://torch.mlverse.org/docs/reference/optim_adam.html)

- adadelta:
  [`optim_adadelta`](https://torch.mlverse.org/docs/reference/optim_adadelta.html)

- adagrad:
  [`optim_adagrad`](https://torch.mlverse.org/docs/reference/optim_adagrad.html)

- rmsprop:
  [`optim_rmsprop`](https://torch.mlverse.org/docs/reference/optim_rmsprop.html)

- rprop:
  [`optim_rprop`](https://torch.mlverse.org/docs/reference/optim_rprop.html)

- sgd:
  [`optim_sgd`](https://torch.mlverse.org/docs/reference/optim_sgd.html)

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

# create optimizer object
opt <- config_optimizer(type = "adagrad",
                        lr_decay = 1e-04,
                        weight_decay = 0.1,
                        verbose = TRUE)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris, optimizer = opt)

}
#> set adagrad optimizer with following values 
#> lr_decay: [1e-04] 
#> weight_decay: [0.1] 
#> initial_accumulator_value: [0] 
#> eps: [1e-10] 
#> Loss at epoch 1: 4.216852, lr: 0.01000

#> Loss at epoch 2: 0.157141, lr: 0.01000
#> Loss at epoch 3: 0.145019, lr: 0.01000
#> Loss at epoch 4: 0.136237, lr: 0.01000
#> Loss at epoch 5: 0.132625, lr: 0.01000
#> Loss at epoch 6: 0.131908, lr: 0.01000
#> Loss at epoch 7: 0.133369, lr: 0.01000
#> Loss at epoch 8: 0.139022, lr: 0.01000
#> Loss at epoch 9: 0.131191, lr: 0.01000
#> Loss at epoch 10: 0.129604, lr: 0.01000
#> Loss at epoch 11: 0.132381, lr: 0.01000
#> Loss at epoch 12: 0.131529, lr: 0.01000
#> Loss at epoch 13: 0.129357, lr: 0.01000
#> Loss at epoch 14: 0.127578, lr: 0.01000
#> Loss at epoch 15: 0.125515, lr: 0.01000
#> Loss at epoch 16: 0.134795, lr: 0.01000
#> Loss at epoch 17: 0.129663, lr: 0.01000
#> Loss at epoch 18: 0.127623, lr: 0.01000
#> Loss at epoch 19: 0.125675, lr: 0.01000
#> Loss at epoch 20: 0.124247, lr: 0.01000
#> Loss at epoch 21: 0.123027, lr: 0.01000
#> Loss at epoch 22: 0.124162, lr: 0.01000
#> Loss at epoch 23: 0.121896, lr: 0.01000
#> Loss at epoch 24: 0.125213, lr: 0.01000
#> Loss at epoch 25: 0.126171, lr: 0.01000
#> Loss at epoch 26: 0.123042, lr: 0.01000
#> Loss at epoch 27: 0.120113, lr: 0.01000
#> Loss at epoch 28: 0.122067, lr: 0.01000
#> Loss at epoch 29: 0.119107, lr: 0.01000
#> Loss at epoch 30: 0.122505, lr: 0.01000
#> Loss at epoch 31: 0.123301, lr: 0.01000
#> Loss at epoch 32: 0.127342, lr: 0.01000
#> Loss at epoch 33: 0.122316, lr: 0.01000
#> Loss at epoch 34: 0.118554, lr: 0.01000
#> Loss at epoch 35: 0.120448, lr: 0.01000
#> Loss at epoch 36: 0.124632, lr: 0.01000
#> Loss at epoch 37: 0.126454, lr: 0.01000
#> Loss at epoch 38: 0.119805, lr: 0.01000
#> Loss at epoch 39: 0.117733, lr: 0.01000
#> Loss at epoch 40: 0.121600, lr: 0.01000
#> Loss at epoch 41: 0.117136, lr: 0.01000
#> Loss at epoch 42: 0.117968, lr: 0.01000
#> Loss at epoch 43: 0.119234, lr: 0.01000
#> Loss at epoch 44: 0.119292, lr: 0.01000
#> Loss at epoch 45: 0.117126, lr: 0.01000
#> Loss at epoch 46: 0.114422, lr: 0.01000
#> Loss at epoch 47: 0.119548, lr: 0.01000
#> Loss at epoch 48: 0.117122, lr: 0.01000
#> Loss at epoch 49: 0.118077, lr: 0.01000
#> Loss at epoch 50: 0.119129, lr: 0.01000
#> Loss at epoch 51: 0.114879, lr: 0.01000
#> Loss at epoch 52: 0.117239, lr: 0.01000
#> Loss at epoch 53: 0.117520, lr: 0.01000
#> Loss at epoch 54: 0.116476, lr: 0.01000
#> Loss at epoch 55: 0.115694, lr: 0.01000
#> Loss at epoch 56: 0.115998, lr: 0.01000
#> Loss at epoch 57: 0.120413, lr: 0.01000
#> Loss at epoch 58: 0.113942, lr: 0.01000
#> Loss at epoch 59: 0.115640, lr: 0.01000
#> Loss at epoch 60: 0.115105, lr: 0.01000
#> Loss at epoch 61: 0.115010, lr: 0.01000
#> Loss at epoch 62: 0.114765, lr: 0.01000
#> Loss at epoch 63: 0.113056, lr: 0.01000
#> Loss at epoch 64: 0.119288, lr: 0.01000
#> Loss at epoch 65: 0.112949, lr: 0.01000
#> Loss at epoch 66: 0.114677, lr: 0.01000
#> Loss at epoch 67: 0.116163, lr: 0.01000
#> Loss at epoch 68: 0.117011, lr: 0.01000
#> Loss at epoch 69: 0.112602, lr: 0.01000
#> Loss at epoch 70: 0.119381, lr: 0.01000
#> Loss at epoch 71: 0.115507, lr: 0.01000
#> Loss at epoch 72: 0.114732, lr: 0.01000
#> Loss at epoch 73: 0.116947, lr: 0.01000
#> Loss at epoch 74: 0.118656, lr: 0.01000
#> Loss at epoch 75: 0.113956, lr: 0.01000
#> Loss at epoch 76: 0.111115, lr: 0.01000
#> Loss at epoch 77: 0.115680, lr: 0.01000
#> Loss at epoch 78: 0.113466, lr: 0.01000
#> Loss at epoch 79: 0.112957, lr: 0.01000
#> Loss at epoch 80: 0.116062, lr: 0.01000
#> Loss at epoch 81: 0.111303, lr: 0.01000
#> Loss at epoch 82: 0.112095, lr: 0.01000
#> Loss at epoch 83: 0.112903, lr: 0.01000
#> Loss at epoch 84: 0.113131, lr: 0.01000
#> Loss at epoch 85: 0.114929, lr: 0.01000
#> Loss at epoch 86: 0.111662, lr: 0.01000
#> Loss at epoch 87: 0.115718, lr: 0.01000
#> Loss at epoch 88: 0.111661, lr: 0.01000
#> Loss at epoch 89: 0.114583, lr: 0.01000
#> Loss at epoch 90: 0.114628, lr: 0.01000
#> Loss at epoch 91: 0.116678, lr: 0.01000
#> Loss at epoch 92: 0.115223, lr: 0.01000
#> Loss at epoch 93: 0.114077, lr: 0.01000
#> Loss at epoch 94: 0.113078, lr: 0.01000
#> Loss at epoch 95: 0.115988, lr: 0.01000
#> Loss at epoch 96: 0.110276, lr: 0.01000
#> Loss at epoch 97: 0.110310, lr: 0.01000
#> Loss at epoch 98: 0.112597, lr: 0.01000
#> Loss at epoch 99: 0.110904, lr: 0.01000
#> Loss at epoch 100: 0.113247, lr: 0.01000
# }
```
