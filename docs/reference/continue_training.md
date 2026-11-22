# Continues training of a model generated with [`dnn`](dnn.md), [`cnn`](cnn.md) or [`mmn`](mmn.md) for additional epochs.

If the training/validation loss is still decreasing at the end of the
training, it is often a sign that the NN has not yet converged. You can
use this function to continue training instead of re-training the entire
model.

## Usage

``` r
continue_training(model, ...)

# S3 method for class 'citodnn'
continue_training(
  model,
  epochs = 32,
  data = NULL,
  device = NULL,
  changed_params = NULL,
  init_optimizer = TRUE,
  X = NULL,
  Y = NULL,
  weights = NULL,
  ...
)

# S3 method for class 'citodnnBootstrap'
continue_training(
  model,
  epochs = 32,
  data = NULL,
  device = NULL,
  changed_params = NULL,
  parallel = FALSE,
  init_optimizer = TRUE,
  X = NULL,
  Y = NULL,
  ...
)

# S3 method for class 'citocnn'
continue_training(
  model,
  epochs = 32,
  X = NULL,
  Y = NULL,
  weights = NULL,
  device = NULL,
  changed_params = NULL,
  init_optimizer = TRUE,
  ...
)

# S3 method for class 'citommn'
continue_training(
  model,
  epochs = 32,
  dataList = NULL,
  device = NULL,
  changed_params = NULL,
  init_optimizer = TRUE,
  weights = NULL,
  ...
)
```

## Arguments

- model:

  a model created by [`dnn`](dnn.md), [`cnn`](cnn.md) or [`mmn`](mmn.md)

- ...:

  class-specific arguments

- epochs:

  additional epochs the training should continue for

- data:

  matrix or data.frame. If not provided data from original training will
  be used

- device:

  can be used to overwrite device used in previous training

- changed_params:

  list of arguments to change compared to original training setup, see
  [`dnn`](dnn.md) which parameter can be changed

- init_optimizer:

  re-initialize optimizer or not

- X:

  Predictor data. If not provided X from original training will be used

- Y:

  Target data. If not provided Y from original training will be used

- weights:

  observation weights (vector or matrix) passed to the likelihood. If
  not provided, the weights from the original training are reused when
  continuing on the original data

- parallel:

  train bootstrapped model in parallel

- dataList:

  A list containing the data for training the model. The list should
  contain all variables used in the formula. If not provided dataList
  from original training will be used

## Value

a model of class citodnn, citodnnBootstrap, citocnn or citommn created
by [`dnn`](dnn.md), [`cnn`](cnn.md) or [`mmn`](mmn.md)

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)
validation_set<- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,], epochs = 32)

# continue training for another 32 epochs
nn.fit<- continue_training(nn.fit,epochs = 32)

# Use model on validation set
predictions <- predict(nn.fit, iris[validation_set,])
}
#> Loss at epoch 1: 5.656469, lr: 0.01000

#> Loss at epoch 2: 0.304695, lr: 0.01000
#> Loss at epoch 3: 0.247588, lr: 0.01000
#> Loss at epoch 4: 0.247688, lr: 0.01000
#> Loss at epoch 5: 0.197667, lr: 0.01000
#> Loss at epoch 6: 0.183831, lr: 0.01000
#> Loss at epoch 7: 0.149110, lr: 0.01000
#> Loss at epoch 8: 0.319102, lr: 0.01000
#> Loss at epoch 9: 0.564000, lr: 0.01000
#> Loss at epoch 10: 0.181594, lr: 0.01000
#> Loss at epoch 11: 0.310423, lr: 0.01000
#> Loss at epoch 12: 0.176109, lr: 0.01000
#> Loss at epoch 13: 0.131800, lr: 0.01000
#> Loss at epoch 14: 0.273026, lr: 0.01000
#> Loss at epoch 15: 0.128291, lr: 0.01000
#> Loss at epoch 16: 0.117457, lr: 0.01000
#> Loss at epoch 17: 0.277310, lr: 0.01000
#> Loss at epoch 18: 0.139877, lr: 0.01000
#> Loss at epoch 19: 0.175280, lr: 0.01000
#> Loss at epoch 20: 0.151616, lr: 0.01000
#> Loss at epoch 21: 0.123418, lr: 0.01000
#> Loss at epoch 22: 0.207998, lr: 0.01000
#> Loss at epoch 23: 0.212171, lr: 0.01000
#> Loss at epoch 24: 0.201210, lr: 0.01000
#> Loss at epoch 25: 0.137739, lr: 0.01000
#> Loss at epoch 26: 0.232378, lr: 0.01000
#> Loss at epoch 27: 0.182911, lr: 0.01000
#> Loss at epoch 28: 0.132849, lr: 0.01000
#> Loss at epoch 29: 0.169551, lr: 0.01000
#> Loss at epoch 30: 0.330250, lr: 0.01000
#> Loss at epoch 31: 0.137505, lr: 0.01000
#> Loss at epoch 32: 0.157123, lr: 0.01000
#> Loss at epoch 33: 0.165622, lr: 0.01000

#> Loss at epoch 34: 0.359787, lr: 0.01000
#> Loss at epoch 35: 0.127529, lr: 0.01000
#> Loss at epoch 36: 0.189185, lr: 0.01000
#> Loss at epoch 37: 0.168838, lr: 0.01000
#> Loss at epoch 38: 0.202628, lr: 0.01000
#> Loss at epoch 39: 0.124156, lr: 0.01000
#> Loss at epoch 40: 0.163396, lr: 0.01000
#> Loss at epoch 41: 0.169515, lr: 0.01000
#> Loss at epoch 42: 0.241171, lr: 0.01000
#> Loss at epoch 43: 0.145793, lr: 0.01000
#> Loss at epoch 44: 0.128160, lr: 0.01000
#> Loss at epoch 45: 0.174631, lr: 0.01000
#> Loss at epoch 46: 0.154899, lr: 0.01000
#> Loss at epoch 47: 0.115280, lr: 0.01000
#> Loss at epoch 48: 0.149199, lr: 0.01000
#> Loss at epoch 49: 0.174235, lr: 0.01000
#> Loss at epoch 50: 0.163682, lr: 0.01000
#> Loss at epoch 51: 0.124556, lr: 0.01000
#> Loss at epoch 52: 0.172014, lr: 0.01000
#> Loss at epoch 53: 0.116780, lr: 0.01000
#> Loss at epoch 54: 0.119284, lr: 0.01000
#> Loss at epoch 55: 0.175356, lr: 0.01000
#> Loss at epoch 56: 0.108428, lr: 0.01000
#> Loss at epoch 57: 0.139362, lr: 0.01000
#> Loss at epoch 58: 0.122645, lr: 0.01000
#> Loss at epoch 59: 0.266000, lr: 0.01000
#> Loss at epoch 60: 0.219633, lr: 0.01000
#> Loss at epoch 61: 0.273199, lr: 0.01000
#> Loss at epoch 62: 0.171184, lr: 0.01000
#> Loss at epoch 63: 0.182435, lr: 0.01000
#> Loss at epoch 64: 0.139387, lr: 0.01000
# }
```
