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
  continuing on the original data (citodnn and citocnn only)

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
#> Loss at epoch 1: 5.179155, lr: 0.01000

#> Loss at epoch 2: 0.242789, lr: 0.01000
#> Loss at epoch 3: 0.138856, lr: 0.01000
#> Loss at epoch 4: 0.230187, lr: 0.01000
#> Loss at epoch 5: 0.165665, lr: 0.01000
#> Loss at epoch 6: 0.178567, lr: 0.01000
#> Loss at epoch 7: 0.158081, lr: 0.01000
#> Loss at epoch 8: 0.220098, lr: 0.01000
#> Loss at epoch 9: 0.442415, lr: 0.01000
#> Loss at epoch 10: 0.200429, lr: 0.01000
#> Loss at epoch 11: 0.151000, lr: 0.01000
#> Loss at epoch 12: 0.130765, lr: 0.01000
#> Loss at epoch 13: 0.132458, lr: 0.01000
#> Loss at epoch 14: 0.191742, lr: 0.01000
#> Loss at epoch 15: 0.155525, lr: 0.01000
#> Loss at epoch 16: 0.181862, lr: 0.01000
#> Loss at epoch 17: 0.192423, lr: 0.01000
#> Loss at epoch 18: 0.194776, lr: 0.01000
#> Loss at epoch 19: 0.218143, lr: 0.01000
#> Loss at epoch 20: 0.141050, lr: 0.01000
#> Loss at epoch 21: 0.136196, lr: 0.01000
#> Loss at epoch 22: 0.113070, lr: 0.01000
#> Loss at epoch 23: 0.145832, lr: 0.01000
#> Loss at epoch 24: 0.136214, lr: 0.01000
#> Loss at epoch 25: 0.189159, lr: 0.01000
#> Loss at epoch 26: 0.184259, lr: 0.01000
#> Loss at epoch 27: 0.153862, lr: 0.01000
#> Loss at epoch 28: 0.223003, lr: 0.01000
#> Loss at epoch 29: 0.152020, lr: 0.01000
#> Loss at epoch 30: 0.160712, lr: 0.01000
#> Loss at epoch 31: 0.166530, lr: 0.01000
#> Loss at epoch 32: 0.158961, lr: 0.01000
#> Loss at epoch 33: 0.165860, lr: 0.01000

#> Loss at epoch 34: 0.117588, lr: 0.01000
#> Loss at epoch 35: 0.182465, lr: 0.01000
#> Loss at epoch 36: 0.141999, lr: 0.01000
#> Loss at epoch 37: 0.206218, lr: 0.01000
#> Loss at epoch 38: 0.148922, lr: 0.01000
#> Loss at epoch 39: 0.209128, lr: 0.01000
#> Loss at epoch 40: 0.316701, lr: 0.01000
#> Loss at epoch 41: 0.208343, lr: 0.01000
#> Loss at epoch 42: 0.162883, lr: 0.01000
#> Loss at epoch 43: 0.143766, lr: 0.01000
#> Loss at epoch 44: 0.140393, lr: 0.01000
#> Loss at epoch 45: 0.139290, lr: 0.01000
#> Loss at epoch 46: 0.134789, lr: 0.01000
#> Loss at epoch 47: 0.137013, lr: 0.01000
#> Loss at epoch 48: 0.117598, lr: 0.01000
#> Loss at epoch 49: 0.129818, lr: 0.01000
#> Loss at epoch 50: 0.137816, lr: 0.01000
#> Loss at epoch 51: 0.241536, lr: 0.01000
#> Loss at epoch 52: 0.168550, lr: 0.01000
#> Loss at epoch 53: 0.280494, lr: 0.01000
#> Loss at epoch 54: 0.132583, lr: 0.01000
#> Loss at epoch 55: 0.162896, lr: 0.01000
#> Loss at epoch 56: 0.150895, lr: 0.01000
#> Loss at epoch 57: 0.121088, lr: 0.01000
#> Loss at epoch 58: 0.120317, lr: 0.01000
#> Loss at epoch 59: 0.126545, lr: 0.01000
#> Loss at epoch 60: 0.157084, lr: 0.01000
#> Loss at epoch 61: 0.119999, lr: 0.01000
#> Loss at epoch 62: 0.150630, lr: 0.01000
#> Loss at epoch 63: 0.260959, lr: 0.01000
#> Loss at epoch 64: 0.149024, lr: 0.01000
# }
```
