# Predict from a fitted dnn model

Predict from a fitted dnn model

## Usage

``` r
# S3 method for class 'citodnn'
predict(
  object,
  newdata = NULL,
  type = c("link", "response", "class"),
  device = NULL,
  batchsize = NULL,
  return_embeddings = FALSE,
  ...
)

# S3 method for class 'citodnnBootstrap'
predict(
  object,
  newdata = NULL,
  type = c("link", "response", "class"),
  device = NULL,
  batchsize = NULL,
  reduce = c("mean", "median", "none"),
  ...
)
```

## Arguments

- object:

  a model created by [`dnn`](dnn.md)

- newdata:

  new data for predictions

- type:

  type of predictions. The default is on the scale of the linear
  predictor, "response" is on the scale of the response, and "class"
  means that class predictions are returned (if it is a classification
  task)

- device:

  device on which network should be trained on.

- batchsize:

  number of samples that are predicted at the same time

- return_embeddings:

  Return embeddings instead of predictions

- ...:

  additional arguments

- reduce:

  predictions from bootstrapped model are by default reduced (mean,
  optional median or none)

## Value

prediction matrix

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)
validation_set<- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])

# Use model on validation set
predictions <- predict(nn.fit, iris[validation_set,])
# Scatterplot
plot(iris[validation_set,]$Sepal.Length,predictions)
# MAE
mean(abs(predictions-iris[validation_set,]$Sepal.Length))
}
#> Loss at epoch 1: 3.319235, lr: 0.01000

#> Loss at epoch 2: 0.207155, lr: 0.01000
#> Loss at epoch 3: 0.191421, lr: 0.01000
#> Loss at epoch 4: 0.248962, lr: 0.01000
#> Loss at epoch 5: 0.191070, lr: 0.01000
#> Loss at epoch 6: 0.174952, lr: 0.01000
#> Loss at epoch 7: 0.232722, lr: 0.01000
#> Loss at epoch 8: 0.226452, lr: 0.01000
#> Loss at epoch 9: 0.177707, lr: 0.01000
#> Loss at epoch 10: 0.245076, lr: 0.01000
#> Loss at epoch 11: 0.154363, lr: 0.01000
#> Loss at epoch 12: 0.209630, lr: 0.01000
#> Loss at epoch 13: 0.202320, lr: 0.01000
#> Loss at epoch 14: 0.170126, lr: 0.01000
#> Loss at epoch 15: 0.147711, lr: 0.01000
#> Loss at epoch 16: 0.157096, lr: 0.01000
#> Loss at epoch 17: 0.164510, lr: 0.01000
#> Loss at epoch 18: 0.138427, lr: 0.01000
#> Loss at epoch 19: 0.217221, lr: 0.01000
#> Loss at epoch 20: 0.263539, lr: 0.01000
#> Loss at epoch 21: 0.165495, lr: 0.01000
#> Loss at epoch 22: 0.141583, lr: 0.01000
#> Loss at epoch 23: 0.161657, lr: 0.01000
#> Loss at epoch 24: 0.132694, lr: 0.01000
#> Loss at epoch 25: 0.167883, lr: 0.01000
#> Loss at epoch 26: 0.175828, lr: 0.01000
#> Loss at epoch 27: 0.139946, lr: 0.01000
#> Loss at epoch 28: 0.190810, lr: 0.01000
#> Loss at epoch 29: 0.213504, lr: 0.01000
#> Loss at epoch 30: 0.168424, lr: 0.01000
#> Loss at epoch 31: 0.148118, lr: 0.01000
#> Loss at epoch 32: 0.183254, lr: 0.01000
#> Loss at epoch 33: 0.143177, lr: 0.01000
#> Loss at epoch 34: 0.196146, lr: 0.01000
#> Loss at epoch 35: 0.123945, lr: 0.01000
#> Loss at epoch 36: 0.113393, lr: 0.01000
#> Loss at epoch 37: 0.129266, lr: 0.01000
#> Loss at epoch 38: 0.197973, lr: 0.01000
#> Loss at epoch 39: 0.125802, lr: 0.01000
#> Loss at epoch 40: 0.143212, lr: 0.01000
#> Loss at epoch 41: 0.229711, lr: 0.01000
#> Loss at epoch 42: 0.169072, lr: 0.01000
#> Loss at epoch 43: 0.146764, lr: 0.01000
#> Loss at epoch 44: 0.140678, lr: 0.01000
#> Loss at epoch 45: 0.139467, lr: 0.01000
#> Loss at epoch 46: 0.134298, lr: 0.01000
#> Loss at epoch 47: 0.172751, lr: 0.01000
#> Loss at epoch 48: 0.153078, lr: 0.01000
#> Loss at epoch 49: 0.184250, lr: 0.01000
#> Loss at epoch 50: 0.137258, lr: 0.01000
#> Loss at epoch 51: 0.160207, lr: 0.01000
#> Loss at epoch 52: 0.161314, lr: 0.01000
#> Loss at epoch 53: 0.145339, lr: 0.01000
#> Loss at epoch 54: 0.228659, lr: 0.01000
#> Loss at epoch 55: 0.206076, lr: 0.01000
#> Loss at epoch 56: 0.140907, lr: 0.01000
#> Loss at epoch 57: 0.184827, lr: 0.01000
#> Loss at epoch 58: 0.143234, lr: 0.01000
#> Loss at epoch 59: 0.123964, lr: 0.01000
#> Loss at epoch 60: 0.146506, lr: 0.01000
#> Loss at epoch 61: 0.118124, lr: 0.01000
#> Loss at epoch 62: 0.133318, lr: 0.01000
#> Loss at epoch 63: 0.112096, lr: 0.01000
#> Loss at epoch 64: 0.104291, lr: 0.01000
#> Loss at epoch 65: 0.113923, lr: 0.01000
#> Loss at epoch 66: 0.141594, lr: 0.01000
#> Loss at epoch 67: 0.124036, lr: 0.01000
#> Loss at epoch 68: 0.110658, lr: 0.01000
#> Loss at epoch 69: 0.176715, lr: 0.01000
#> Loss at epoch 70: 0.131149, lr: 0.01000
#> Loss at epoch 71: 0.124892, lr: 0.01000
#> Loss at epoch 72: 0.174389, lr: 0.01000
#> Loss at epoch 73: 0.102673, lr: 0.01000
#> Loss at epoch 74: 0.131373, lr: 0.01000
#> Loss at epoch 75: 0.225670, lr: 0.01000
#> Loss at epoch 76: 0.112805, lr: 0.01000
#> Loss at epoch 77: 0.148229, lr: 0.01000
#> Loss at epoch 78: 0.127204, lr: 0.01000
#> Loss at epoch 79: 0.231458, lr: 0.01000
#> Loss at epoch 80: 0.234797, lr: 0.01000
#> Loss at epoch 81: 0.160555, lr: 0.01000
#> Loss at epoch 82: 0.126203, lr: 0.01000
#> Loss at epoch 83: 0.109658, lr: 0.01000
#> Loss at epoch 84: 0.135907, lr: 0.01000
#> Loss at epoch 85: 0.142485, lr: 0.01000
#> Loss at epoch 86: 0.103260, lr: 0.01000
#> Loss at epoch 87: 0.097652, lr: 0.01000
#> Loss at epoch 88: 0.127592, lr: 0.01000
#> Loss at epoch 89: 0.121840, lr: 0.01000
#> Loss at epoch 90: 0.136958, lr: 0.01000
#> Loss at epoch 91: 0.133155, lr: 0.01000
#> Loss at epoch 92: 0.162306, lr: 0.01000
#> Loss at epoch 93: 0.305924, lr: 0.01000
#> Loss at epoch 94: 0.222982, lr: 0.01000
#> Loss at epoch 95: 0.120529, lr: 0.01000
#> Loss at epoch 96: 0.138657, lr: 0.01000
#> Loss at epoch 97: 0.166077, lr: 0.01000
#> Loss at epoch 98: 0.128268, lr: 0.01000
#> Loss at epoch 99: 0.151711, lr: 0.01000
#> Loss at epoch 100: 0.168395, lr: 0.01000

#> [1] 0.2630849
# }
```
