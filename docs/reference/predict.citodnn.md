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
#> Loss at epoch 1: 4.573890, lr: 0.01000

#> Loss at epoch 2: 0.385768, lr: 0.01000
#> Loss at epoch 3: 0.243814, lr: 0.01000
#> Loss at epoch 4: 0.223834, lr: 0.01000
#> Loss at epoch 5: 0.172706, lr: 0.01000
#> Loss at epoch 6: 0.259235, lr: 0.01000
#> Loss at epoch 7: 0.277559, lr: 0.01000
#> Loss at epoch 8: 0.139885, lr: 0.01000
#> Loss at epoch 9: 0.214819, lr: 0.01000
#> Loss at epoch 10: 0.123079, lr: 0.01000
#> Loss at epoch 11: 0.187724, lr: 0.01000
#> Loss at epoch 12: 0.156052, lr: 0.01000
#> Loss at epoch 13: 0.194120, lr: 0.01000
#> Loss at epoch 14: 0.141890, lr: 0.01000
#> Loss at epoch 15: 0.148867, lr: 0.01000
#> Loss at epoch 16: 0.324458, lr: 0.01000
#> Loss at epoch 17: 0.161580, lr: 0.01000
#> Loss at epoch 18: 0.189428, lr: 0.01000
#> Loss at epoch 19: 0.139566, lr: 0.01000
#> Loss at epoch 20: 0.126003, lr: 0.01000
#> Loss at epoch 21: 0.199198, lr: 0.01000
#> Loss at epoch 22: 0.161998, lr: 0.01000
#> Loss at epoch 23: 0.153942, lr: 0.01000
#> Loss at epoch 24: 0.147695, lr: 0.01000
#> Loss at epoch 25: 0.252663, lr: 0.01000
#> Loss at epoch 26: 0.162165, lr: 0.01000
#> Loss at epoch 27: 0.170533, lr: 0.01000
#> Loss at epoch 28: 0.151931, lr: 0.01000
#> Loss at epoch 29: 0.247082, lr: 0.01000
#> Loss at epoch 30: 0.187079, lr: 0.01000
#> Loss at epoch 31: 0.130460, lr: 0.01000
#> Loss at epoch 32: 0.120020, lr: 0.01000
#> Loss at epoch 33: 0.105801, lr: 0.01000
#> Loss at epoch 34: 0.126228, lr: 0.01000
#> Loss at epoch 35: 0.178666, lr: 0.01000
#> Loss at epoch 36: 0.171616, lr: 0.01000
#> Loss at epoch 37: 0.133888, lr: 0.01000
#> Loss at epoch 38: 0.132859, lr: 0.01000
#> Loss at epoch 39: 0.214757, lr: 0.01000
#> Loss at epoch 40: 0.140986, lr: 0.01000
#> Loss at epoch 41: 0.119952, lr: 0.01000
#> Loss at epoch 42: 0.129326, lr: 0.01000
#> Loss at epoch 43: 0.159073, lr: 0.01000
#> Loss at epoch 44: 0.216206, lr: 0.01000
#> Loss at epoch 45: 0.131447, lr: 0.01000
#> Loss at epoch 46: 0.260662, lr: 0.01000
#> Loss at epoch 47: 0.171616, lr: 0.01000
#> Loss at epoch 48: 0.193276, lr: 0.01000
#> Loss at epoch 49: 0.177814, lr: 0.01000
#> Loss at epoch 50: 0.113384, lr: 0.01000
#> Loss at epoch 51: 0.254792, lr: 0.01000
#> Loss at epoch 52: 0.253987, lr: 0.01000
#> Loss at epoch 53: 0.157166, lr: 0.01000
#> Loss at epoch 54: 0.125427, lr: 0.01000
#> Loss at epoch 55: 0.112757, lr: 0.01000
#> Loss at epoch 56: 0.142917, lr: 0.01000
#> Loss at epoch 57: 0.122251, lr: 0.01000
#> Loss at epoch 58: 0.158645, lr: 0.01000
#> Loss at epoch 59: 0.164446, lr: 0.01000
#> Loss at epoch 60: 0.112557, lr: 0.01000
#> Loss at epoch 61: 0.120916, lr: 0.01000
#> Loss at epoch 62: 0.100177, lr: 0.01000
#> Loss at epoch 63: 0.198976, lr: 0.01000
#> Loss at epoch 64: 0.118618, lr: 0.01000
#> Loss at epoch 65: 0.112047, lr: 0.01000
#> Loss at epoch 66: 0.153683, lr: 0.01000
#> Loss at epoch 67: 0.142675, lr: 0.01000
#> Loss at epoch 68: 0.119044, lr: 0.01000
#> Loss at epoch 69: 0.128290, lr: 0.01000
#> Loss at epoch 70: 0.156276, lr: 0.01000
#> Loss at epoch 71: 0.115214, lr: 0.01000
#> Loss at epoch 72: 0.186600, lr: 0.01000
#> Loss at epoch 73: 0.134912, lr: 0.01000
#> Loss at epoch 74: 0.163303, lr: 0.01000
#> Loss at epoch 75: 0.114479, lr: 0.01000
#> Loss at epoch 76: 0.211622, lr: 0.01000
#> Loss at epoch 77: 0.205889, lr: 0.01000
#> Loss at epoch 78: 0.139686, lr: 0.01000
#> Loss at epoch 79: 0.103429, lr: 0.01000
#> Loss at epoch 80: 0.177321, lr: 0.01000
#> Loss at epoch 81: 0.148383, lr: 0.01000
#> Loss at epoch 82: 0.146238, lr: 0.01000
#> Loss at epoch 83: 0.190641, lr: 0.01000
#> Loss at epoch 84: 0.145082, lr: 0.01000
#> Loss at epoch 85: 0.128550, lr: 0.01000
#> Loss at epoch 86: 0.126223, lr: 0.01000
#> Loss at epoch 87: 0.148261, lr: 0.01000
#> Loss at epoch 88: 0.136706, lr: 0.01000
#> Loss at epoch 89: 0.100412, lr: 0.01000
#> Loss at epoch 90: 0.208544, lr: 0.01000
#> Loss at epoch 91: 0.146686, lr: 0.01000
#> Loss at epoch 92: 0.157992, lr: 0.01000
#> Loss at epoch 93: 0.172056, lr: 0.01000
#> Loss at epoch 94: 0.188673, lr: 0.01000
#> Loss at epoch 95: 0.166790, lr: 0.01000
#> Loss at epoch 96: 0.137210, lr: 0.01000
#> Loss at epoch 97: 0.111124, lr: 0.01000
#> Loss at epoch 98: 0.147368, lr: 0.01000
#> Loss at epoch 99: 0.130184, lr: 0.01000
#> Loss at epoch 100: 0.116421, lr: 0.01000

#> [1] 0.3158731
# }
```
