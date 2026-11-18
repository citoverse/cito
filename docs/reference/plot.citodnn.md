# Creates graph plot which gives an overview of the network architecture.

Creates graph plot which gives an overview of the network architecture.

## Usage

``` r
# S3 method for class 'citodnn'
plot(x, plot_type = c("ALE", "arch"), node_size = 1, scale_edges = FALSE, ...)

# S3 method for class 'citodnnBootstrap'
plot(
  x,
  plot_type = c("ALE", "arch"),
  node_size = 1,
  scale_edges = FALSE,
  which_model = 1,
  ...
)
```

## Arguments

- x:

  a model created by [`dnn`](dnn.md)

- plot_type:

  plot ALE or architecture

- node_size:

  size of node in plot

- scale_edges:

  edge weight gets scaled according to other weights (layer specific)

- ...:

  no further functionality implemented yet

- which_model:

  which model from the ensemble should be plotted

## Value

A plot made with 'ggraph' + 'igraph' that represents the neural network

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){
library(cito)

set.seed(222)
validation_set<- sample(c(1:nrow(datasets::iris)),25)

# Build and train  Network
nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,])

plot(nn.fit)
}
#> Loss at epoch 1: 5.448213, lr: 0.01000

#> Loss at epoch 2: 0.256391, lr: 0.01000
#> Loss at epoch 3: 0.224851, lr: 0.01000
#> Loss at epoch 4: 0.199845, lr: 0.01000
#> Loss at epoch 5: 0.207671, lr: 0.01000
#> Loss at epoch 6: 0.199086, lr: 0.01000
#> Loss at epoch 7: 0.179258, lr: 0.01000
#> Loss at epoch 8: 0.204761, lr: 0.01000
#> Loss at epoch 9: 0.184657, lr: 0.01000
#> Loss at epoch 10: 0.237917, lr: 0.01000
#> Loss at epoch 11: 0.209842, lr: 0.01000
#> Loss at epoch 12: 0.277750, lr: 0.01000
#> Loss at epoch 13: 0.167967, lr: 0.01000
#> Loss at epoch 14: 0.149861, lr: 0.01000
#> Loss at epoch 15: 0.152976, lr: 0.01000
#> Loss at epoch 16: 0.250413, lr: 0.01000
#> Loss at epoch 17: 0.162631, lr: 0.01000
#> Loss at epoch 18: 0.164008, lr: 0.01000
#> Loss at epoch 19: 0.189430, lr: 0.01000
#> Loss at epoch 20: 0.139306, lr: 0.01000
#> Loss at epoch 21: 0.155455, lr: 0.01000
#> Loss at epoch 22: 0.159127, lr: 0.01000
#> Loss at epoch 23: 0.194863, lr: 0.01000
#> Loss at epoch 24: 0.217389, lr: 0.01000
#> Loss at epoch 25: 0.161585, lr: 0.01000
#> Loss at epoch 26: 0.183323, lr: 0.01000
#> Loss at epoch 27: 0.214789, lr: 0.01000
#> Loss at epoch 28: 0.125333, lr: 0.01000
#> Loss at epoch 29: 0.177430, lr: 0.01000
#> Loss at epoch 30: 0.153871, lr: 0.01000
#> Loss at epoch 31: 0.142046, lr: 0.01000
#> Loss at epoch 32: 0.183660, lr: 0.01000
#> Loss at epoch 33: 0.156611, lr: 0.01000
#> Loss at epoch 34: 0.127325, lr: 0.01000
#> Loss at epoch 35: 0.154049, lr: 0.01000
#> Loss at epoch 36: 0.155599, lr: 0.01000
#> Loss at epoch 37: 0.142751, lr: 0.01000
#> Loss at epoch 38: 0.163961, lr: 0.01000
#> Loss at epoch 39: 0.214390, lr: 0.01000
#> Loss at epoch 40: 0.178946, lr: 0.01000
#> Loss at epoch 41: 0.184023, lr: 0.01000
#> Loss at epoch 42: 0.264345, lr: 0.01000
#> Loss at epoch 43: 0.117405, lr: 0.01000
#> Loss at epoch 44: 0.158093, lr: 0.01000
#> Loss at epoch 45: 0.199588, lr: 0.01000
#> Loss at epoch 46: 0.174158, lr: 0.01000
#> Loss at epoch 47: 0.143385, lr: 0.01000
#> Loss at epoch 48: 0.156476, lr: 0.01000
#> Loss at epoch 49: 0.168904, lr: 0.01000
#> Loss at epoch 50: 0.145329, lr: 0.01000
#> Loss at epoch 51: 0.118444, lr: 0.01000
#> Loss at epoch 52: 0.176269, lr: 0.01000
#> Loss at epoch 53: 0.179569, lr: 0.01000
#> Loss at epoch 54: 0.242511, lr: 0.01000
#> Loss at epoch 55: 0.153843, lr: 0.01000
#> Loss at epoch 56: 0.233522, lr: 0.01000
#> Loss at epoch 57: 0.139065, lr: 0.01000
#> Loss at epoch 58: 0.126802, lr: 0.01000
#> Loss at epoch 59: 0.140868, lr: 0.01000
#> Loss at epoch 60: 0.131983, lr: 0.01000
#> Loss at epoch 61: 0.307741, lr: 0.01000
#> Loss at epoch 62: 0.133301, lr: 0.01000
#> Loss at epoch 63: 0.132552, lr: 0.01000
#> Loss at epoch 64: 0.151220, lr: 0.01000
#> Loss at epoch 65: 0.153869, lr: 0.01000
#> Loss at epoch 66: 0.128606, lr: 0.01000
#> Loss at epoch 67: 0.135683, lr: 0.01000
#> Loss at epoch 68: 0.199559, lr: 0.01000
#> Loss at epoch 69: 0.171522, lr: 0.01000
#> Loss at epoch 70: 0.156211, lr: 0.01000
#> Loss at epoch 71: 0.127242, lr: 0.01000
#> Loss at epoch 72: 0.118136, lr: 0.01000
#> Loss at epoch 73: 0.150886, lr: 0.01000
#> Loss at epoch 74: 0.171071, lr: 0.01000
#> Loss at epoch 75: 0.110202, lr: 0.01000
#> Loss at epoch 76: 0.138654, lr: 0.01000
#> Loss at epoch 77: 0.152562, lr: 0.01000
#> Loss at epoch 78: 0.128503, lr: 0.01000
#> Loss at epoch 79: 0.148810, lr: 0.01000
#> Loss at epoch 80: 0.131882, lr: 0.01000
#> Loss at epoch 81: 0.119503, lr: 0.01000
#> Loss at epoch 82: 0.193478, lr: 0.01000
#> Loss at epoch 83: 0.126298, lr: 0.01000
#> Loss at epoch 84: 0.168286, lr: 0.01000
#> Loss at epoch 85: 0.157873, lr: 0.01000
#> Loss at epoch 86: 0.158569, lr: 0.01000
#> Loss at epoch 87: 0.107234, lr: 0.01000
#> Loss at epoch 88: 0.116558, lr: 0.01000
#> Loss at epoch 89: 0.119927, lr: 0.01000
#> Loss at epoch 90: 0.257737, lr: 0.01000
#> Loss at epoch 91: 0.111558, lr: 0.01000
#> Loss at epoch 92: 0.160242, lr: 0.01000
#> Loss at epoch 93: 0.163679, lr: 0.01000
#> Loss at epoch 94: 0.106388, lr: 0.01000
#> Loss at epoch 95: 0.106353, lr: 0.01000
#> Loss at epoch 96: 0.126579, lr: 0.01000
#> Loss at epoch 97: 0.139948, lr: 0.01000
#> Loss at epoch 98: 0.135301, lr: 0.01000
#> Loss at epoch 99: 0.163574, lr: 0.01000
#> Loss at epoch 100: 0.264516, lr: 0.01000
#> Categorical features are not yet supported.

# }
```
