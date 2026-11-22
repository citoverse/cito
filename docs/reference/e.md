# Embeddings

Can be used to create an embedding structure for categorical variables
in the function interface

## Usage

``` r
e(dim = 1L, weights = NULL, train = TRUE, lambda = 0, alpha = 1)
```

## Arguments

- dim:

  integer, embedding dimension

- weights:

  matrix, to use custom embedding matrices

- train:

  logical, should the embeddings be trained or not

- lambda:

  regularization strength on the embeddings

- alpha:

  mix between L1 and L2 regularization

## Value

A list with the settings of the embedding; it is only meaningful inside
the formula of [`dnn`](dnn.md).

## Details

The e() structure must be used in the function interface. Although not
document in the function signature, the first argument to the e()
structure is the categorical variable that codes a group in the data, as
in

predictors + e(group, ...)

For more details, see the example below

## Examples

``` r
# \donttest{
if(torch::torch_is_installed()){

# The following example shows that groups with similar responses will cluster in embedding space
set.seed(123)

n = 10000 # observations
m = 100 # groups / individuals
k = 10 # cluster of groups with the same behavior

dat = data.frame(f1 = runif(n),
                 f2 = runif(n),
                 f3 = runif(n),
                 ind = factor(rep(1:m, each = n/m)),
                 cluster = rep(1:k, each = n/k),
                 response = NA)

slopes = matrix(runif(3*k, min = -10,max = 10), nrow = k, ncol = 3)

for(i in 1:k) dat$response[dat$cluster == i] =
  as.matrix(dat[dat$cluster == i, 1:3]) %*% slopes[i,] + rnorm(n/k, sd = 0.2)

mod <- dnn(response~f1+f2+f3 + e(ind,dim = 2),
           data = dat, epochs = 200L, optimizer = config_optimizer("adam"))

embeddings = coef(mod)[[1]][[1]] # extract embeddings
plot(embeddings, col = c(rep(1:k, each = m/k))) # plot clusters in embedding space
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

ace = conditionalEffects(mod) # extract conditional effects
# now average conditional effects per cluster
ind_ace =
  sapply(1:m, function(ind) {
    tmp = ace[[1]]$result[dat$ind==ind,,]
    return(diag(apply(tmp, 2:3, mean)))
  })

# to create biplot, multiply beta of each cluster with coordinates
coord = ind_ace %*% embeddings/m
arrows(x0 = rep(0, 3), x1 = coord[,1], y0 = rep(0,3), y1 =coord[,2])

}
#> Loss at epoch 1: 28.808414, lr: 0.01000

#> Loss at epoch 2: 27.579052, lr: 0.01000
#> Loss at epoch 3: 26.136549, lr: 0.01000
#> Loss at epoch 4: 24.097060, lr: 0.01000
#> Loss at epoch 5: 21.627275, lr: 0.01000
#> Loss at epoch 6: 19.017823, lr: 0.01000
#> Loss at epoch 7: 16.401340, lr: 0.01000
#> Loss at epoch 8: 13.457431, lr: 0.01000
#> Loss at epoch 9: 10.490485, lr: 0.01000
#> Loss at epoch 10: 8.334878, lr: 0.01000
#> Loss at epoch 11: 6.931188, lr: 0.01000
#> Loss at epoch 12: 5.618284, lr: 0.01000
#> Loss at epoch 13: 4.600126, lr: 0.01000
#> Loss at epoch 14: 3.913898, lr: 0.01000
#> Loss at epoch 15: 3.413744, lr: 0.01000
#> Loss at epoch 16: 3.110019, lr: 0.01000
#> Loss at epoch 17: 2.812318, lr: 0.01000
#> Loss at epoch 18: 2.545130, lr: 0.01000
#> Loss at epoch 19: 2.234862, lr: 0.01000
#> Loss at epoch 20: 1.989430, lr: 0.01000
#> Loss at epoch 21: 1.685810, lr: 0.01000
#> Loss at epoch 22: 1.475590, lr: 0.01000
#> Loss at epoch 23: 1.467252, lr: 0.01000
#> Loss at epoch 24: 1.221288, lr: 0.01000
#> Loss at epoch 25: 1.096738, lr: 0.01000
#> Loss at epoch 26: 1.036046, lr: 0.01000
#> Loss at epoch 27: 1.039389, lr: 0.01000
#> Loss at epoch 28: 1.031590, lr: 0.01000
#> Loss at epoch 29: 1.017889, lr: 0.01000
#> Loss at epoch 30: 0.920405, lr: 0.01000
#> Loss at epoch 31: 0.912439, lr: 0.01000
#> Loss at epoch 32: 0.893330, lr: 0.01000
#> Loss at epoch 33: 0.870546, lr: 0.01000
#> Loss at epoch 34: 0.832111, lr: 0.01000
#> Loss at epoch 35: 0.899931, lr: 0.01000
#> Loss at epoch 36: 0.908461, lr: 0.01000
#> Loss at epoch 37: 0.851893, lr: 0.01000
#> Loss at epoch 38: 0.809910, lr: 0.01000
#> Loss at epoch 39: 0.764929, lr: 0.01000
#> Loss at epoch 40: 0.734078, lr: 0.01000
#> Loss at epoch 41: 0.746736, lr: 0.01000
#> Loss at epoch 42: 0.747699, lr: 0.01000
#> Loss at epoch 43: 0.730525, lr: 0.01000
#> Loss at epoch 44: 0.800820, lr: 0.01000
#> Loss at epoch 45: 0.760904, lr: 0.01000
#> Loss at epoch 46: 0.685496, lr: 0.01000
#> Loss at epoch 47: 0.724847, lr: 0.01000
#> Loss at epoch 48: 0.638501, lr: 0.01000
#> Loss at epoch 49: 0.621704, lr: 0.01000
#> Loss at epoch 50: 0.699206, lr: 0.01000
#> Loss at epoch 51: 0.708276, lr: 0.01000
#> Loss at epoch 52: 0.635200, lr: 0.01000
#> Loss at epoch 53: 0.636321, lr: 0.01000
#> Loss at epoch 54: 0.639015, lr: 0.01000
#> Loss at epoch 55: 0.601725, lr: 0.01000
#> Loss at epoch 56: 0.562358, lr: 0.01000
#> Loss at epoch 57: 0.523153, lr: 0.01000
#> Loss at epoch 58: 0.538413, lr: 0.01000
#> Loss at epoch 59: 0.503697, lr: 0.01000
#> Loss at epoch 60: 0.503141, lr: 0.01000
#> Loss at epoch 61: 0.476629, lr: 0.01000
#> Loss at epoch 62: 0.471042, lr: 0.01000
#> Loss at epoch 63: 0.526554, lr: 0.01000
#> Loss at epoch 64: 0.553730, lr: 0.01000
#> Loss at epoch 65: 0.523286, lr: 0.01000
#> Loss at epoch 66: 0.487354, lr: 0.01000
#> Loss at epoch 67: 0.460857, lr: 0.01000
#> Loss at epoch 68: 0.476241, lr: 0.01000
#> Loss at epoch 69: 0.439576, lr: 0.01000
#> Loss at epoch 70: 0.460905, lr: 0.01000
#> Loss at epoch 71: 0.453706, lr: 0.01000
#> Loss at epoch 72: 0.420935, lr: 0.01000
#> Loss at epoch 73: 0.393940, lr: 0.01000
#> Loss at epoch 74: 0.449222, lr: 0.01000
#> Loss at epoch 75: 0.429993, lr: 0.01000
#> Loss at epoch 76: 0.402848, lr: 0.01000
#> Loss at epoch 77: 0.375280, lr: 0.01000
#> Loss at epoch 78: 0.359321, lr: 0.01000
#> Loss at epoch 79: 0.404511, lr: 0.01000
#> Loss at epoch 80: 0.393714, lr: 0.01000
#> Loss at epoch 81: 0.388586, lr: 0.01000
#> Loss at epoch 82: 0.334882, lr: 0.01000
#> Loss at epoch 83: 0.332760, lr: 0.01000
#> Loss at epoch 84: 0.312033, lr: 0.01000
#> Loss at epoch 85: 0.346529, lr: 0.01000
#> Loss at epoch 86: 0.331494, lr: 0.01000
#> Loss at epoch 87: 0.281968, lr: 0.01000
#> Loss at epoch 88: 0.288941, lr: 0.01000
#> Loss at epoch 89: 0.280467, lr: 0.01000
#> Loss at epoch 90: 0.303269, lr: 0.01000
#> Loss at epoch 91: 0.286400, lr: 0.01000
#> Loss at epoch 92: 0.280039, lr: 0.01000
#> Loss at epoch 93: 0.237098, lr: 0.01000
#> Loss at epoch 94: 0.229388, lr: 0.01000
#> Loss at epoch 95: 0.267268, lr: 0.01000
#> Loss at epoch 96: 0.240142, lr: 0.01000
#> Loss at epoch 97: 0.221005, lr: 0.01000
#> Loss at epoch 98: 0.215475, lr: 0.01000
#> Loss at epoch 99: 0.236126, lr: 0.01000
#> Loss at epoch 100: 0.252091, lr: 0.01000
#> Loss at epoch 101: 0.202551, lr: 0.01000
#> Loss at epoch 102: 0.205654, lr: 0.01000
#> Loss at epoch 103: 0.200615, lr: 0.01000
#> Loss at epoch 104: 0.215997, lr: 0.01000
#> Loss at epoch 105: 0.183831, lr: 0.01000
#> Loss at epoch 106: 0.190352, lr: 0.01000
#> Loss at epoch 107: 0.241495, lr: 0.01000
#> Loss at epoch 108: 0.237090, lr: 0.01000
#> Loss at epoch 109: 0.248107, lr: 0.01000
#> Loss at epoch 110: 0.196178, lr: 0.01000
#> Loss at epoch 111: 0.209498, lr: 0.01000
#> Loss at epoch 112: 0.195294, lr: 0.01000
#> Loss at epoch 113: 0.200690, lr: 0.01000
#> Loss at epoch 114: 0.295335, lr: 0.01000
#> Loss at epoch 115: 0.182344, lr: 0.01000
#> Loss at epoch 116: 0.198195, lr: 0.01000
#> Loss at epoch 117: 0.267625, lr: 0.01000
#> Loss at epoch 118: 0.199162, lr: 0.01000
#> Loss at epoch 119: 0.163652, lr: 0.01000
#> Loss at epoch 120: 0.158686, lr: 0.01000
#> Loss at epoch 121: 0.166704, lr: 0.01000
#> Loss at epoch 122: 0.162565, lr: 0.01000
#> Loss at epoch 123: 0.166294, lr: 0.01000
#> Loss at epoch 124: 0.206432, lr: 0.01000
#> Loss at epoch 125: 0.160008, lr: 0.01000
#> Loss at epoch 126: 0.215209, lr: 0.01000
#> Loss at epoch 127: 0.163689, lr: 0.01000
#> Loss at epoch 128: 0.161233, lr: 0.01000
#> Loss at epoch 129: 0.174699, lr: 0.01000
#> Loss at epoch 130: 0.153034, lr: 0.01000
#> Loss at epoch 131: 0.179113, lr: 0.01000
#> Loss at epoch 132: 0.184850, lr: 0.01000
#> Loss at epoch 133: 0.181077, lr: 0.01000
#> Loss at epoch 134: 0.207208, lr: 0.01000
#> Loss at epoch 135: 0.151543, lr: 0.01000
#> Loss at epoch 136: 0.167370, lr: 0.01000
#> Loss at epoch 137: 0.132924, lr: 0.01000
#> Loss at epoch 138: 0.124959, lr: 0.01000
#> Loss at epoch 139: 0.137884, lr: 0.01000
#> Loss at epoch 140: 0.157829, lr: 0.01000
#> Loss at epoch 141: 0.207818, lr: 0.01000
#> Loss at epoch 142: 0.159209, lr: 0.01000
#> Loss at epoch 143: 0.157718, lr: 0.01000
#> Loss at epoch 144: 0.157310, lr: 0.01000
#> Loss at epoch 145: 0.139731, lr: 0.01000
#> Loss at epoch 146: 0.139482, lr: 0.01000
#> Loss at epoch 147: 0.153456, lr: 0.01000
#> Loss at epoch 148: 0.214660, lr: 0.01000
#> Loss at epoch 149: 0.175453, lr: 0.01000
#> Loss at epoch 150: 0.146519, lr: 0.01000
#> Loss at epoch 151: 0.121811, lr: 0.01000
#> Loss at epoch 152: 0.121329, lr: 0.01000
#> Loss at epoch 153: 0.131863, lr: 0.01000
#> Loss at epoch 154: 0.122491, lr: 0.01000
#> Loss at epoch 155: 0.209654, lr: 0.01000
#> Loss at epoch 156: 0.179252, lr: 0.01000
#> Loss at epoch 157: 0.208965, lr: 0.01000
#> Loss at epoch 158: 0.163047, lr: 0.01000
#> Loss at epoch 159: 0.151977, lr: 0.01000
#> Loss at epoch 160: 0.161357, lr: 0.01000
#> Loss at epoch 161: 0.189848, lr: 0.01000
#> Loss at epoch 162: 0.184438, lr: 0.01000
#> Loss at epoch 163: 0.157250, lr: 0.01000
#> Loss at epoch 164: 0.115647, lr: 0.01000
#> Loss at epoch 165: 0.109501, lr: 0.01000
#> Loss at epoch 166: 0.137830, lr: 0.01000
#> Loss at epoch 167: 0.162000, lr: 0.01000
#> Loss at epoch 168: 0.116460, lr: 0.01000
#> Loss at epoch 169: 0.116286, lr: 0.01000
#> Loss at epoch 170: 0.101002, lr: 0.01000
#> Loss at epoch 171: 0.131016, lr: 0.01000
#> Loss at epoch 172: 0.142925, lr: 0.01000
#> Loss at epoch 173: 0.117518, lr: 0.01000
#> Loss at epoch 174: 0.115240, lr: 0.01000
#> Loss at epoch 175: 0.127956, lr: 0.01000
#> Loss at epoch 176: 0.121842, lr: 0.01000
#> Loss at epoch 177: 0.122701, lr: 0.01000
#> Loss at epoch 178: 0.119916, lr: 0.01000
#> Loss at epoch 179: 0.097183, lr: 0.01000
#> Loss at epoch 180: 0.105021, lr: 0.01000
#> Loss at epoch 181: 0.141565, lr: 0.01000
#> Loss at epoch 182: 0.114594, lr: 0.01000
#> Loss at epoch 183: 0.094901, lr: 0.01000
#> Loss at epoch 184: 0.096753, lr: 0.01000
#> Loss at epoch 185: 0.090006, lr: 0.01000
#> Loss at epoch 186: 0.106048, lr: 0.01000
#> Loss at epoch 187: 0.088255, lr: 0.01000
#> Loss at epoch 188: 0.097012, lr: 0.01000
#> Loss at epoch 189: 0.125291, lr: 0.01000
#> Loss at epoch 190: 0.131998, lr: 0.01000
#> Loss at epoch 191: 0.101352, lr: 0.01000
#> Loss at epoch 192: 0.123844, lr: 0.01000
#> Loss at epoch 193: 0.125428, lr: 0.01000
#> Loss at epoch 194: 0.125173, lr: 0.01000
#> Loss at epoch 195: 0.096052, lr: 0.01000
#> Loss at epoch 196: 0.117838, lr: 0.01000
#> Loss at epoch 197: 0.145444, lr: 0.01000
#> Loss at epoch 198: 0.161404, lr: 0.01000
#> Loss at epoch 199: 0.135585, lr: 0.01000
#> Loss at epoch 200: 0.127798, lr: 0.01000
#> Conditional effects calculated on 5000 of 10000 observations. Use dnn(ce = <number of rows>) to change this.

# }
```
