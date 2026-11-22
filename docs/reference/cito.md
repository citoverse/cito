# 'cito': Building and training neural networks

The 'cito' package provides a user-friendly interface for training and
interpreting deep neural networks (DNN). 'cito' simplifies the fitting
of DNNs by supporting the familiar formula syntax, hyperparameter tuning
under cross-validation, and helps to detect and handle convergence
problems. DNNs can be trained on CPU, GPU and MacOS GPUs. In addition,
'cito' has many downstream functionalities such as various explainable
AI (xAI) metrics (e.g. variable importance, partial dependence plots,
accumulated local effect plots, and effect estimates) to interpret
trained DNNs. 'cito' optionally provides confidence intervals (and
p-values) for all xAI metrics and predictions. At the same time, 'cito'
is computationally efficient because it is based on the deep learning
framework 'torch'. The 'torch' package is native to R, so no Python
installation or other API is required for this package.

## Details

Cito is built around its main function [`dnn`](dnn.md), which creates
and trains a deep neural network. Various tools for analyzing the
trained neural network are available.

## Installation

in order to install cito please follow these steps:

`install.packages("cito")`

[`library(torch)`](https://torch.mlverse.org/docs)

`install_torch(reinstall = TRUE)`

[`library(cito)`](https://citoverse.github.io/cito/)

## cito functions and typical workflow

- [`dnn`](dnn.md): train deep neural network

- [`cnn`](cnn.md), [`mmn`](mmn.md): train convolutional and multi-modal
  neural networks

- [`raster_patches`](raster_patches.md): train [`cnn()`](cnn.md) and
  [`mmn()`](mmn.md) on windows read from one large GeoTIFF

- [`analyze_training`](analyze_training.md): check for convergence by
  comparing training loss with baseline loss

- [`continue_training`](continue_training.md): continues training of an
  existing cito dnn model for additional epochs

- [`summary.citodnn`](summary.citodnn.md): extract xAI metrics/effects
  to understand how predictions are made

- [`PDP`](PDP.md): plot the partial dependency plot for a specific
  feature

- [`ALE`](ALE.md): plot the accumulated local effect plot for a specific
  feature

Check out the vignettes for more details on training NN and how a
typical workflow with 'cito' could look like.

## See also

Useful links:

- <https://citoverse.github.io/cito/>

- Report bugs at <https://github.com/citoverse/cito/issues>

## Author

**Maintainer**: Maximilian Pichler
<maximilian.pichler@biologie.uni-regensburg.de>
([ORCID](https://orcid.org/0000-0003-2252-8327))

Authors:

- Maximilian Pichler <maximilian.pichler@biologie.uni-regensburg.de>
  ([ORCID](https://orcid.org/0000-0003-2252-8327))

- Christian Amesöder <Christian.Amesoeder@informatik.uni-regensburg.de>

Other contributors:

- Florian Hartig <florian.hartig@biologie.uni-regensburg.de>
  ([ORCID](https://orcid.org/0000-0002-6255-9059)) \[contributor\]

- Armin Schenk <armin.schenk99@gmail.com> \[contributor\]

## Examples

``` r
# \donttest{

if(torch::torch_is_installed()){
library(cito)

# Example workflow in cito

## Build and train  Network
### cross-entropy (softmax link) for multi-class responses (e.g., Species)
nn.fit<- dnn(Species~., data = datasets::iris, loss = "cross-entropy")

## The training loss is below the baseline loss but at the end of the
## training the loss was still decreasing, so continue training for another 50
## epochs
nn.fit <- continue_training(nn.fit, epochs = 50L)

# Structure of Neural Network
print(nn.fit)

# Plot Neural Network
if(requireNamespace("ggraph", quietly = TRUE)) plot(nn.fit, plot_type = "arch")
## 4 Input nodes (first layer) because of 4 features
## 3 Output nodes (last layer) because of 3 response species (one node for each
## level in the response variable).
## The layers between the input and output layer are called hidden layers (two
## of them)

## We now want to understand how the predictions are made, what are the
## important features? The summary function automatically calculates feature
## importance (the interpretation is similar to an anova) and calculates
## average conditional effects that are similar to linear effects:
summary(nn.fit)

## To visualize the effect (response-feature effect), we can use the ALE and
## PDP functions

# Partial dependencies
PDP(nn.fit, variable = "Petal.Length")

# Accumulated local effect plots
ALE(nn.fit, variable = "Petal.Length")



# Per se, it is difficult to get confidence intervals for our xAI metrics (or
# for the predictions). But we can use bootstrapping to obtain uncertainties
# for all cito outputs:
## Re-fit the neural network with bootstrapping
nn.fit<- dnn(Species~.,
             data = datasets::iris,
             loss = "cross-entropy",
             epochs = 150L,
             verbose = FALSE,
             bootstrap = 20L)
## convergence can be tested via the analyze_training function
analyze_training(nn.fit)

## Summary for xAI metrics (can take some time):
summary(nn.fit, importance = "permutation", type = "link")
## Now with standard errors and p-values
## Note: Take the p-values with a grain of salt! We do not know yet if they are
## correct (e.g. if you use regularization, they are likely conservative == too
## large)

## Predictions with bootstrapping:
dim(predict(nn.fit))
## predictions are by default averaged (over the bootstrap samples)

## Multinomial and conditional logit regression
m = dnn(Species~., data = iris, loss = "clogit", lr = 0.01)
m = dnn(Species~., data = iris, loss = "multinomial", lr = 0.01)

Y = t(stats::rmultinom(100, 10, prob = c(0.2, 0.2, 0.5)))
m = dnn(cbind(X1, X2, X3)~., data = data.frame(Y, A = as.factor(runif(100))),
        loss = "multinomial", lr = 0.01)
## conditional logit for size > 1 is not supported yet


# Hyperparameter tuning (experimental feature)
hidden_values = matrix(c(5, 2,
                         4, 2,
                         10,2,
                         15,2), 4, 2, byrow = TRUE)
## Potential architectures we want to test, first column == number of nodes
print(hidden_values)

nn.fit = dnn(Species~.,
             data = iris,
             epochs = 30L,
             loss = "cross-entropy",
             hidden = tune(values = hidden_values),
             lr = tune(0.00001, 0.1) # tune lr between range 0.00001 and 0.1
             )
## Tuning results:
print(nn.fit$tuning)

# test = Inf means that tuning was cancelled after only one fit (within the CV)


# Advanced: Custom loss functions and additional parameters
## Normal Likelihood with sd parameter:
custom_loss = function(pred, true) {
  logLik = torch::distr_normal(pred,
                               scale = torch::nnf_relu(scale)+
                                 0.001)$log_prob(true)
  return(-logLik$mean())
}

nn.fit<- dnn(Sepal.Length~.,
             data = datasets::iris,
             loss = custom_loss,
             verbose = FALSE,
             custom_parameters = list(scale = 1.0)
)
nn.fit$loss$parameters$scale

## Multivariate normal likelihood with parametrized covariance matrix
## Sigma = L*L^t + D
## Helper function to build covariance matrix
create_cov = function(LU, Diag) {
  return(torch::torch_matmul(LU, LU$t()) + torch::torch_diag(Diag$exp()+0.01))
}

custom_loss_MVN = function(true, pred) {
  Sigma = create_cov(SigmaPar, SigmaDiag)
  logLik = torch::distr_multivariate_normal(pred,
                                            covariance_matrix = Sigma)$
    log_prob(true)
  return(-logLik$mean())
}


nn.fit<- dnn(cbind(Sepal.Length, Sepal.Width, Petal.Length)~.,
             data = datasets::iris,
             lr = 0.01,
             verbose = FALSE,
             loss = custom_loss_MVN,
             custom_parameters =
               list(SigmaDiag =  rep(0, 3),
                    SigmaPar = matrix(rnorm(6, sd = 0.001), 3, 2))
)
as.matrix(create_cov(nn.fit$loss$parameters$SigmaPar,
                     nn.fit$loss$parameters$SigmaDiag))


# Embeddings and fixed effects
## Species enters as an embedding: each species gets two learned coordinates,
## from which the network learns how the species differ in their response to
## all predictors. Petal.Width enters as a fixed (linear) effect.
nn.fit = dnn(Sepal.Length ~ Sepal.Width + Petal.Length + fe(Petal.Width) + e(Species, dim = 2),
             data = datasets::iris, loss = "mse", epochs = 50L, verbose = FALSE)

## the linear effect of Petal.Width
coef(nn.fit)$fixed_effects

## the embeddings, one row per species
coef(nn.fit)$net_parameters$e_1.weight

## architecture: Species feeds its two embedding dimensions, Petal.Width goes
## straight to the output
if(requireNamespace("ggraph", quietly = TRUE)) plot(nn.fit, plot_type = "arch")
}
#> Loss at epoch 1: 1.015505, lr: 0.01000

#> Loss at epoch 2: 0.854792, lr: 0.01000
#> Loss at epoch 3: 0.742751, lr: 0.01000
#> Loss at epoch 4: 0.658226, lr: 0.01000
#> Loss at epoch 5: 0.599510, lr: 0.01000
#> Loss at epoch 6: 0.553689, lr: 0.01000
#> Loss at epoch 7: 0.512316, lr: 0.01000
#> Loss at epoch 8: 0.480680, lr: 0.01000
#> Loss at epoch 9: 0.452225, lr: 0.01000
#> Loss at epoch 10: 0.434102, lr: 0.01000
#> Loss at epoch 11: 0.413984, lr: 0.01000
#> Loss at epoch 12: 0.388556, lr: 0.01000
#> Loss at epoch 13: 0.377465, lr: 0.01000
#> Loss at epoch 14: 0.353613, lr: 0.01000
#> Loss at epoch 15: 0.335876, lr: 0.01000
#> Loss at epoch 16: 0.323514, lr: 0.01000
#> Loss at epoch 17: 0.311373, lr: 0.01000
#> Loss at epoch 18: 0.294426, lr: 0.01000
#> Loss at epoch 19: 0.291936, lr: 0.01000
#> Loss at epoch 20: 0.274842, lr: 0.01000
#> Loss at epoch 21: 0.260082, lr: 0.01000
#> Loss at epoch 22: 0.245517, lr: 0.01000
#> Loss at epoch 23: 0.240497, lr: 0.01000
#> Loss at epoch 24: 0.226433, lr: 0.01000
#> Loss at epoch 25: 0.219159, lr: 0.01000
#> Loss at epoch 26: 0.219359, lr: 0.01000
#> Loss at epoch 27: 0.222146, lr: 0.01000
#> Loss at epoch 28: 0.200722, lr: 0.01000
#> Loss at epoch 29: 0.198911, lr: 0.01000
#> Loss at epoch 30: 0.190348, lr: 0.01000
#> Loss at epoch 31: 0.187133, lr: 0.01000
#> Loss at epoch 32: 0.179013, lr: 0.01000
#> Loss at epoch 33: 0.167868, lr: 0.01000
#> Loss at epoch 34: 0.165628, lr: 0.01000
#> Loss at epoch 35: 0.168766, lr: 0.01000
#> Loss at epoch 36: 0.160818, lr: 0.01000
#> Loss at epoch 37: 0.158546, lr: 0.01000
#> Loss at epoch 38: 0.170270, lr: 0.01000
#> Loss at epoch 39: 0.145624, lr: 0.01000
#> Loss at epoch 40: 0.140192, lr: 0.01000
#> Loss at epoch 41: 0.140016, lr: 0.01000
#> Loss at epoch 42: 0.148305, lr: 0.01000
#> Loss at epoch 43: 0.136880, lr: 0.01000
#> Loss at epoch 44: 0.126149, lr: 0.01000
#> Loss at epoch 45: 0.127899, lr: 0.01000
#> Loss at epoch 46: 0.125772, lr: 0.01000
#> Loss at epoch 47: 0.123943, lr: 0.01000
#> Loss at epoch 48: 0.124371, lr: 0.01000
#> Loss at epoch 49: 0.119979, lr: 0.01000
#> Loss at epoch 50: 0.130684, lr: 0.01000
#> Loss at epoch 51: 0.121181, lr: 0.01000
#> Loss at epoch 52: 0.112529, lr: 0.01000
#> Loss at epoch 53: 0.104285, lr: 0.01000
#> Loss at epoch 54: 0.118650, lr: 0.01000
#> Loss at epoch 55: 0.105205, lr: 0.01000
#> Loss at epoch 56: 0.105175, lr: 0.01000
#> Loss at epoch 57: 0.104891, lr: 0.01000
#> Loss at epoch 58: 0.099950, lr: 0.01000
#> Loss at epoch 59: 0.096674, lr: 0.01000
#> Loss at epoch 60: 0.107314, lr: 0.01000
#> Loss at epoch 61: 0.104438, lr: 0.01000
#> Loss at epoch 62: 0.109164, lr: 0.01000
#> Loss at epoch 63: 0.095913, lr: 0.01000
#> Loss at epoch 64: 0.099512, lr: 0.01000
#> Loss at epoch 65: 0.103657, lr: 0.01000
#> Loss at epoch 66: 0.096764, lr: 0.01000
#> Loss at epoch 67: 0.088812, lr: 0.01000
#> Loss at epoch 68: 0.102676, lr: 0.01000
#> Loss at epoch 69: 0.082099, lr: 0.01000
#> Loss at epoch 70: 0.107412, lr: 0.01000
#> Loss at epoch 71: 0.087972, lr: 0.01000
#> Loss at epoch 72: 0.081982, lr: 0.01000
#> Loss at epoch 73: 0.084536, lr: 0.01000
#> Loss at epoch 74: 0.092863, lr: 0.01000
#> Loss at epoch 75: 0.082408, lr: 0.01000
#> Loss at epoch 76: 0.081081, lr: 0.01000
#> Loss at epoch 77: 0.092977, lr: 0.01000
#> Loss at epoch 78: 0.084702, lr: 0.01000
#> Loss at epoch 79: 0.087957, lr: 0.01000
#> Loss at epoch 80: 0.112584, lr: 0.01000
#> Loss at epoch 81: 0.077197, lr: 0.01000
#> Loss at epoch 82: 0.083423, lr: 0.01000
#> Loss at epoch 83: 0.078252, lr: 0.01000
#> Loss at epoch 84: 0.089181, lr: 0.01000
#> Loss at epoch 85: 0.083328, lr: 0.01000
#> Loss at epoch 86: 0.078072, lr: 0.01000
#> Loss at epoch 87: 0.075089, lr: 0.01000
#> Loss at epoch 88: 0.075273, lr: 0.01000
#> Loss at epoch 89: 0.080742, lr: 0.01000
#> Loss at epoch 90: 0.073873, lr: 0.01000
#> Loss at epoch 91: 0.077193, lr: 0.01000
#> Loss at epoch 92: 0.084743, lr: 0.01000
#> Loss at epoch 93: 0.076336, lr: 0.01000
#> Loss at epoch 94: 0.095381, lr: 0.01000
#> Loss at epoch 95: 0.074340, lr: 0.01000
#> Loss at epoch 96: 0.076168, lr: 0.01000
#> Loss at epoch 97: 0.076900, lr: 0.01000
#> Loss at epoch 98: 0.066934, lr: 0.01000
#> Loss at epoch 99: 0.070210, lr: 0.01000
#> Loss at epoch 100: 0.084270, lr: 0.01000
#> Loss at epoch 101: 0.064228, lr: 0.01000

#> Loss at epoch 102: 0.074129, lr: 0.01000
#> Loss at epoch 103: 0.070503, lr: 0.01000
#> Loss at epoch 104: 0.072558, lr: 0.01000
#> Loss at epoch 105: 0.072872, lr: 0.01000
#> Loss at epoch 106: 0.077591, lr: 0.01000
#> Loss at epoch 107: 0.091114, lr: 0.01000
#> Loss at epoch 108: 0.075569, lr: 0.01000
#> Loss at epoch 109: 0.092581, lr: 0.01000
#> Loss at epoch 110: 0.083211, lr: 0.01000
#> Loss at epoch 111: 0.075237, lr: 0.01000
#> Loss at epoch 112: 0.065031, lr: 0.01000
#> Loss at epoch 113: 0.073442, lr: 0.01000
#> Loss at epoch 114: 0.074192, lr: 0.01000
#> Loss at epoch 115: 0.071644, lr: 0.01000
#> Loss at epoch 116: 0.064721, lr: 0.01000
#> Loss at epoch 117: 0.080918, lr: 0.01000
#> Loss at epoch 118: 0.070205, lr: 0.01000
#> Loss at epoch 119: 0.092782, lr: 0.01000
#> Loss at epoch 120: 0.071285, lr: 0.01000
#> Loss at epoch 121: 0.054058, lr: 0.01000
#> Loss at epoch 122: 0.076727, lr: 0.01000
#> Loss at epoch 123: 0.064280, lr: 0.01000
#> Loss at epoch 124: 0.071735, lr: 0.01000
#> Loss at epoch 125: 0.067350, lr: 0.01000
#> Loss at epoch 126: 0.072123, lr: 0.01000
#> Loss at epoch 127: 0.059716, lr: 0.01000
#> Loss at epoch 128: 0.065493, lr: 0.01000
#> Loss at epoch 129: 0.065521, lr: 0.01000
#> Loss at epoch 130: 0.070376, lr: 0.01000
#> Loss at epoch 131: 0.065378, lr: 0.01000
#> Loss at epoch 132: 0.069262, lr: 0.01000
#> Loss at epoch 133: 0.059379, lr: 0.01000
#> Loss at epoch 134: 0.069803, lr: 0.01000
#> Loss at epoch 135: 0.075693, lr: 0.01000
#> Loss at epoch 136: 0.069064, lr: 0.01000
#> Loss at epoch 137: 0.102483, lr: 0.01000
#> Loss at epoch 138: 0.068728, lr: 0.01000
#> Loss at epoch 139: 0.066321, lr: 0.01000
#> Loss at epoch 140: 0.076674, lr: 0.01000
#> Loss at epoch 141: 0.058992, lr: 0.01000
#> Loss at epoch 142: 0.070005, lr: 0.01000
#> Loss at epoch 143: 0.064598, lr: 0.01000
#> Loss at epoch 144: 0.068274, lr: 0.01000
#> Loss at epoch 145: 0.075432, lr: 0.01000
#> Loss at epoch 146: 0.088818, lr: 0.01000
#> Loss at epoch 147: 0.066915, lr: 0.01000
#> Loss at epoch 148: 0.065721, lr: 0.01000
#> Loss at epoch 149: 0.075572, lr: 0.01000
#> Loss at epoch 150: 0.071678, lr: 0.01000
#> dnn(formula = Species ~ Sepal.Length + Sepal.Width + Petal.Length + 
#>     Petal.Width, data = datasets::iris, loss = "cross-entropy")
#> An `nn_module` containing 2,953 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • l_1: <nn_linear> #250 parameters
#> • l_2: <nn_selu> #0 parameters
#> • l_3: <nn_linear> #2,550 parameters
#> • l_4: <nn_selu> #0 parameters
#> • l_5: <nn_linear> #153 parameters



#> Loss at epoch 1: 0.645088, lr: 0.01000

#> Loss at epoch 2: 0.540452, lr: 0.01000
#> Loss at epoch 3: 0.510175, lr: 0.01000
#> Loss at epoch 4: 0.478642, lr: 0.01000
#> Loss at epoch 5: 0.453327, lr: 0.01000
#> Loss at epoch 6: 0.433280, lr: 0.01000
#> Loss at epoch 7: 0.412819, lr: 0.01000
#> Loss at epoch 8: 0.396388, lr: 0.01000
#> Loss at epoch 9: 0.380423, lr: 0.01000
#> Loss at epoch 10: 0.364339, lr: 0.01000
#> Loss at epoch 11: 0.354364, lr: 0.01000
#> Loss at epoch 12: 0.342938, lr: 0.01000
#> Loss at epoch 13: 0.338027, lr: 0.01000
#> Loss at epoch 14: 0.322217, lr: 0.01000
#> Loss at epoch 15: 0.322535, lr: 0.01000
#> Loss at epoch 16: 0.310728, lr: 0.01000
#> Loss at epoch 17: 0.302647, lr: 0.01000
#> Loss at epoch 18: 0.299226, lr: 0.01000
#> Loss at epoch 19: 0.283111, lr: 0.01000
#> Loss at epoch 20: 0.280036, lr: 0.01000
#> Loss at epoch 21: 0.276949, lr: 0.01000
#> Loss at epoch 22: 0.268862, lr: 0.01000
#> Loss at epoch 23: 0.267003, lr: 0.01000
#> Loss at epoch 24: 0.256385, lr: 0.01000
#> Loss at epoch 25: 0.250732, lr: 0.01000
#> Loss at epoch 26: 0.245234, lr: 0.01000
#> Loss at epoch 27: 0.244054, lr: 0.01000
#> Loss at epoch 28: 0.240241, lr: 0.01000
#> Loss at epoch 29: 0.229093, lr: 0.01000
#> Loss at epoch 30: 0.226340, lr: 0.01000
#> Loss at epoch 31: 0.227314, lr: 0.01000
#> Loss at epoch 32: 0.213507, lr: 0.01000
#> Loss at epoch 33: 0.211934, lr: 0.01000
#> Loss at epoch 34: 0.206185, lr: 0.01000
#> Loss at epoch 35: 0.201768, lr: 0.01000
#> Loss at epoch 36: 0.199187, lr: 0.01000
#> Loss at epoch 37: 0.188786, lr: 0.01000
#> Loss at epoch 38: 0.194131, lr: 0.01000
#> Loss at epoch 39: 0.184979, lr: 0.01000
#> Loss at epoch 40: 0.182501, lr: 0.01000
#> Loss at epoch 41: 0.178936, lr: 0.01000
#> Loss at epoch 42: 0.183904, lr: 0.01000
#> Loss at epoch 43: 0.170357, lr: 0.01000
#> Loss at epoch 44: 0.166213, lr: 0.01000
#> Loss at epoch 45: 0.164801, lr: 0.01000
#> Loss at epoch 46: 0.157319, lr: 0.01000
#> Loss at epoch 47: 0.153236, lr: 0.01000
#> Loss at epoch 48: 0.155165, lr: 0.01000
#> Loss at epoch 49: 0.152869, lr: 0.01000
#> Loss at epoch 50: 0.146145, lr: 0.01000
#> Loss at epoch 51: 0.145557, lr: 0.01000
#> Loss at epoch 52: 0.153433, lr: 0.01000
#> Loss at epoch 53: 0.137900, lr: 0.01000
#> Loss at epoch 54: 0.134789, lr: 0.01000
#> Loss at epoch 55: 0.138737, lr: 0.01000
#> Loss at epoch 56: 0.131434, lr: 0.01000
#> Loss at epoch 57: 0.135783, lr: 0.01000
#> Loss at epoch 58: 0.125407, lr: 0.01000
#> Loss at epoch 59: 0.123102, lr: 0.01000
#> Loss at epoch 60: 0.118288, lr: 0.01000
#> Loss at epoch 61: 0.123514, lr: 0.01000
#> Loss at epoch 62: 0.116626, lr: 0.01000
#> Loss at epoch 63: 0.126807, lr: 0.01000
#> Loss at epoch 64: 0.117658, lr: 0.01000
#> Loss at epoch 65: 0.114796, lr: 0.01000
#> Loss at epoch 66: 0.109634, lr: 0.01000
#> Loss at epoch 67: 0.106671, lr: 0.01000
#> Loss at epoch 68: 0.108341, lr: 0.01000
#> Loss at epoch 69: 0.105626, lr: 0.01000
#> Loss at epoch 70: 0.104673, lr: 0.01000
#> Loss at epoch 71: 0.102118, lr: 0.01000
#> Loss at epoch 72: 0.102703, lr: 0.01000
#> Loss at epoch 73: 0.101469, lr: 0.01000
#> Loss at epoch 74: 0.099630, lr: 0.01000
#> Loss at epoch 75: 0.096496, lr: 0.01000
#> Loss at epoch 76: 0.095857, lr: 0.01000
#> Loss at epoch 77: 0.091528, lr: 0.01000
#> Loss at epoch 78: 0.092093, lr: 0.01000
#> Loss at epoch 79: 0.097916, lr: 0.01000
#> Loss at epoch 80: 0.091633, lr: 0.01000
#> Loss at epoch 81: 0.089921, lr: 0.01000
#> Loss at epoch 82: 0.087267, lr: 0.01000
#> Loss at epoch 83: 0.089085, lr: 0.01000
#> Loss at epoch 84: 0.092812, lr: 0.01000
#> Loss at epoch 85: 0.085197, lr: 0.01000
#> Loss at epoch 86: 0.083999, lr: 0.01000
#> Loss at epoch 87: 0.079803, lr: 0.01000
#> Loss at epoch 88: 0.081322, lr: 0.01000
#> Loss at epoch 89: 0.088043, lr: 0.01000
#> Loss at epoch 90: 0.078430, lr: 0.01000
#> Loss at epoch 91: 0.079827, lr: 0.01000
#> Loss at epoch 92: 0.080868, lr: 0.01000
#> Loss at epoch 93: 0.075651, lr: 0.01000
#> Loss at epoch 94: 0.076514, lr: 0.01000
#> Loss at epoch 95: 0.084253, lr: 0.01000
#> Loss at epoch 96: 0.085107, lr: 0.01000
#> Loss at epoch 97: 0.072431, lr: 0.01000
#> Loss at epoch 98: 0.076644, lr: 0.01000
#> Loss at epoch 99: 0.074707, lr: 0.01000
#> Loss at epoch 100: 0.076683, lr: 0.01000
#> Loss at epoch 1: 1.077847, lr: 0.01000

#> Loss at epoch 2: 0.887549, lr: 0.01000
#> Loss at epoch 3: 0.753153, lr: 0.01000
#> Loss at epoch 4: 0.690777, lr: 0.01000
#> Loss at epoch 5: 0.619362, lr: 0.01000
#> Loss at epoch 6: 0.575299, lr: 0.01000
#> Loss at epoch 7: 0.539180, lr: 0.01000
#> Loss at epoch 8: 0.500205, lr: 0.01000
#> Loss at epoch 9: 0.473611, lr: 0.01000
#> Loss at epoch 10: 0.447891, lr: 0.01000
#> Loss at epoch 11: 0.440715, lr: 0.01000
#> Loss at epoch 12: 0.419725, lr: 0.01000
#> Loss at epoch 13: 0.402861, lr: 0.01000
#> Loss at epoch 14: 0.400176, lr: 0.01000
#> Loss at epoch 15: 0.369791, lr: 0.01000
#> Loss at epoch 16: 0.352016, lr: 0.01000
#> Loss at epoch 17: 0.332771, lr: 0.01000
#> Loss at epoch 18: 0.336436, lr: 0.01000
#> Loss at epoch 19: 0.312062, lr: 0.01000
#> Loss at epoch 20: 0.300581, lr: 0.01000
#> Loss at epoch 21: 0.291023, lr: 0.01000
#> Loss at epoch 22: 0.289441, lr: 0.01000
#> Loss at epoch 23: 0.273692, lr: 0.01000
#> Loss at epoch 24: 0.282456, lr: 0.01000
#> Loss at epoch 25: 0.250863, lr: 0.01000
#> Loss at epoch 26: 0.245295, lr: 0.01000
#> Loss at epoch 27: 0.236609, lr: 0.01000
#> Loss at epoch 28: 0.232860, lr: 0.01000
#> Loss at epoch 29: 0.227213, lr: 0.01000
#> Loss at epoch 30: 0.213300, lr: 0.01000
#> Loss at epoch 31: 0.227634, lr: 0.01000
#> Loss at epoch 32: 0.228728, lr: 0.01000
#> Loss at epoch 33: 0.200585, lr: 0.01000
#> Loss at epoch 34: 0.191979, lr: 0.01000
#> Loss at epoch 35: 0.185630, lr: 0.01000
#> Loss at epoch 36: 0.190121, lr: 0.01000
#> Loss at epoch 37: 0.178824, lr: 0.01000
#> Loss at epoch 38: 0.178120, lr: 0.01000
#> Loss at epoch 39: 0.175231, lr: 0.01000
#> Loss at epoch 40: 0.181816, lr: 0.01000
#> Loss at epoch 41: 0.166392, lr: 0.01000
#> Loss at epoch 42: 0.157937, lr: 0.01000
#> Loss at epoch 43: 0.152786, lr: 0.01000
#> Loss at epoch 44: 0.159809, lr: 0.01000
#> Loss at epoch 45: 0.145262, lr: 0.01000
#> Loss at epoch 46: 0.140221, lr: 0.01000
#> Loss at epoch 47: 0.145633, lr: 0.01000
#> Loss at epoch 48: 0.141166, lr: 0.01000
#> Loss at epoch 49: 0.124587, lr: 0.01000
#> Loss at epoch 50: 0.138022, lr: 0.01000
#> Loss at epoch 51: 0.154743, lr: 0.01000
#> Loss at epoch 52: 0.127986, lr: 0.01000
#> Loss at epoch 53: 0.124700, lr: 0.01000
#> Loss at epoch 54: 0.124996, lr: 0.01000
#> Loss at epoch 55: 0.134888, lr: 0.01000
#> Loss at epoch 56: 0.117694, lr: 0.01000
#> Loss at epoch 57: 0.119171, lr: 0.01000
#> Loss at epoch 58: 0.126081, lr: 0.01000
#> Loss at epoch 59: 0.116349, lr: 0.01000
#> Loss at epoch 60: 0.109682, lr: 0.01000
#> Loss at epoch 61: 0.110488, lr: 0.01000
#> Loss at epoch 62: 0.107829, lr: 0.01000
#> Loss at epoch 63: 0.105076, lr: 0.01000
#> Loss at epoch 64: 0.108016, lr: 0.01000
#> Loss at epoch 65: 0.135103, lr: 0.01000
#> Loss at epoch 66: 0.125658, lr: 0.01000
#> Loss at epoch 67: 0.106039, lr: 0.01000
#> Loss at epoch 68: 0.110909, lr: 0.01000
#> Loss at epoch 69: 0.118666, lr: 0.01000
#> Loss at epoch 70: 0.097300, lr: 0.01000
#> Loss at epoch 71: 0.092556, lr: 0.01000
#> Loss at epoch 72: 0.092477, lr: 0.01000
#> Loss at epoch 73: 0.088827, lr: 0.01000
#> Loss at epoch 74: 0.107223, lr: 0.01000
#> Loss at epoch 75: 0.110257, lr: 0.01000
#> Loss at epoch 76: 0.092394, lr: 0.01000
#> Loss at epoch 77: 0.091957, lr: 0.01000
#> Loss at epoch 78: 0.106473, lr: 0.01000
#> Loss at epoch 79: 0.087948, lr: 0.01000
#> Loss at epoch 80: 0.092484, lr: 0.01000
#> Loss at epoch 81: 0.087046, lr: 0.01000
#> Loss at epoch 82: 0.098362, lr: 0.01000
#> Loss at epoch 83: 0.088943, lr: 0.01000
#> Loss at epoch 84: 0.094020, lr: 0.01000
#> Loss at epoch 85: 0.093961, lr: 0.01000
#> Loss at epoch 86: 0.086197, lr: 0.01000
#> Loss at epoch 87: 0.084912, lr: 0.01000
#> Loss at epoch 88: 0.103865, lr: 0.01000
#> Loss at epoch 89: 0.080484, lr: 0.01000
#> Loss at epoch 90: 0.080030, lr: 0.01000
#> Loss at epoch 91: 0.081781, lr: 0.01000
#> Loss at epoch 92: 0.091265, lr: 0.01000
#> Loss at epoch 93: 0.083017, lr: 0.01000
#> Loss at epoch 94: 0.102536, lr: 0.01000
#> Loss at epoch 95: 0.088064, lr: 0.01000
#> Loss at epoch 96: 0.084960, lr: 0.01000
#> Loss at epoch 97: 0.095921, lr: 0.01000
#> Loss at epoch 98: 0.080171, lr: 0.01000
#> Loss at epoch 99: 0.087543, lr: 0.01000
#> Loss at epoch 100: 0.073870, lr: 0.01000
#> Loss at epoch 1: 3.573249, lr: 0.01000

#> Loss at epoch 2: 3.308973, lr: 0.01000
#> Loss at epoch 3: 3.278158, lr: 0.01000
#> Loss at epoch 4: 3.269300, lr: 0.01000
#> Loss at epoch 5: 3.265998, lr: 0.01000
#> Loss at epoch 6: 3.263842, lr: 0.01000
#> Loss at epoch 7: 3.256435, lr: 0.01000
#> Loss at epoch 8: 3.250657, lr: 0.01000
#> Loss at epoch 9: 3.245152, lr: 0.01000
#> Loss at epoch 10: 3.243515, lr: 0.01000
#> Loss at epoch 11: 3.238880, lr: 0.01000
#> Loss at epoch 12: 3.238438, lr: 0.01000
#> Loss at epoch 13: 3.230115, lr: 0.01000
#> Loss at epoch 14: 3.225333, lr: 0.01000
#> Loss at epoch 15: 3.221183, lr: 0.01000
#> Loss at epoch 16: 3.214350, lr: 0.01000
#> Loss at epoch 17: 3.210951, lr: 0.01000
#> Loss at epoch 18: 3.210054, lr: 0.01000
#> Loss at epoch 19: 3.202419, lr: 0.01000
#> Loss at epoch 20: 3.195159, lr: 0.01000
#> Loss at epoch 21: 3.186826, lr: 0.01000
#> Loss at epoch 22: 3.186802, lr: 0.01000
#> Loss at epoch 23: 3.175924, lr: 0.01000
#> Loss at epoch 24: 3.171505, lr: 0.01000
#> Loss at epoch 25: 3.167885, lr: 0.01000
#> Loss at epoch 26: 3.163211, lr: 0.01000
#> Loss at epoch 27: 3.149705, lr: 0.01000
#> Loss at epoch 28: 3.139630, lr: 0.01000
#> Loss at epoch 29: 3.127796, lr: 0.01000
#> Loss at epoch 30: 3.124144, lr: 0.01000
#> Loss at epoch 31: 3.109341, lr: 0.01000
#> Loss at epoch 32: 3.109427, lr: 0.01000
#> Loss at epoch 33: 3.086516, lr: 0.01000
#> Loss at epoch 34: 3.076760, lr: 0.01000
#> Loss at epoch 35: 3.064479, lr: 0.01000
#> Loss at epoch 36: 3.052951, lr: 0.01000
#> Loss at epoch 37: 3.038013, lr: 0.01000
#> Loss at epoch 38: 3.017009, lr: 0.01000
#> Loss at epoch 39: 3.001326, lr: 0.01000
#> Loss at epoch 40: 2.983455, lr: 0.01000
#> Loss at epoch 41: 2.970453, lr: 0.01000
#> Loss at epoch 42: 2.948536, lr: 0.01000
#> Loss at epoch 43: 2.928132, lr: 0.01000
#> Loss at epoch 44: 2.906130, lr: 0.01000
#> Loss at epoch 45: 2.883587, lr: 0.01000
#> Loss at epoch 46: 2.863344, lr: 0.01000
#> Loss at epoch 47: 2.835524, lr: 0.01000
#> Loss at epoch 48: 2.812481, lr: 0.01000
#> Loss at epoch 49: 2.791929, lr: 0.01000
#> Loss at epoch 50: 2.766065, lr: 0.01000
#> Loss at epoch 51: 2.753487, lr: 0.01000
#> Loss at epoch 52: 2.719023, lr: 0.01000
#> Loss at epoch 53: 2.694892, lr: 0.01000
#> Loss at epoch 54: 2.664784, lr: 0.01000
#> Loss at epoch 55: 2.649655, lr: 0.01000
#> Loss at epoch 56: 2.626337, lr: 0.01000
#> Loss at epoch 57: 2.607745, lr: 0.01000
#> Loss at epoch 58: 2.579975, lr: 0.01000
#> Loss at epoch 59: 2.556982, lr: 0.01000
#> Loss at epoch 60: 2.538649, lr: 0.01000
#> Loss at epoch 61: 2.513613, lr: 0.01000
#> Loss at epoch 62: 2.502361, lr: 0.01000
#> Loss at epoch 63: 2.485762, lr: 0.01000
#> Loss at epoch 64: 2.471307, lr: 0.01000
#> Loss at epoch 65: 2.448349, lr: 0.01000
#> Loss at epoch 66: 2.438235, lr: 0.01000
#> Loss at epoch 67: 2.419751, lr: 0.01000
#> Loss at epoch 68: 2.411421, lr: 0.01000
#> Loss at epoch 69: 2.397049, lr: 0.01000
#> Loss at epoch 70: 2.383281, lr: 0.01000
#> Loss at epoch 71: 2.375100, lr: 0.01000
#> Loss at epoch 72: 2.365220, lr: 0.01000
#> Loss at epoch 73: 2.353513, lr: 0.01000
#> Loss at epoch 74: 2.345630, lr: 0.01000
#> Loss at epoch 75: 2.343278, lr: 0.01000
#> Loss at epoch 76: 2.332319, lr: 0.01000
#> Loss at epoch 77: 2.327057, lr: 0.01000
#> Loss at epoch 78: 2.320742, lr: 0.01000
#> Loss at epoch 79: 2.314568, lr: 0.01000
#> Loss at epoch 80: 2.311985, lr: 0.01000
#> Loss at epoch 81: 2.305813, lr: 0.01000
#> Loss at epoch 82: 2.302001, lr: 0.01000
#> Loss at epoch 83: 2.299080, lr: 0.01000
#> Loss at epoch 84: 2.296899, lr: 0.01000
#> Loss at epoch 85: 2.291546, lr: 0.01000
#> Loss at epoch 86: 2.289355, lr: 0.01000
#> Loss at epoch 87: 2.284846, lr: 0.01000
#> Loss at epoch 88: 2.282638, lr: 0.01000
#> Loss at epoch 89: 2.279380, lr: 0.01000
#> Loss at epoch 90: 2.276558, lr: 0.01000
#> Loss at epoch 91: 2.275306, lr: 0.01000
#> Loss at epoch 92: 2.272885, lr: 0.01000
#> Loss at epoch 93: 2.270229, lr: 0.01000
#> Loss at epoch 94: 2.268153, lr: 0.01000
#> Loss at epoch 95: 2.266985, lr: 0.01000
#> Loss at epoch 96: 2.265092, lr: 0.01000
#> Loss at epoch 97: 2.263398, lr: 0.01000
#> Loss at epoch 98: 2.261550, lr: 0.01000
#> Loss at epoch 99: 2.260955, lr: 0.01000
#> Loss at epoch 100: 2.259774, lr: 0.01000
#>      [,1] [,2]
#> [1,]    5    2
#> [2,]    4    2
#> [3,]   10    2
#> [4,]   15    2
#> Starting hyperparameter tuning...
#> Fitting final model...
#> # A tibble: 10 × 6
#>    steps  test train models hidden          lr
#>    <int> <dbl> <dbl> <lgl>  <list>       <dbl>
#>  1     1  18.7     0 NA     <dbl [2]> 0.0769  
#>  2     2  31.8     0 NA     <dbl [2]> 0.0647  
#>  3     3  25.7     0 NA     <dbl [2]> 0.0409  
#>  4     4 154.      0 NA     <dbl [2]> 0.000946
#>  5     5  57.1     0 NA     <dbl [2]> 0.0194  
#>  6     6  42.4     0 NA     <dbl [2]> 0.0959  
#>  7     7  37.1     0 NA     <dbl [2]> 0.0706  
#>  8     8  57.9     0 NA     <dbl [2]> 0.0444  
#>  9     9  40.7     0 NA     <dbl [2]> 0.0201  
#> 10    10  28.7     0 NA     <dbl [2]> 0.0794  




# }
```
