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
### softmax is used for multi-class responses (e.g., Species)
nn.fit<- dnn(Species~., data = datasets::iris, loss = "cross-entropy")

## The training loss is below the baseline loss but at the end of the
## training the loss was still decreasing, so continue training for another 50
## epochs
nn.fit <- continue_training(nn.fit, epochs = 50L)

# Sturcture of Neural Network
print(nn.fit)

# Plot Neural Network
plot(nn.fit)
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
}
#> Loss at epoch 1: 1.000792, lr: 0.01000

#> Loss at epoch 2: 0.843469, lr: 0.01000
#> Loss at epoch 3: 0.728029, lr: 0.01000
#> Loss at epoch 4: 0.644354, lr: 0.01000
#> Loss at epoch 5: 0.590123, lr: 0.01000
#> Loss at epoch 6: 0.553017, lr: 0.01000
#> Loss at epoch 7: 0.500041, lr: 0.01000
#> Loss at epoch 8: 0.474793, lr: 0.01000
#> Loss at epoch 9: 0.467093, lr: 0.01000
#> Loss at epoch 10: 0.421626, lr: 0.01000
#> Loss at epoch 11: 0.413133, lr: 0.01000
#> Loss at epoch 12: 0.383560, lr: 0.01000
#> Loss at epoch 13: 0.365484, lr: 0.01000
#> Loss at epoch 14: 0.358255, lr: 0.01000
#> Loss at epoch 15: 0.337952, lr: 0.01000
#> Loss at epoch 16: 0.334408, lr: 0.01000
#> Loss at epoch 17: 0.311903, lr: 0.01000
#> Loss at epoch 18: 0.301889, lr: 0.01000
#> Loss at epoch 19: 0.299873, lr: 0.01000
#> Loss at epoch 20: 0.289223, lr: 0.01000
#> Loss at epoch 21: 0.273431, lr: 0.01000
#> Loss at epoch 22: 0.251682, lr: 0.01000
#> Loss at epoch 23: 0.244824, lr: 0.01000
#> Loss at epoch 24: 0.236544, lr: 0.01000
#> Loss at epoch 25: 0.227809, lr: 0.01000
#> Loss at epoch 26: 0.219184, lr: 0.01000
#> Loss at epoch 27: 0.231609, lr: 0.01000
#> Loss at epoch 28: 0.208096, lr: 0.01000
#> Loss at epoch 29: 0.200328, lr: 0.01000
#> Loss at epoch 30: 0.198290, lr: 0.01000
#> Loss at epoch 31: 0.177845, lr: 0.01000
#> Loss at epoch 32: 0.174957, lr: 0.01000
#> Loss at epoch 33: 0.169651, lr: 0.01000
#> Loss at epoch 34: 0.171616, lr: 0.01000
#> Loss at epoch 35: 0.160210, lr: 0.01000
#> Loss at epoch 36: 0.159997, lr: 0.01000
#> Loss at epoch 37: 0.152599, lr: 0.01000
#> Loss at epoch 38: 0.155932, lr: 0.01000
#> Loss at epoch 39: 0.154753, lr: 0.01000
#> Loss at epoch 40: 0.141751, lr: 0.01000
#> Loss at epoch 41: 0.152290, lr: 0.01000
#> Loss at epoch 42: 0.147723, lr: 0.01000
#> Loss at epoch 43: 0.133380, lr: 0.01000
#> Loss at epoch 44: 0.130007, lr: 0.01000
#> Loss at epoch 45: 0.127333, lr: 0.01000
#> Loss at epoch 46: 0.131297, lr: 0.01000
#> Loss at epoch 47: 0.125624, lr: 0.01000
#> Loss at epoch 48: 0.128484, lr: 0.01000
#> Loss at epoch 49: 0.128296, lr: 0.01000
#> Loss at epoch 50: 0.117567, lr: 0.01000
#> Loss at epoch 51: 0.115320, lr: 0.01000
#> Loss at epoch 52: 0.117917, lr: 0.01000
#> Loss at epoch 53: 0.112088, lr: 0.01000
#> Loss at epoch 54: 0.120179, lr: 0.01000
#> Loss at epoch 55: 0.107038, lr: 0.01000
#> Loss at epoch 56: 0.102974, lr: 0.01000
#> Loss at epoch 57: 0.114743, lr: 0.01000
#> Loss at epoch 58: 0.106880, lr: 0.01000
#> Loss at epoch 59: 0.102930, lr: 0.01000
#> Loss at epoch 60: 0.101183, lr: 0.01000
#> Loss at epoch 61: 0.106111, lr: 0.01000
#> Loss at epoch 62: 0.095014, lr: 0.01000
#> Loss at epoch 63: 0.100248, lr: 0.01000
#> Loss at epoch 64: 0.097392, lr: 0.01000
#> Loss at epoch 65: 0.095697, lr: 0.01000
#> Loss at epoch 66: 0.102193, lr: 0.01000
#> Loss at epoch 67: 0.093405, lr: 0.01000
#> Loss at epoch 68: 0.094699, lr: 0.01000
#> Loss at epoch 69: 0.086617, lr: 0.01000
#> Loss at epoch 70: 0.089818, lr: 0.01000
#> Loss at epoch 71: 0.094055, lr: 0.01000
#> Loss at epoch 72: 0.097066, lr: 0.01000
#> Loss at epoch 73: 0.085571, lr: 0.01000
#> Loss at epoch 74: 0.084119, lr: 0.01000
#> Loss at epoch 75: 0.093795, lr: 0.01000
#> Loss at epoch 76: 0.091884, lr: 0.01000
#> Loss at epoch 77: 0.090705, lr: 0.01000
#> Loss at epoch 78: 0.099037, lr: 0.01000
#> Loss at epoch 79: 0.085440, lr: 0.01000
#> Loss at epoch 80: 0.069022, lr: 0.01000
#> Loss at epoch 81: 0.099314, lr: 0.01000
#> Loss at epoch 82: 0.089358, lr: 0.01000
#> Loss at epoch 83: 0.085280, lr: 0.01000
#> Loss at epoch 84: 0.087741, lr: 0.01000
#> Loss at epoch 85: 0.092466, lr: 0.01000
#> Loss at epoch 86: 0.088161, lr: 0.01000
#> Loss at epoch 87: 0.077284, lr: 0.01000
#> Loss at epoch 88: 0.077921, lr: 0.01000
#> Loss at epoch 89: 0.095528, lr: 0.01000
#> Loss at epoch 90: 0.074031, lr: 0.01000
#> Loss at epoch 91: 0.092261, lr: 0.01000
#> Loss at epoch 92: 0.073852, lr: 0.01000
#> Loss at epoch 93: 0.083681, lr: 0.01000
#> Loss at epoch 94: 0.068270, lr: 0.01000
#> Loss at epoch 95: 0.084204, lr: 0.01000
#> Loss at epoch 96: 0.069424, lr: 0.01000
#> Loss at epoch 97: 0.075997, lr: 0.01000
#> Loss at epoch 98: 0.073621, lr: 0.01000
#> Loss at epoch 99: 0.075233, lr: 0.01000
#> Loss at epoch 100: 0.083552, lr: 0.01000
#> Loss at epoch 101: 0.074416, lr: 0.01000

#> Loss at epoch 102: 0.069428, lr: 0.01000
#> Loss at epoch 103: 0.076510, lr: 0.01000
#> Loss at epoch 104: 0.076383, lr: 0.01000
#> Loss at epoch 105: 0.079851, lr: 0.01000
#> Loss at epoch 106: 0.077582, lr: 0.01000
#> Loss at epoch 107: 0.074387, lr: 0.01000
#> Loss at epoch 108: 0.071717, lr: 0.01000
#> Loss at epoch 109: 0.077853, lr: 0.01000
#> Loss at epoch 110: 0.071093, lr: 0.01000
#> Loss at epoch 111: 0.065614, lr: 0.01000
#> Loss at epoch 112: 0.111904, lr: 0.01000
#> Loss at epoch 113: 0.079243, lr: 0.01000
#> Loss at epoch 114: 0.075562, lr: 0.01000
#> Loss at epoch 115: 0.077556, lr: 0.01000
#> Loss at epoch 116: 0.065359, lr: 0.01000
#> Loss at epoch 117: 0.075541, lr: 0.01000
#> Loss at epoch 118: 0.070239, lr: 0.01000
#> Loss at epoch 119: 0.068680, lr: 0.01000
#> Loss at epoch 120: 0.065131, lr: 0.01000
#> Loss at epoch 121: 0.075199, lr: 0.01000
#> Loss at epoch 122: 0.070085, lr: 0.01000
#> Loss at epoch 123: 0.092045, lr: 0.01000
#> Loss at epoch 124: 0.075939, lr: 0.01000
#> Loss at epoch 125: 0.074401, lr: 0.01000
#> Loss at epoch 126: 0.079905, lr: 0.01000
#> Loss at epoch 127: 0.068428, lr: 0.01000
#> Loss at epoch 128: 0.071126, lr: 0.01000
#> Loss at epoch 129: 0.066209, lr: 0.01000
#> Loss at epoch 130: 0.064975, lr: 0.01000
#> Loss at epoch 131: 0.068669, lr: 0.01000
#> Loss at epoch 132: 0.078434, lr: 0.01000
#> Loss at epoch 133: 0.082002, lr: 0.01000
#> Loss at epoch 134: 0.078731, lr: 0.01000
#> Loss at epoch 135: 0.077436, lr: 0.01000
#> Loss at epoch 136: 0.065429, lr: 0.01000
#> Loss at epoch 137: 0.073675, lr: 0.01000
#> Loss at epoch 138: 0.062480, lr: 0.01000
#> Loss at epoch 139: 0.072366, lr: 0.01000
#> Loss at epoch 140: 0.061792, lr: 0.01000
#> Loss at epoch 141: 0.065277, lr: 0.01000
#> Loss at epoch 142: 0.064844, lr: 0.01000
#> Loss at epoch 143: 0.063390, lr: 0.01000
#> Loss at epoch 144: 0.064933, lr: 0.01000
#> Loss at epoch 145: 0.086699, lr: 0.01000
#> Loss at epoch 146: 0.070615, lr: 0.01000
#> Loss at epoch 147: 0.068577, lr: 0.01000
#> Loss at epoch 148: 0.057564, lr: 0.01000
#> Loss at epoch 149: 0.072669, lr: 0.01000
#> Loss at epoch 150: 0.069225, lr: 0.01000
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



#> Loss at epoch 1: 0.642680, lr: 0.01000

#> Loss at epoch 2: 0.604908, lr: 0.01000
#> Loss at epoch 3: 0.575487, lr: 0.01000
#> Loss at epoch 4: 0.546488, lr: 0.01000
#> Loss at epoch 5: 0.520514, lr: 0.01000
#> Loss at epoch 6: 0.497126, lr: 0.01000
#> Loss at epoch 7: 0.473825, lr: 0.01000
#> Loss at epoch 8: 0.447309, lr: 0.01000
#> Loss at epoch 9: 0.425270, lr: 0.01000
#> Loss at epoch 10: 0.410153, lr: 0.01000
#> Loss at epoch 11: 0.391405, lr: 0.01000
#> Loss at epoch 12: 0.375785, lr: 0.01000
#> Loss at epoch 13: 0.363008, lr: 0.01000
#> Loss at epoch 14: 0.350705, lr: 0.01000
#> Loss at epoch 15: 0.340230, lr: 0.01000
#> Loss at epoch 16: 0.324034, lr: 0.01000
#> Loss at epoch 17: 0.313032, lr: 0.01000
#> Loss at epoch 18: 0.302297, lr: 0.01000
#> Loss at epoch 19: 0.288020, lr: 0.01000
#> Loss at epoch 20: 0.282210, lr: 0.01000
#> Loss at epoch 21: 0.274804, lr: 0.01000
#> Loss at epoch 22: 0.264087, lr: 0.01000
#> Loss at epoch 23: 0.259426, lr: 0.01000
#> Loss at epoch 24: 0.251006, lr: 0.01000
#> Loss at epoch 25: 0.243698, lr: 0.01000
#> Loss at epoch 26: 0.237018, lr: 0.01000
#> Loss at epoch 27: 0.232993, lr: 0.01000
#> Loss at epoch 28: 0.222183, lr: 0.01000
#> Loss at epoch 29: 0.222885, lr: 0.01000
#> Loss at epoch 30: 0.208773, lr: 0.01000
#> Loss at epoch 31: 0.206012, lr: 0.01000
#> Loss at epoch 32: 0.197773, lr: 0.01000
#> Loss at epoch 33: 0.192819, lr: 0.01000
#> Loss at epoch 34: 0.187792, lr: 0.01000
#> Loss at epoch 35: 0.181931, lr: 0.01000
#> Loss at epoch 36: 0.178029, lr: 0.01000
#> Loss at epoch 37: 0.172696, lr: 0.01000
#> Loss at epoch 38: 0.167781, lr: 0.01000
#> Loss at epoch 39: 0.163345, lr: 0.01000
#> Loss at epoch 40: 0.161846, lr: 0.01000
#> Loss at epoch 41: 0.151633, lr: 0.01000
#> Loss at epoch 42: 0.152716, lr: 0.01000
#> Loss at epoch 43: 0.150755, lr: 0.01000
#> Loss at epoch 44: 0.150338, lr: 0.01000
#> Loss at epoch 45: 0.140482, lr: 0.01000
#> Loss at epoch 46: 0.138550, lr: 0.01000
#> Loss at epoch 47: 0.137999, lr: 0.01000
#> Loss at epoch 48: 0.131653, lr: 0.01000
#> Loss at epoch 49: 0.128602, lr: 0.01000
#> Loss at epoch 50: 0.132828, lr: 0.01000
#> Loss at epoch 51: 0.124458, lr: 0.01000
#> Loss at epoch 52: 0.118968, lr: 0.01000
#> Loss at epoch 53: 0.120668, lr: 0.01000
#> Loss at epoch 54: 0.112987, lr: 0.01000
#> Loss at epoch 55: 0.112808, lr: 0.01000
#> Loss at epoch 56: 0.109660, lr: 0.01000
#> Loss at epoch 57: 0.112381, lr: 0.01000
#> Loss at epoch 58: 0.109906, lr: 0.01000
#> Loss at epoch 59: 0.102357, lr: 0.01000
#> Loss at epoch 60: 0.099804, lr: 0.01000
#> Loss at epoch 61: 0.099913, lr: 0.01000
#> Loss at epoch 62: 0.100493, lr: 0.01000
#> Loss at epoch 63: 0.097146, lr: 0.01000
#> Loss at epoch 64: 0.094450, lr: 0.01000
#> Loss at epoch 65: 0.092289, lr: 0.01000
#> Loss at epoch 66: 0.091554, lr: 0.01000
#> Loss at epoch 67: 0.093704, lr: 0.01000
#> Loss at epoch 68: 0.093189, lr: 0.01000
#> Loss at epoch 69: 0.087317, lr: 0.01000
#> Loss at epoch 70: 0.086101, lr: 0.01000
#> Loss at epoch 71: 0.088625, lr: 0.01000
#> Loss at epoch 72: 0.083799, lr: 0.01000
#> Loss at epoch 73: 0.081474, lr: 0.01000
#> Loss at epoch 74: 0.083640, lr: 0.01000
#> Loss at epoch 75: 0.078293, lr: 0.01000
#> Loss at epoch 76: 0.079954, lr: 0.01000
#> Loss at epoch 77: 0.080666, lr: 0.01000
#> Loss at epoch 78: 0.079187, lr: 0.01000
#> Loss at epoch 79: 0.077950, lr: 0.01000
#> Loss at epoch 80: 0.078211, lr: 0.01000
#> Loss at epoch 81: 0.077823, lr: 0.01000
#> Loss at epoch 82: 0.074846, lr: 0.01000
#> Loss at epoch 83: 0.079448, lr: 0.01000
#> Loss at epoch 84: 0.071144, lr: 0.01000
#> Loss at epoch 85: 0.069736, lr: 0.01000
#> Loss at epoch 86: 0.074044, lr: 0.01000
#> Loss at epoch 87: 0.068769, lr: 0.01000
#> Loss at epoch 88: 0.068477, lr: 0.01000
#> Loss at epoch 89: 0.071221, lr: 0.01000
#> Loss at epoch 90: 0.066842, lr: 0.01000
#> Loss at epoch 91: 0.069920, lr: 0.01000
#> Loss at epoch 92: 0.067206, lr: 0.01000
#> Loss at epoch 93: 0.066341, lr: 0.01000
#> Loss at epoch 94: 0.069623, lr: 0.01000
#> Loss at epoch 95: 0.063408, lr: 0.01000
#> Loss at epoch 96: 0.068439, lr: 0.01000
#> Loss at epoch 97: 0.062418, lr: 0.01000
#> Loss at epoch 98: 0.065210, lr: 0.01000
#> Loss at epoch 99: 0.072858, lr: 0.01000
#> Loss at epoch 100: 0.068683, lr: 0.01000
#> Loss at epoch 1: 1.041954, lr: 0.01000

#> Loss at epoch 2: 0.921608, lr: 0.01000
#> Loss at epoch 3: 0.825877, lr: 0.01000
#> Loss at epoch 4: 0.765165, lr: 0.01000
#> Loss at epoch 5: 0.688122, lr: 0.01000
#> Loss at epoch 6: 0.640561, lr: 0.01000
#> Loss at epoch 7: 0.592972, lr: 0.01000
#> Loss at epoch 8: 0.550812, lr: 0.01000
#> Loss at epoch 9: 0.519388, lr: 0.01000
#> Loss at epoch 10: 0.491109, lr: 0.01000
#> Loss at epoch 11: 0.470940, lr: 0.01000
#> Loss at epoch 12: 0.447176, lr: 0.01000
#> Loss at epoch 13: 0.426492, lr: 0.01000
#> Loss at epoch 14: 0.410585, lr: 0.01000
#> Loss at epoch 15: 0.394398, lr: 0.01000
#> Loss at epoch 16: 0.387315, lr: 0.01000
#> Loss at epoch 17: 0.358664, lr: 0.01000
#> Loss at epoch 18: 0.363604, lr: 0.01000
#> Loss at epoch 19: 0.336838, lr: 0.01000
#> Loss at epoch 20: 0.333819, lr: 0.01000
#> Loss at epoch 21: 0.314092, lr: 0.01000
#> Loss at epoch 22: 0.294557, lr: 0.01000
#> Loss at epoch 23: 0.283134, lr: 0.01000
#> Loss at epoch 24: 0.268334, lr: 0.01000
#> Loss at epoch 25: 0.265036, lr: 0.01000
#> Loss at epoch 26: 0.255478, lr: 0.01000
#> Loss at epoch 27: 0.248125, lr: 0.01000
#> Loss at epoch 28: 0.246690, lr: 0.01000
#> Loss at epoch 29: 0.220406, lr: 0.01000
#> Loss at epoch 30: 0.218995, lr: 0.01000
#> Loss at epoch 31: 0.211211, lr: 0.01000
#> Loss at epoch 32: 0.205809, lr: 0.01000
#> Loss at epoch 33: 0.204228, lr: 0.01000
#> Loss at epoch 34: 0.194784, lr: 0.01000
#> Loss at epoch 35: 0.185452, lr: 0.01000
#> Loss at epoch 36: 0.189052, lr: 0.01000
#> Loss at epoch 37: 0.170412, lr: 0.01000
#> Loss at epoch 38: 0.173709, lr: 0.01000
#> Loss at epoch 39: 0.170945, lr: 0.01000
#> Loss at epoch 40: 0.158677, lr: 0.01000
#> Loss at epoch 41: 0.157716, lr: 0.01000
#> Loss at epoch 42: 0.153734, lr: 0.01000
#> Loss at epoch 43: 0.151485, lr: 0.01000
#> Loss at epoch 44: 0.156553, lr: 0.01000
#> Loss at epoch 45: 0.137842, lr: 0.01000
#> Loss at epoch 46: 0.140933, lr: 0.01000
#> Loss at epoch 47: 0.133758, lr: 0.01000
#> Loss at epoch 48: 0.133273, lr: 0.01000
#> Loss at epoch 49: 0.132954, lr: 0.01000
#> Loss at epoch 50: 0.130681, lr: 0.01000
#> Loss at epoch 51: 0.125372, lr: 0.01000
#> Loss at epoch 52: 0.122442, lr: 0.01000
#> Loss at epoch 53: 0.123693, lr: 0.01000
#> Loss at epoch 54: 0.121925, lr: 0.01000
#> Loss at epoch 55: 0.121622, lr: 0.01000
#> Loss at epoch 56: 0.115993, lr: 0.01000
#> Loss at epoch 57: 0.120076, lr: 0.01000
#> Loss at epoch 58: 0.110667, lr: 0.01000
#> Loss at epoch 59: 0.110613, lr: 0.01000
#> Loss at epoch 60: 0.104328, lr: 0.01000
#> Loss at epoch 61: 0.110951, lr: 0.01000
#> Loss at epoch 62: 0.102370, lr: 0.01000
#> Loss at epoch 63: 0.107298, lr: 0.01000
#> Loss at epoch 64: 0.111824, lr: 0.01000
#> Loss at epoch 65: 0.112514, lr: 0.01000
#> Loss at epoch 66: 0.117718, lr: 0.01000
#> Loss at epoch 67: 0.099495, lr: 0.01000
#> Loss at epoch 68: 0.097359, lr: 0.01000
#> Loss at epoch 69: 0.095918, lr: 0.01000
#> Loss at epoch 70: 0.111480, lr: 0.01000
#> Loss at epoch 71: 0.097787, lr: 0.01000
#> Loss at epoch 72: 0.098198, lr: 0.01000
#> Loss at epoch 73: 0.093798, lr: 0.01000
#> Loss at epoch 74: 0.088523, lr: 0.01000
#> Loss at epoch 75: 0.085269, lr: 0.01000
#> Loss at epoch 76: 0.095244, lr: 0.01000
#> Loss at epoch 77: 0.102064, lr: 0.01000
#> Loss at epoch 78: 0.097386, lr: 0.01000
#> Loss at epoch 79: 0.084039, lr: 0.01000
#> Loss at epoch 80: 0.088636, lr: 0.01000
#> Loss at epoch 81: 0.085057, lr: 0.01000
#> Loss at epoch 82: 0.083539, lr: 0.01000
#> Loss at epoch 83: 0.087730, lr: 0.01000
#> Loss at epoch 84: 0.115399, lr: 0.01000
#> Loss at epoch 85: 0.078624, lr: 0.01000
#> Loss at epoch 86: 0.077649, lr: 0.01000
#> Loss at epoch 87: 0.090330, lr: 0.01000
#> Loss at epoch 88: 0.094252, lr: 0.01000
#> Loss at epoch 89: 0.072074, lr: 0.01000
#> Loss at epoch 90: 0.085394, lr: 0.01000
#> Loss at epoch 91: 0.087729, lr: 0.01000
#> Loss at epoch 92: 0.079488, lr: 0.01000
#> Loss at epoch 93: 0.077160, lr: 0.01000
#> Loss at epoch 94: 0.091311, lr: 0.01000
#> Loss at epoch 95: 0.088667, lr: 0.01000
#> Loss at epoch 96: 0.076489, lr: 0.01000
#> Loss at epoch 97: 0.090377, lr: 0.01000
#> Loss at epoch 98: 0.073660, lr: 0.01000
#> Loss at epoch 99: 0.082882, lr: 0.01000
#> Loss at epoch 100: 0.072192, lr: 0.01000
#> Loss at epoch 1: 3.557764, lr: 0.01000

#> Loss at epoch 2: 3.308642, lr: 0.01000
#> Loss at epoch 3: 3.274878, lr: 0.01000
#> Loss at epoch 4: 3.273276, lr: 0.01000
#> Loss at epoch 5: 3.269602, lr: 0.01000
#> Loss at epoch 6: 3.258938, lr: 0.01000
#> Loss at epoch 7: 3.259117, lr: 0.01000
#> Loss at epoch 8: 3.250010, lr: 0.01000
#> Loss at epoch 9: 3.249795, lr: 0.01000
#> Loss at epoch 10: 3.248078, lr: 0.01000
#> Loss at epoch 11: 3.242347, lr: 0.01000
#> Loss at epoch 12: 3.238866, lr: 0.01000
#> Loss at epoch 13: 3.235480, lr: 0.01000
#> Loss at epoch 14: 3.231714, lr: 0.01000
#> Loss at epoch 15: 3.224512, lr: 0.01000
#> Loss at epoch 16: 3.224098, lr: 0.01000
#> Loss at epoch 17: 3.213248, lr: 0.01000
#> Loss at epoch 18: 3.212830, lr: 0.01000
#> Loss at epoch 19: 3.206559, lr: 0.01000
#> Loss at epoch 20: 3.201039, lr: 0.01000
#> Loss at epoch 21: 3.197436, lr: 0.01000
#> Loss at epoch 22: 3.192628, lr: 0.01000
#> Loss at epoch 23: 3.185266, lr: 0.01000
#> Loss at epoch 24: 3.180265, lr: 0.01000
#> Loss at epoch 25: 3.177464, lr: 0.01000
#> Loss at epoch 26: 3.166990, lr: 0.01000
#> Loss at epoch 27: 3.161127, lr: 0.01000
#> Loss at epoch 28: 3.153812, lr: 0.01000
#> Loss at epoch 29: 3.149565, lr: 0.01000
#> Loss at epoch 30: 3.143549, lr: 0.01000
#> Loss at epoch 31: 3.134075, lr: 0.01000
#> Loss at epoch 32: 3.123114, lr: 0.01000
#> Loss at epoch 33: 3.116488, lr: 0.01000
#> Loss at epoch 34: 3.103397, lr: 0.01000
#> Loss at epoch 35: 3.092855, lr: 0.01000
#> Loss at epoch 36: 3.086442, lr: 0.01000
#> Loss at epoch 37: 3.074081, lr: 0.01000
#> Loss at epoch 38: 3.066444, lr: 0.01000
#> Loss at epoch 39: 3.048586, lr: 0.01000
#> Loss at epoch 40: 3.036232, lr: 0.01000
#> Loss at epoch 41: 3.023283, lr: 0.01000
#> Loss at epoch 42: 3.010240, lr: 0.01000
#> Loss at epoch 43: 2.994427, lr: 0.01000
#> Loss at epoch 44: 2.985291, lr: 0.01000
#> Loss at epoch 45: 2.963230, lr: 0.01000
#> Loss at epoch 46: 2.945317, lr: 0.01000
#> Loss at epoch 47: 2.922581, lr: 0.01000
#> Loss at epoch 48: 2.912794, lr: 0.01000
#> Loss at epoch 49: 2.883504, lr: 0.01000
#> Loss at epoch 50: 2.863654, lr: 0.01000
#> Loss at epoch 51: 2.852189, lr: 0.01000
#> Loss at epoch 52: 2.821250, lr: 0.01000
#> Loss at epoch 53: 2.797081, lr: 0.01000
#> Loss at epoch 54: 2.781896, lr: 0.01000
#> Loss at epoch 55: 2.759186, lr: 0.01000
#> Loss at epoch 56: 2.731999, lr: 0.01000
#> Loss at epoch 57: 2.713169, lr: 0.01000
#> Loss at epoch 58: 2.693273, lr: 0.01000
#> Loss at epoch 59: 2.669815, lr: 0.01000
#> Loss at epoch 60: 2.647569, lr: 0.01000
#> Loss at epoch 61: 2.623310, lr: 0.01000
#> Loss at epoch 62: 2.602955, lr: 0.01000
#> Loss at epoch 63: 2.587594, lr: 0.01000
#> Loss at epoch 64: 2.563069, lr: 0.01000
#> Loss at epoch 65: 2.545920, lr: 0.01000
#> Loss at epoch 66: 2.526538, lr: 0.01000
#> Loss at epoch 67: 2.507672, lr: 0.01000
#> Loss at epoch 68: 2.496987, lr: 0.01000
#> Loss at epoch 69: 2.473560, lr: 0.01000
#> Loss at epoch 70: 2.462135, lr: 0.01000
#> Loss at epoch 71: 2.445650, lr: 0.01000
#> Loss at epoch 72: 2.438689, lr: 0.01000
#> Loss at epoch 73: 2.418812, lr: 0.01000
#> Loss at epoch 74: 2.410201, lr: 0.01000
#> Loss at epoch 75: 2.399029, lr: 0.01000
#> Loss at epoch 76: 2.387134, lr: 0.01000
#> Loss at epoch 77: 2.379141, lr: 0.01000
#> Loss at epoch 78: 2.366836, lr: 0.01000
#> Loss at epoch 79: 2.359399, lr: 0.01000
#> Loss at epoch 80: 2.351885, lr: 0.01000
#> Loss at epoch 81: 2.345700, lr: 0.01000
#> Loss at epoch 82: 2.339253, lr: 0.01000
#> Loss at epoch 83: 2.333450, lr: 0.01000
#> Loss at epoch 84: 2.326963, lr: 0.01000
#> Loss at epoch 85: 2.321367, lr: 0.01000
#> Loss at epoch 86: 2.315550, lr: 0.01000
#> Loss at epoch 87: 2.311197, lr: 0.01000
#> Loss at epoch 88: 2.307295, lr: 0.01000
#> Loss at epoch 89: 2.302387, lr: 0.01000
#> Loss at epoch 90: 2.299849, lr: 0.01000
#> Loss at epoch 91: 2.293640, lr: 0.01000
#> Loss at epoch 92: 2.293107, lr: 0.01000
#> Loss at epoch 93: 2.288502, lr: 0.01000
#> Loss at epoch 94: 2.285827, lr: 0.01000
#> Loss at epoch 95: 2.282153, lr: 0.01000
#> Loss at epoch 96: 2.281443, lr: 0.01000
#> Loss at epoch 97: 2.277360, lr: 0.01000
#> Loss at epoch 98: 2.274772, lr: 0.01000
#> Loss at epoch 99: 2.272979, lr: 0.01000
#> Loss at epoch 100: 2.271738, lr: 0.01000
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
#>  1     1  19.3     0 NA     <dbl [2]> 0.0769  
#>  2     2  22.4     0 NA     <dbl [2]> 0.0647  
#>  3     3  73.8     0 NA     <dbl [2]> 0.0409  
#>  4     4 137.      0 NA     <dbl [2]> 0.000946
#>  5     5  58.7     0 NA     <dbl [2]> 0.0194  
#>  6     6  33.7     0 NA     <dbl [2]> 0.0959  
#>  7     7  21.5     0 NA     <dbl [2]> 0.0706  
#>  8     8  47.0     0 NA     <dbl [2]> 0.0444  
#>  9     9  42.0     0 NA     <dbl [2]> 0.0201  
#> 10    10  37.2     0 NA     <dbl [2]> 0.0794  


#>            [,1]       [,2]       [,3]
#> [1,] 0.32859024 0.03327374 0.07044561
#> [2,] 0.03327374 0.17262027 0.02405240
#> [3,] 0.07044561 0.02405240 0.23674124
# }
```
