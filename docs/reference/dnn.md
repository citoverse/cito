# DNN

fits a custom deep neural network using the Multilayer Perceptron
architecture. `dnn()` supports the formula syntax and allows to
customize the neural network to a maximal degree.

## Usage

``` r
dnn(
  formula = NULL,
  data = NULL,
  hidden = c(50L, 50L),
  activation = "selu",
  bias = TRUE,
  dropout = 0,
  custom = NULL,
  loss = c("mse", "mae", "cross-entropy", "bernoulli", "gaussian", "binomial", "poisson",
    "mvp", "nbinom", "multinomial", "clogit", "softmax"),
  custom_parameters = NULL,
  optimizer = c("sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop", "ignite_adam"),
  lr = 0.01,
  lr_scheduler = NULL,
  alpha = 0.5,
  lambda = 0,
  validation = 0,
  batchsize = NULL,
  shuffle = TRUE,
  epochs = 100,
  weights = NULL,
  early_stopping = Inf,
  burnin = Inf,
  baseloss = NULL,
  device = c("cpu", "cuda", "mps"),
  plot = TRUE,
  verbose = TRUE,
  bootstrap = NULL,
  bootstrap_parallel = FALSE,
  bootstrap_blocking_variable = NULL,
  tuning = config_tuning(),
  hooks = NULL,
  ce = TRUE,
  ce_method = c("analytical", "finite"),
  X = NULL,
  Y = NULL
)
```

## Arguments

- formula:

  an object of class
  "[`formula`](https://rdrr.io/r/stats/formula.html)": a description of
  the model that should be fitted

- data:

  matrix or data.frame with features/predictors and response variable

- hidden:

  hidden units in layers, length of hidden corresponds to number of
  layers

- activation:

  activation functions, can be of length one, or a vector of different
  activation functions for each layer

- bias:

  whether use biases in the layers, can be of length one, or a vector
  (number of hidden layers + 1 (last layer)) of logicals for each layer.

- dropout:

  dropout rate, probability of a node getting left out during training
  (see
  [`nn_dropout`](https://torch.mlverse.org/docs/reference/nn_dropout.html))

- custom:

  custom network architecture, replaces the default MLP that is built
  from hidden, activation, bias, and dropout (these are ignored if
  custom is provided). Either an
  [`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  generator, an instantiated nn_module, or a list(module = generator,
  args = list(...), output_dim = integer). See details

- loss:

  loss after which network should be optimized. Can also be distribution
  from the stats package or own function, see details

- custom_parameters:

  List of parameters/variables to be optimized. Can be used in a custom
  loss function. See Vignette for example.

- optimizer:

  which optimizer used for training the network, for more adjustments to
  optimizer see [`config_optimizer`](config_optimizer.md)

- lr:

  learning rate given to optimizer

- lr_scheduler:

  learning rate scheduler created with
  [`config_lr_scheduler`](config_lr_scheduler.md)

- alpha:

  add L1/L2 regularization to training \\(1 - \alpha) \* \|weights\| +
  \alpha \|\|weights\|\|^2\\ will get added for each layer. Must be
  between 0 and 1

- lambda:

  strength of regularization: lambda penalty, \\\lambda \* (L1 + L2)\\
  (see alpha)

- validation:

  percentage of data set that should be taken as validation set (chosen
  randomly). Alternatively, a vector containing the indices of the
  validation samples can be provided.

- batchsize:

  number of samples that are used to calculate one learning rate step,
  default is 10% of the training data

- shuffle:

  if TRUE, data in each batch gets reshuffled every epoch

- epochs:

  epochs the training goes on for

- weights:

  weights or other values (can be also a matrix) that the likelihood has
  access to

- early_stopping:

  if set to an integer, training stops when the loss has increased for
  the specified number of epochs in a row. The validation loss is used
  when a validation set is available, otherwise the training loss is
  used.

- burnin:

  training is aborted if the training loss is not below the baseline
  loss after burnin epochs

- baseloss:

  baseloss, if null baseloss corresponds to intercept only models

- device:

  device on which network should be trained on. mps correspond to M1/M2
  GPU devices.

- plot:

  plot training loss

- verbose:

  print training and validation loss of epochs

- bootstrap:

  bootstrap neural network or not, numeric corresponds to number of
  bootstrap samples

- bootstrap_parallel:

  parallelize (CPU) bootstrapping

- bootstrap_blocking_variable:

  variable/vector that will be used for blocked bootstrapping (should be
  a factor)

- tuning:

  tuning options created with [`config_tuning`](config_tuning.md)

- hooks:

  list of functions that are executed after each epoch (can be used to
  calculate summary statistics after each epoch)

- ce:

  whether to calculate the conditional effects after training. TRUE
  calculates them on at most 5000 observations, and on fewer if the
  model has many features (the stored effects grow with the number of
  features squared); a number sets the number of observations directly.
  FALSE turns them off, in which case
  [`conditionalEffects`](conditionalEffects.md) and the "ce"/"ale"
  importances calculate them on demand.

- ce_method:

  analytical or finite method

- X:

  Feature matrix or data.frame, alternative data interface

- Y:

  Response vector, factor, matrix or data.frame, alternative data
  interface

## Value

An S3 object of class `"citodnn"` is returned. It is a list containing
everything there is to know about the model and its training process.
The list consists of the following attributes:

- net:

  An object of class "nn_module". Originates from the torch package and
  represents the core object of this workflow.

- call:

  The original function call.

- loss:

  An object of class "nn_module". Contains all relevant information for
  the loss function, e.g. parameters and a function (format_Y) that
  transforms target data.

- data:

  A list. Contains the data used for the training of the model.

- model_properties:

  A list of properties, that define the architecture of the model.

- training_properties:

  A list of all training hyperparameters used the last time the model
  was trained.

- losses:

  A data.frame containing training and validation losses of each epoch.

- best_epoch_net_state_dict:

  Serialized state dict of net from the best training epoch.

- best_epoch_loss_state_dict:

  Serialized state dict of loss from the best training epoch.

- last_epoch_net_state_dict:

  Serialized state dict of net from the last training epoch.

- last_epoch_loss_state_dict:

  Serialized state dict of loss from the last training epoch.

- use_model_epoch:

  String, either "best" or "last". Determines whether the parameters
  (e.g. weights, biases) from the best or the last training epoch are
  used (e.g. for prediction).

- loaded_model_epoch:

  String, shows from which training epoch the parameters are currently
  loaded in `net` and `loss`.

## How neural networks work

In Multilayer Perceptron (MLP) networks, each neuron is connected to
every neuron in the previous layer and every neuron in the subsequent
layer. The value of each neuron is computed using a weighted sum of the
outputs from the previous layer, followed by the application of an
activation function. Specifically, the value of a neuron is calculated
as the weighted sum of the outputs of the neurons in the previous layer,
combined with a bias term. This sum is then passed through an activation
function, which introduces non-linearity into the network. The
calculated value of each neuron becomes the input for the neurons in the
next layer, and the process continues until the output layer is reached.
The choice of activation function and the specific weight values
determine the network's ability to learn and approximate complex
relationships between inputs and outputs.

Therefore the value of each neuron can be calculated using: \\ a
(\sum_j{ w_j \* a_j})\\. Where \\w_j\\ is the weight and \\a_j\\ is the
value from neuron j to the current one. a() is the activation function,
e.g. \\ relu(x) = max(0,x)\\

## Training and convergence of neural networks

Ensuring convergence can be tricky when training neural networks. Their
training is sensitive to a combination of the learning rate (how much
the weights are updated in each optimization step), the batch size (a
random subset of the data is used in each optimization step), and the
number of epochs (number of optimization steps). Typically, the learning
rate should be decreased with the size of the neural networks (depth of
the network and width of the hidden layers). We provide a baseline loss
(intercept only model) that can give hints about an appropriate learning
rate:

![Learning rates](figures/learningrates.jpg)

If the training loss of the model doesn't fall below the baseline loss,
the learning rate is either too high or too low. If this happens, try
higher and lower learning rates.

A common strategy is to try (manually) a few different learning rates to
see if the learning rate is on the right scale.

See the troubleshooting vignette
([`vignette("B-Training_neural_networks")`](../articles/B-Training_neural_networks.md))
for more help on training and debugging neural networks.

## Loss functions / Likelihoods

We support loss functions and likelihoods for different tasks:

|  |  |  |  |
|----|----|----|----|
| Name | Explanation | Example / Task | Data Type |
| mse | mean squared error | Regression, predicting continuous values | `numeric` (vector, matrix, or data.frame) |
| mae | mean absolute error | Regression, predicting continuous values | `numeric` (vector, matrix, or data.frame) |
| cross-entropy | categorical cross entropy | Multi-class, species classification | `factor` or `character` (vector, or 1-column matrix/data.frame) |
| softmax | categorical cross entropy (deprecated alias for `cross-entropy`) | Multi-class, species classification | `factor` or `character` (vector, or 1-column matrix/data.frame) |
| gaussian | Normal likelihood | Regression, residual error is also estimated (similar to [`stats::lm()`](https://rdrr.io/r/stats/lm.html)) | `numeric` (vector, matrix, or data.frame) |
| binomial | Binomial likelihood | Classification/Logistic regression, mortality | `factor`/`character` (vector or 1-column matrix/data.frame), a 0/1 vector, or `integer` matrix/data.frame with exactly 2 columns (successes, failures) |
| bernoulli | Bernoulli likelihood | Classification, binary (0/1) outcome per observation | `integer` (0/1 only), vector, matrix, or data.frame |
| poisson | Poisson likelihood | Regression, count data, e.g. species abundances | `numeric` (vector, matrix, or data.frame); not enforced as integer |
| nbinom | Negative binomial likelihood | Regression, count data with dispersion parameter | `numeric` (vector, matrix, or data.frame); not enforced as integer |
| mvp | multivariate probit model | joint species distribution model, multi species (presence absence) | `integer` (0/1 only) matrix/data.frame with \\\ge\\2 columns; vectors/factors not accepted |
| multinomial | Multinomial likelihood | step selection in animal movement models | `factor`/`character` (vector or 1-column matrix/data.frame), or `integer` matrix/data.frame with \\\ge\\2 columns |
| clogit | conditional binomial | step selection in animal movement models | `factor`/`character` (vector or 1-column matrix/data.frame), or `integer` matrix/data.frame with \\\ge\\2 columns |

## Finding the right architecture

As with the learning rate, there is no definitive guide to choosing the
right architecture for the right task. However, there are some general
rules/recommendations: In general, wider, and deeper neural networks can
improve generalization - but this is a double-edged sword because it
also increases the risk of overfitting. So, if you increase the width
and depth of the network, you should also add regularization (e.g., by
increasing the lambda parameter, which corresponds to the regularization
strength). Furthermore, in [Pichler & Hartig,
2023](https://arxiv.org/abs/2306.10551), we investigated the effects of
the hyperparameters on the prediction performance as a function of the
data size. For example, we found that the `selu` activation function
outperforms `relu` for small data sizes (\<100 observations).

We recommend starting with moderate sizes (like the defaults), and if
the model doesn't generalize/converge, try larger networks along with a
regularization that helps minimize the risk of overfitting (see
[`vignette("B-Training_neural_networks")`](../articles/B-Training_neural_networks.md)
).

## Overfitting

Overfitting means that the model fits the training data well, but
generalizes poorly to new observations. We can use the validation
argument to detect overfitting. If the validation loss starts to
increase again at a certain point, it often means that the models are
starting to overfit your training data:

![Overfitting](figures/overfitting.jpg)

**Solutions**:

- Re-train with epochs = point where model started to overfit

- Early stopping, stop training when model starts to overfit, can be
  specified using the `early_stopping=…` argument

- Use regularization (dropout or elastic-net, see next section)

## Regularization

Elastic Net regularization combines the strengths of L1 (Lasso) and L2
(Ridge) regularization. It introduces a penalty term that encourages
sparse weight values while maintaining overall weight shrinkage. By
controlling the sparsity of the learned model, Elastic Net
regularization helps avoid overfitting while allowing for meaningful
feature selection. We advise using elastic net (e.g. lambda = 0.001 and
alpha = 0.2).

Dropout regularization helps prevent overfitting by randomly disabling a
portion of neurons during training. This technique encourages the
network to learn more robust and generalized representations, as it
prevents individual neurons from relying too heavily on specific input
patterns. Dropout has been widely adopted as a simple yet effective
regularization method in deep learning.

By utilizing these regularization methods in your neural network
training with the cito package, you can improve generalization
performance and enhance the network's ability to handle unseen data.
These techniques act as valuable tools in mitigating overfitting and
promoting more robust and reliable model performance.

## Uncertainty

We can use bootstrapping to generate uncertainties for all outputs.
Bootstrapping can be enabled by setting `bootstrap = ...` to the number
of bootstrap samples to be used. Note, however, that the computational
cost can be excessive.

In some cases it may be worthwhile to parallelize bootstrapping, for
example if you have a GPU and the neural network is small.
Parallelization for bootstrapping can be enabled by setting the
`bootstrap_parallel = ...` argument to the desired number of calls to
run in parallel.

## Custom Optimizer and Learning Rate Schedulers

When training a network, you have the flexibility to customize the
optimizer settings and learning rate scheduler to optimize the learning
process. In the cito package, you can initialize these configurations
using the [`config_lr_scheduler`](config_lr_scheduler.md) and
[`config_optimizer`](config_optimizer.md) functions.

[`config_lr_scheduler`](config_lr_scheduler.md) allows you to define a
specific learning rate scheduler that controls how the learning rate
changes over time during training. This is beneficial in scenarios where
you want to adaptively adjust the learning rate to improve convergence
or avoid getting stuck in local optima.

Similarly, the [`config_optimizer`](config_optimizer.md) function
enables you to specify the optimizer for your network. Different
optimizers, such as stochastic gradient descent (SGD), Adam, or RMSprop,
offer various strategies for updating the network's weights and biases
during training. Choosing the right optimizer can significantly impact
the training process and the final performance of your neural network.

## Hyperparameter tuning

We have implemented experimental support for hyperparameter tuning. We
can mark hyperparameters that should be tuned by cito by setting their
values to [`tune()`](tune.md), for example `dnn (..., lr = tune()`.
[`tune()`](tune.md) is a function that creates a range of random values
for the given hyperparameter. You can change the maximum and minimum
range of the potential hyperparameters or pass custom values to the
`tune(values = c(....))` function. The following table lists the
hyperparameters that can currently be tuned:

|  |  |  |
|----|----|----|
| Hyperparameter | Example | Details |
| hidden | `dnn(…,hidden=tune(10, 20, fixed=’depth’))` | Depth and width can be both tuned or only one of them, if both of them should be tuned, vectors for lower and upper \#' boundaries must be provided (first = number of nodes) |
| bias | `dnn(…, bias=tune())` | Should the bias be turned on or off for all hidden layers |
| lambda | `dnn(…, lambda = tune(0.0001, 0.1))` | lambda will be tuned within the range (0.0001, 0.1) |
| alpha | `dnn(…, lambda = tune(0.2, 0.4))` | alpha will be tuned within the range (0.2, 0.4) |
| activation | `dnn(…, activation = tune())` | activation functions of the hidden layers will be tuned |
| dropout | `dnn(…, dropout = tune())` | Dropout rate will be tuned (globally for all layers) |
| lr | `dnn(…, lr = tune())` | Learning rate will be tuned |
| batchsize | `dnn(…, batchsize = tune())` | batch size will be tuned |

The hyperparameters are tuned by random search (i.e., random values for
the hyperparameters within a specified range) and by cross-validation.
The exact tuning regime can be specified with
[config_tuning](config_tuning.md).

Note that hyperparameter tuning can be expensive. We have implemented an
option to parallelize hyperparameter tuning, including parallelization
over one or more GPUs (the hyperparameter evaluation is parallelized,
not the CV). This can be especially useful for small models. For
example, if you have 4 GPUs, 20 CPU cores, and 20 steps (random samples
from the random search), you could run
`dnn(..., device="cuda",lr = tune(), batchsize=tune(), tuning=config_tuning(parallel=20, NGPU=4)`,
which will distribute 20 model fits across 4 GPUs, so that each GPU will
process 5 models (in parallel).

As this is an experimental feature, we welcome feature requests and bug
reports on our github site.

For the custom values, all hyperparameters except for the hidden layers
require a vector of values. Hidden layers expect a two-column matrix
where the first column is the number of hidden nodes and the second
column corresponds to the number of hidden layers.

## Custom architectures

Instead of the default MLP, a custom
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
can be passed via the `custom` argument. The module must fulfill the
following contract:

- If the initialize method has arguments named `input`, `output`, or
  `embeddings`, they are filled in automatically by cito (unless
  provided in `args`): `input` = number of columns of the model.matrix
  built from the formula (for cnn() the input shape), `output` = the
  output dimension expected by the loss (within mmn(): `output_dim`),
  `embeddings` = named vector with the number of levels of each e()
  term. This way the number of predictors does not have to be hard coded
  in the custom module.

- forward(x) receives a 2D float tensor (batch, ncol(X), i.e. the
  columns of the model.matrix built from the formula) and must return a
  2D tensor (batch, output), where output matches the dimension expected
  by the loss (checked at fit time).

- The output must be on the link scale, i.e. the raw output of the last
  linear layer. Do not apply softmax/sigmoid/exp in the forward method,
  the loss functions apply the (inverse) link internally (e.g. the
  cross-entropy loss applies softmax, the poisson loss applies exp) and
  `predict(..., type = "response")` returns predictions on the response
  scale via the loss' invlink. Applying the link twice prevents the
  model from converging properly.

- If embeddings (e()) are used in the formula, forward(x, z) receives a
  second 2D long tensor with the 1-based factor codes (one column per
  e() term, in formula order). Embedding layers, including their
  initialization and regularization, must be handled inside the custom
  module, e(weights=), e(train=), e(lambda=), and e(alpha=) are
  therefore not supported.

- The forward method should be differentiable with respect to x if
  conditionalEffects/summary are to be used.

- lambda/alpha regularization is applied to all parameters of the custom
  module with more than one dimension.

- If a generator is passed (recommended), the model can be saved/loaded
  with saveRDS as usual, but the generator's enclosing environment must
  not capture torch tensors. If an instantiated module is passed, it is
  serialized via
  [`torch_serialize`](https://torch.mlverse.org/docs/reference/torch_serialize.html)
  internally.

- Architecture hyperparameters (hidden, activation, bias, dropout) are
  ignored and cannot be tuned; tuning of training hyperparameters (lr,
  lambda, alpha, batchsize) is still supported.

## Activation functions

Supported activation functions: "relu", "leaky_relu", "tanh", "elu",
"rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6",
"sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink",
"hardshrink", "log_sigmoid"

## Training on graphic cards

If you have an NVIDIA CUDA-enabled device and have installed the CUDA
toolkit version 11.3 and cuDNN 8.4, you can take advantage of GPU
acceleration for training your neural networks. It is crucial to have
these specific versions installed, as other versions may not be
compatible. For detailed installation instructions and more information
on utilizing GPUs for training, please refer to the [mlverse: 'torch'
documentation](https://torch.mlverse.org/docs/articles/installation.html).

Note: GPU training is optional, and the package can still be used for
training on CPU even without CUDA and cuDNN installations.

## See also

[`predict.citodnn`](predict.citodnn.md),
[`plot.citodnn`](plot.citodnn.md),
[`coef.citodnn`](coef.citodnn.md),[`print.citodnn`](print.citodnn.md),
[`summary.citodnn`](summary.citodnn.md),
[`continue_training`](continue_training.md),
[`analyze_training`](analyze_training.md), [`PDP`](PDP.md),
[`ALE`](ALE.md),

## Author

Christian Amesoeder, Maximilian Pichler

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
#> Loss at epoch 1: 1.047610, lr: 0.01000

#> Loss at epoch 2: 0.892782, lr: 0.01000
#> Loss at epoch 3: 0.788463, lr: 0.01000
#> Loss at epoch 4: 0.692052, lr: 0.01000
#> Loss at epoch 5: 0.644701, lr: 0.01000
#> Loss at epoch 6: 0.589922, lr: 0.01000
#> Loss at epoch 7: 0.557884, lr: 0.01000
#> Loss at epoch 8: 0.515625, lr: 0.01000
#> Loss at epoch 9: 0.479033, lr: 0.01000
#> Loss at epoch 10: 0.458168, lr: 0.01000
#> Loss at epoch 11: 0.441431, lr: 0.01000
#> Loss at epoch 12: 0.408867, lr: 0.01000
#> Loss at epoch 13: 0.399429, lr: 0.01000
#> Loss at epoch 14: 0.387338, lr: 0.01000
#> Loss at epoch 15: 0.377321, lr: 0.01000
#> Loss at epoch 16: 0.359182, lr: 0.01000
#> Loss at epoch 17: 0.336992, lr: 0.01000
#> Loss at epoch 18: 0.314337, lr: 0.01000
#> Loss at epoch 19: 0.312227, lr: 0.01000
#> Loss at epoch 20: 0.293291, lr: 0.01000
#> Loss at epoch 21: 0.283489, lr: 0.01000
#> Loss at epoch 22: 0.272252, lr: 0.01000
#> Loss at epoch 23: 0.271839, lr: 0.01000
#> Loss at epoch 24: 0.250646, lr: 0.01000
#> Loss at epoch 25: 0.246924, lr: 0.01000
#> Loss at epoch 26: 0.235966, lr: 0.01000
#> Loss at epoch 27: 0.241137, lr: 0.01000
#> Loss at epoch 28: 0.219607, lr: 0.01000
#> Loss at epoch 29: 0.212032, lr: 0.01000
#> Loss at epoch 30: 0.218194, lr: 0.01000
#> Loss at epoch 31: 0.212571, lr: 0.01000
#> Loss at epoch 32: 0.200118, lr: 0.01000
#> Loss at epoch 33: 0.187104, lr: 0.01000
#> Loss at epoch 34: 0.182861, lr: 0.01000
#> Loss at epoch 35: 0.175508, lr: 0.01000
#> Loss at epoch 36: 0.174161, lr: 0.01000
#> Loss at epoch 37: 0.166152, lr: 0.01000
#> Loss at epoch 38: 0.170598, lr: 0.01000
#> Loss at epoch 39: 0.182429, lr: 0.01000
#> Loss at epoch 40: 0.152212, lr: 0.01000
#> Loss at epoch 41: 0.180620, lr: 0.01000
#> Loss at epoch 42: 0.148277, lr: 0.01000
#> Loss at epoch 43: 0.143725, lr: 0.01000
#> Loss at epoch 44: 0.142058, lr: 0.01000
#> Loss at epoch 45: 0.145207, lr: 0.01000
#> Loss at epoch 46: 0.132275, lr: 0.01000
#> Loss at epoch 47: 0.124470, lr: 0.01000
#> Loss at epoch 48: 0.126993, lr: 0.01000
#> Loss at epoch 49: 0.125357, lr: 0.01000
#> Loss at epoch 50: 0.126574, lr: 0.01000
#> Loss at epoch 51: 0.120332, lr: 0.01000
#> Loss at epoch 52: 0.123630, lr: 0.01000
#> Loss at epoch 53: 0.129695, lr: 0.01000
#> Loss at epoch 54: 0.120108, lr: 0.01000
#> Loss at epoch 55: 0.130742, lr: 0.01000
#> Loss at epoch 56: 0.113400, lr: 0.01000
#> Loss at epoch 57: 0.109455, lr: 0.01000
#> Loss at epoch 58: 0.113352, lr: 0.01000
#> Loss at epoch 59: 0.107271, lr: 0.01000
#> Loss at epoch 60: 0.103485, lr: 0.01000
#> Loss at epoch 61: 0.108765, lr: 0.01000
#> Loss at epoch 62: 0.110835, lr: 0.01000
#> Loss at epoch 63: 0.107924, lr: 0.01000
#> Loss at epoch 64: 0.117210, lr: 0.01000
#> Loss at epoch 65: 0.104804, lr: 0.01000
#> Loss at epoch 66: 0.106462, lr: 0.01000
#> Loss at epoch 67: 0.126777, lr: 0.01000
#> Loss at epoch 68: 0.097991, lr: 0.01000
#> Loss at epoch 69: 0.105106, lr: 0.01000
#> Loss at epoch 70: 0.107571, lr: 0.01000
#> Loss at epoch 71: 0.084662, lr: 0.01000
#> Loss at epoch 72: 0.101372, lr: 0.01000
#> Loss at epoch 73: 0.099413, lr: 0.01000
#> Loss at epoch 74: 0.088655, lr: 0.01000
#> Loss at epoch 75: 0.090058, lr: 0.01000
#> Loss at epoch 76: 0.105790, lr: 0.01000
#> Loss at epoch 77: 0.098646, lr: 0.01000
#> Loss at epoch 78: 0.100589, lr: 0.01000
#> Loss at epoch 79: 0.094730, lr: 0.01000
#> Loss at epoch 80: 0.097224, lr: 0.01000
#> Loss at epoch 81: 0.088540, lr: 0.01000
#> Loss at epoch 82: 0.088708, lr: 0.01000
#> Loss at epoch 83: 0.101753, lr: 0.01000
#> Loss at epoch 84: 0.085440, lr: 0.01000
#> Loss at epoch 85: 0.089618, lr: 0.01000
#> Loss at epoch 86: 0.081203, lr: 0.01000
#> Loss at epoch 87: 0.083068, lr: 0.01000
#> Loss at epoch 88: 0.091987, lr: 0.01000
#> Loss at epoch 89: 0.087156, lr: 0.01000
#> Loss at epoch 90: 0.118454, lr: 0.01000
#> Loss at epoch 91: 0.104999, lr: 0.01000
#> Loss at epoch 92: 0.087815, lr: 0.01000
#> Loss at epoch 93: 0.083065, lr: 0.01000
#> Loss at epoch 94: 0.084915, lr: 0.01000
#> Loss at epoch 95: 0.100002, lr: 0.01000
#> Loss at epoch 96: 0.078273, lr: 0.01000
#> Loss at epoch 97: 0.098475, lr: 0.01000
#> Loss at epoch 98: 0.083712, lr: 0.01000
#> Loss at epoch 99: 0.083797, lr: 0.01000
#> Loss at epoch 100: 0.081820, lr: 0.01000
#> Loss at epoch 101: 0.082601, lr: 0.01000

#> Loss at epoch 102: 0.076694, lr: 0.01000
#> Loss at epoch 103: 0.091089, lr: 0.01000
#> Loss at epoch 104: 0.077123, lr: 0.01000
#> Loss at epoch 105: 0.091445, lr: 0.01000
#> Loss at epoch 106: 0.091220, lr: 0.01000
#> Loss at epoch 107: 0.118371, lr: 0.01000
#> Loss at epoch 108: 0.106468, lr: 0.01000
#> Loss at epoch 109: 0.069542, lr: 0.01000
#> Loss at epoch 110: 0.096720, lr: 0.01000
#> Loss at epoch 111: 0.074798, lr: 0.01000
#> Loss at epoch 112: 0.075740, lr: 0.01000
#> Loss at epoch 113: 0.068093, lr: 0.01000
#> Loss at epoch 114: 0.078388, lr: 0.01000
#> Loss at epoch 115: 0.065940, lr: 0.01000
#> Loss at epoch 116: 0.066176, lr: 0.01000
#> Loss at epoch 117: 0.079676, lr: 0.01000
#> Loss at epoch 118: 0.080898, lr: 0.01000
#> Loss at epoch 119: 0.088275, lr: 0.01000
#> Loss at epoch 120: 0.068800, lr: 0.01000
#> Loss at epoch 121: 0.085497, lr: 0.01000
#> Loss at epoch 122: 0.071093, lr: 0.01000
#> Loss at epoch 123: 0.079509, lr: 0.01000
#> Loss at epoch 124: 0.079955, lr: 0.01000
#> Loss at epoch 125: 0.087661, lr: 0.01000
#> Loss at epoch 126: 0.080534, lr: 0.01000
#> Loss at epoch 127: 0.082294, lr: 0.01000
#> Loss at epoch 128: 0.067994, lr: 0.01000
#> Loss at epoch 129: 0.096183, lr: 0.01000
#> Loss at epoch 130: 0.074553, lr: 0.01000
#> Loss at epoch 131: 0.076242, lr: 0.01000
#> Loss at epoch 132: 0.066735, lr: 0.01000
#> Loss at epoch 133: 0.065233, lr: 0.01000
#> Loss at epoch 134: 0.073242, lr: 0.01000
#> Loss at epoch 135: 0.067636, lr: 0.01000
#> Loss at epoch 136: 0.076862, lr: 0.01000
#> Loss at epoch 137: 0.083534, lr: 0.01000
#> Loss at epoch 138: 0.075017, lr: 0.01000
#> Loss at epoch 139: 0.064435, lr: 0.01000
#> Loss at epoch 140: 0.081610, lr: 0.01000
#> Loss at epoch 141: 0.064009, lr: 0.01000
#> Loss at epoch 142: 0.066052, lr: 0.01000
#> Loss at epoch 143: 0.062426, lr: 0.01000
#> Loss at epoch 144: 0.063256, lr: 0.01000
#> Loss at epoch 145: 0.067600, lr: 0.01000
#> Loss at epoch 146: 0.073859, lr: 0.01000
#> Loss at epoch 147: 0.073222, lr: 0.01000
#> Loss at epoch 148: 0.084544, lr: 0.01000
#> Loss at epoch 149: 0.089969, lr: 0.01000
#> Loss at epoch 150: 0.074522, lr: 0.01000
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



#> Loss at epoch 1: 0.673099, lr: 0.01000

#> Loss at epoch 2: 0.586612, lr: 0.01000
#> Loss at epoch 3: 0.551332, lr: 0.01000
#> Loss at epoch 4: 0.521364, lr: 0.01000
#> Loss at epoch 5: 0.495815, lr: 0.01000
#> Loss at epoch 6: 0.471680, lr: 0.01000
#> Loss at epoch 7: 0.452111, lr: 0.01000
#> Loss at epoch 8: 0.431466, lr: 0.01000
#> Loss at epoch 9: 0.418337, lr: 0.01000
#> Loss at epoch 10: 0.402363, lr: 0.01000
#> Loss at epoch 11: 0.384449, lr: 0.01000
#> Loss at epoch 12: 0.372305, lr: 0.01000
#> Loss at epoch 13: 0.360876, lr: 0.01000
#> Loss at epoch 14: 0.349842, lr: 0.01000
#> Loss at epoch 15: 0.341812, lr: 0.01000
#> Loss at epoch 16: 0.327649, lr: 0.01000
#> Loss at epoch 17: 0.319970, lr: 0.01000
#> Loss at epoch 18: 0.315298, lr: 0.01000
#> Loss at epoch 19: 0.305130, lr: 0.01000
#> Loss at epoch 20: 0.297186, lr: 0.01000
#> Loss at epoch 21: 0.292309, lr: 0.01000
#> Loss at epoch 22: 0.283399, lr: 0.01000
#> Loss at epoch 23: 0.275238, lr: 0.01000
#> Loss at epoch 24: 0.271337, lr: 0.01000
#> Loss at epoch 25: 0.263307, lr: 0.01000
#> Loss at epoch 26: 0.256266, lr: 0.01000
#> Loss at epoch 27: 0.262010, lr: 0.01000
#> Loss at epoch 28: 0.243615, lr: 0.01000
#> Loss at epoch 29: 0.240985, lr: 0.01000
#> Loss at epoch 30: 0.237142, lr: 0.01000
#> Loss at epoch 31: 0.237555, lr: 0.01000
#> Loss at epoch 32: 0.225110, lr: 0.01000
#> Loss at epoch 33: 0.224458, lr: 0.01000
#> Loss at epoch 34: 0.215281, lr: 0.01000
#> Loss at epoch 35: 0.211877, lr: 0.01000
#> Loss at epoch 36: 0.208955, lr: 0.01000
#> Loss at epoch 37: 0.206568, lr: 0.01000
#> Loss at epoch 38: 0.195595, lr: 0.01000
#> Loss at epoch 39: 0.194365, lr: 0.01000
#> Loss at epoch 40: 0.189517, lr: 0.01000
#> Loss at epoch 41: 0.193385, lr: 0.01000
#> Loss at epoch 42: 0.180241, lr: 0.01000
#> Loss at epoch 43: 0.180931, lr: 0.01000
#> Loss at epoch 44: 0.172148, lr: 0.01000
#> Loss at epoch 45: 0.168916, lr: 0.01000
#> Loss at epoch 46: 0.161902, lr: 0.01000
#> Loss at epoch 47: 0.165100, lr: 0.01000
#> Loss at epoch 48: 0.156666, lr: 0.01000
#> Loss at epoch 49: 0.154791, lr: 0.01000
#> Loss at epoch 50: 0.151878, lr: 0.01000
#> Loss at epoch 51: 0.150796, lr: 0.01000
#> Loss at epoch 52: 0.150205, lr: 0.01000
#> Loss at epoch 53: 0.144378, lr: 0.01000
#> Loss at epoch 54: 0.138802, lr: 0.01000
#> Loss at epoch 55: 0.142683, lr: 0.01000
#> Loss at epoch 56: 0.134676, lr: 0.01000
#> Loss at epoch 57: 0.132630, lr: 0.01000
#> Loss at epoch 58: 0.133897, lr: 0.01000
#> Loss at epoch 59: 0.125627, lr: 0.01000
#> Loss at epoch 60: 0.124523, lr: 0.01000
#> Loss at epoch 61: 0.120150, lr: 0.01000
#> Loss at epoch 62: 0.121673, lr: 0.01000
#> Loss at epoch 63: 0.114689, lr: 0.01000
#> Loss at epoch 64: 0.114484, lr: 0.01000
#> Loss at epoch 65: 0.114619, lr: 0.01000
#> Loss at epoch 66: 0.113901, lr: 0.01000
#> Loss at epoch 67: 0.108983, lr: 0.01000
#> Loss at epoch 68: 0.109720, lr: 0.01000
#> Loss at epoch 69: 0.105502, lr: 0.01000
#> Loss at epoch 70: 0.098955, lr: 0.01000
#> Loss at epoch 71: 0.104303, lr: 0.01000
#> Loss at epoch 72: 0.104899, lr: 0.01000
#> Loss at epoch 73: 0.105508, lr: 0.01000
#> Loss at epoch 74: 0.097000, lr: 0.01000
#> Loss at epoch 75: 0.097277, lr: 0.01000
#> Loss at epoch 76: 0.092530, lr: 0.01000
#> Loss at epoch 77: 0.095517, lr: 0.01000
#> Loss at epoch 78: 0.091596, lr: 0.01000
#> Loss at epoch 79: 0.095261, lr: 0.01000
#> Loss at epoch 80: 0.091921, lr: 0.01000
#> Loss at epoch 81: 0.088120, lr: 0.01000
#> Loss at epoch 82: 0.088884, lr: 0.01000
#> Loss at epoch 83: 0.091654, lr: 0.01000
#> Loss at epoch 84: 0.087390, lr: 0.01000
#> Loss at epoch 85: 0.093177, lr: 0.01000
#> Loss at epoch 86: 0.084196, lr: 0.01000
#> Loss at epoch 87: 0.081539, lr: 0.01000
#> Loss at epoch 88: 0.087324, lr: 0.01000
#> Loss at epoch 89: 0.081075, lr: 0.01000
#> Loss at epoch 90: 0.087649, lr: 0.01000
#> Loss at epoch 91: 0.077732, lr: 0.01000
#> Loss at epoch 92: 0.081827, lr: 0.01000
#> Loss at epoch 93: 0.089344, lr: 0.01000
#> Loss at epoch 94: 0.082772, lr: 0.01000
#> Loss at epoch 95: 0.071802, lr: 0.01000
#> Loss at epoch 96: 0.076497, lr: 0.01000
#> Loss at epoch 97: 0.080083, lr: 0.01000
#> Loss at epoch 98: 0.078343, lr: 0.01000
#> Loss at epoch 99: 0.080494, lr: 0.01000
#> Loss at epoch 100: 0.076998, lr: 0.01000
#> Loss at epoch 1: 1.131531, lr: 0.01000

#> Loss at epoch 2: 0.919546, lr: 0.01000
#> Loss at epoch 3: 0.832303, lr: 0.01000
#> Loss at epoch 4: 0.738153, lr: 0.01000
#> Loss at epoch 5: 0.675164, lr: 0.01000
#> Loss at epoch 6: 0.617698, lr: 0.01000
#> Loss at epoch 7: 0.596311, lr: 0.01000
#> Loss at epoch 8: 0.531553, lr: 0.01000
#> Loss at epoch 9: 0.499873, lr: 0.01000
#> Loss at epoch 10: 0.472194, lr: 0.01000
#> Loss at epoch 11: 0.472169, lr: 0.01000
#> Loss at epoch 12: 0.435488, lr: 0.01000
#> Loss at epoch 13: 0.405205, lr: 0.01000
#> Loss at epoch 14: 0.397699, lr: 0.01000
#> Loss at epoch 15: 0.373819, lr: 0.01000
#> Loss at epoch 16: 0.344326, lr: 0.01000
#> Loss at epoch 17: 0.362608, lr: 0.01000
#> Loss at epoch 18: 0.327755, lr: 0.01000
#> Loss at epoch 19: 0.311631, lr: 0.01000
#> Loss at epoch 20: 0.296480, lr: 0.01000
#> Loss at epoch 21: 0.287739, lr: 0.01000
#> Loss at epoch 22: 0.280704, lr: 0.01000
#> Loss at epoch 23: 0.272791, lr: 0.01000
#> Loss at epoch 24: 0.250329, lr: 0.01000
#> Loss at epoch 25: 0.244502, lr: 0.01000
#> Loss at epoch 26: 0.242433, lr: 0.01000
#> Loss at epoch 27: 0.226342, lr: 0.01000
#> Loss at epoch 28: 0.227408, lr: 0.01000
#> Loss at epoch 29: 0.239694, lr: 0.01000
#> Loss at epoch 30: 0.210975, lr: 0.01000
#> Loss at epoch 31: 0.206775, lr: 0.01000
#> Loss at epoch 32: 0.197558, lr: 0.01000
#> Loss at epoch 33: 0.192611, lr: 0.01000
#> Loss at epoch 34: 0.183896, lr: 0.01000
#> Loss at epoch 35: 0.183691, lr: 0.01000
#> Loss at epoch 36: 0.187726, lr: 0.01000
#> Loss at epoch 37: 0.179555, lr: 0.01000
#> Loss at epoch 38: 0.164799, lr: 0.01000
#> Loss at epoch 39: 0.161865, lr: 0.01000
#> Loss at epoch 40: 0.206064, lr: 0.01000
#> Loss at epoch 41: 0.162717, lr: 0.01000
#> Loss at epoch 42: 0.160399, lr: 0.01000
#> Loss at epoch 43: 0.151961, lr: 0.01000
#> Loss at epoch 44: 0.142115, lr: 0.01000
#> Loss at epoch 45: 0.144716, lr: 0.01000
#> Loss at epoch 46: 0.134994, lr: 0.01000
#> Loss at epoch 47: 0.136796, lr: 0.01000
#> Loss at epoch 48: 0.139099, lr: 0.01000
#> Loss at epoch 49: 0.128870, lr: 0.01000
#> Loss at epoch 50: 0.127726, lr: 0.01000
#> Loss at epoch 51: 0.137342, lr: 0.01000
#> Loss at epoch 52: 0.142199, lr: 0.01000
#> Loss at epoch 53: 0.120015, lr: 0.01000
#> Loss at epoch 54: 0.136557, lr: 0.01000
#> Loss at epoch 55: 0.129173, lr: 0.01000
#> Loss at epoch 56: 0.122449, lr: 0.01000
#> Loss at epoch 57: 0.119311, lr: 0.01000
#> Loss at epoch 58: 0.112996, lr: 0.01000
#> Loss at epoch 59: 0.119444, lr: 0.01000
#> Loss at epoch 60: 0.106306, lr: 0.01000
#> Loss at epoch 61: 0.103052, lr: 0.01000
#> Loss at epoch 62: 0.111367, lr: 0.01000
#> Loss at epoch 63: 0.105712, lr: 0.01000
#> Loss at epoch 64: 0.098682, lr: 0.01000
#> Loss at epoch 65: 0.105603, lr: 0.01000
#> Loss at epoch 66: 0.096138, lr: 0.01000
#> Loss at epoch 67: 0.095587, lr: 0.01000
#> Loss at epoch 68: 0.096322, lr: 0.01000
#> Loss at epoch 69: 0.094421, lr: 0.01000
#> Loss at epoch 70: 0.113112, lr: 0.01000
#> Loss at epoch 71: 0.098762, lr: 0.01000
#> Loss at epoch 72: 0.098612, lr: 0.01000
#> Loss at epoch 73: 0.089817, lr: 0.01000
#> Loss at epoch 74: 0.087835, lr: 0.01000
#> Loss at epoch 75: 0.106064, lr: 0.01000
#> Loss at epoch 76: 0.096136, lr: 0.01000
#> Loss at epoch 77: 0.080880, lr: 0.01000
#> Loss at epoch 78: 0.094191, lr: 0.01000
#> Loss at epoch 79: 0.089933, lr: 0.01000
#> Loss at epoch 80: 0.093061, lr: 0.01000
#> Loss at epoch 81: 0.106057, lr: 0.01000
#> Loss at epoch 82: 0.082139, lr: 0.01000
#> Loss at epoch 83: 0.083490, lr: 0.01000
#> Loss at epoch 84: 0.097702, lr: 0.01000
#> Loss at epoch 85: 0.083351, lr: 0.01000
#> Loss at epoch 86: 0.077428, lr: 0.01000
#> Loss at epoch 87: 0.096786, lr: 0.01000
#> Loss at epoch 88: 0.077784, lr: 0.01000
#> Loss at epoch 89: 0.092930, lr: 0.01000
#> Loss at epoch 90: 0.077674, lr: 0.01000
#> Loss at epoch 91: 0.087673, lr: 0.01000
#> Loss at epoch 92: 0.101678, lr: 0.01000
#> Loss at epoch 93: 0.077219, lr: 0.01000
#> Loss at epoch 94: 0.079223, lr: 0.01000
#> Loss at epoch 95: 0.085558, lr: 0.01000
#> Loss at epoch 96: 0.073956, lr: 0.01000
#> Loss at epoch 97: 0.084609, lr: 0.01000
#> Loss at epoch 98: 0.104687, lr: 0.01000
#> Loss at epoch 99: 0.084520, lr: 0.01000
#> Loss at epoch 100: 0.096416, lr: 0.01000
#> Loss at epoch 1: 3.682381, lr: 0.01000

#> Loss at epoch 2: 3.294621, lr: 0.01000
#> Loss at epoch 3: 3.256830, lr: 0.01000
#> Loss at epoch 4: 3.241283, lr: 0.01000
#> Loss at epoch 5: 3.236913, lr: 0.01000
#> Loss at epoch 6: 3.228450, lr: 0.01000
#> Loss at epoch 7: 3.228770, lr: 0.01000
#> Loss at epoch 8: 3.222266, lr: 0.01000
#> Loss at epoch 9: 3.220139, lr: 0.01000
#> Loss at epoch 10: 3.211467, lr: 0.01000
#> Loss at epoch 11: 3.208660, lr: 0.01000
#> Loss at epoch 12: 3.200411, lr: 0.01000
#> Loss at epoch 13: 3.199195, lr: 0.01000
#> Loss at epoch 14: 3.191670, lr: 0.01000
#> Loss at epoch 15: 3.187047, lr: 0.01000
#> Loss at epoch 16: 3.177722, lr: 0.01000
#> Loss at epoch 17: 3.170024, lr: 0.01000
#> Loss at epoch 18: 3.163408, lr: 0.01000
#> Loss at epoch 19: 3.154720, lr: 0.01000
#> Loss at epoch 20: 3.147918, lr: 0.01000
#> Loss at epoch 21: 3.137136, lr: 0.01000
#> Loss at epoch 22: 3.130076, lr: 0.01000
#> Loss at epoch 23: 3.123847, lr: 0.01000
#> Loss at epoch 24: 3.111760, lr: 0.01000
#> Loss at epoch 25: 3.097186, lr: 0.01000
#> Loss at epoch 26: 3.086954, lr: 0.01000
#> Loss at epoch 27: 3.075425, lr: 0.01000
#> Loss at epoch 28: 3.066186, lr: 0.01000
#> Loss at epoch 29: 3.053875, lr: 0.01000
#> Loss at epoch 30: 3.037676, lr: 0.01000
#> Loss at epoch 31: 3.017720, lr: 0.01000
#> Loss at epoch 32: 3.004156, lr: 0.01000
#> Loss at epoch 33: 2.988850, lr: 0.01000
#> Loss at epoch 34: 2.972227, lr: 0.01000
#> Loss at epoch 35: 2.953804, lr: 0.01000
#> Loss at epoch 36: 2.941721, lr: 0.01000
#> Loss at epoch 37: 2.914810, lr: 0.01000
#> Loss at epoch 38: 2.891287, lr: 0.01000
#> Loss at epoch 39: 2.867359, lr: 0.01000
#> Loss at epoch 40: 2.853650, lr: 0.01000
#> Loss at epoch 41: 2.830839, lr: 0.01000
#> Loss at epoch 42: 2.797697, lr: 0.01000
#> Loss at epoch 43: 2.781916, lr: 0.01000
#> Loss at epoch 44: 2.745592, lr: 0.01000
#> Loss at epoch 45: 2.725676, lr: 0.01000
#> Loss at epoch 46: 2.695169, lr: 0.01000
#> Loss at epoch 47: 2.675290, lr: 0.01000
#> Loss at epoch 48: 2.646004, lr: 0.01000
#> Loss at epoch 49: 2.619965, lr: 0.01000
#> Loss at epoch 50: 2.597354, lr: 0.01000
#> Loss at epoch 51: 2.568193, lr: 0.01000
#> Loss at epoch 52: 2.549519, lr: 0.01000
#> Loss at epoch 53: 2.523628, lr: 0.01000
#> Loss at epoch 54: 2.504871, lr: 0.01000
#> Loss at epoch 55: 2.489016, lr: 0.01000
#> Loss at epoch 56: 2.463736, lr: 0.01000
#> Loss at epoch 57: 2.444420, lr: 0.01000
#> Loss at epoch 58: 2.429753, lr: 0.01000
#> Loss at epoch 59: 2.418805, lr: 0.01000
#> Loss at epoch 60: 2.402865, lr: 0.01000
#> Loss at epoch 61: 2.390542, lr: 0.01000
#> Loss at epoch 62: 2.380021, lr: 0.01000
#> Loss at epoch 63: 2.369290, lr: 0.01000
#> Loss at epoch 64: 2.358917, lr: 0.01000
#> Loss at epoch 65: 2.350258, lr: 0.01000
#> Loss at epoch 66: 2.340402, lr: 0.01000
#> Loss at epoch 67: 2.333662, lr: 0.01000
#> Loss at epoch 68: 2.330305, lr: 0.01000
#> Loss at epoch 69: 2.322404, lr: 0.01000
#> Loss at epoch 70: 2.316494, lr: 0.01000
#> Loss at epoch 71: 2.310920, lr: 0.01000
#> Loss at epoch 72: 2.310121, lr: 0.01000
#> Loss at epoch 73: 2.302551, lr: 0.01000
#> Loss at epoch 74: 2.297861, lr: 0.01000
#> Loss at epoch 75: 2.294823, lr: 0.01000
#> Loss at epoch 76: 2.290777, lr: 0.01000
#> Loss at epoch 77: 2.287246, lr: 0.01000
#> Loss at epoch 78: 2.284240, lr: 0.01000
#> Loss at epoch 79: 2.281092, lr: 0.01000
#> Loss at epoch 80: 2.278792, lr: 0.01000
#> Loss at epoch 81: 2.276934, lr: 0.01000
#> Loss at epoch 82: 2.273517, lr: 0.01000
#> Loss at epoch 83: 2.273334, lr: 0.01000
#> Loss at epoch 84: 2.270286, lr: 0.01000
#> Loss at epoch 85: 2.268805, lr: 0.01000
#> Loss at epoch 86: 2.266817, lr: 0.01000
#> Loss at epoch 87: 2.264881, lr: 0.01000
#> Loss at epoch 88: 2.262718, lr: 0.01000
#> Loss at epoch 89: 2.261000, lr: 0.01000
#> Loss at epoch 90: 2.259238, lr: 0.01000
#> Loss at epoch 91: 2.257553, lr: 0.01000
#> Loss at epoch 92: 2.256529, lr: 0.01000
#> Loss at epoch 93: 2.254707, lr: 0.01000
#> Loss at epoch 94: 2.254000, lr: 0.01000
#> Loss at epoch 95: 2.252978, lr: 0.01000
#> Loss at epoch 96: 2.251548, lr: 0.01000
#> Loss at epoch 97: 2.250273, lr: 0.01000
#> Loss at epoch 98: 2.249917, lr: 0.01000
#> Loss at epoch 99: 2.247842, lr: 0.01000
#> Loss at epoch 100: 2.246752, lr: 0.01000
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
#>  1     1  56.3     0 NA     <dbl [2]> 0.0769  
#>  2     2  38.0     0 NA     <dbl [2]> 0.0647  
#>  3     3  20.7     0 NA     <dbl [2]> 0.0409  
#>  4     4 134.      0 NA     <dbl [2]> 0.000946
#>  5     5  46.2     0 NA     <dbl [2]> 0.0194  
#>  6     6  33.3     0 NA     <dbl [2]> 0.0959  
#>  7     7  47.8     0 NA     <dbl [2]> 0.0706  
#>  8     8  48.6     0 NA     <dbl [2]> 0.0444  
#>  9     9  40.2     0 NA     <dbl [2]> 0.0201  
#> 10    10  28.4     0 NA     <dbl [2]> 0.0794  


#>            [,1]       [,2]       [,3]
#> [1,] 0.33127803 0.04741610 0.07248808
#> [2,] 0.04741610 0.17655137 0.03152749
#> [3,] 0.07248808 0.03152749 0.21995127
# }
```
