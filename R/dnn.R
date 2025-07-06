#' DNN
#'
#' fits a custom deep neural network using the Multilayer Perceptron architecture. `dnn()` supports the formula syntax and allows to customize the neural network to a maximal degree.
#'
#' @param formula an object of class "\code{\link[stats]{formula}}": a description of the model that should be fitted
#' @param data matrix or data.frame with features/predictors and response variable
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of different activation functions for each layer
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param dropout dropout rate, probability of a node getting left out during training (see \code{\link[torch]{nn_dropout}})
#' @param loss loss after which network should be optimized. Can also be distribution from the stats package or own function, see details
#' @param custom_parameters List of parameters/variables to be optimized. Can be used in a custom loss function. See Vignette for example.
#' @param optimizer which optimizer used for training the network, for more adjustments to optimizer see \code{\link{config_optimizer}}
#' @param lr learning rate given to optimizer
#' @param lr_scheduler learning rate scheduler created with \code{\link{config_lr_scheduler}}
#' @param alpha add L1/L2 regularization to training  \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2} will get added for each layer. Must be between 0 and 1
#' @param lambda strength of regularization: lambda penalty, \eqn{\lambda * (L1 + L2)} (see alpha)
#' @param validation percentage of data set that should be taken as validation set (chosen randomly)
#' @param batchsize number of samples that are used to calculate one learning rate step, default is 10% of the training data
#' @param shuffle if TRUE, data in each batch gets reshuffled every epoch
#' @param epochs epochs the training goes on for
#' @param early_stopping if set to integer, training will stop if loss has gotten higher for defined number of epochs in a row, will use validation loss is available.
#' @param burnin training is aborted if the trainings loss is not below the baseline loss after burnin epochs
#' @param baseloss baseloss, if null baseloss corresponds to intercept only models
#' @param device device on which network should be trained on. mps correspond to M1/M2 GPU devices.
#' @param plot plot training loss
#' @param verbose print training and validation loss of epochs
#' @param bootstrap bootstrap neural network or not, numeric corresponds to number of bootstrap samples
#' @param bootstrap_parallel parallelize (CPU) bootstrapping
#' @param tuning tuning options created with \code{\link{config_tuning}}
#' @param X Feature matrix or data.frame, alternative data interface
#' @param Y Response vector, factor, matrix or data.frame, alternative data interface
#'
#' @details
#'
#' # Activation functions
#'
#' Supported activation functions:  "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#'
#' # Loss functions / Likelihoods
#'
#' We support loss functions and likelihoods for different tasks:
#'
#' | Name| Explanation| Example / Task|
#' | :--- | :--- | :--- |
#' | mse | mean squared error |Regression, predicting continuous values|
#' | mae | mean absolute error | Regression, predicting continuous values |
#' | cross-entropy | categorical cross entropy |Multi-class, species classification|
#' | gaussian | Normal likelihood | Regression, residual error is also estimated (similar to `stats::lm()`)	|
#' | binomial | Binomial likelihood | Classification/Logistic regression, mortality|
#' | poisson | Poisson likelihood |Regression, count data, e.g. species abundances|
#' | nbinom | Negative binomial likelihood | Regression, count data with dispersion parameter |
#' | mvp | multivariate probit model | joint species distribution model, multi species (presence absence) |
#' | multinomial | Multinomial likelihood | step selection in animal movement models |
#' | clogit | conditional binomial | step selection in animal movement models |
#'
#' # Training and convergence of neural networks
#'
#' Ensuring convergence can be tricky when training neural networks. Their training is sensitive to a combination of the learning rate (how much the weights are updated in each optimization step), the batch size (a random subset of the data is used in each optimization step), and the number of epochs (number of optimization steps). Typically, the learning rate should be decreased with the size of the neural networks (depth of the network and width of the hidden layers). We provide a baseline loss (intercept only model) that can give hints about an appropriate learning rate:
#'
#' ![](learningrates.jpg "Learning rates")
#'
#' If the training loss of the model doesn't fall below the baseline loss, the learning rate is either too high or too low. If this happens, try higher and lower learning rates.
#'
#' A common strategy is to try (manually) a few different learning rates to see if the learning rate is on the right scale.
#'
#' See the troubleshooting vignette (\code{vignette("B-Training_neural_networks")}) for more help on training and debugging neural networks.
#'
#' # Finding the right architecture
#'
#' As with the learning rate, there is no definitive guide to choosing the right architecture for the right task. However, there are some general rules/recommendations: In general, wider, and deeper neural networks can improve generalization - but this is a double-edged sword because it also increases the risk of overfitting. So, if you increase the width and depth of the network, you should also add regularization (e.g., by increasing the lambda parameter, which corresponds to the regularization strength). Furthermore, in [Pichler & Hartig, 2023](https://arxiv.org/abs/2306.10551), we investigated the effects of the hyperparameters on the prediction performance as a function of the data size. For example, we found that the `selu` activation function outperforms `relu` for small data sizes (<100 observations).
#'
#' We recommend starting with moderate sizes (like the defaults), and if the model doesn't generalize/converge, try larger networks along with a regularization that helps minimize the risk of overfitting (see \code{vignette("B-Training_neural_networks")} ).
#'
#' # Overfitting
#'
#' Overfitting means that the model fits the training data well, but generalizes poorly to new observations. We can use the validation argument to detect overfitting. If the validation loss starts to increase again at a certain point, it often means that the models are starting to overfit your training data:
#'
#' ![](overfitting.jpg "Overfitting")
#'
#' **Solutions**:
#'
#' \itemize{
#'   \item Re-train with epochs = point where model started to overfit
#'   \item Early stopping, stop training when model starts to overfit, can be specified using the `early_stopping=…` argument
#'   \item Use regularization (dropout or elastic-net, see next section)
#' }
#'
#' # Regularization
#'
#' Elastic Net regularization combines the strengths of L1 (Lasso) and L2 (Ridge) regularization. It introduces a penalty term that encourages sparse weight values while maintaining overall weight shrinkage. By controlling the sparsity of the learned model, Elastic Net regularization helps avoid overfitting while allowing for meaningful feature selection. We advise using elastic net (e.g. lambda = 0.001 and alpha = 0.2).
#'
#' Dropout regularization helps prevent overfitting by randomly disabling a portion of neurons during training. This technique encourages the network to learn more robust and generalized representations, as it prevents individual neurons from relying too heavily on specific input patterns. Dropout has been widely adopted as a simple yet effective regularization method in deep learning.
#'
#' By utilizing these regularization methods in your neural network training with the cito package, you can improve generalization performance and enhance the network's ability to handle unseen data. These techniques act as valuable tools in mitigating overfitting and promoting more robust and reliable model performance.
#'
#' # Uncertainty
#'
#' We can use bootstrapping to generate uncertainties for all outputs. Bootstrapping can be enabled by setting `bootstrap = ...` to the number of bootstrap samples to be used. Note, however, that the computational cost can be excessive.
#'
#' In some cases it may be worthwhile to parallelize bootstrapping, for example if you have a GPU and the neural network is small. Parallelization for bootstrapping can be enabled by setting the `bootstrap_parallel = ...` argument to the desired number of calls to run in parallel.
#'
#' # Custom Optimizer and Learning Rate Schedulers
#'
#' When training a network, you have the flexibility to customize the optimizer settings and learning rate scheduler to optimize the learning process. In the cito package, you can initialize these configurations using the \code{\link{config_lr_scheduler}} and \code{\link{config_optimizer}} functions.
#'
#' \code{\link{config_lr_scheduler}} allows you to define a specific learning rate scheduler that controls how the learning rate changes over time during training. This is beneficial in scenarios where you want to adaptively adjust the learning rate to improve convergence or avoid getting stuck in local optima.
#'
#' Similarly, the \code{\link{config_optimizer}} function enables you to specify the optimizer for your network. Different optimizers, such as stochastic gradient descent (SGD), Adam, or RMSprop, offer various strategies for updating the network's weights and biases during training. Choosing the right optimizer can significantly impact the training process and the final performance of your neural network.
#'
#'
#' # Hyperparameter tuning
#'
#' We have implemented experimental support for hyperparameter tuning. We can mark hyperparameters that should be tuned by cito by setting their values to `tune()`, for example `dnn (..., lr = tune()`. [tune()] is a function that creates a range of random values for the given hyperparameter. You can change the maximum and minimum range of the potential hyperparameters or pass custom values to the `tune(values = c(....))` function. The following table lists the hyperparameters that can currently be tuned:
#'
#'   | Hyperparameter| Example| Details |
#'   | :--- | :--- | :--- |
#'   | hidden | `dnn(…,hidden=tune(10, 20, fixed=’depth’))` |Depth and width can be both tuned or only one of them, if both of them should be tuned, vectors for lower and upper #' boundaries must be provided (first = number of nodes)|
#'   | bias | `dnn(…, bias=tune())` | Should the bias be turned on or off for all hidden layers |
#'   | lambda | `dnn(…, lambda = tune(0.0001, 0.1))` |lambda will be tuned within the range (0.0001, 0.1)|
#'   | alpha | `dnn(…, lambda = tune(0.2, 0.4))` |alpha will be tuned within the range (0.2, 0.4)|
#'   | activation | `dnn(…, activation = tune())`  |	activation functions of the hidden layers will be tuned|
#'   | dropout | `dnn(…, dropout = tune())`  | Dropout rate will be tuned (globally for all layers)|
#'   | lr | `dnn(…, lr = tune())`  |Learning rate will be tuned|
#'   | batchsize | `dnn(…, batchsize = tune())`  | batch size will be tuned |
#'   | epochs | `dnn(…, batchsize = tune())`  | batchsize will be tuned |
#'
#'   The hyperparameters are tuned by random search (i.e., random values for the hyperparameters within a specified range) and by cross-validation. The exact tuning regime can be specified with [config_tuning].
#'
#' Note that hyperparameter tuning can be expensive. We have implemented an option to parallelize hyperparameter tuning, including parallelization over one or more GPUs (the hyperparameter evaluation is parallelized, not the CV). This can be especially useful for small models. For example, if you have 4 GPUs, 20 CPU cores, and 20 steps (random samples from the random search), you could run `dnn(..., device="cuda",lr = tune(), batchsize=tune(), tuning=config_tuning(parallel=20, NGPU=4)`, which will distribute 20 model fits across 4 GPUs, so that each GPU will process 5 models (in parallel).
#'
#' As this is an experimental feature, we welcome feature requests and bug reports on our github site.
#'
#' For the custom values, all hyperparameters except for the hidden layers require a vector of values. Hidden layers expect a two-column matrix where the first column is the number of hidden nodes and the second column corresponds to the number of hidden layers.
#'
#'
#'# How neural networks work
#' In Multilayer Perceptron (MLP) networks, each neuron is connected to every neuron in the previous layer and every neuron in the subsequent layer. The value of each neuron is computed using a weighted sum of the outputs from the previous layer, followed by the application of an activation function. Specifically, the value of a neuron is calculated as the weighted sum of the outputs of the neurons in the previous layer, combined with a bias term. This sum is then passed through an activation function, which introduces non-linearity into the network. The calculated value of each neuron becomes the input for the neurons in the next layer, and the process continues until the output layer is reached. The choice of activation function and the specific weight values determine the network's ability to learn and approximate complex relationships between inputs and outputs.
#'
#' Therefore the value of each neuron can be calculated using: \eqn{ a (\sum_j{ w_j * a_j})}. Where \eqn{w_j} is the weight and \eqn{a_j} is the value from neuron j to the current one. a() is the activation function, e.g. \eqn{ relu(x) = max(0,x)}
#'
#'
#' # Training on graphic cards
#'
#' If you have an NVIDIA CUDA-enabled device and have installed the CUDA toolkit version 11.3 and cuDNN 8.4, you can take advantage of GPU acceleration for training your neural networks. It is crucial to have these specific versions installed, as other versions may not be compatible.
#' For detailed installation instructions and more information on utilizing GPUs for training, please refer to the [mlverse: 'torch' documentation](https://torch.mlverse.org/docs/articles/installation.html).
#'
#' Note: GPU training is optional, and the package can still be used for training on CPU even without CUDA and cuDNN installations.
#'
#' @return an S3 object of class \code{"citodnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call}
#' \item{loss}{A list which contains relevant information for the target variable and the used loss function}
#' \item{data}{Contains data used for training the model}
#' \item{weights}{List of weights for each training epoch}
#' \item{use_model_epoch}{Integer, which defines which model from which training epoch should be used for prediction. 1 = best model, 2 = last model}
#' \item{loaded_model_epoch}{Integer, shows which model from which epoch is loaded currently into model$net.}
#' \item{model_properties}{A list of properties of the neural network, contains number of input nodes, number of output nodes, size of hidden layers, activation functions, whether bias is included and if dropout layers are included.}
#' \item{training_properties}{A list of all training parameters that were used the last time the model was trained. It consists of learning rate, information about an learning rate scheduler, information about the optimizer, number of epochs, whether early stopping was used, if plot was active, lambda and alpha for L1/L2 regularization, batchsize, shuffle, was the data set split into validation and training, which formula was used for training and at which epoch did the training stop.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' @import checkmate
#' @example /inst/examples/dnn-example.R
#' @author Christian Amesoeder, Maximilian Pichler
#' @seealso \code{\link{predict.citodnn}}, \code{\link{plot.citodnn}},  \code{\link{coef.citodnn}},\code{\link{print.citodnn}}, \code{\link{summary.citodnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}, \code{\link{PDP}}, \code{\link{ALE}},
#' @export
dnn <- function(formula = NULL,
                data = NULL,
                hidden = c(50L, 50L),
                activation = "selu",
                bias = TRUE,
                dropout = 0.0,
                loss = c("mse", "mae", "cross-entropy", "gaussian", "binomial", "poisson", "mvp", "nbinom", "multinomial", "clogit"),
                custom_parameters = NULL,
                optimizer = c("sgd","adam","adadelta", "adagrad", "rmsprop", "rprop", "ignite_adam"),
                lr = 0.01,
                lr_scheduler = NULL,
                alpha = 0.5,
                lambda = 0.0,
                validation = 0,
                batchsize = NULL,
                shuffle = TRUE,
                epochs = 100,
                early_stopping = Inf,
                burnin = Inf,
                baseloss = NULL,
                device = c("cpu","cuda", "mps"),
                plot = TRUE,
                verbose = TRUE,
                bootstrap = NULL,
                bootstrap_parallel = FALSE,
                tuning = config_tuning(),
                hooks = NULL,
                X = NULL,
                Y = NULL) {

  tuner = check_hyperparameters(hidden = hidden ,
                                bias = bias,
                                activation = activation,
                                lambda = lambda,
                                alpha = alpha,
                                dropout =dropout,
                                batchsize = batchsize,
                                epochs = epochs,
                                lr = lr)

  checkmate::qassert(custom_parameters, c("0", "L+"))
  checkmate::qassert(validation, "N1[0,1)")
  checkmate::qassert(shuffle, "B1")
  checkmate::qassert(early_stopping, "N1[1,]")
  checkmate::qassert(burnin, "N1[1,]")
  checkmate::qassert(baseloss, c("0", "N1"))
  checkmate::qassert(device, "S+[3,)")
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")


  device <- match.arg(device)

  # if(!is.function(loss) & !inherits(loss,"family")){
  #   loss <- match.arg(loss)
  #
  #   if((device == "mps") & (loss %in% c("poisson", "nbinom", "multinomial"))) {
  #     message("`poisson`, `nbinom`, and `multinomial` are not yet supported for `device=mps`, switching to `device=cpu`")
  #     device = "cpu"
  #   }
  # }
  #
  # if(inherits(loss,"family")) {
  #   if((device == "mps") & (loss$family %in% c("poisson", "nbinom"))) {
  #     message("`poisson` or `nbinom` are not yet supported for `device=mps`, switching to `device=cpu`")
  #     device = "cpu"
  #   }
  # }

  device_old <- device
  device <- check_device(device)


  tmp_data = get_X_Y(formula, X, Y, data)
  old_formula = tmp_data$old_formula
  X = tmp_data$X
  Y = tmp_data$Y
  Z = tmp_data$Z
  formula = tmp_data$formula
  Z_formula = tmp_data$Z_terms
  data = tmp_data$data

  if(!is.null(Z)) {

    Z_args = list()
    for(i in 1:length(tmp_data$Z_args)) {
      Z_args = append(Z_args, eval(tmp_data$Z_args[[i]]))
    }

    embeddings = list(inputs = ncol(Z),
                      dims = tmp_data$Zlvls, # input embeddings
                      args = Z_args )
  } else {
    embeddings = NULL
  }

  # Only return the model properties if no Y specified (Used in mmn())
  if(is.null(Y)) {
    if(any(inherits(hidden, "tune"), inherits(activation, "tune"), inherits(bias, "tune"), inherits(dropout, "tune"))) stop("Tuning of hyperparameters is not supported within mmn().")
    model_properties <- list(input = ncol(X),
                             hidden = hidden,
                             activation = activation,
                             bias = bias,
                             dropout = dropout,
                             embeddings = embeddings)
    class(model_properties) <- "citodnn_properties"
    return(model_properties)
  }

  if(is.character(loss)) loss <- match.arg(loss)
  loss_obj <- get_loss_new(loss, Y, custom_parameters)

  response_column <- NULL
  if(inherits(loss_obj, "cross-entropy loss")) response_column = as.character(LHSForm(formula)) # add response_column to loss_obj instead of out

  X_torch <- torch::torch_tensor(X)
  Y_torch <- loss_obj$format_Y(Y)
  if(!is.null(embeddings)) {
    Z_torch = torch::torch_tensor(Z, dtype = torch::torch_long())
  } else {
    Z_torch = NULL
  }

  if(is.null(batchsize)) batchsize = round(0.1*dim(Y_torch)[1])

  ### Hyperparameter tuning ###
  #TODO: Adjust for new loss objects
  if(length(tuner) != 0 ) {
    parameters = as.list(match.call())
    parameters[!nzchar(names(parameters))] = NULL
    #parameters$hidden = hidden
    model = tuning_function(tuner, parameters, loss.fkt,loss_obj, X, Y, Z, data, old_formula, tuning, Y_torch, loss, device_old) # add Z....
    return(model)
  }


  if(is.null(bootstrap) || !bootstrap) {

    if(is.null(baseloss)) baseloss <- loss_obj$baseloss

    if(validation != 0) {
      n_samples <- dim(Y_torch)[1]
      valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
      train <- c(1:n_samples)[-valid]
      if(is.null(Z_torch)) {
        train_dl <- get_data_loader(X_torch[train,], Y_torch[train,], batch_size = batchsize, shuffle = shuffle)
        valid_dl <- get_data_loader(X_torch[valid,], Y_torch[valid,], batch_size = batchsize, shuffle = shuffle)
      } else {
        train_dl <- get_data_loader(X_torch[train,], Z_torch[train,], Y_torch[train,], batch_size = batchsize, shuffle = shuffle)
        valid_dl <- get_data_loader(X_torch[valid,], Z_torch[valid,], Y_torch[valid,], batch_size = batchsize, shuffle = shuffle)
      }
    } else {
      if(is.null(Z_torch)) {
        train_dl <- get_data_loader(X_torch, Y_torch, batch_size = batchsize, shuffle = shuffle)
      } else {
        train_dl <- get_data_loader(X_torch, Z_torch, Y_torch, batch_size = batchsize, shuffle = shuffle)
      }
      valid_dl <- NULL
    }

    model_properties <- list(input = ncol(X),
                             output = loss_obj$y_dim,
                             hidden = hidden,
                             activation = activation,
                             bias = bias,
                             dropout = dropout,
                             embeddings = embeddings)
    class(model_properties) <- "citodnn_properties"

    net <- build_dnn(model_properties)

    training_properties <- list(optimizer = optimizer,
                                lr = lr,
                                lr_scheduler = lr_scheduler,
                                alpha = alpha,
                                lambda = lambda,
                                validation = validation,
                                batchsize = batchsize,
                                shuffle = shuffle,
                                epochs = epochs, #redundant?
                                early_stopping = early_stopping,
                                burnin = burnin,
                                baseloss = baseloss,
                                device = device_old,
                                plot = plot,
                                verbose = verbose,
                                formula = formula, #redundant: in out$call
                                embeddings = embeddings, #redundant: in model_properties
                                hooks = hooks) #unnecessary: does nothing?

    out <- list()
    class(out) <- "citodnn"
    out$net <- net
    out$call <- match.call()
    out$call$formula <- stats::terms.formula(formula, data=data) #suggestion: put formula and Z_formula back together to get nice looking formula of what the network actually does (e.g. for print.citodnn)
    out$Z_formula = Z_formula #redundant: inferable from old_formula
    out$old_formula = old_formula #redundant: see above
    out$loss <- loss_obj
    out$response_column <- response_column #redundant: inferable from old_formula
    out$data <- list(X = X, Y = as.matrix(Y_torch), data = data, Z = Z) #redundant to save X, Y and Z?
    # levels should be only saved for features in the model, otherwise we get warnings from the predict function
    data_tmp = data[, labels(stats::terms(formula, data = data)), drop=FALSE]
    out$data$xlvls <- lapply(data_tmp[,sapply(data_tmp, is.factor), drop = F], function(j) levels(j) )
    out$data$xlvls <- out$data$xlvls[!names(out$data$xlvls) %in% as.character(formula[[2]])]
    if(validation != 0) out$data <- append(out$data, list(validation = valid))
    out$model_properties <- model_properties
    out$training_properties <- training_properties
    # below unneccessary?
    out$weights <- list()
    out$use_model_epoch <- 2
    out$loaded_model_epoch <- torch::torch_tensor(0)

    out <- train_model(model = out, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)

  } else {
    out <- list()
    class(out) <- "citodnnBootstrap"

    if(bootstrap_parallel == FALSE) {
      pb = progress::progress_bar$new(total = bootstrap, format = "[:bar] :percent :eta", width = round(getOption("width")/2))
      models = list()

      for(b in 1:bootstrap) {
        indices <- sample(nrow(data),replace = TRUE)
        #Alternative that doesn't require future editing when new arguments are added to dnn() (e.g. burnin, baseloss are missing)
        #call <- match.call()
        #call$data <- data[indices,]
        #call$plot <- FALSE
        #call$verbose <- FALSE
        #call$bootstrap <- NULL
        #m = eval(call)
        m = do.call(dnn, args = list(
          formula = old_formula, data = data[indices,], loss = loss, hidden = hidden, activation = activation,
          bias = bias, validation = validation, lambda = lambda, alpha = alpha, lr = lr, dropout = dropout, hooks = hooks,
          optimizer = optimizer, batchsize = batchsize, shuffle = shuffle, epochs = epochs, plot = FALSE, verbose = FALSE,
          bootstrap = NULL, device = device_old, custom_parameters = custom_parameters, lr_scheduler = lr_scheduler, early_stopping = early_stopping,
          bootstrap_parallel = FALSE
        ))
        m$data$indices = indices
        m$data$original = list(data = data, X = X, Y = Y_transformed, Z = Z) #X, Y, Z redundant
        pb$tick()
        models[[b]] = m
      }

    } else {
      if(is.logical(bootstrap_parallel)) {
          bootstrap_parallel = parallel::detectCores() -1
      }
      if(is.numeric(bootstrap_parallel)) {
        backend = parabar::start_backend(bootstrap_parallel)
        parabar::export(backend, ls(environment()), environment())
        parabar::evaluate(backend, {
          require(torch)
          torch::torch_set_num_threads(2L)
          torch::torch_set_num_interop_threads(2L)
        })
      }
      parabar::configure_bar(type = "modern", format = "[:bar] :percent :eta", width = round(getOption("width")/2))
      models <- parabar::par_lapply(backend, 1:bootstrap, function(i) {
        indices <- sample(nrow(data),replace = TRUE)
        m = do.call(dnn, args = list(
          formula = old_formula, data = data[indices,], loss = loss, hidden = hidden, activation = activation,
          bias = bias, validation = validation,lambda = lambda, alpha = alpha,lr = lr, dropout = dropout, hooks = hooks,
          optimizer = optimizer,batchsize = batchsize,shuffle = shuffle, epochs = epochs, plot = FALSE, verbose = FALSE,
          bootstrap = NULL, device = device_old, custom_parameters = custom_parameters, lr_scheduler = lr_scheduler, early_stopping = early_stopping,
          bootstrap_parallel = FALSE
        ))
        m$data$indices = indices
        m$data$original = list(data = data, X = X, Y = Y_transformed, Z = Z) #X, Y, Z redundant
        m
      })
      if(!is.null(backend)) parabar::stop_backend(backend)

    }

    out$models <- models
    #out$call for print.citodnnBootstrap ?
    out$old_formula = old_formula #old_formula into out$call ?
    out$loss = loss_obj
    out$data <- list(X = X, Y = as.matrix(Y_torch), data = data, Z = Z) #X, Y, Z redundant
    out$device = device_old
    out$response_column = response_column #redundant: inferable from old_formula

    out$successfull = any(!sapply(models, function(m) m$successfull) == 0)

  }
  return(out)
}



#' Print class citodnn
#'
#' @param x a model created by \code{\link{dnn}}
#' @param ... additional arguments
#' @example /inst/examples/print.citodnn-example.R
#' @return original object x gets returned
#' @export
print.citodnn <- function(x,...){
  x <- check_model(x)
  print(x$call)
  print(x$net)
  return(invisible(x))
}

#' @rdname print.citodnn
#' @export
print.citodnnBootstrap <- function(x,...){
  #print call?
  x$models <- lapply(x$models, check_model)
  print(x$models[[1]]$net)
  return(invisible(x))
}

#' Extract Model Residuals
#'
#' Returns residuals of training set.
#'
#' @param object a model created by \code{\link{dnn}}
#' @param ... no additional arguments implemented
#' @return residuals of training set
#' @export
residuals.citodnn <- function(object,...){
  object <- check_model(object)
  out <- data.frame(
    true = object$data$Y, #If Y is removed to reduce redundancy, infer Y here from data
    pred = stats::predict(object, object$data$data)
  )
  return(out)
}





#' Summarize Neural Network of class citodnn
#'
#' Performs a Feature Importance calculation based on Permutations
#'
#' @details
#'
#' Performs the feature importance calculation as suggested by  Fisher, Rudin, and Dominici (2018), and the mean and standard deviation of the average conditional Effects as suggested by Pichler & Hartig (2023).
#'
#' Feature importances are in their interpretation similar to a ANOVA. Main and interaction effects are absorbed into the features. Also, feature importances are prone to collinearity between features, i.e. if two features are collinear, the importances might be overestimated.
#'
#' Average conditional effects (ACE) are similar to marginal effects and approximate linear effects, i.e. their interpretation is similar to effects in a linear regression model.
#'
#' The standard deviation of the ACE informs about the non-linearity of the feature effects. Higher values correlate with stronger non-linearities.
#'
#' For each feature n permutation get done and original and permuted predictive mean squared error (\eqn{e_{perm}} & \eqn{e_{orig}}) get evaluated with \eqn{ FI_j= e_{perm}/e_{orig}}. Based on Mean Squared Error.
#'
#' @param object a model of class citodnn created by \code{\link{dnn}}
#' @param n_permute number of permutations performed. Default is \eqn{3 * \sqrt{n}}, where n euqals then number of samples in the training set
#' @param device for calculating variable importance and conditional effects
#' @param adjust_se adjust standard errors for importance (standard errors are multiplied with 1/sqrt(3) )
#' @param type on what scale should the average conditional effects be calculated ("response" or "link")
#' @param ... additional arguments
#' @return summary.citodnn returns an object of class "summary.citodnn", a list with components
#' @export
summary.citodnn <- function(object, n_permute = NULL, device = NULL, type = "response", ...){
  object <- check_model(object)
  out <- list()
  class(out) <- "summary.citodnn"
  if(is.null(device)) device = object$training_properties$device
  out$importance <- get_importance(object, n_permute, device)
  for(i in 2:ncol(out$importance)) out$importance[,i] = out$importance[,i] - 1
  out$conditionalEffects = conditionalEffects(object, device = device, type = type)
  out$ACE = sapply(out$conditionalEffects, function(R) diag(R$mean))
  if(!is.matrix(out$ACE )) out$ACE= matrix(out$ACE , ncol = 1L)
  colnames(out$ACE) = paste0("Response_", 1:ncol(out$ACE))
  out$ASCE = sapply(out$conditionalEffects, function(R) diag(R$sd))
  if(!is.matrix(out$ASCE )) out$ASCE = matrix(out$ASCE , ncol = 1L)
  colnames(out$ASCE) = paste0("Response_", 1:ncol(out$ASCE))
  rownames(out$ASCE) = rownames(out$ACE)
  return(out)
}



#' Print method for class summary.citodnn
#'
#' @param x a summary object created by \code{\link{summary.citodnn}}
#' @param ... additional arguments
#' @return List with Matrices for importance, average CE, absolute sum of CE, and standard deviation of the CE
#' @export
print.summary.citodnn <- function(x, ... ){
  out = list()
  cat("Summary of Deep Neural Network Model\n")
  cat("\nFeature Importance:\n")
  print(x$importance)
  cat("\nAverage Conditional Effects:\n")
  print(x$ACE)
  cat("\nStandard Deviation of Conditional Effects:\n")
  print(x$ASCE)
  out$importance = x$importance
  out$ACE = x$ACE
  out$ASCE = x$ASCE
  return(invisible(out))
}



#' @rdname summary.citodnn
#' @export
summary.citodnnBootstrap <- function(object, n_permute = NULL, device = NULL, adjust_se = FALSE, type = "response", ...){
  object$models <- lapply(object$models, check_model)
  out <- list()
  class(out) <- "summary.citodnnBootstrap"
  if(is.null(device)) device = object$device
  out$importance <- lapply(object$models, function(m) get_importance(m, n_permute, device = device, ...))
  out$conditionalEffects <- lapply(object$models, function(m) conditionalEffects(m, device = device, type = type))

  if(!is.null(out$importance[[1]])){
    res_imps = list()
    for(i in 2:ncol(out$importance[[1]])) {
      tmp = sapply(out$importance, function(X) X[,i, drop=FALSE])
      if(inherits(tmp, "list")) tmp = do.call(cbind, tmp)
      imps = apply(tmp, 1, mean) - 1
      imps_se = apply(tmp, 1, stats::sd)

      if(adjust_se) imps_se = imps_se * 1/sqrt(3)

      coefmat = cbind(
        as.vector(as.matrix(imps)),
        as.vector(as.matrix(imps_se)),
        as.vector(as.matrix(imps/imps_se)),
        as.vector(as.matrix(stats::pnorm(abs(imps/imps_se), lower.tail = FALSE)*2))
      )
      colnames(coefmat) = c("Importance", "Std.Err", "Z value", "Pr(>|z|)")
      resp = object$loss$responses[i-1]
      if(inherits(loss_obj, "cross-entropy loss")) resp = object$response_column
      rownames(coefmat) = paste0(out$importance[[1]]$variable, " \U2192 ", resp)
      if(ncol(out$importance[[1]]) > 2) coefmat = rbind(coefmat, c(NA))
      res_imps[[i-1]] = coefmat
    }
    out$importance_coefmat = do.call(rbind, res_imps)
  } else {
    out$importance_coefmat = NULL
  }


  res_ACE = list()

  for(i in 1:length(out$conditionalEffects[[1]])) {
    tmp = sapply(1:length(out$conditionalEffects), function(j) diag(out$conditionalEffects[[j]][[i]]$mean))
    if(inherits(tmp, "list")) tmp = do.call(cbind, tmp)
    if(is.vector(tmp)) tmp = matrix(tmp, nrow = 1L)
    eff = apply(tmp, 1, mean)
    eff_se = apply(tmp, 1, stats::sd)

    coefmat = cbind(
      as.vector(as.matrix(eff)),
      as.vector(as.matrix(eff_se)),
      as.vector(as.matrix(eff/eff_se)),
      as.vector(as.matrix(stats::pnorm(abs(eff/eff_se), lower.tail = FALSE)*2))
    )
    colnames(coefmat) = c("ACE", "Std.Err", "Z value", "Pr(>|z|)")
    rownames(coefmat) = paste0( rownames(out$conditionalEffects[[1]][[1]]$mean), " \U2192 ", object$loss$responses[i])
    if(i != length(out$conditionalEffects[[1]])) coefmat = rbind(coefmat, c(NA))
    res_ACE[[i]] = coefmat
  }
  out$ACE_coefmat = do.call(rbind, res_ACE)


  res_ASCE = list()

  for(i in 1:length(out$conditionalEffects[[1]])) {
    tmp = sapply(1:length(out$conditionalEffects), function(j) diag(out$conditionalEffects[[j]][[i]]$sd))
    if(inherits(tmp, "list")) tmp = do.call(cbind, tmp)
    if(is.vector(tmp)) tmp = matrix(tmp, nrow = 1L)
    eff = apply(tmp, 1, mean)
    eff_se = apply(tmp, 1, stats::sd)

    coefmat = cbind(
      as.vector(as.matrix(eff)),
      as.vector(as.matrix(eff_se)),
      as.vector(as.matrix(eff/eff_se)),
      as.vector(as.matrix(stats::pnorm(abs(eff/eff_se), lower.tail = FALSE)*2))
    )
    colnames(coefmat) = c("ACE", "Std.Err", "Z value", "Pr(>|z|)")
    rownames(coefmat) = paste0(rownames(out$conditionalEffects[[1]][[1]]$mean), " \U2192 ", object$loss$responses[i])
    if(i != length(out$conditionalEffects[[1]])) coefmat = rbind(coefmat, c(NA))
    res_ASCE[[i]] = coefmat
  }
  out$ASCE_coefmat = do.call(rbind, res_ASCE)

  return(out)
}


#' @rdname print.summary.citodnn
#' @export
print.summary.citodnnBootstrap <- function(x, ... ){
  out = list()
  cat("Summary of Deep Neural Network Model\n\n")
  #cat("\t##########################################################\n")
  cli::cli_h3(cli::style_bold("Feature Importance \n"))
  #cat("\t##########################################################\n")
  #cat("\nFeature Importance:\n")

  if(!is.null(x$importance_coefmat)) stats::printCoefmat(x$importance_coefmat, signif.stars = getOption("show.signif.stars"), digits = 3, na.print = "")


  #cat("\n\n\t##########################################################\n")
  cat("\n\n")
  cli::cli_h3(cli::style_bold("Average Conditional Effects"))
  #cat("\t##########################################################\n")

  #cat("\nAverage Conditional Effects:\n")
  stats::printCoefmat(x$ACE_coefmat, signif.stars = getOption("show.signif.stars"), digits = 3, na.print = "")

  #cat("\n\n\t##########################################################\n")
  cat("\n\n")
  cli::cli_h3(cli::style_bold("Standard Deviation of Conditional Effects \n"))
  #cat("\t##########################################################\n")
  #cat("\nAbsolute sum of Conditional Effects:\n")

  res_ASCE = list()

  stats::printCoefmat(x$ASCE_coefmat, signif.stars = getOption("show.signif.stars"), digits = 3, na.print = "")


  out$importance = x$res_imps
  out$ACE = x$res_ACE
  out$AbsCE = x$res_ASCE
  return(invisible(out))
}


#' Returns list of parameters the neural network model currently has in use
#'
#' @param object a model created by \code{\link{dnn}}
#' @param ... nothing implemented yet
#' @return list of weights of neural network
#'
#' @example /inst/examples/coef.citodnn-example.R
#' @export
coef.citodnn <- function(object,...){
  return(object$weights[object$use_model_epoch])
}

#' @rdname coef.citodnn
#' @export
coef.citodnnBootstrap <- function(object, ...) {
  return(lapply(object$models, stats::coef))
}


#' Predict from a fitted dnn model
#'
#' @param object a model created by \code{\link{dnn}}
#' @param newdata new data for predictions
#' @param type type of predictions. The default is on the scale of the linear predictor, "response" is on the scale of the response, and "class" means that class predictions are returned (if it is a classification task)
#' @param device device on which network should be trained on.
#' @param batchsize number of samples that are predicted at the same time
#' @param reduce predictions from bootstrapped model are by default reduced (mean, optional median or none)
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @example /inst/examples/predict.citodnn-example.R
#' @export
predict.citodnn <- function(object,
                            newdata = NULL,
                            type=c("link", "response", "class"),
                            device = NULL,
                            batchsize = NULL, ...) {

  checkmate::assert( checkmate::checkNull(newdata),
                     checkmate::checkMatrix(newdata),
                     checkmate::checkDataFrame(newdata),
                     checkmate::checkScalarNA(newdata))

  object <- check_model(object)

  type <- match.arg(type)

  if(is.null(device)) device <- object$training_properties$device
  device <- check_device(device)

  object$net$to(device = device)
  object$loss$to(device = device)

  if(is.null(batchsize)) batchsize <- object$training_properties$batchsize

  if(type %in% c("response", "class")) {
    link <- object$loss$invlink
  }else{
    link = function(a) a
  }

  Z = NULL

  if(is.null(newdata)){
    sample_names <- rownames(object$data$X)
    X = torch::torch_tensor(object$data$X)
    if(!is.null(object$model_properties$embeddings)) {
      Z = torch::torch_tensor(object$data$Z, dtype = torch::torch_long())
    }
  } else {
    sample_names <- rownames(newdata)
    if(is.data.frame(newdata)) X <- torch::torch_tensor(stats::model.matrix(stats::as.formula(stats::delete.response(object$call$formula)), newdata, xlev = object$data$xlvls)[,-1,drop=FALSE])
    else X <- torch::torch_tensor(stats::model.matrix(stats::as.formula(stats::delete.response(object$call$formula)), data.frame(newdata), xlev = object$data$xlvls)[,-1,drop=FALSE])

    if(!is.null(object$model_properties$embeddings)) {
      tmp = do.call(cbind, lapply(object$Z_formula, function(term) {
        if(!is.factor(newdata[[term]])) stop(paste0(term, " must be factor."))
        #TODO: check if newdata[[term]] has the same factor levels as object$data[[term]]
        newdata[[term]] |> as.integer()
      }) )
      Z = torch::torch_tensor(tmp, dtype = torch::torch_long())
    }
  }

  if(is.null(Z)) {
    dl <- get_data_loader(X, batch_size = batchsize, shuffle = FALSE)
  } else {
    dl <- get_data_loader(X, Z, batch_size = batchsize, shuffle = FALSE)
  }

  pred <- NULL
  coro::loop(for(b in dl) {
    if(is.null(pred)) {
      if(is.null(Z)) pred <- torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE)))$to(device="cpu"))
      else pred <- torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE), b[[2]]$to(device = device, non_blocking = TRUE)))$to(device="cpu"))
    } else {
      if(is.null(Z)) pred <- rbind(pred, torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE)))$to(device="cpu")))
      else pred <- rbind(pred, torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE), b[[2]]$to(device = device, non_blocking = TRUE)))$to(device="cpu")))
    }
  })

  if(!is.null(sample_names)) rownames(pred) <- sample_names

  if(!is.null(object$loss$responses)) {
    colnames(pred) <- object$loss$responses
    if(type == "class") pred <- factor(apply(pred, 1, function(x) object$loss$responses[which.max(x)]), levels = object$loss$responses)
  }

  return(pred)
}




#' @rdname predict.citodnn
#' @export
predict.citodnnBootstrap <- function(object,
                                     newdata = NULL,
                                     type=c("link", "response", "class"),
                                     device = NULL,
                                     batchsize = NULL,
                                     reduce = c("mean", "median", "none"),...) {

  checkmate::assert( checkmate::checkNull(newdata),
                     checkmate::checkMatrix(newdata),
                     checkmate::checkDataFrame(newdata),
                     checkmate::checkScalarNA(newdata))

  predictions = lapply(object$models, function(m) stats::predict(m, newdata = newdata, type = type, device = device, batchsize = batchsize))
  predictions = abind::abind(predictions, along = 0L)
  reduce <- match.arg(reduce)
  if(reduce == "mean") {
    return(apply(predictions, 2:3, mean))
  } else if(reduce == "median") {
    return(apply(predictions, 2:3, stats::median))
  } else {
    return(predictions)
  }
}




#' Creates graph plot which gives an overview of the network architecture.
#'
#' @param x a model created by \code{\link{dnn}}
#' @param node_size size of node in plot
#' @param scale_edges edge weight gets scaled according to other weights (layer specific)
#' @param ... no further functionality implemented yet
#' @return A plot made with 'ggraph' + 'igraph' that represents the neural network
#' @example /inst/examples/plot.citodnn-example.R
#' @export
plot.citodnn<- function(x, node_size = 1, scale_edges = FALSE,...){

  if(x$net$has_embeddings) {
    cat("Models with embeddings layers detected. Embedding layers can be currently not visualized.")
    return(invisible(NULL))
  }

  sapply(c("igraph","ggraph","ggplot2"),function(x)
    if (!requireNamespace(x, quietly = TRUE)) {
      stop(
        paste0("Package \"",x,"\" must be installed to use this function."),
        call. = FALSE
      )
    })
  checkmate::qassert(node_size, "R+[0,)")
  checkmate::qassert(scale_edges, "B1")

  weights <- coef.citodnn(x)
  input <- ncol(weights[[1]][1][[1]])
  structure <- data.frame(expand.grid(from=paste0("1;",c(1:input)),
                                      to = paste0("2;",c(1:(nrow(weights[[1]][1][[1]]))))),
                          value = scale(c(t(weights[[1]][1][[1]][1:input])), center=scale_edges,scale= scale_edges))
  x_pos<- c(rep(1,input))
  y_pos<- c(0,rep(1:input,each=2) *c(1,-1))[1:input]
  num_layer <-  2

  if(length(weights[[1]])>1){
    for (i in 2:length(weights[[1]])){
      if (grepl("weight",names(weights[[1]][i]))){
        structure <- rbind(structure, data.frame(expand.grid(from=paste0(num_layer,";",c(1:(ncol(weights[[1]][i][[1]])))),
                                                             to = paste0(num_layer + 1,";",c(1:(nrow(weights[[1]][i][[1]]))))),
                                                 value= scale(c(t(weights[[1]][i][[1]])), center=scale_edges,scale= scale_edges)))
        x_pos <- c(x_pos, rep(num_layer, x$model_properties$hidden[num_layer-1]))
        y_pos <- c(y_pos, c(0,rep(1:x$model_properties$hidden[num_layer-1],each=2) *c(1,-1))[1:x$model_properties$hidden[num_layer-1]])
        num_layer <- num_layer + 1

      }
    }
  }
  x_pos <- c(x_pos, rep(num_layer,x$model_properties$output))
  y_pos <- c(y_pos, c(0,rep(1:input,each=2) *c(1,-1))[1:x$model_properties$output])


  graph<- igraph::graph_from_data_frame(structure)
  layout <- ggraph::create_layout(graph, layout= "manual", x = x_pos, y = y_pos)

  p<- ggraph::ggraph(layout)+
    ggraph::geom_edge_link( ggplot2::aes(width = abs(structure$value))) +
    ggraph::geom_node_point(size = node_size)
  print(p)
}


#' @rdname plot.citodnn
#' @param which_model which model from the ensemble should be plotted
#' @export
plot.citodnnBootstrap <- function(x, node_size = 1, scale_edges = FALSE,which_model = 1,...){
 plot(x$models[[which_model]], node_size = node_size, scale_edges = scale_edges)
}





