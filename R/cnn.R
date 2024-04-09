#' CNN
#'
#' @description
#'
#' fits a custom convolutional neural network.
#' @param X predictor: array with dimension 3, 4 or 5 for 1D-, 2D- or 3D-convolutions, respectively. The first dimension are the samples, the second dimension the channels and the third - fifth dimension are the input dimensions
#' @param Y response: vector, factor, numerical matrix or logical matrix
#' @param architecture 'citoarchitecture' object created by \code{\link{create_architecture}}
#' @param loss loss after which network should be optimized. Can also be distribution from the stats package or own function, see details
#' @param optimizer which optimizer used for training the network, for more adjustments to optimizer see \code{\link{config_optimizer}}
#' @param lr learning rate given to optimizer
#' @param alpha add L1/L2 regularization to training  \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2} will get added for each layer. Must be between 0 and 1
#' @param lambda strength of regularization: lambda penalty, \eqn{\lambda * (L1 + L2)} (see alpha)
#' @param validation percentage of data set that should be taken as validation set (chosen randomly)
#' @param batchsize number of samples that are used to calculate one learning rate step
#' @param burnin training is aborted if the trainings loss is not below the baseline loss after burnin epochs
#' @param shuffle if TRUE, data in each batch gets reshuffled every epoch
#' @param epochs epochs the training goes on for
#' @param early_stopping if set to integer, training will stop if loss has gotten higher for defined number of epochs in a row, will use validation loss if available.
#' @param lr_scheduler learning rate scheduler created with \code{\link{config_lr_scheduler}}
#' @param custom_parameters List of parameters/variables to be optimized. Can be used in a custom loss function. See Vignette for example.
#' @param device device on which network should be trained on.
#' @param plot plot training loss
#' @param verbose print training and validation loss of epochs
#'
#' @details
#'
#' # Convolutional neural networks:
#' Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for processing structured grid data, such as images.
#' The characterizing parts of the architecture are convolutional layers, pooling layers and fully-connected (linear) layers:
#' \itemize{
#'    \item Convolutional layers are the core building blocks of CNNs. They consist of filters (also called kernels), which are small, learnable matrices. These filters slide over the input data to perform element-wise multiplication, producing feature maps that capture local patterns and features. Multiple filters are used to detect different features in parallel. They help the network learn hierarchical representations of the input data by capturing low-level features (edges, textures) and gradually combining them (in subsequent convolutional layers) to form higher-level features.
#'    \item Pooling layers are used to downsample the spatial dimensions of the feature maps while retaining important information. Max pooling is a common pooling operation, where the maximum value in a local region of the input is retained, reducing the size of the feature maps.
#'    \item Fully-connected (linear) layers connect every neuron in one layer to every neuron in the next layer. These layers are found at the end of the network and are responsible for combining high-level features to make final predictions.
#' }
#'
#' # Loss functions / Likelihoods
#'
#' We support loss functions and likelihoods for different tasks:
#'
#' | Name| Explanation| Example / Task|
#' | :--- | :--- | :--- |
#' | mse | mean squared error |Regression, predicting continuous values|
#' | mae | mean absolute error | Regression, predicting continuous values |
#' | softmax | categorical cross entropy |Multi-class, species classification|
#' | cross-entropy | categorical cross entropy |Multi-class, species classification|
#' | gaussian | Normal likelihood | Regression, residual error is also estimated (similar to `stats::lm()`)	|
#' | binomial | Binomial likelihood | Classification/Logistic regression, mortality|
#' | Poisson | Poisson likelihood |Regression, count data, e.g. species abundances|
#'
#' # Training and convergence of neural networks
#'
#' Ensuring convergence can be tricky when training neural networks. Their training is sensitive to a combination of the learning rate (how much the weights are updated in each optimization step), the batch size (a random subset of the data is used in each optimization step), and the number of epochs (number of optimization steps). Typically, the learning rate should be decreased with the size of the neural networks (amount of learnable parameters). We provide a baseline loss (intercept only model) that can give hints about an appropriate learning rate:
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
#' In the case of 2D and 3D inputs whole feature maps are disabled. Since the torch package doesn't currently support feature map-wise dropout for 1D inputs, instead random neurons in the feature maps are disabled similar to dropout in linear layers.
#'
#' By utilizing these regularization methods in your neural network training with the cito package, you can improve generalization performance and enhance the network's ability to handle unseen data. These techniques act as valuable tools in mitigating overfitting and promoting more robust and reliable model performance.
#'
#' # Custom Optimizer and Learning Rate Schedulers
#'
#' When training a network, you have the flexibility to customize the optimizer settings and learning rate scheduler to optimize the learning process. In the cito package, you can initialize these configurations using the \code{\link{config_lr_scheduler}} and \code{\link{config_optimizer}} functions.
#'
#' \code{\link{config_lr_scheduler}} allows you to define a specific learning rate scheduler that controls how the learning rate changes over time during training. This is beneficial in scenarios where you want to adaptively adjust the learning rate to improve convergence or avoid getting stuck in local optima.
#'
#' Similarly, the \code{\link{config_optimizer}} function enables you to specify the optimizer for your network. Different optimizers, such as stochastic gradient descent (SGD), Adam, or RMSprop, offer various strategies for updating the network's weights and biases during training. Choosing the right optimizer can significantly impact the training process and the final performance of your neural network.
#'
#' # Training on graphic cards
#'
#' If you have an NVIDIA CUDA-enabled device and have installed the CUDA toolkit version 11.3 and cuDNN 8.4, you can take advantage of GPU acceleration for training your neural networks. It is crucial to have these specific versions installed, as other versions may not be compatible.
#' For detailed installation instructions and more information on utilizing GPUs for training, please refer to the [mlverse: 'torch' documentation](https://torch.mlverse.org/docs/articles/installation.html).
#'
#' Note: GPU training is optional, and the package can still be used for training on CPU even without CUDA and cuDNN installations.
#'
#' @return an S3 object of class \code{"citocnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call}
#' \item{loss}{A list which contains relevant information for the target variable and the used loss function}
#' \item{data}{Contains data used for training the model}
#' \item{weights}{List of weights for each training epoch}
#' \item{use_model_epoch}{Integer, which defines which model from which training epoch should be used for prediction.}
#' \item{loaded_model_epoch}{Integer, shows which model from which epoch is loaded currently into model$net.}
#' \item{model_properties}{A list of properties of the neural network, contains number of input nodes, number of output nodes, size of hidden layers, activation functions, whether bias is included and if dropout layers are included.}
#' \item{training_properties}{A list of all training parameters that were used the last time the model was trained. It consists of learning rate, information about an learning rate scheduler, information about the optimizer, number of epochs, whether early stopping was used, if plot was active, lambda and alpha for L1/L2 regularization, batchsize, shuffle, was the data set split into validation and training, which formula was used for training and at which epoch did the training stop.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' @import checkmate
#' @author Armin Schenk, Maximilian Pichler
#' @seealso \code{\link{predict.citocnn}}, \code{\link{plot.citocnn}},  \code{\link{coef.citocnn}}, \code{\link{print.citocnn}}, \code{\link{summary.citocnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#'
cnn <- function(X,
                Y = NULL,
                architecture,
                loss = c("mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson"),
                optimizer = c("sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop"),
                lr = 0.01,
                alpha = 0.5,
                lambda = 0.0,
                validation = 0.0,
                batchsize = 32L,
                burnin = 10,
                shuffle = TRUE,
                epochs = 100,
                early_stopping = NULL,
                lr_scheduler = NULL,
                custom_parameters = NULL,
                device = c("cpu", "cuda", "mps"),
                plot = TRUE,
                verbose = TRUE) {

  #Data
  checkmate::assert(checkmate::checkArray(X, min.d = 3, max.d = 5))
  checkmate::assert(checkmate::checkFactor(Y), checkmate::checkNumeric(Y),
                    checkmate::checkMatrix(Y, mode = "numeric"), checkmate::checkMatrix(Y, mode = "logical"),
                    checkmate::checkNull(Y))

  if(!inherits(architecture, "citoarchitecture")) stop("architecture is not an object of class 'citoarchitecture'. See ?create_architecture for more info.")

  #Training
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(alpha, "R1[0,1]")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(batchsize, "X1[1,)")
  checkmate::qassert(shuffle, "B1")
  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(early_stopping, c("0","X1[1,)"))
  checkmate::qassert(custom_parameters, c("0", "L+"))
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")
  checkmate::qassert(device, "S+[3,)")


  # No training if no Y specified (E.g. used in mmn())
  if(is.null(Y)) {

    input_shape <- dim(X)[-1]

    architecture <- adjust_architecture(architecture = architecture, input_dim = length(input_shape)-1)

    net <- build_cnn(input_shape = input_shape,
                     output_shape = NULL,
                     architecture = architecture)

    model_properties <- list(input = input_shape,
                             output = NULL,
                             architecture = architecture)

    out <- list()
    class(out) <- "citocnn"
    out$net <- net
    out$call <- match.call()
    out$data <- list(X = X, Y = NULL)
    out$model_properties <- model_properties
    return(out)
  }

  device <- match.arg(device)
  device_old <- device
  device <- check_device(device)

  if(!is.function(loss) & !inherits(loss,"family")) {
    loss <- match.arg(loss)
  }

  loss_obj <- get_loss(loss)
  if(!is.null(loss_obj$parameter)) loss_obj$parameter <- list(scale = loss_obj$parameter)
  if(!is.null(custom_parameters)){
    if(!inherits(custom_parameters,"list")){
      warning("custom_parameters has to be list")
    } else {
      custom_parameters <- lapply(custom_parameters, function(x) torch::torch_tensor(x, requires_grad = TRUE, device = device))
      loss_obj$parameter <- append(loss_obj$parameter, unlist(custom_parameters))
    }
  }

  old_X <- X
  old_Y <- Y
  targets <- format_targets(Y, loss_obj)
  Y <- targets$Y
  Y_base <- targets$Y_base
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls

  X <- torch::torch_tensor(X, dtype = torch::torch_float32())

  loss.fkt <- loss_obj$loss
  if(!is.null(loss_obj$parameter)) list2env(loss_obj$parameter,envir = environment(fun= loss.fkt))
  base_loss = as.numeric(loss.fkt(loss_obj$link(Y_base), Y)$mean())

  if(validation != 0) {
    n_samples <- dim(X)[1]
    valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
    train <- c(1:n_samples)[-valid]
    train_dl <- get_data_loader(X[train,], Y[train,], batch_size = batchsize, shuffle = shuffle)
    valid_dl <- get_data_loader(X[valid,], Y[valid,], batch_size = batchsize, shuffle = shuffle)
  } else {
    train_dl <- get_data_loader(X, Y, batch_size = batchsize, shuffle = shuffle)
    valid_dl <- NULL
  }

  input_shape <- dim(X)[-1]

  architecture <- adjust_architecture(architecture = architecture, input_dim = length(input_shape)-1)

  net <- build_cnn(input_shape = input_shape,
                   output_shape = y_dim,
                   architecture = architecture)

  model_properties <- list(input = input_shape,
                           output = y_dim,
                           architecture = architecture)

  training_properties <- list(lr = lr,
                              lr_scheduler = lr_scheduler,
                              optimizer = optimizer,
                              epochs = epochs,
                              early_stopping = early_stopping,
                              plot = plot,
                              validation = validation,
                              lambda = lambda,
                              alpha = alpha,
                              batchsize = batchsize,
                              shuffle = shuffle)

  out <- list()
  class(out) <- "citocnn"
  out$net <- net
  out$call <- match.call()
  out$loss <- loss_obj
  out$data <- list(X = old_X, Y = old_Y)
  if(!is.null(ylvls)) out$data$ylvls <- ylvls
  if(validation != 0) out$data <- append(out$data, list(validation = valid))
  out$base_loss <- base_loss
  out$weights <- list()
  out$use_model_epoch <- 1
  out$loaded_model_epoch <- 0
  out$model_properties <- model_properties
  out$training_properties <- training_properties
  out$device <- device_old
  out$burnin = burnin



  ### training loop ###
  out <- train_model(model = out,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)


  return(out)
}

#' Predict from a fitted cnn model
#'
#' @param object a model created by \code{\link{cnn}}
#' @param newdata new data for predictions
#' @param type which value should be calculated, either raw response, output of link function or predicted class (in case of classification)
#' @param device device on which network should be trained on.
#' @param batchsize number of samples that are predicted at the same time
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @export
predict.citocnn <- function(object,
                            newdata = NULL,
                            type=c("link", "response", "class"),
                            device = c("cpu","cuda", "mps"),
                            batchsize, ...) {

  checkmate::assert(checkmate::checkNull(newdata),
                    checkmate::checkArray(newdata, min.d = 3, max.d = 5))

  object <- check_model(object)

  type <- match.arg(type)

  device <- match.arg(device)

  if(type %in% c("link", "class")) {
    link <- object$loss$invlink
  }else{
    link = function(a) a
  }

  device <- check_device(device)

  object$net$to(device = device)

  if(is.null(newdata)){
    newdata = torch::torch_tensor(object$data$X, dtype = torch::torch_float32())
  } else if(all(dim(newdata)[-1] == dim(object$data$X)[-1])) {
    newdata <- torch::torch_tensor(newdata, dtype = torch::torch_float32())
  } else {
    stop(paste0("Wrong dimension of newdata: [", paste(dim(newdata), collapse = ", "), "]   Correct input dimension: [", paste(c("N", dim(object$data$X)[-1]), collapse = ", "), "]"))
  }

  dl <- get_data_loader(newdata, batch_size = batchsize, shuffle = FALSE)

  pred <- NULL
  coro::loop(for(b in dl) {
    if(is.null(pred)) pred <- torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE)))$to(device="cpu"))
    else pred <- rbind(pred, torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE)))$to(device="cpu")))
  })

  if(!is.null(dimnames(newdata))) rownames(pred) <- dimnames(newdata)[[1]]

  if(!is.null(object$data$ylvls)) {
    colnames(pred) <- object$data$ylvls
    if(type == "class") pred <- factor(apply(pred,1, function(x) object$data$ylvls[which.max(x)]), levels = object$data$ylvls)
  }

  return(pred)
}

#' Print class citocnn
#'
#' @param x a model created by \code{\link{cnn}}
#' @param ... additional arguments
#' @return original object x
#'
#' @export
print.citocnn <- function(x, ...){
  x <- check_model(x)
  print(x$call)
  print(x$model_properties$architecture, x$model_properties$input, x$model_properties$output)
  return(invisible(x))
}

#' Summary citocnn
#' @description
#'
#' currently the same as the print.citocnn method.
#'
#' @param object a model created by \code{\link{cnn}}
#' @param ... additional arguments
#' @return original object x
#'
#' @export
summary.citocnn <- function(object, ...){
  return(print(object))
}

#' Plot the CNN architecture
#'
#' @param x a model created by \code{\link{cnn}}
#' @param ... additional arguments
#' @return original object x
#'
#' @export
plot.citocnn <- function(x, ...){
  x <- check_model(x)
  plot(x$model_properties$architecture, x$model_properties$input, x$model_properties$output)
  return(invisible(x))
}

#' Returns list of parameters the neural network model currently has in use
#'
#' @param object a model created by \code{\link{cnn}}
#' @param ... nothing implemented yet
#' @return list of weights of neural network
#'
#' @export
coef.citocnn <- function(object,...){
  return(object$weights[object$use_model_epoch])
}




#' CNN architecture
#'
#' @description
#'
#' creates a 'citoarchitecture' object that is used by \code{\link{cnn}}.
#'
#' @param ... objects of class 'citolayer' created by \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}} or \code{\link{transfer}}
#' @param default_n_neurons (int) default value: amount of neurons in a linear layer
#' @param default_n_kernels (int) default value: amount of kernels in a convolutional layer
#' @param default_kernel_size (int or tuple) default value: size of the kernels in convolutional and pooling layers. Use a tuple if the kernel size isn't equal in all dimensions
#' @param default_stride (int or tuple) default value: stride of the kernels in convolutional and pooling layers. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param default_padding (int or tuple) default value: zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#' @param default_dilation (int or tuple) default value: dilation of the kernels in convolutional and maxPooling layers. Use a tuple if the dilation isn't equal in all dimensions
#' @param default_bias (boolean) default value: if TRUE, adds a learnable bias to neurons of linear and kernels of convolutional layers
#' @param default_activation (string) default value: activation function that is used after linear and convolutional layers. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#' @param default_normalization (boolean) default value: if TRUE, batch normalization is used after linear and convolutional layers
#' @param default_dropout (float) default value: dropout rate of linear and convolutional layers. Set to 0 for no dropout
#'
#' @details
#' This function creates a 'citoarchitecture' object that provides the \code{\link{cnn}} function with all information about the architecture of the CNN that will be created and trained.
#' The final architecture consists of the layers in the sequence they were passed to this function.
#' All parameters of the 'citolayer' objects, that are still NULL because they haven't been specified at the creation of the layer, are filled with the given default parameters for their specific layer type (linear, conv, maxPool, avgPool).
#' The default values can be changed by either passing a list with the values for specific layer types (in which case the defaults of layer types which aren't in the list remain the same)
#' or by passing a single value (in which case the defaults for all layer types is set to that value).

#' @return S3 object of class \code{"citoarchitecture"}
#' @import checkmate
#' @author Armin Schenk
#' @seealso \code{\link{cnn}}, \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}}, \code{\link{transfer}}, \code{\link{print.citoarchitecture}}, \code{\link{plot.citoarchitecture}}

create_architecture <- function(...,
                                default_n_neurons = 10,
                                default_n_kernels = 10,
                                default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
                                default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
                                default_padding = list(conv = 0, maxPool = 0, avgPool = 0),
                                default_dilation = list(conv = 1, maxPool = 1),
                                default_bias = list(conv = TRUE, linear = TRUE),
                                default_activation = list(conv = "relu", linear = "relu"),
                                default_normalization = list(conv = FALSE, linear = FALSE),
                                default_dropout = list(conv = 0.0, linear = 0.0)) {

  checkmate::qassert(default_n_neurons, "X1")
  checkmate::qassert(default_n_kernels, "X1")
  check_listable_parameter(default_kernel_size, "X<=3[1,)")
  check_listable_parameter(default_stride, c("X<=3[1,)","0"))
  check_listable_parameter(default_padding, "X<=3[0,)")
  check_listable_parameter(default_dilation, "X<=3[1,)")
  check_listable_parameter(default_bias, "B1")
  check_listable_parameter(default_activation, "S1[1,)")
  check_listable_parameter(default_normalization, "B1")
  check_listable_parameter(default_dropout, "R1[0,1)")

  default_n_neurons <- list(linear = default_n_neurons)
  default_n_kernels <- list(conv = default_n_kernels)
  default_kernel_size <- fill_defaults(default_kernel_size, list(conv = 3, maxPool = 2, avgPool = 2))
  default_stride <- fill_defaults(default_stride, list(conv = 1, maxPool = NULL, avgPool = NULL))
  default_padding <- fill_defaults(default_padding, list(conv = 0, maxPool = 0, avgPool = 0))
  default_dilation <- fill_defaults(default_dilation, list(conv = 1, maxPool = 1))
  default_bias <- fill_defaults(default_bias, list(conv = TRUE, linear = TRUE))
  default_activation <- fill_defaults(default_activation, list(conv = "relu", linear = "relu"))
  default_normalization <- fill_defaults(default_normalization, list(conv = FALSE, linear = FALSE))
  default_dropout <- fill_defaults(default_dropout, list(conv = 0.0, linear = 0.0))

  fill_layer <- function(layer) {
    type <- class(layer)[1]
    for(parameter in names(layer)) {
      if(is.null(layer[[parameter]])) {
        eval(parse(text=paste0("if(!is.null(default_", parameter,"[[type]])) layer[[parameter]] <- default_", parameter, "[[type]]")))
      }
    }
    return(layer)
  }

  architecture <- list()

  transfer <- FALSE
  flattened <- FALSE
  for(layer in list(...)) {
    if(!inherits(layer, "citolayer")) stop("Objects must be of class citolayer")
    if(inherits(layer, "transfer")) {
      if(length(architecture) != 0) stop("There mustn't be any layers before a transfer layer")
      layer$replace_classifier <- length(list(...)) > 1
      architecture <- append(architecture, list(layer))
      transfer <- TRUE
    } else {
      if(transfer && inherits(layer, c("conv", "maxPool", "avgPool"))) stop("Only linear layers are allowed after a transfer layer")
      if(inherits(layer, "linear")) {
        flattened <- TRUE
      } else {
        if(flattened) stop("Only linear layers are allowed after a linear layer")
      }
      architecture <- append(architecture, list(fill_layer(layer)))
    }
  }

  class(architecture) <- "citoarchitecture"

  return(architecture)
}

fill_defaults <- function(passed_defaults, default_defaults) {
  if(inherits(passed_defaults, "list")) {
    for(entry in names(default_defaults)) {
      if(!entry %in% names(passed_defaults)) {
        passed_defaults[[entry]] <- default_defaults[[entry]]
      }
    }
    return(passed_defaults)

  } else {
    for(entry in names(default_defaults)) {
      default_defaults[[entry]] <- passed_defaults
    }
    return(default_defaults)
  }
}


#' Linear layer
#'
#' @description
#'
#' creates a 'linear' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param n_neurons (int) amount of hidden neurons in this layer
#' @param bias (boolean) if TRUE, adds a learnable bias to the neurons of this layer
#' @param activation (string) activation function that is used after this layer. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#' @param normalization (boolean) if TRUE, batch normalization is used after this layer
#' @param dropout (float) dropout rate of this layer. Set to 0 for no dropout
#'
#' @details
#' This function creates a 'linear' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"linear" "citolayer"}
#' @import checkmate
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
linear <- function(n_neurons = NULL,
                   bias = NULL,
                   activation = NULL,
                   normalization = NULL,
                   dropout = NULL) {

  layer <- list(n_neurons = n_neurons,
                bias = bias,
                activation = activation,
                normalization = normalization,
                dropout = dropout)
  class(layer) <- c("linear", "citolayer")
  return(layer)
}

#' Convolutional layer
#'
#' @description
#'
#' creates a 'conv' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param n_kernels (int) amount of kernels in this layer
#' @param kernel_size (int or tuple) size of the kernels in this layer. Use a tuple if the kernel size isn't equal in all dimensions
#' @param stride (int or tuple) stride of the kernels in this layer. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param padding (int or tuple) zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#' @param dilation (int or tuple) dilation of the kernels in this layer. Use a tuple if the dilation isn't equal in all dimensions
#' @param bias (boolean) if TRUE, adds a learnable bias to the kernels of this layer
#' @param activation (string) activation function that is used after this layer. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#' @param normalization (boolean) if TRUE, batch normalization is used after this layer
#' @param dropout (float) dropout rate of this layer. Set to 0 for no dropout
#'
#' @details
#' This function creates a 'conv' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"conv" "citolayer"}
#' @import checkmate
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
conv <- function(n_kernels = NULL,
                 kernel_size = NULL,
                 stride = NULL,
                 padding = NULL,
                 dilation = NULL,
                 bias = NULL,
                 activation = NULL,
                 normalization = NULL,
                 dropout = NULL) {

  layer <- list(n_kernels = n_kernels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = bias,
                activation = activation,
                normalization = normalization,
                dropout = dropout)
  class(layer) <- c("conv", "citolayer")
  return(layer)
}

#' Average pooling layer
#'
#' @description
#'
#' creates a 'avgPool' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param kernel_size (int or tuple) size of the kernel in this layer. Use a tuple if the kernel size isn't equal in all dimensions
#' @param stride (int or tuple) stride of the kernel in this layer. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param padding (int or tuple) zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#'
#' @details
#' This function creates a 'avgPool' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"avgPool" "citolayer"}
#' @import checkmate
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
avgPool <- function(kernel_size = NULL,
                    stride = NULL,
                    padding = NULL) {

  layer <- list(kernel_size = kernel_size,
                stride = stride,
                padding = padding)
  class(layer) <- c("avgPool", "citolayer")
  return(layer)
}

#' Maximum pooling layer
#'
#' @description
#'
#' creates a 'maxPool' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param kernel_size (int or tuple) size of the kernel in this layer. Use a tuple if the kernel size isn't equal in all dimensions
#' @param stride (int or tuple) stride of the kernel in this layer. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param padding (int or tuple) zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#' @param dilation (int or tuple) dilation of the kernel in this layer. Use a tuple if the dilation isn't equal in all dimensions
#'
#' @details
#' This function creates a 'maxPool' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"maxPool" "citolayer"}
#' @import checkmate
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
maxPool <- function(kernel_size = NULL,
                    stride = NULL,
                    padding = NULL,
                    dilation = NULL) {

  layer <- list(kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation)
  class(layer) <- c("maxPool", "citolayer")
  return(layer)
}

#' Transfer learning
#'
#' @description
#'
#' creates a 'transfer' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param name The name of the pretrained model
#' @param pretrained if FALSE, random weights are used instead of the pretrained weights
#' @param freeze if TRUE, the weights of the pretrained model (except the "classifier" part at the end) aren't changed in the training anymore. Only works if pretrained=TRUE
#'
#' @details
#' This function creates a 'transfer' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' With this object the pretrained models that are available in the 'torchvision' package can be used in cito.
#' When 'freeze' is set to TRUE, only the weights of the last part of the network (consisting of one or more linear layers) are adjusted in the training.
#' There mustn't be any other citolayers before the transfer citolayer object when calling \code{\link{create_architecture}}.
#' If there are any citolayers after the transfer citolayer, the linear classifier part of the pretrained model is replaced with the specified citolayers.

#' @return S3 object of class \code{"transfer" "citolayer"}
#' @import checkmate
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
transfer <- function(name = c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"),
                     pretrained = TRUE,
                     freeze = TRUE) {

  if(identical(name, c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"))) {
    name <- "alexnet"
  }

  name <- match.arg(name)

  layer <- list(name = name,
                pretrained = pretrained,
                freeze = pretrained & freeze,
                replace_classifier = FALSE)
  class(layer) <- c("transfer", "citolayer")
  return(layer)
}

#' Print class citoarchitecture
#'
#' @param x an object created by \code{\link{create_architecture}}
#' @param input_shape a vector with the dimensions of a single sample (e.g. c(3,28,28))
#' @param output_shape the number of nodes in the output layer
#' @param ... additional arguments
#' @return original object
print.citoarchitecture <- function(x, input_shape, output_shape, ...) {
  x <- adjust_architecture(x, length(input_shape)-1)
  need_output_layer <- TRUE

  for(layer in x) {
    if(inherits(layer, "transfer")) {
      if(!(length(input_shape) == 3 && input_shape[1] == 3)) stop("The pretrained models only work on 2 dimensional data with 3 channels: [3, x, y]")
      need_output_layer <- layer$replace_classifier
    }
    input_shape <- print(layer, input_shape, output_shape)
  }

  if(need_output_layer) {
    output_layer <- linear(n_neurons=output_shape, bias = TRUE,
                           activation="Depends on loss", normalization=FALSE, dropout=0)
    print(output_layer, input_shape)
  }
  cat("-------------------------------------------------------------------------------\n")
}

#' Print linear layer
#'
#' @param x an object of class linear
#' @param input_shape input shape
#' @param ... further arguments, not supported yet
print.linear <- function(x, input_shape, ...) {

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Linear     |Input: ", prod(input_shape), "\n"))
  cat(paste0("           |Output: ", x[["n_neurons"]], "\n"))
  cat(paste0("           |Bias: ", x[["bias"]], "\n"))
  if(x[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  cat(paste0("           |Activation: ", x[["activation"]], "\n"))
  if(x[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", x[["dropout"]], "\n"))
  }

  return(invisible(x[["n_neurons"]]))
}

#' Print conv layer
#'
#' @param x an object of class conv
#' @param input_shape input shape
#' @param ... further arguments, not supported yet
print.conv <- function(x, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = x[["n_kernels"]],
                                   kernel_size = x[["kernel_size"]],
                                   stride = x[["stride"]],
                                   padding = x[["padding"]],
                                   dilation = x[["dilation"]])

  kernel_size <- paste(x[["kernel_size"]], collapse = "x")
  stride <- paste(x[["stride"]], collapse = "x")
  padding <- paste(x[["padding"]], collapse = "x")
  dilation <- paste(x[["dilation"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Convolution|Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")\n"))
  cat(paste0("           |Bias: ", x[["bias"]], "\n"))
  if(x[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  cat(paste0("           |Activation: ", x[["activation"]], "\n"))
  if(x[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", x[["dropout"]], "\n"))
  }

  return(invisible(output_shape))
}


#' Print pooling layer
#'
#' @param x an object of class avgPool
#' @param input_shape input shape
#' @param ... further arguments, not supported yet
print.avgPool <- function(x, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = x[["kernel_size"]],
                                   stride = x[["stride"]],
                                   padding = x[["padding"]],
                                   dilation = rep(1,length(input_shape)-1))

  kernel_size <- paste(x[["kernel_size"]], collapse = "x")
  stride <- paste(x[["stride"]], collapse = "x")
  padding <- paste(x[["padding"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("AvgPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ")\n"))

  return(invisible(output_shape))
}

#' Print pooling layer
#'
#' @param x an object of class maxPool
#' @param input_shape input shape
#' @param ... further arguments, not supported yet
print.maxPool <- function(x, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = x[["kernel_size"]],
                                   stride = x[["stride"]],
                                   padding = x[["padding"]],
                                   dilation = x[["dilation"]])

  kernel_size <- paste(x[["kernel_size"]], collapse = "x")
  stride <- paste(x[["stride"]], collapse = "x")
  padding <- paste(x[["padding"]], collapse = "x")
  dilation <- paste(x[["dilation"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("MaxPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")\n"))

  return(invisible(output_shape))
}


#' Print transfer model
#'
#' @param x an object of class transfer
#' @param input_shape input shape
#' @param output_shape output shape
#' @param ... further arguments, not supported yet
print.transfer <- function(x, input_shape, output_shape, ...) {

  if(x$replace_classifier) {
    output_shape <- get_transfer_output_shape(x$name)
  }

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Transfer   |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Network: ", x[["name"]] , "\n"))
  cat(paste0("           |Pretrained: ", x[["pretrained"]] , "\n"))
  if(x[["pretrained"]]) cat(paste0("           |Weights frozen: ", x[["freeze"]] , "\n"))

  return(invisible(output_shape))
}


#' Plot the CNN architecture
#'
#' @param x an object of class citoarchitecture created by \code{\link{create_architecture}}
#' @param input_shape a vector with the dimensions of a single sample (e.g. c(3,28,28))
#' @param output_shape the number of nodes in the output layer
#' @param ... additional arguments
#' @return nothing
plot.citoarchitecture <- function(x, input_shape, output_shape, ...) {
  x <- adjust_architecture(x, length(input_shape)-1)

  transfer_only <- length(x) == 1 && inherits(x[[1]], "transfer")

  text <- c(paste0("Input size: ", paste(input_shape, collapse = "x")), "")
  type <- c("data", "arrow")

  for(layer in x) {
    if(inherits(layer, "transfer")) {
      if(!(length(input_shape) == 3 && input_shape[1] == 3)) stop("The pretrained models only work on 2 dimensional data with 3 channels: [3, x, y]")
      tmp <- paste0("Transfer network: ", layer[["name"]], " (pretrained weights: ", layer[["pretrained"]])
      if(layer[["pretrained"]]) {
        tmp <- paste0(tmp, ", frozen weights: ", layer[["freeze"]], ")")
      } else {
        tmp <- paste0(tmp, ")")
      }
      text <- c(text, tmp)
      type <- c(type, "transfer")
      if(!transfer_only) {
        input_shape <- get_transfer_output_shape(layer[["name"]])
        text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
        type <- c(type, "arrow")
      } else {
        text <- c(text, "")
        type <- c(type, "arrow")
      }
    } else if(inherits(layer, "linear")) {
      text <- c(text, paste0("Linear layer with ", layer[["n_neurons"]], " neurons"))
      type <- c(type, "linear")
      if(layer[["normalization"]]) {
        text <- c(text, "Batch normalization")
        type <- c(type, "arrow")
      }
      text <- c(text, paste0("Activation: ", layer[["activation"]]))
      type <- c(type, "arrow")
      if(layer[["dropout"]] > 0) {
        text <- c(text, paste0("Dropout rate: ", layer[["dropout"]]))
        type <- c(type, "arrow")
      }
    } else if(inherits(layer, "conv")) {
      kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
      stride <- paste(layer[["stride"]], collapse = "x")
      padding <- paste(layer[["padding"]], collapse = "x")
      dilation <- paste(layer[["dilation"]], collapse = "x")
      text <- c(text, paste0("Convolutional layer with ", layer[["n_kernels"]], " kernels (size=", kernel_size, ", stride=", stride, ", padding=", padding, ", dilation=", dilation, ")"))
      type <- c(type, "conv")
      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = layer[["n_kernels"]],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")
      if(layer[["normalization"]]) {
        text <- c(text, "Batch normalization")
        type <- c(type, "arrow")
      }
      text <- c(text, paste0("Activation: ", layer[["activation"]]))
      type <- c(type, "arrow")
      if(layer[["dropout"]] > 0) {
        text <- c(text, paste0("Dropout rate: ", layer[["dropout"]]))
        type <- c(type, "arrow")
      }
    } else if(inherits(layer, "maxPool")) {
      kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
      stride <- paste(layer[["stride"]], collapse = "x")
      padding <- paste(layer[["padding"]], collapse = "x")
      dilation <- paste(layer[["dilation"]], collapse = "x")
      text <- c(text, paste0("Maximum pooling layer (kernel size=", kernel_size, ", stride=", stride, ", padding=", padding, ", dilation=", dilation, ")"))
      type <- c(type, "pool")
      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")
    } else if(inherits(layer, "avgPool")) {
      kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
      stride <- paste(layer[["stride"]], collapse = "x")
      padding <- paste(layer[["padding"]], collapse = "x")
      text <- c(text, paste0("Average pooling layer (kernel size=", kernel_size, ", stride=", stride, ", padding=", padding, ")"))
      type <- c(type, "pool")
      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = rep(1,length(input_shape)-1))
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")
    }
  }

  if(!transfer_only) {
    text <- c(text, paste0("Linear layer with ", output_shape, " neurons"), "")
    type <- c(type, "linear", "arrow")
  }

  text <- c(text, paste0("Output size: ", paste(output_shape, collapse = "x")))
  type <- c(type, "data")


  n <- length(text)
  height <- 1/n
  ybottom <- 1-height
  ytop <- 1
  graphics::plot.new()
  for(i in 1:n) {
    if(type[i] == "arrow") {
      graphics::arrows(0.5, ytop, 0.5, ybottom, length = 0.5*graphics::par("pin")[2]/(2*n-1))
      graphics::text(0.5, ytop-height/2, text[i], pos = 4)
    } else {
      color <- switch(type[i],
                      "transfer" = "lightyellow",
                      "conv" = "lightblue",
                      "linear" = "lightgreen",
                      "pool" = "pink",
                      "data" = "orange")
      graphics::rect(0, ybottom, 1, ytop, col = color)
      graphics::text(0.5, ytop-height/2, text[i])
    }
    ybottom <- ybottom-height
    ytop <- ytop-height
  }
}

