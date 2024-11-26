#' Train a Convolutional Neural Network (CNN)
#'
#' This function trains a Convolutional Neural Network (CNN) on the provided input data `X` and the target data `Y` using the specified architecture, loss function, and optimizer.
#'
#' @param X An array of input data with a minimum of 3 and a maximum of 5 dimensions. The first dimension represents the samples, the second dimension represents the channels, and the third to fifth dimensions represent the input dimensions.
#' @param Y The target data. It can be a factor, numeric vector, or a numeric or logical matrix.
#' @param architecture An object of class 'citoarchitecture'. See \code{\link{create_architecture}} for more information.
#' @param loss The loss function to be used. Options include "mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson", "nbinom", "mvp", "multinomial", and "clogit". You can also specify your own loss function. See Details for more information. Default is "mse".
#' @param optimizer The optimizer to be used. Options include "sgd", "adam", "adadelta", "adagrad", "rmsprop", and "rprop". See \code{\link{config_optimizer}} for further adjustments to the optimizer. Default is "sgd".
#' @param lr Learning rate for the optimizer. Default is 0.01.
#' @param alpha Alpha value for L1/L2 regularization. Default is 0.5.
#' @param lambda Lambda value for L1/L2 regularization. Default is 0.0.
#' @param validation Proportion of the data to be used for validation. Default is 0.0.
#' @param batchsize Batch size for training. Default is 32.
#' @param burnin Number of epochs after which the training stops if the loss is still above the base loss. Default is Inf.
#' @param shuffle Whether to shuffle the data before each epoch. Default is TRUE.
#' @param epochs Number of epochs to train the model. Default is 100.
#' @param early_stopping Number of epochs with no improvement after which training will be stopped. Default is NULL.
#' @param lr_scheduler Learning rate scheduler. See \code{\link{config_lr_scheduler}} for creating a learning rate scheduler. Default is NULL.
#' @param custom_parameters Parameters for the custom loss function. See the vignette for an example. Default is NULL.
#' @param device Device to be used for training. Options are "cpu", "cuda", and "mps". Default is "cpu".
#' @param plot Whether to plot the training progress. Default is TRUE.
#' @param verbose Whether to print detailed training progress. Default is TRUE.
#'
#'
#' @details
#'
#' # Convolutional Neural Networks:
#' Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for processing structured data, such as images.
#' The key components of a CNN are convolutional layers, pooling layers and fully-connected (linear) layers:
#' \itemize{
#'    \item **Convolutional layers** are the core building blocks of CNNs. They consist of filters (also called kernels), which are small, learnable matrices. These filters slide over the input data to perform element-wise multiplication, producing feature maps that capture local patterns and features. Multiple filters are used to detect different features in parallel. They help the network learn hierarchical representations of the input data by capturing low-level features (edges, textures) and gradually combining them (in subsequent convolutional layers) to form higher-level features.
#'    \item **Pooling layers** reduce the size of the feature maps created by convolutional layers, while retaining important information. A common type is max pooling, which keeps the highest value in a region, simplifying the data while preserving essential features.
#'    \item **Fully-connected (linear) layers** connect every neuron in one layer to every neuron in the next layer. These layers are found at the end of the network and are responsible for combining high-level features to make final predictions.
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
#' | poisson | Poisson likelihood |Regression, count data, e.g. species abundances|
#' | nbinom | Negative binomial likelihood | Regression, count data with dispersion parameter |
#' | mvp | multivariate probit model | joint species distribution model, multi species (presence absence) |
#' | multinomial | Multinomial likelihood | step selection in animal movement models |
#' | clogit | conditional binomial | step selection in animal movement models |
#'
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
#'
#' @return An S3 object of class \code{"citocnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_sequential" "nn_module", originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call.}
#' \item{loss}{A list which contains relevant information for the target variable and the used loss function.}
#' \item{data}{Contains the data used for the training of the model.}
#' \item{base_loss}{The loss of the intercept-only model.}
#' \item{weights}{List of parameters (weights and biases) of the models from the best and the last training epoch.}
#' \item{buffers}{List of buffers (e.g. running mean and variance of batch normalization layers) of the models from the best and the last training epoch.}
#' \item{use_model_epoch}{Integer, defines whether the model from the best (= 1) or the last (= 2) training epoch should be used for prediction.}
#' \item{loaded_model_epoch}{Integer, shows whether the parameters and buffers of the model from the best (= 1) or the last (= 2) training epoch are currently loaded in \code{net}.}
#' \item{model_properties}{A list of properties, that define the architecture of the model.}
#' \item{training_properties}{A list of all the training parameters used the last time the model was trained.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch.}
#' @import checkmate
#' @example /inst/examples/cnn-example.R
#' @author Armin Schenk, Maximilian Pichler
#' @seealso \code{\link{predict.citocnn}}, \code{\link{print.citocnn}}, \code{\link{plot.citocnn}}, \code{\link{summary.citocnn}}, \code{\link{coef.citocnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
cnn <- function(X,
                Y = NULL,
                architecture,
                loss = c("mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson", "mvp", "nbinom", "multinomial", "clogit"),
                optimizer = c("sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop"),
                lr = 0.01,
                alpha = 0.5,
                lambda = 0.0,
                validation = 0.0,
                batchsize = 32L,
                burnin = Inf,
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


  # Only return the model properties if no Y specified (Used in mmn())
  if(is.null(Y)) {

    input_shape <- dim(X)[-1]

    architecture <- adjust_architecture(architecture = architecture, input_dim = length(input_shape)-1)

    model_properties <- list(input = input_shape,
                             architecture = architecture)
    class(model_properties) <- "citocnn_properties"
    return(model_properties)
  }

  device <- match.arg(device)

  if(!is.function(loss) & !inherits(loss,"family")) {
    loss <- match.arg(loss)

    if((device == "mps") & (loss %in% c("poisson", "nbinom", "multinomial"))) {
      message("`poisson`, `nbinom`, and `multinomial` are not yet supported for `device=mps`, switching to `device=cpu`")
      device = "cpu"
    }
  }

  if(inherits(loss,"family")) {
    if((device == "mps") & (loss$family %in% c("poisson", "nbinom"))) {
      message("`poisson` or `nbinom` are not yet supported for `device=mps`, switching to `device=cpu`")
      device = "cpu"
    }
  }

  device_old <- device
  device <- check_device(device)

  loss_obj <- get_loss(loss, device = device, X = X, Y = Y)
  if(!is.null(loss_obj$parameter)) loss_obj$parameter <- list(parameter = loss_obj$parameter)
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
  base_loss = as.numeric(loss.fkt(torch::torch_tensor(loss_obj$link(Y_base$cpu()), dtype = Y_base$dtype)$to(device = device), Y$to(device = device))$mean()$cpu())

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



  model_properties <- list(input = input_shape,
                           output = y_dim,
                           architecture = architecture)
  class(model_properties) <- "citocnn_properties"

  net <- build_cnn(model_properties)


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
  out$buffers <- list()
  out$use_model_epoch <- 2
  out$loaded_model_epoch <- torch::torch_tensor(0)
  out$model_properties <- model_properties
  out$training_properties <- training_properties
  out$device <- device_old
  out$burnin <- burnin #Add to training_properties



  ### training loop ###
  out <- train_model(model = out,epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)


  return(out)
}

#' Predict with a fitted CNN model
#'
#' This function generates predictions from a Convolutional Neural Network (CNN) model that was created using the \code{\link{cnn}} function.
#'
#' @param object a model created by \code{\link{cnn}}.
#' @param newdata A multidimensional array representing the new data for which predictions are to be made. The dimensions of \code{newdata} should match those of the training data, except for the first dimension which represents the number of samples. If \code{NULL}, the function uses the data the model was trained on.
#' @param type A character string specifying the type of prediction to be made. Options are:
#' \itemize{
#'   \item \code{"link"}: Scale of the linear predictor.
#'   \item \code{"response"}: Scale of the response.
#'   \item \code{"class"}: The predicted class labels (for classification tasks).
#' }
#' @param device Device to be used for making predictions. Options are "cpu", "cuda", and "mps". If \code{NULL}, the function uses the same device that was used when training the model. Default is \code{NULL}.
#' @param batchsize An integer specifying the number of samples to be processed at the same time. If \code{NULL}, the function uses the same batchsize that was used when training the model. Default is \code{NULL}.
#' @param ... Additional arguments (currently not used).
#' @return A matrix of predictions. If \code{type} is \code{"class"}, a factor of predicted class labels is returned.
#'
#' @example /inst/examples/predict.citocnn-example.R
#' @export
predict.citocnn <- function(object,
                            newdata = NULL,
                            type=c("link", "response", "class"),
                            device = NULL,
                            batchsize = NULL,
                            ...) {

  checkmate::assert(checkmate::checkNull(newdata),
                    checkmate::checkArray(newdata, min.d = 3, max.d = 5))

  object <- check_model(object)

  type <- match.arg(type)

  if(is.null(device)) device <- object$device
  device <- check_device(device)

  if(is.null(batchsize)) batchsize <- object$training_properties$batchsize

  if(type %in% c("response","class")) {
    link <- object$loss$invlink
  }else{
    link = function(a) a
  }

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

#' Print a fitted CNN model
#'
#' This function prints the architecture of a Convolutional Neural Network (CNN) model created using the \code{\link{cnn}} function.
#'
#' @param x A model created by \code{\link{cnn}}.
#' @param ... Additional arguments (currently not used).
#' @return The original model object \code{x}, returned invisibly.
#' @example /inst/examples/print.citocnn-example.R
#' @export
print.citocnn <- function(x, ...){
  x <- check_model(x)
  print(x$call)
  print(x$model_properties$architecture, x$model_properties$input, x$model_properties$output)
  return(invisible(x))
}

#' Summarize a fitted CNN model
#'
#' This function provides a summary of a Convolutional Neural Network (CNN) model created using the \code{\link{cnn}} function. It currently replicates the output of the \code{\link{print.citocnn}} method.
#'
#' @param object A model created by \code{\link{cnn}}.
#' @param ... Additional arguments (currently not used).
#' @return The original model object \code{object}, returned invisibly.
#' @export
summary.citocnn <- function(object, ...){
  return(print(object))
}

#' Plot a fitted CNN model
#'
#' This function plots the architecture of a Convolutional Neural Network (CNN) model created using the \code{\link{cnn}} function.
#'
#' @param x A model created by \code{\link{cnn}}.
#' @param ... Additional arguments (currently not used).
#' @return The original model object \code{x}, returned invisibly.
#' @example /inst/examples/plot.citocnn-example.R
#' @export
plot.citocnn <- function(x, ...){
  x <- check_model(x)
  plot(x$model_properties$architecture, x$model_properties$input, x$model_properties$output)
  return(invisible(x))
}

#' Retrieve parameters of a fitted CNN model
#'
#' This function returns the list of parameters (weights and biases) and buffers (e.g. running mean and variance of batch normalization layers) currently in use by the neural network model created using the \code{\link{cnn}} function.
#'
#' @param object A model created by \code{\link{cnn}}.
#' @param ... Additional arguments (currently not used).
#' @return A list with two components:
#' \itemize{
#'   \item \code{parameters}: A list of the model's weights and biases for the currently used model epoch.
#'   \item \code{buffers}: A list of buffers (e.g., running statistics) for the currently used model epoch.
#' }
#' @example /inst/examples/coef.citocnn-example.R
#' @export
coef.citocnn <- function(object,...){
  coefs <- list()
  coefs$parameters <- object$weights[object$use_model_epoch]
  coefs$buffers <- object$buffers[object$use_model_epoch]
  return(coefs)
}




#' Create a CNN Architecture
#'
#' This function constructs a \code{citoarchitecture} object that defines the architecture of a Convolutional Neural Network (CNN). The \code{citoarchitecture} object can be used by the \code{\link{cnn}} function to specify the structure of the network, including layer types, parameters, and default values.
#'
#' @param ... Objects of class \code{citolayer} created by \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}}, or \code{\link{transfer}}. These layers define the architecture of the CNN.
#' @param default_n_neurons (integer) Default number of neurons in a linear layer. Default is 10.
#' @param default_n_kernels (integer) Default number of kernels in a convolutional layer. Default is 10.
#' @param default_kernel_size (integer or tuple) Default size of kernels in convolutional and pooling layers. Can be a single integer or a tuple if sizes differ across dimensions. Default is \code{list(conv = 3, maxPool = 2, avgPool = 2)}.
#' @param default_stride (integer or tuple) Default stride of kernels in convolutional and pooling layers. Can be a single integer, a tuple if strides differ across dimensions, or \code{NULL} to use the kernel size. Default is \code{list(conv = 1, maxPool = NULL, avgPool = NULL)}.
#' @param default_padding (integer or tuple) Default zero-padding added to both sides of the input. Can be a single integer or a tuple if padding differs across dimensions. Default is \code{list(conv = 0, maxPool = 0, avgPool = 0)}.
#' @param default_dilation (integer or tuple) Default dilation of kernels in convolutional and max pooling layers. Can be a single integer or a tuple if dilation differs across dimensions. Default is \code{list(conv = 1, maxPool = 1)}.
#' @param default_bias (boolean) Default value indicating if a learnable bias should be added to neurons of linear layers and kernels of convolutional layers. Default is \code{list(conv = TRUE, linear = TRUE)}.
#' @param default_activation (character) Default activation function used after linear and convolutional layers. Supported activation functions include "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid". Default is \code{list(conv = "selu", linear = "selu")}.
#' @param default_normalization (boolean) Default value indicating if batch normalization should be applied after linear and convolutional layers. Default is \code{list(conv = FALSE, linear = FALSE)}.
#' @param default_dropout (numeric) Default dropout rate for linear and convolutional layers. Set to 0 for no dropout. Default is \code{list(conv = 0.0, linear = 0.0)}.
#'
#' @details
#' This function creates a \code{citoarchitecture} object that outlines the CNN's architecture based on the provided layers and default parameters. The final architecture consists of layers in the order they are provided. Any unspecified parameters in the \code{citolayer} objects are filled with the provided default values for their respective layer types. Defaults can be specified for each layer type individually or for all layers at once.
#'
#' @return An S3 object of class \code{"citoarchitecture"} that encapsulates the architecture of the CNN.
#' @import checkmate
#' @example /inst/examples/cnnarchitecture-example.R
#' @author Armin Schenk
#' @seealso \code{\link{cnn}}, \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}}, \code{\link{transfer}}, \code{\link{print.citoarchitecture}}, \code{\link{plot.citoarchitecture}}
#' @export
create_architecture <- function(...,
                                default_n_neurons = 10,
                                default_n_kernels = 10,
                                default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
                                default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
                                default_padding = list(conv = 0, maxPool = 0, avgPool = 0),
                                default_dilation = list(conv = 1, maxPool = 1),
                                default_bias = list(conv = TRUE, linear = TRUE),
                                default_activation = list(conv = "selu", linear = "selu"),
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
  default_activation <- fill_defaults(default_activation, list(conv = "selu", linear = "selu"))
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


#' Create a Linear Layer for a CNN Architecture
#'
#' This function creates a \code{linear} layer object of class \code{citolayer} for use in constructing a Convolutional Neural Network (CNN) architecture. The resulting layer object can be passed to the \code{\link{create_architecture}} function to define the structure of the network.
#'
#' @param n_neurons (integer) The number of hidden neurons in this layer.
#' @param bias (boolean) If \code{TRUE}, a learnable bias is added to the neurons of this layer.
#' @param activation (character) The activation function applied after this layer. Supported activation functions include "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid".
#' @param normalization (boolean) If \code{TRUE}, batch normalization is applied after this layer.
#' @param dropout (numeric) The dropout rate for this layer. Set to 0 to disable dropout.
#'
#' @details
#' This function creates a \code{linear} layer object, which is used to define a linear layer in a CNN architecture. Parameters not specified (and thus set to \code{NULL}) will be filled with default values provided to the \code{\link{create_architecture}} function.
#'
#' @return An S3 object of class \code{"linear" "citolayer"}, representing a linear layer in the CNN architecture.
#'
#' @import checkmate
#' @example /inst/examples/linear-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
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

#' Create a Convolutional Layer for a CNN Architecture
#'
#' This function creates a \code{conv} layer object of class \code{citolayer} for use in constructing a Convolutional Neural Network (CNN) architecture. The resulting layer object can be passed to the \code{\link{create_architecture}} function to define the structure of the network.
#'
#' @param n_kernels (integer) The number of kernels (or filters) in this layer.
#' @param kernel_size (integer or tuple) The size of the kernels in this layer. Use a tuple if the kernel size is different in each dimension.
#' @param stride (integer or tuple) The stride of the kernels in this layer. If \code{NULL}, the stride is set to the kernel size. Use a tuple if the stride is different in each dimension.
#' @param padding (integer or tuple) The amount of zero-padding added to the input on both sides. Use a tuple if the padding is different in each dimension.
#' @param dilation (integer or tuple) The dilation of the kernels in this layer. Use a tuple if the dilation is different in each dimension.
#' @param bias (boolean) If \code{TRUE}, a learnable bias is added to the kernels of this layer.
#' @param activation (character) The activation function applied after this layer. Supported activation functions include "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid".
#' @param normalization (boolean) If \code{TRUE}, batch normalization is applied after this layer.
#' @param dropout (numeric) The dropout rate for this layer. Set to 0 to disable dropout.
#'
#' @details
#' This function creates a \code{conv} layer object, which is used to define a convolutional layer in a CNN architecture. Parameters that are not specified (and thus set to \code{NULL}) will be filled with default values provided to the \code{\link{create_architecture}} function.
#'
#' @return An S3 object of class \code{"conv" "citolayer"}, representing a convolutional layer in the CNN architecture.
#'
#' @import checkmate
#' @example /inst/examples/conv-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
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

' Create an Average Pooling Layer for a CNN Architecture
#'
#' This function creates an \code{avgPool} layer object of class \code{citolayer} for use in constructing a Convolutional Neural Network (CNN) architecture. The resulting layer object can be passed to the \code{\link{create_architecture}} function to define the structure of the network.
#'
#' @param kernel_size (integer or tuple) The size of the kernel in this layer. Use a tuple if the kernel size differs across dimensions.
#' @param stride (integer or tuple) The stride of the kernel in this layer. If \code{NULL}, the stride is set to the kernel size. Use a tuple if the stride differs across dimensions.
#' @param padding (integer or tuple) The amount of zero-padding added to the input on both sides. Use a tuple if the padding differs across dimensions.
#'
#' @details
#' This function creates an \code{avgPool} layer object, which represents an average pooling layer in a CNN architecture. Parameters not specified (and thus set to \code{NULL}) will be filled with default values provided to the \code{\link{create_architecture}} function.
#'
#' @return An S3 object of class \code{"avgPool" "citolayer"}, representing an average pooling layer in the CNN architecture.
#'
#' @import checkmate
#' @example /inst/examples/avgPool-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
avgPool <- function(kernel_size = NULL,
                    stride = NULL,
                    padding = NULL) {

  layer <- list(kernel_size = kernel_size,
                stride = stride,
                padding = padding)
  class(layer) <- c("avgPool", "citolayer")
  return(layer)
}

#' Create a Maximum Pooling Layer for a CNN Architecture
#'
#' This function creates a \code{maxPool} layer object of class \code{citolayer} for use in constructing a Convolutional Neural Network (CNN) architecture. The resulting layer object can be passed to the \code{\link{create_architecture}} function to define the structure of the network.
#'
#' @param kernel_size (integer or tuple) The size of the kernel in this layer. Use a tuple if the kernel size varies across dimensions.
#' @param stride (integer or tuple) The stride of the kernel in this layer. If \code{NULL}, the stride is set to the kernel size. Use a tuple if the stride differs across dimensions.
#' @param padding (integer or tuple) The amount of zero-padding added to the input on both sides. Use a tuple if the padding differs across dimensions.
#' @param dilation (integer or tuple) The dilation of the kernel in this layer. Use a tuple if the dilation varies across dimensions.
#'
#' @details
#' This function creates a \code{maxPool} layer object, which represents a maximum pooling layer in a CNN architecture. Parameters not specified (and thus set to \code{NULL}) will be filled with default values provided to the \code{\link{create_architecture}} function.
#'
#' @return An S3 object of class \code{"maxPool" "citolayer"}, representing a maximum pooling layer in the CNN architecture.
#'
#' @import checkmate
#' @example /inst/examples/maxPool-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
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

#' Include a Pretrained Model in a CNN Architecture
#'
#' This function creates a \code{transfer} layer object of class \code{citolayer} for use in constructing a Convolutional Neural Network (CNN) architecture. The resulting layer object allows the use of pretrained models available in the 'torchvision' package within cito.
#'
#' @param name (character) The name of the pretrained model. Available options include: "alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2".
#' @param pretrained (boolean) If \code{TRUE}, the model uses its pretrained weights. If \code{FALSE}, random weights are initialized.
#' @param freeze (boolean) If \code{TRUE}, the weights of the pretrained model (except the "classifier" part at the end) are not updated during training. This setting only applies if \code{pretrained = TRUE}.
#'
#' @details
#' This function creates a \code{transfer} layer object, which represents a pretrained model of the \code{torchvision} package with the linear "classifier" part removed. This allows the pretrained features of the model to be utilized while enabling customization of the classifier. When using this function with \code{\link{create_architecture}}, only linear layers can be added after the \code{transfer} layer. These linear layers define the "classifier" part of the network. If no linear layers are provided following the \code{transfer} layer, the default classifier will consist of a single output layer.
#'
#' Additionally, the \code{pretrained} argument specifies whether to use the pretrained weights or initialize the model with random weights. If \code{freeze} is set to \code{TRUE}, only the weights of the final linear layers (the "classifier") are updated during training, while the rest of the pretrained model remains unchanged. Note that \code{freeze} has no effect unless \code{pretrained} is set to \code{TRUE}.
#'
#' @return An S3 object of class \code{"transfer" "citolayer"}, representing a pretrained model of the \code{torchvision} package in the CNN architecture.
#'
#' @import checkmate
#' @example /inst/examples/transfer-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
transfer <- function(name = c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"),
                     pretrained = TRUE,
                     freeze = TRUE) {

  if(identical(name, c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"))) {
    name <- "alexnet"
  }

  name <- match.arg(name)

  layer <- list(name = name,
                pretrained = pretrained,
                freeze = pretrained & freeze)
  class(layer) <- c("transfer", "citolayer")
  return(layer)
}

#' Print method for citoarchitecture objects
#'
#' This method provides a visual representation of the network architecture defined by an object of class \code{citoarchitecture}, including information about each layer's configuration. It helps in understanding the structure of the architecture defined by \code{\link{create_architecture}}.
#'
#' @param x An object of class \code{citoarchitecture}, created by \code{\link{create_architecture}}.
#' @param input_shape A numeric vector specifying the dimensions of a single sample (e.g., \code{c(3, 28, 28)} for an RGB image with height and width of 28 pixels). This argument is required for a detailed output.
#' @param output_shape An integer specifying the number of nodes in the output layer. If \code{NULL}, no output layer is printed.
#' @param ... Additional arguments (currently not used).
#'
#' @return The original \code{citoarchitecture} object, returned invisibly.
#'
#' @example /inst/examples/print.citoarchitecture-example.R
#' @export
print.citoarchitecture <- function(x, input_shape, output_shape = NULL, ...) {

  if (missing(input_shape)) {
    message("For a more detailed output specify the input_shape (and output_shape) argument(s)! See ?print.citoarchitecture for more information!")
    class(x) <- "list"
    print(x)
  } else {
    x <- adjust_architecture(x, length(input_shape)-1)

    for(layer in x) {
      if(inherits(layer, "transfer")) {
        if(!(length(input_shape) == 3 && input_shape[1] == 3)) stop("The pretrained models only work on 2 dimensional data with 3 channels: [3, x, y]")
      }
      input_shape <- print(layer, input_shape)
    }

    if(!is.null(output_shape)) {
      output_layer <- linear(n_neurons=output_shape, bias = TRUE,
                             activation="Depends on loss", normalization=FALSE, dropout=0)
      print(output_layer, input_shape)
    }
    cat("-------------------------------------------------------------------------------\n")
    return(invisible(x))
  }
}

#' @export
print.linear <- function(layer, input_shape, ...) {

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Linear     |Input: ", prod(input_shape), "\n"))
  cat(paste0("           |Output: ", layer[["n_neurons"]], "\n"))
  cat(paste0("           |Bias: ", layer[["bias"]], "\n"))
  if(layer[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  cat(paste0("           |Activation: ", layer[["activation"]], "\n"))
  if(layer[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", layer[["dropout"]], "\n"))
  }

  return(invisible(layer[["n_neurons"]]))
}

#' @export
print.conv <- function(layer, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = layer[["n_kernels"]],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = layer[["dilation"]])

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")
  dilation <- paste(layer[["dilation"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Convolution|Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")\n"))
  cat(paste0("           |Bias: ", layer[["bias"]], "\n"))
  if(layer[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  cat(paste0("           |Activation: ", layer[["activation"]], "\n"))
  if(layer[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", layer[["dropout"]], "\n"))
  }

  return(invisible(output_shape))
}

#' @export
print.avgPool <- function(layer, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = rep(1,length(input_shape)-1))

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("AvgPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ")\n"))

  return(invisible(output_shape))
}

#' @export
print.maxPool <- function(layer, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = layer[["dilation"]])

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")
  dilation <- paste(layer[["dilation"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("MaxPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")\n"))

  return(invisible(output_shape))
}

#' @export
print.transfer <- function(layer, input_shape, ...) {

  output_shape <- get_transfer_output_shape(layer$name)

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Transfer   |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Network: ", layer[["name"]] , "\n"))
  cat(paste0("           |Pretrained: ", layer[["pretrained"]] , "\n"))
  if(layer[["pretrained"]]) cat(paste0("           |Weights frozen: ", layer[["freeze"]] , "\n"))

  return(invisible(output_shape))
}


#' Plot method for citoarchitecture objects
#'
#' This method provides a visual representation of the network architecture defined by an object of class \code{citoarchitecture}, including information about each layer's configuration. It helps in understanding the structure of the architecture defined by \code{\link{create_architecture}}.
#'
#' @param x An object of class \code{citoarchitecture}, created by \code{\link{create_architecture}}.
#' @param input_shape A numeric vector specifying the dimensions of a single sample (e.g., \code{c(3, 28, 28)} for an RGB image with height and width of 28 pixels). This argument is required for a detailed output.
#' @param output_shape An integer specifying the number of nodes in the output layer. If \code{NULL}, no output layer is printed.
#' @param ... Additional arguments (currently not used).
#'
#' @return The original \code{citoarchitecture} object, returned invisibly.
#'
#'
#' @example /inst/examples/plot.citoarchitecture-example.R
#' @export
plot.citoarchitecture <- function(x, input_shape, output_shape = NULL, ...) {
  x <- adjust_architecture(x, length(input_shape)-1)

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

      input_shape <- get_transfer_output_shape(layer[["name"]])
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")

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
      input_shape <- layer[["n_neurons"]]
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

  if(!is.null(output_shape)) {
    text <- c(text, paste0("Linear layer with ", output_shape, " neurons"), "")
    type <- c(type, "linear", "arrow")
    input_shape <- output_shape
  }


  text <- c(text, paste0("Output size: ", prod(input_shape)))
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
