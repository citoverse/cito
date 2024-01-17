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
#' | poisson | Poisson likelihood |Regression, count data, e.g. species abundances|
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
#' \item{weigths}{List of weights for each training epoch}
#' \item{use_model_epoch}{Integer, which defines which model from which training epoch should be used for prediction.}
#' \item{loaded_model_epoch}{Integer, shows which model from which epoch is loaded currently into model$net.}
#' \item{model_properties}{A list of properties of the neural network, contains number of input nodes, number of output nodes, size of hidden layers, activation functions, whether bias is included and if dropout layers are included.}
#' \item{training_properties}{A list of all training parameters that were used the last time the model was trained. It consists of learning rate, information about an learning rate scheduler, information about the optimizer, number of epochs, whether early stopping was used, if plot was active, lambda and alpha for L1/L2 regularization, batchsize, shuffle, was the data set split into validation and training, which formula was used for training and at which epoch did the training stop.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch}
#' @import checkmate
#' @example /inst/examples/cnn-example.R
#' @author Armin Schenk, Maximilian Pichler
#' @seealso \code{\link{predict.citocnn}}, \code{\link{plot.citocnn}},  \code{\link{coef.citocnn}}, \code{\link{print.citocnn}}, \code{\link{summary.citocnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
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
  out$use_model_epoch <- 1
  out$loaded_model_epoch <- 0
  out$model_properties <- model_properties
  out$training_properties <- training_properties
  out$device <- device_old



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
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @example /inst/examples/predict.citocnn-example.R
#' @export
predict.citocnn <- function(object, newdata = NULL, type=c("link", "response", "class"), device = c("cpu","cuda", "mps"), ...) {

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


  ### TO DO: use dataloaders via get_data_loader function
  if(is.null(newdata)){
    newdata = torch::torch_tensor(object$data$X, device = device, dtype = torch::torch_float32())
  } else if(all(dim(newdata)[-1] == dim(object$data$X)[-1])) {
    newdata <- torch::torch_tensor(newdata, device = device, dtype = torch::torch_float32())
  } else {
    stop(paste0("Wrong dimension of newdata: [", paste(dim(newdata), collapse = ", "), "]   Correct input dimension: [", paste(c("N", dim(object$data$X)[-1]), collapse = ", "), "]"))
  }

  pred <- torch::as_array(link(object$net(newdata))$to(device="cpu"))

  if(!is.null(object$data$ylvls)) {
    colnames(pred) <- object$data$ylvls
    if(type == "class") pred <- factor(apply(pred,1, function(x) object$data$ylvls[which.max(x)]), levels = object$data$ylvls)
  }
  if(!is.null(dimnames(newdata))) rownames(pred) <- dimnames(newdata)[[1]]

  return(pred)
}

#' Print class citocnn
#'
#' @param x a model created by \code{\link{cnn}}
#' @param ... additional arguments
#' @return original object x
#'
#' @example /inst/examples/print.citocnn-example.R
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
#' @example /inst/examples/plot.citocnn-example.R
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
#' @example /inst/examples/coef.citocnn-example.R
#' @export
coef.citocnn <- function(object,...){
  return(object$weights[object$use_model_epoch])
}

