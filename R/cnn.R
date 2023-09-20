#' CNN
#'
#' @description
#'
#' fits a custom convolutional neural network.
#' @param X array with dimension 3, 4 or 5 for 1D-, 2D- or 3D-convolutions, respectively. The first dimension are the samples, the second dimension the channels and the third - fifth dimension are the input dimensions
#' @param Y vector (regression), factor (classification), numerical matrix (regression) or logical matrix (multi-label classification)
#' @param layers vector/list with elements of type list that describe the architecture of the network. See examples for more details
#' @param n_kernels (int) default value: amount of kernels in a convolutional layer
#' @param kernel_size (int or tuple) default value: size of the kernels in a convolutional or pooling layer. Use a list to set different defaults for convolutional and pooling layers
#' @param stride (int or tuple) default value: stride of the kernels in a convolutional or pooling layer. NULL sets the stride equal to the kernel size. Use a list to set different defaults for convolutional and pooling layers
#' @param padding (int or tuple) default value: zero-padding added to both sides of the input. Use a list to set different defaults for convolutional and pooling layers
#' @param dilation (int or tuple) default value: dilation of the kernels in a convolutional layer
#' @param n_neurons (int) default value: amount of neurons in a linear layer
#' @param bias (boolean) default value: if TRUE, adds a learnable bias to linear and convolutional layers. Use a list to set different defaults for linear and convolutional layers
#' @param activation (string) default value: activation function that is used after each linear and convolutional layer. Use a list to set different defaults for linear and convolutional layers
#' @param normalization (boolean) default value: if TRUE, batch normalization is used after linear and convolutional layers. Use a list to set different defaults for linear and convolutional layers
#' @param dropout (float) default value: dropout rate of linear and convolutional layers. Set to 0 for no dropout. Use a list to set different defaults for linear and convolutional layers
#' @param lambda (float) default value: Controls the strength of the L1/L2 regularization of linear and convolutional layers. Set to 0 for no regularization. Use a list to set different defaults for linear and convolutional layers
#' @param alpha (float) default value: Controls the proportion of the L1 and L2 term of the L1/L2 regularization of linear and convolutional layers. Set to 0 for only L1 regularization, to 1 for only L2 regularization or to NA for no regularization. Use a list to set different defaults for linear and convolutional layers
#' @param loss loss after which network should be optimized. Can also be distribution from the stats package or own function
#' @param optimizer which optimizer used for training the network, for more adjustments to optimizer see \code{\link{config_optimizer}}
#' @param lr learning rate given to optimizer
#' @param validation percentage of data set that should be taken as validation set (chosen randomly)
#' @param batchsize number of samples that are used to calculate one learning rate step
#' @param shuffle if TRUE, data in each batch gets reshuffled every epoch
#' @param epochs epochs the training goes on for
#' @param early_stopping if set to integer, training will stop if loss has gotten higher for defined number of epochs in a row, will use validation loss is available.
#' @param lr_scheduler learning rate scheduler created with \code{\link{config_lr_scheduler}}
#' @param custom_parameters List of parameters/variables to be optimized. Can be used in a custom loss function. See Vignette for example.
#' @param device device on which network should be trained on.
#' @param plot plot training loss
#' @param verbose print training and validation loss of epochs
#'
#' @details
#'
#' Training on graphic cards:
#' If you want to train on your cuda device, you have to install the NVIDIA CUDA toolkit version 11.3. and cuDNN 8.4. beforehand. Make sure that you have exactly these versions installed, since it does not work with other version.
#' For more information see [mlverse: 'torch'](https://torch.mlverse.org/docs/articles/installation.html)
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
#' @example /inst/examples/dnn-example.R
#' @seealso \code{\link{predict.citodnn}}, \code{\link{plot.citodnn}},  \code{\link{coef.citodnn}},\code{\link{print.citodnn}}, \code{\link{summary.citodnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}, \code{\link{PDP}}, \code{\link{ALE}},
#' @export
cnn <- function(X,
                Y,
                layers = NULL, #Default einfuegen
                n_kernels = 10,
                kernel_size = list(conv = 3, pool = 2),
                stride = list(conv = 1, pool = NULL),
                padding = 0,
                dilation = 1,
                n_neurons = 10,
                bias = TRUE,
                activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
                normalization = FALSE,
                dropout = 0.0,
                lambda = 0.0,
                alpha = 0.5,
                loss = c("mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson"),
                optimizer = c("sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop"),
                lr = 0.01,
                validation = 0.0,
                batchsize = 32L,
                shuffle = TRUE,
                epochs = 100,
                early_stopping = NULL,
                lr_scheduler = NULL,
                custom_parameters = NULL,
                device = c("cpu", "cuda"),
                plot = TRUE,
                verbose = TRUE) {

  if(identical(activation, c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                              "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                              "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"))) activation <- "relu"

  #Data
  checkmate::assert(checkmate::checkArray(X, min.d = 3, max.d = 5))
  checkmate::assert(checkmate::checkFactor(Y), checkmate::checkNumeric(Y),
                    checkmate::checkMatrix(Y, mode = "numeric"), checkmate::checkMatrix(Y, mode = "logical"))

  #Architecture
  check_listable_parameter(activation, "S1[1,)")
  check_listable_parameter(normalization, "B1")
  check_listable_parameter(lambda, "R1[0,)")
  check_listable_parameter(alpha, "r1[0,1]")
  check_listable_parameter(dropout, "R1[0,1)")
  check_listable_parameter(bias, "B1")
  checkmate::qassert(n_kernels, "X1")
  check_listable_parameter(kernel_size, "X<=3[1,)")
  check_listable_parameter(stride, c("X<=3[1,)","0"))
  check_listable_parameter(padding, "X<=3[0,)")
  check_listable_parameter(dilation, "X<=3[1,)")

  #Training
  checkmate::qassert(lr, "R+[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(batchsize, "X1[1,)")
  checkmate::qassert(shuffle, "B1")
  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(early_stopping, c("0","X1[1,)"))
  checkmate::qassert(custom_parameters, c("0", "L+"))
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")


  if(!is.function(loss) & !inherits(loss,"family")) {
    loss <- match.arg(loss)
  }

  device <- match.arg(device)

  if(device == "cuda") {
    if(torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")
    } else {
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  } else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }

  input_shape <- dim(X)[-1]


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

  targets <- format_targets(Y, loss_obj)
  Y <- targets$Y
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls

  X <- torch::torch_tensor(X, dtype = torch::torch_float32())

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

  layers <- fill_layer_parameters(layers = layers,
                                  input_dim = length(input_shape)-1,
                                  n_kernels = n_kernels,
                                  kernel_size = kernel_size,
                                  stride = stride,
                                  padding = padding,
                                  dilation = dilation,
                                  n_neurons = n_neurons,
                                  bias = bias,
                                  activation = activation,
                                  normalization = normalization,
                                  dropout = dropout,
                                  lambda = lambda,
                                  alpha = alpha)

  regularization_parameters <- get_regularization_parameters(layers)
  lambda <- c(regularization_parameters$lambda, get_default_parameter(lambda, "linear", 0.0))
  alpha <- c(regularization_parameters$alpha, get_default_parameter(alpha, "linear", 0.5))

  net <- build_cnn(input_shape = input_shape,
                   output_shape = y_dim,
                   layers = layers)

  model_properties <- list(input = input_shape,
                           output = y_dim,
                           layers = layers)

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
  out$data <- list(X = X, Y = Y)
  if(validation != 0) out$data <- append(out$data, list(validation = valid))
  out$weights <- list()
  out$use_model_epoch <- 0
  out$loaded_model_epoch <- 0
  out$model_properties <- model_properties
  out$training_properties <- training_properties



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
#' @example /inst/examples/predict.citodnn-example.R
#' @export
predict.citocnn <- function(object, newdata = NULL, type=c("link", "response", "class"), device = c("cpu","cuda"), ...) {

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

  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  }else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }

  object$net$to(device = device)


  ### TO DO: use dataloaders via get_data_loader function
  if(is.null(newdata)){
    newdata = torch::torch_tensor(object$data$X, device = device)
  } else if(all(dim(newdata)[-1] == dim(object$data$X)[-1])) {
    newdata <- torch::torch_tensor(newdata, device = device)
  } else {
    stop(paste0("Wrong dimension of newdata: [", paste(dim(newdata), collapse = ", "), "]   Correct input dimension: [", paste(c("N", dim(object$data$X)[-1]), collapse = ", "), "]"))
  }

  pred <- torch::as_array(link(object$net(newdata))$to(device="cpu"))

  #if(!is.null(object$data$ylvls)) colnames(pred) <- object$data$ylvls
  if(type == "class") pred <- as.factor(apply(pred,1, function(x) object$data$ylvls[which.max(x)]))

  #rownames(pred) <- rownames(newdata)


  return(pred)
}

#' Printing object of class 'citocnn'
#'
#' Prints the method call that created the 'citocnn' object and the structure of the CNN
#'
#' @param x a model created by \code{\link{cnn}}
#' @param ... additional arguments (none implemented yet)
#' @example /inst/examples/print.citodnn-example.R
#' @return original object x
#' @export
print.citocnn <- function(x, ...){
  x <- check_model(x)
  print(x$call)
  input_shape <- x$model_properties$input
  for(layer in x$model_properties$layers) {
    input_shape <- print_layer(layer, input_shape)
  }
  output_layer <- list("linear", n_neurons=x$model_properties$output_shape, activation="",
                       lambda=x$training_properties$lambda[length(x$training_properties$lambda)],
                       alpha=x$training_properties$alpha[length(x$training_properties$alpha)],
                       normalization=FALSE, dropout=0)
  print_layer(output_layer, input_shape)
  cat("-------------------------------------------------------------------------------\n")
  return(invisible(x))
}

print_layer <- function(layer, input_shape) {
  return(switch(layer[[1]],
                "conv" = print_convLayer(layer, input_shape),
                "linear" = print_linearLayer(layer, input_shape),
                "maxPool" = print_maxPoolLayer(layer, input_shape),
                "avgPool" = print_avgPoolLayer(layer, input_shape)))
}


print_convLayer <- function(layer, input_shape) {

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
  cat(paste0("           |Activation: ", layer[["activation"]], "\n"))
  if(layer[["lambda"]]>0 & !is.na(layer[["alpha"]])) {
    cat(paste0("           |L1/L2 Regularization: lambda=", layer[["lambda"]], ", alpha=", layer[["alpha"]], "\n"))
  }
  if(layer[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  if(layer[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", layer[["dropout"]], "\n"))
  }

  return(invisible(output_shape))
}

print_linearLayer <- function(layer, input_shape) {

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Linear     |Input: ", prod(input_shape), "\n"))
  cat(paste0("           |Output: ", layer[["n_neurons"]], "\n"))
  cat(paste0("           |Activation: ", layer[["activation"]], "\n"))
  if(layer[["lambda"]]>0 & !is.na(layer[["alpha"]])) {
    cat(paste0("           |L1/L2 Regularization: lambda=", layer[["lambda"]], ", alpha=", layer[["alpha"]], "\n"))
  }
  if(layer[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  if(layer[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", layer[["dropout"]], "\n"))
  }

  return(invisible(layer[["n_neurons"]]))
}

print_maxPoolLayer <- function(layer, input_shape) {

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

print_avgPoolLayer <- function(layer, input_shape) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = 1)

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("AvgPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ")\n"))

  return(invisible(output_shape))
}


#' Creates plot which gives an overview of the network architecture.
#'
#' @param x a model created by \code{\link{cnn}}
#' @param ... no further functionality implemented yet
#' @return A plot that represents the neural network
#' @example /inst/examples/plot.citodnn-example.R
#' @export
plot.citocnn <- function(x, ...) {
  object <- check_model(x)
  net <- x$net

  n_modules <- length(net$modules[-1])

  height <- 1/n_modules
  ybottom <- 1-height
  ytop <- 1
  graphics::plot.new()
  for(module in net$modules[-1]) {
    if(inherits(module, c("nn_conv_nd", "nn_linear", "nn_max_pool_nd", "nn_avg_pool_nd"))) {
      if(inherits(module, "nn_conv_nd")) {
        color <- "lightblue"
        kernel_size <- paste(module$kernel_size, collapse = "x")
        stride <- paste(module$stride, collapse = "x")
        padding <- paste(module$padding, collapse = "x")
        dilation <- paste(module$dilation, collapse = "x")
        text <- paste0("Convolution layer with ", kernel_size, " kernel (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")")
      } else if(inherits(module, "nn_linear")) {
        color <- "lightgreen"
        neurons <- dim(module$weight)[1]
        text <- paste0("Linear layer with ", neurons, " neurons")
      } else if(inherits(module, "nn_max_pool_nd")) {
        color <- "pink"
        kernel_size <- paste(module$kernel_size, collapse = "x")
        stride <- paste(module$stride, collapse = "x")
        padding <- paste(module$padding, collapse = "x")
        dilation <- paste(module$dilation, collapse = "x")
        text <- paste0("MaxPool with ", kernel_size, " kernel (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")")
      } else if(inherits(module, "nn_avg_pool_nd")) {
        color <- "pink"
        kernel_size <- paste(module$kernel_size, collapse = "x")
        stride <- paste(module$stride, collapse = "x")
        padding <- paste(module$padding, collapse = "x")
        text <- paste0("AvgPool with ", kernel_size, " kernel (stride=", stride, ", padding=", padding, ")")
      }
      graphics::rect(0, ybottom, 1, ytop, col = color)
      graphics::text(0.5, ytop-height/2, text)
    } else {
      if(inherits(module,c("nn_relu","nn_leaky_relu","nn_tanh","nn_elu","nn_rrelu","nn_prelu","nn_softplus","nn_celu","nn_selu","nn_gelu",
                     "nn_relu6","nn_sigmoid","nn_softsign","nn_hardtanh","nn_tanhshrink","nn_softshrink","nn_hardshrink","nn_log_sigmoid"))) {
        text <- paste0("Activation: ", gsub("nn_","", class(module)[1]))
      } else if(inherits(module, "nn_batch_norm_")) {
        text <- "Batch normalization"
      } else if(inherits(module, "nn_dropout")) {
        text <- paste0("Dropout layer (p=", module$p, ")")
      } else if(inherits(module, "nn_flatten")) {
        text <- "Flatten output"
      }
      graphics::arrows(0.5, ytop, 0.5, ybottom, length = 0.5*graphics::par("pin")[2]/(2*n_modules-1))
      graphics::text(0.5, ytop-height/2, text, pos = 4)
    }



    ybottom <- ybottom-height
    ytop <- ytop-height
  }
}



