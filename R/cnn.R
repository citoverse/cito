#' Train a Convolutional Neural Network (CNN)
#'
#' This function trains a Convolutional Neural Network (CNN) on the provided input data `X` and the target data `Y` using the specified architecture, loss function, and optimizer.
#'
#' @param X An array of input data with a minimum of 3 and a maximum of 5 dimensions. The first dimension represents the samples, the second dimension represents the channels, and the third to fifth dimensions represent the input dimensions. As an alternative, you can provide the relative or absolute path to the folder containing the images. In this case, the images will be normalized by dividing them by 255.0.
#' @param Y The target data. The allowed formats of the target data differ between loss functions. See \code{\link{dnn}} for more information.
#' @param architecture An object of class 'citoarchitecture'. See \code{\link{create_architecture}} for more information.
#' @param loss The loss function to be used. Options include "mse", "mae", "cross-entropy", "bernoulli", "gaussian", "binomial", "poisson", "nbinom", "mvp", "multinomial", and "clogit". You can also specify your own loss function. See Details for more information. Default is "mse".
#' @param custom_parameters Parameters for the custom loss function. See the vignette for an example. Default is NULL.
#' @param optimizer The optimizer to be used. Options include "sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop", and "ignite_adam". See \code{\link{config_optimizer}} for further adjustments to the optimizer. Default is "sgd".
#' @param lr Learning rate for the optimizer. Default is 0.01.
#' @param lr_scheduler Learning rate scheduler. See \code{\link{config_lr_scheduler}} for creating a learning rate scheduler. Default is NULL.
#' @param alpha Alpha value for L1/L2 regularization. Default is 0.5.
#' @param lambda Lambda value for L1/L2 regularization. Default is 0.0.
#' @param validation Proportion of the data to be used for validation. Alternatively, a vector containing the indices of the validation samples can be provided. Default is 0.0.
#' @param batchsize Batch size for training. If NULL, batchsize is 10% of the training data. Default is NULL.
#' @param shuffle Whether to shuffle the data before each epoch. Default is TRUE.
#' @param data_augmentation A list of functions used for data augmentation. Elements must be either functions or strings corresponding to inbuilt data augmentation functions. See details for more information.
#' @param epochs Number of epochs to train the model. Default is 100.
#' @param early_stopping Number of epochs with no improvement after which training will be stopped. Default is Inf.
#' @param burnin Number of epochs after which the training stops if the loss is still above the baseloss. Default is Inf.
#' @param baseloss Baseloss used for burnin and plot. If NULL, the baseloss corresponds to intercept only models. Default is NULL.
#' @param device Device to be used for training. Options are "cpu", "cuda", and "mps". Default is "cpu".
#' @param plot Whether to plot the training progress. Default is TRUE.
#' @param verbose Whether to print detailed training progress. Default is TRUE.
#'
#' @return An S3 object of class \code{"citocnn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_module". Originates from the torch package and represents the core object of this workflow.}
#' \item{call}{The original function call.}
#' \item{loss}{An object of class "nn_module". Contains all relevant information for the loss function, e.g. parameters and a function (format_Y) that transforms target data.}
#' \item{data}{A list. Contains the data used for the training of the model.}
#' \item{model_properties}{A list of properties, that define the architecture of the model.}
#' \item{training_properties}{A list of all training hyperparameters used the last time the model was trained.}
#' \item{losses}{A data.frame containing training and validation losses of each epoch.}
#' \item{best_epoch_net_state_dict}{Serialized state dict of net from the best training epoch.}
#' \item{best_epoch_loss_state_dict}{Serialized state dict of loss from the best training epoch.}
#' \item{last_epoch_net_state_dict}{Serialized state dict of net from the last training epoch.}
#' \item{last_epoch_net_state_dict}{Serialized state dict of loss from the last training epoch.}
#' \item{use_model_epoch}{String, either "best" or "last". Determines whether the parameters (e.g. weights, biases) from the best or the last training epoch are used (e.g. for prediction).}
#' \item{loaded_model_epoch}{String, shows from which training epoch the parameters are currently loaded in \code{net} and \code{loss}.}
#'
#'
#' @inheritSection dnn Training and convergence of neural networks
#'
#' @details
#'
#' # Details:
#'
#'
#' Also check \code{\link{dnn}} for details to common arguments.
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
#' The architecture of the CNN that will be created and trained by this function is defined by an object of class 'citoarchitecture'. See \code{\link{create_architecture}} for detailed information on how to define and customize your CNN architecture.
#'
#' # Data Augmentation
#' Data augmentation is a technique used to improve the generalization of convolutional neural networks (CNNs) by increasing the diversity of the training data through random transformations. This function supports data augmentation through the \code{data_augmentation} argument, which accepts a list containing either user-defined functions or the names of cito's built-in data augmentation functions.
#' Each user-defined function must take a \code{torch_tensor} as input and return a \code{torch_tensor} with the same shape. The input tensor must have 3 to 5 dimensions with the following structure:
#' \itemize{
#'    \item Dimension 1: singleton batch dimension (i.e., size 1),
#'    \item Dimension 2: channel dimension,
#'    \item Dimensions 3 to 5: spatial dimensions (e.g., X, Y, Z).
#' }
#' During training, the data loader re-loads each sample at every epoch, applying all provided augmentation functions sequentially each time the sample is accessed. This allows transformations to vary across epochs if the functions include randomness (e.g., randomly flipping a spatial axis) helping the model learn invariance to such changes.
#' In addition to custom functions, the list can contain the names (as strings) of the following built-in augmentation methods:
#' \itemize{
#'   \item \code{"rotate90"}:
#'     \itemize{
#'       \item For 2D convolutions: randomly applies one of the 4 possible 90° rotations. The X and Y dimensions have to be equal.
#'       \item For 3D convolutions:
#'         \itemize{
#'           \item If X, Y, and Z dimensions are equal: randomly applies one of the 24 possible 90° rotations.
#'           \item If only two spatial dimensions are equal: randomly applies one of the 4 possible 90° rotations in the plane of the two spatial dimensions.
#'         }
#'       \item Not available for 1D convolutions.
#'     }
#'   \item \code{"flip"}: Randomly flips each spatial dimension independently with 50% probability.
#'   \item \code{"noise"}: Adds a small amount of normally distributed noise to the tensor.
#' }
#'
#' @import checkmate
#' @example /inst/examples/cnn-example.R
#' @author Armin Schenk
#' @seealso \code{\link{predict.citocnn}}, \code{\link{print.citocnn}}, \code{\link{plot.citocnn}}, \code{\link{summary.citocnn}}, \code{\link{coef.citocnn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
cnn <- function(X,
                Y = NULL,
                architecture,
                loss = c("mse", "mae", "cross-entropy", "bernoulli", "gaussian", "binomial", "poisson", "mvp", "nbinom", "multinomial", "clogit", "softmax"),
                custom_parameters = NULL,
                optimizer = c("sgd","adam","adadelta", "adagrad", "rmsprop", "rprop", "ignite_adam"),
                lr = 0.01,
                lr_scheduler = NULL,
                alpha = 0.5,
                lambda = 0.0,
                validation = 0.0,
                batchsize = NULL,
                shuffle = TRUE,
                data_augmentation = NULL,
                epochs = 100,
                early_stopping = Inf,
                burnin = Inf,
                baseloss = NULL,
                device = c("cpu", "cuda", "mps"),
                plot = TRUE,
                verbose = TRUE) {

  if(!inherits(architecture, "citoarchitecture")) stop("architecture is not an object of class 'citoarchitecture'. See ?create_architecture for more info.")

  checkmate::assert(checkmate::checkArray(X, mode = "numeric", min.d = 3, max.d = 5, any.missing = FALSE), checkmate::check_character(X))
  checkmate::qassert(custom_parameters, c("0", "L+"))
  checkmate::qassert(lr, "N1(0,)")
  checkmate::qassert(alpha, "N1[0,1]")
  checkmate::qassert(lambda, "N1[0,)")
  checkmate::qassert(validation, c("N1[0,1)","X>1[1,)"))
  checkmate::qassert(batchsize, c("0", "X1[1,)"))
  checkmate::qassert(shuffle, "B1")
  checkmate::qassert(data_augmentation, c("0", "L+"))
  checkmate::qassert(epochs, "X1[0,)")
  checkmate::qassert(early_stopping, "N1[1,]")
  checkmate::qassert(burnin, "N1[1,]")
  checkmate::qassert(baseloss, c("0", "N1"))
  checkmate::qassert(device, "S+[3,)")
  checkmate::qassert(plot, "B1")
  checkmate::qassert(verbose, "B1")

  # Only return the model properties if no Y specified (Used in mmn())
  if(is.null(Y)) {

    # TODO: X handling at the beginning, Y at the end (MAX)
    if(is.character(X)) {
      X = list.files(X, full.names = TRUE)
      ds = get_data_loader(X, batch_size = 1L, shuffle = FALSE, from_folder = TRUE)
      input_shape = dim(ds$dataset$.getbatch(1)[[1]])[-1]
    } else {
      input_shape <- dim(X)[-1]
    }

    architecture <- adjust_architecture(architecture = architecture, input_dim = length(input_shape)-1)

    model_properties <- list(input = input_shape,
                             architecture = architecture)
    class(model_properties) <- "citocnn_properties"
    return(model_properties)
  }

  device <- match.arg(device)

  # if(!is.function(loss) & !inherits(loss,"family")) {
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

  if(is.character(loss)) loss <- match.arg(loss)
  loss_obj <- get_loss(loss, Y, custom_parameters, baseloss)
  baseloss <- loss_obj$baseloss

  X_old <- X
  Y_old <- Y

  from_folder = FALSE

  if(is.character(X)) {
    X = list.files(X, full.names = TRUE)
    from_folder = TRUE
  } else {
    X <- torch::torch_tensor(X, dtype = torch::torch_float32())
  }

  Y <- loss_obj$format_Y(Y)
  if(is.null(batchsize)) batchsize = round(0.1*dim(Y)[1])

  if(!is.null(data_augmentation)) data_augmentation <- check_data_augmentation(data_augmentation)

  if(length(validation) == 1 && validation == 0) {
    train_dl <- get_data_loader(X, Y, batch_size = batchsize, shuffle = shuffle, from_folder = from_folder, data_augmentation = data_augmentation)
    valid_dl <- NULL
  } else {
    n_samples <- dim(Y)[1]
    if(length(validation) > 1) {
      if(any(validation>n_samples)) stop("Validation indices mustn't exceed the number of samples.")
      valid <- validation
    } else {
      valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
    }
    train <- c(1:n_samples)[-valid]
    train_dl <- get_data_loader(X[train, drop=F], Y[train, drop=F], batch_size = batchsize, shuffle = shuffle, from_folder = from_folder, data_augmentation = data_augmentation)
    valid_dl <- get_data_loader(X[valid, drop=F], Y[valid, drop=F], batch_size = batchsize, shuffle = shuffle, from_folder = from_folder)
  }

  # TODO infer form the train_dl!
  input_shape <- dim(train_dl$dataset$.getbatch(1)[[1]])[-1] #dim(X)[-1]
  architecture <- adjust_architecture(architecture = architecture, input_dim = length(input_shape)-1)

  model_properties <- list(input = input_shape,
                           output = loss_obj$y_dim,
                           architecture = architecture)
  class(model_properties) <- "citocnn_properties"

  net <- build_cnn(model_properties)

  training_properties <- list(optimizer = optimizer,
                              lr = lr,
                              lr_scheduler = lr_scheduler,
                              alpha = alpha,
                              lambda = lambda,
                              validation = validation,
                              batchsize = batchsize,
                              shuffle = shuffle,
                              data_augmentation = data_augmentation,
                              epochs = epochs, #redundant?
                              early_stopping = early_stopping,
                              burnin = burnin,
                              baseloss = baseloss,
                              device = device_old,
                              plot = plot,
                              verbose = verbose)

  out <- list()
  class(out) <- "citocnn"
  out$net <- net
  out$call <- match.call()
  out$loss <- loss_obj
  out$data <- list(X = X_old, Y = Y_old)
  if(length(validation) > 1 || validation != 0) out$data <- append(out$data, list(validation = valid))
  out$model_properties <- model_properties
  out$training_properties <- training_properties

  out <- train_model(model = out, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl)

  return(out)
}

#' Predict with a fitted CNN model
#'
#' This function generates predictions from a Convolutional Neural Network (CNN) model that was created using the \code{\link{cnn}} function.
#'
#' @param object a model created by \code{\link{cnn}}.
#' @param newdata A multidimensional array representing the new data for which predictions are to be made. The dimensions of \code{newdata} should match those of the training data, except for the first dimension which represents the number of samples. As an alternative, you can provide the relative or absolute path to the folder containing the images. In this case, the images will be normalized by dividing them by 255.0. If \code{NULL}, the function uses the data the model was trained on.
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

  checkmate::assert(checkmate::checkNull(newdata), checkmate::checkCharacter(newdata),
                    checkmate::checkArray(newdata, mode = "numeric", min.d = 3, max.d = 5, any.missing = FALSE))

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

  from_folder = FALSE
  sample_names = NULL

  #TODO: get sample_names from files in folder
  if(is.null(newdata)){
    if(is.character(object$data$X)) {
      newdata = list.files(object$data$X, full.names = TRUE)
      from_folder = TRUE
    } else {
      sample_names <- dimnames(object$data$X)[[1]]
      newdata = torch::torch_tensor(object$data$X, dtype = torch::torch_float32())
    }
  } else if(is.character(newdata)) {
    newdata = list.files(newdata, full.names = TRUE)
    from_folder = TRUE
  } else if(all(dim(newdata)[-1] == object$model_properties$input)) {
    sample_names <- dimnames(newdata)[[1]]
    newdata <- torch::torch_tensor(newdata, dtype = torch::torch_float32())
  } else {
    stop(paste0("Wrong dimension of newdata: [", paste(dim(newdata), collapse = ", "), "]   Correct input dimension: [", paste(c("N", dim(object$data$X)[-1]), collapse = ", "), "]"))
  }

  dl <- get_data_loader(newdata, batch_size = batchsize, shuffle = FALSE, from_folder = from_folder)

  pred <- NULL
  coro::loop(for(b in dl) {
    if(is.null(pred)) pred <- torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE)))$to(device="cpu"))
    else pred <- rbind(pred, torch::as_array(link(object$net(b[[1]]$to(device = device, non_blocking= TRUE)))$to(device="cpu")))
  })

  if(!is.null(sample_names)) rownames(pred) <- sample_names

  if(!is.null(object$loss$responses)) {
    colnames(pred) <- object$loss$responses
    if(type == "class") pred <- factor(apply(pred, 1, function(x) object$loss$responses[which.max(x)]), levels = object$loss$responses)
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
#' @return A list with up to three components:
#' \itemize{
#'   \item \code{net_parameters}: A list of the model's weights and biases for the currently used model epoch.
#'   \item \code{net_buffers}: A list of buffers (e.g., running statistics) for the currently used model epoch.
#'   \item \code{loss_parameters}: A list of the loss function's parameters for the currently used model epoch.
#' }
#' @example /inst/examples/coef.citocnn-example.R
#' @export
coef.citocnn <- function(object, ...) {
  object <- check_model(object)
  out <- list()
  out$net_parameters <- lapply(object$net$parameters, function(x) torch::as_array(x$to(device = "cpu")))
  if(!is.null(object$net$buffers)) out$net_buffers <- lapply(object$net$buffers, function(x) torch::as_array(x$to(device = "cpu")))
  if(!is.null(object$loss$parameters)) out$loss_parameters <- lapply(object$loss$parameters, function(x) torch::as_array(x$to(device = "cpu")))
  return(out)
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
#' @param rgb (boolean) If \code{FALSE}, the pretrained weights of the first convolutional layer are averaged across the channel dimension. This is useful if your data has 3 channels but isn't an RGB image. This setting only applies if \code{pretrained = TRUE}.
#'
#' @details
#' This function creates a \code{transfer} layer object, which represents a pretrained model of the \code{torchvision} package with the linear "classifier" part removed. This allows the pretrained features of the model to be utilized while enabling customization of the classifier. When using this function with \code{\link{create_architecture}}, only linear layers can be added after the \code{transfer} layer. These linear layers define the "classifier" part of the network. If no linear layers are provided following the \code{transfer} layer, the default classifier will consist of a single output layer.
#'
#' Additionally, the \code{pretrained} argument specifies whether to use the pretrained weights or initialize the model with random weights. If \code{freeze} is set to \code{TRUE}, only the weights of the final linear layers (the "classifier") are updated during training, while the rest of the pretrained model remains unchanged. Note that \code{freeze} has no effect unless \code{pretrained} is set to \code{TRUE}.
#'
#' If your data has three channels but is not an RGB image set \code{rgb} to \code{FALSE} to average the pretrained weights of the first convolutional layer, so that each channel is treated equally. This is also done if your data has more or less channels than 3.
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
                     freeze = TRUE,
                     rgb = TRUE) {

  if(identical(name, c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"))) {
    name <- "alexnet"
  }

  name <- match.arg(name)

  layer <- list(name = name,
                pretrained = pretrained,
                freeze = pretrained & freeze,
                rgb = rgb)
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
        if(length(input_shape) != 3) stop("The pretrained models only work for 2D convolutions: [n_channels, x, y]")
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
      if(length(input_shape) != 3) stop("The pretrained models only work for 2D convolutions: [n_channels, x, y]")
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

augment_flip <- function(x) {
  for(i in 3:x$dim()) {
    if(runif(1)>0.5) x <- x$flip(i)
  }
  return(x)
}

augment_rotate90 <- function(x) {
  if(x$ndim == 3) {
    stop("1D data: Rotations not possible.")
  } else if (x$ndim == 4) {
    if(dim(x)[3] != dim(x)[4]) stop("2D data: X and Y dimension must be equal for rotations.")
    return(x$rot90(sample(0:3,1), c(3,4)))
  } else if (x$ndim == 5) {
    planes <- list(c(3,4), c(3,5), c(4,5))
    equal_dims <- sapply(planes, function(y) dim(x)[y[1]]==dim(x)[y[2]])
    if(!any(equal_dims)) {
      stop("3D data: At least 2 dimensions of X, Y and Z must be equal for rotations.")
    } else if(all(equal_dims)) {
      orientation <- list(function(y) y,
                          function(y) y$rot90(1, c(3,5)),
                          function(y) y$rot90(2, c(3,5)),
                          function(y) y$rot90(3, c(3,5)),
                          function(y) y$rot90(1, c(4,5)),
                          function(y) y$rot90(3, c(4,5)))
      x <- orientation[[sample(1:6,1)]](x)
      return(x$rot90(sample(0:3,1), c(3,4)))
    } else {
      return(x$rot90(sample(0:3,1), planes[[which(equal_dims)]]))
    }
  }
}

augment_noise <- function(x, std = 0.01) {
  return(x + torch::torch_randn_like(x) * std)
}













