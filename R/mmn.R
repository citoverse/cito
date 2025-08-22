#' Train a Multi-Modal Neural Network (MMN)
#'
#' This function trains a Multi-Modal Neural Network (MMN) which consists of a combination of DNNs and CNNs.
#'
#' @param formula A formula object specifying the model structure. See examples for more information.
#' @param dataList A list containing the data for training the model. The list should contain all variables used in the formula.
#' @param fusion_hidden A numeric vector specifying the number of units (nodes) in each hidden layer of the fusion network. The length of this vector determines the number of hidden layers created, with each element specifying the number of units in the corresponding layer.
#' @param fusion_activation A character vector specifying the activation function(s) applied after each hidden layer in the fusion network. If a single character string is provided, the same activation function will be applied to all hidden layers. Alternatively, a character vector of the same length as \code{fusion_hidden} can be provided to apply different activation functions to each layer. Available options include: \code{"relu"}, \code{"leaky_relu"}, \code{"tanh"}, \code{"elu"}, \code{"rrelu"}, \code{"prelu"}, \code{"softplus"}, \code{"celu"}, \code{"selu"}, \code{"gelu"}, \code{"relu6"}, \code{"sigmoid"}, \code{"softsign"}, \code{"hardtanh"}, \code{"tanhshrink"}, \code{"softshrink"}, \code{"hardshrink"}, \code{"log_sigmoid"}.
#' @param fusion_bias A logical value or a vector indicating whether to include bias terms in each layer of the fusion network. If a single logical value is provided, it will apply to all layers. To specify bias inclusion for each layer individually, provide a logical vector of length \code{length(fusion_hidden) + 1}, where each element corresponds to a hidden layer, and the final element controls whether a bias term is added to the output layer.
#' @param fusion_dropout The dropout rate(s) to apply to each hidden layer in the fusion network. This can be a single numeric value (between 0 and 1) to apply the same dropout rate to all hidden layers, or a numeric vector of length \code{length(fusion_hidden)} to set different dropout rates for each layer individually. The dropout rate is not applied to the output layer.
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
#' @return An S3 object of class \code{"citommn"} is returned. It is a list containing everything there is to know about the model and its training process.
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
#' @details
#'
#' # Details:
#'
#' Also check \code{\link{dnn}} and \code{\link{cnn}} for details to common arguments.
#'
#' # MMN architecture:
#'
#' ![](MMN.png "MMN architecture")
#'
#' The MMN combines multiple CNNs and DNNs. This allows the model to process data in different formats (e.g., DNN+CNN for tabular data and images, or CNN+CNN for images with different spatial resolutions).
#' The architecture of the MMN is defined by the arguments \code{formula}, \code{fusion_hidden}, \code{fusion_activation}, \code{fusion_bias} and \code{fusion_dropout}:
#' \itemize{
#'    \item \code{formula} specifies the architecture of the individual networks as well as their respective inputs, and the target data of the MMN (which specifies the shape of the output layer).
#'    \item \code{fusion_hidden}, \code{fusion_activation}, \code{fusion_bias} and \code{fusion_dropout} define the architecture of the DNN that fuses the outputs of the individual networks. See \code{\link{dnn}} for details.
#' }
#'
#' **`mmn(Y ~ dnn(X=tabular_data1) + dnn(~., data=tabular_data2) + cnn(X=image_data), dataList=mmn_data, ...)`**
#'
#' In this example, **Y** (left side of ~) is the target data of the MMN. On the right side of ~ you can specify as many DNNs and CNNs as required.
#' The specification works exactly as in \code{\link{dnn}} and \code{\link{cnn}} with the following restrictions:
#'
#' \itemize{
#'    \item Only specify arguments that relate to the architecture and input data of the network (bold arguments mandatory):
#'        \itemize{
#'            \item dnn(): **formula**, **data**, hidden, activation, bias, dropout, (**X**, alternatively to formula and data)
#'            \item cnn(): **X**, **architecture**
#'        }
#'    \item Arguments relating to the training (e.g. loss, lr, epochs, ...) have to be passed to mmn() instead.
#'    \item The names of the data variables (in this example: Y, tabular_data1, tabular_data2, image_data) must be available in \code{dataList} (named list).
#' }
#'
#' @import checkmate
#' @example /inst/examples/mmn-example.R
#' @author Armin Schenk
#' @seealso \code{\link{predict.citommn}}, \code{\link{print.citommn}}, \code{\link{summary.citommn}}, \code{\link{coef.citommn}}, \code{\link{continue_training}}, \code{\link{analyze_training}}
#' @export
mmn <- function(formula,
                dataList = NULL,
                fusion_hidden = c(50L, 50L),
                fusion_activation = "relu",
                fusion_bias = TRUE,
                fusion_dropout = 0.0,
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

  checkmate::assert(checkmate::checkList(dataList), checkmate::checkNull(dataList))
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

  formula <- stats::terms(formula)
  check_mmn_formula(formula)

  if(!inherits(formula[[2]], "name")) stop(paste0("In '", deparse1(formula), "': Left side of ~ (target data) has to be assigned to a name that exists in dataList. E.g., mmn(some_name ~ dnn(...) + cnn(...), dataList=list(some_name=your_data, ...), ...)"))
  if(!as.character(formula[[2]]) %in% names(dataList)) stop(paste0("In '", deparse1(formula), "': Couldn't find '", as.character(formula[[2]]), "' in names of dataList."))
  Y <- eval(formula[[2]], envir = dataList)

  if(is.character(loss)) loss <- match.arg(loss)
  loss_obj <- get_loss(loss, Y, custom_parameters, baseloss)
  baseloss <- loss_obj$baseloss

  X <- format_input_data(formula[[3]], dataList)

  from_folder = FALSE
  if(any(sapply(X, is.character))) from_folder = TRUE

  X <- lapply(X, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    return(x)
    })

  Y <- loss_obj$format_Y(Y)
  if(is.null(batchsize)) batchsize = round(0.1*dim(Y)[1])

  if(!is.null(data_augmentation)) data_augmentation <- check_data_augmentation(data_augmentation)

  if(length(validation) == 1 && validation == 0) {
    train_dl <- do.call(get_data_loader, append(X, list(Y, batch_size = batchsize, shuffle = shuffle, from_folder = from_folder, data_augmentation = data_augmentation)))
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
    train_dl <- do.call(get_data_loader, append(lapply(append(X, Y), function(x) x[train, drop=F]), list(batch_size = batchsize, shuffle = shuffle, from_folder = from_folder, data_augmentation = data_augmentation)))
    valid_dl <- do.call(get_data_loader, append(lapply(append(X, Y), function(x) x[valid, drop=F]), list(batch_size = batchsize, shuffle = shuffle, from_folder = from_folder)))
  }

  model_properties <- list()
  model_properties$subModules <- get_model_properties(formula[[3]], dataList)
  model_properties$fusion <- list(output = loss_obj$y_dim,
                                  hidden = fusion_hidden,
                                  activation = fusion_activation,
                                  bias = fusion_bias,
                                  dropout = fusion_dropout)
  class(model_properties) <- "citommn_properties"

  net <- build_mmn(model_properties)

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
  class(out) <- "citommn"
  out$net <- net
  out$call <- match.call()
  out$call$formula <- formula
  out$loss <- loss_obj
  out$data <- list(dataList=dataList)
  if(length(validation) > 1 || validation != 0) out$data <- append(out$data, list(validation = valid))
  out$model_properties <- model_properties
  out$training_properties <- training_properties

  out <- train_model(model = out, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl)

  return(out)
}

#' Predict with a fitted MMN model
#'
#' This function generates predictions from a Multi-Modal Neural Network (MMN) model that was created using the \code{\link{mmn}} function.
#'
#' @param object a model created by \code{\link{mmn}}
#' @param newdata A list containing the new data for which predictions are to be made. The dimensions of the elements in \code{newdata} should match those of the training data, except for the respective first dimensions which represents the number of samples. If \code{NULL}, the function uses the data the model was trained on.
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
#' @export
predict.citommn <- function(object,
                            newdata = NULL,
                            type=c("link", "response", "class"),
                            device = NULL,
                            batchsize = NULL,
                            ...) {

  checkmate::assert(checkmate::checkNull(newdata),
                    checkmate::checkList(newdata))

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

  if(is.null(newdata)){
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = object$data$dataList)
  } else {
    #TODO check whether elements in newdata have the same dimensions as they had in object$data$dataList
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = newdata)
  }

  from_folder = FALSE
  if(any(sapply(newdata, is.character))) from_folder = TRUE

  newdata <- lapply(newdata, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    return(x)
  })

  dl <- do.call(get_data_loader, append(newdata, list(batch_size = batchsize, shuffle = FALSE, from_folder=from_folder)))

  pred <- NULL
  coro::loop(for(b in dl) {
    b <- lapply(b, function(x) x$to(device=device, non_blocking= TRUE))
    if(is.null(pred)) pred <- torch::as_array(link(object$net(b))$to(device="cpu"))
    else pred <- rbind(pred, torch::as_array(link(object$net(b))$to(device="cpu")))
  })

  #TODO: find a way to get sample_names as in predict.citodnn and predict.citocnn

  if(!is.null(object$loss$responses)) {
    colnames(pred) <- object$loss$responses
    if(type == "class") pred <- factor(apply(pred, 1, function(x) object$loss$responses[which.max(x)]), levels = object$loss$responses)
  }

  return(pred)
}

#' Print a fitted MMN model
#'
#' This function prints the architecture of a Multi-Modal Neural Network (MMN) model created using the \code{\link{mmn}} function.
#'
#' @param x A model created by \code{\link{mmn}}.
#' @param ... Additional arguments (currently not used).
#' @return The original model object \code{x}, returned invisibly.
#' @export
print.citommn <- function(x, ...){
  x <- check_model(x)
  print(x$call)
  print(x$net)
  return(invisible(x))
}

#' Summarize a fitted MMN model
#'
#' This function provides a summary of a Multi-Modal Neural Network (MMN) model created using the \code{\link{mmn}} function. It currently replicates the output of the \code{\link{print.citommn}} method.
#'
#' @param object A model created by \code{\link{mmn}}.
#' @param ... Additional arguments (currently not used).
#' @return The original model object \code{object}, returned invisibly.
#' @export
summary.citommn <- function(object, ...){
  return(print(object))
}

#' Retrieve parameters of a fitted MMN model
#'
#' This function returns the list of parameters (weights and biases) and buffers (e.g. running mean and variance of batch normalization layers) currently in use by the neural network model created using the \code{\link{mmn}} function.
#'
#' @param object A model created by \code{\link{mmn}}.
#' @param ... Additional arguments (currently not used).
#' @return A list with two components:
#' \itemize{
#'   \item \code{net_parameters}: A list of the model's weights and biases for the currently used model epoch.
#'   \item \code{net_buffers}: A list of buffers (e.g., running statistics) for the currently used model epoch.
#'   \item \code{loss_parameters}: A list of the loss function's parameters for the currently used model epoch.
#' }
#' @export
coef.citommn <- function(object,...){
  object <- check_model(object)
  out <- list()
  out$net_parameters <- lapply(object$net$parameters, function(x) torch::as_array(x$to(device = "cpu")))
  if(!is.null(object$net$buffers)) out$net_buffers <- lapply(object$net$buffers, function(x) torch::as_array(x$to(device = "cpu")))
  if(!is.null(object$loss$parameters)) out$loss_parameters <- lapply(object$loss$parameters, function(x) torch::as_array(x$to(device = "cpu")))
  return(out)
}

format_input_data <- function(formula, dataList) {

  inner <- function(term, input_data) {
    if(term[[1]] == "+") {
      input_data <- inner(term[[2]], input_data)
      input_data <- inner(term[[3]], input_data)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      X <- NULL
      formula <- NULL
      data <- NULL
      if(!is.null(call$X)) {
        if(!inherits(call$X, "name")) stop(paste0("In '", deparse1(term), "': X has to be assigned to a name that exists in dataList. E.g., mmn(Y ~ dnn(X=some_name, ...) + cnn(...), dataList=list(Y=Y, some_name=your_data, ...), ...)"))
        if(!as.character(call$X) %in% names(dataList)) stop(paste0("In '", deparse1(term), "': Couldn't find '", as.character(call$X), "' in names of dataList."))
        X <- eval(call$X, envir = dataList)
      } else if(!is.null(call$formula)) {
        formula <- stats::formula(call$formula)
        if(is.null(call$data)) stop(paste0("In '", deparse1(term), "': When using 'formula', 'data' has to be specified."))
        if(!inherits(call$data, "name")) stop(paste0("In '", deparse1(term), "': data has to be assigned to a name that exists in dataList. E.g., mmn(Y ~ dnn(~ A + e(B), data = some_name, ...) + cnn(...), dataList=list(Y=Y, some_name=your_data, ...), ...)"))
        if(!as.character(call$data) %in% names(dataList)) stop(paste0("In '", deparse1(term), "': Couldn't find '", as.character(call$data), "' in names of dataList."))
        data <- data.frame(eval(call$data, envir = dataList))
      }
      temp <- get_X_Y(formula, X, NULL, data)
      input_data <- append(input_data, list(torch::torch_tensor(temp$X, dtype = torch::torch_float32())))
      if (!is.null(temp$Z)) input_data <- append(input_data, list(torch::torch_tensor(temp$Z, dtype = torch::torch_long())))
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      if(!inherits(call$X, "name")) stop(paste0("In '", deparse1(term), "': X has to be assigned to a name that exists in dataList. E.g., mmn(Y ~ cnn(X=some_name, ...) + dnn(...), dataList=list(Y=Y, some_name=your_data, ...), ...)"))
      if(!as.character(call$X) %in% names(dataList)) stop(paste0("In '", deparse1(term), "': Couldn't find '", as.character(call$X), "' in names of dataList."))
      if(is.character(eval(call$X, envir = dataList))) input_data <- append(input_data, list(eval(call$X, envir = dataList)))
      else input_data <- append(input_data, list(torch::torch_tensor(eval(call$X, envir = dataList), dtype = torch::torch_float32())))
    }
    return(input_data)
  }

  return(inner(formula, list()))
}

get_model_properties <- function(formula, dataList) {
  inner <- function(term, model_properties) {
    if(term[[1]] == "+") {
      model_properties <- inner(term[[2]], model_properties)
      model_properties <- inner(term[[3]], model_properties)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      model_properties <- append(model_properties, list(eval(call, envir = dataList)))
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      model_properties <- append(model_properties, list(eval(call, envir = dataList)))
    }
    return(model_properties)
  }

  return(inner(formula, list()))
}

check_mmn_formula <- function(formula) {

  inner <- function(term) {
    if(length(term) == 1) {
      stop(paste0("In '", deparse1(formula), "': 'formula' in mmn() must not contain symbols outside of dnn() or cnn() structures. Therefore, '", deparse1(term), "' is not allowed."))
    }
    if(term[[1]] == "+") {
      inner(term[[2]])
      inner(term[[3]])
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      if(is.null(call$formula) & is.null(call$X)) stop(paste0("In '", deparse1(term), "': Either 'formula' or 'X' must be specified."))
      if(!is.null(call$Y)) stop(paste0("In '", deparse1(term), "': 'Y' must not be specified."))
      if(!is.null(call$formula)) {
        if(call$formula[[1]] != "~") stop(paste0("In '", deparse1(term), "': ~ missing in 'formula'."))
        if(length(call$formula) != 2) stop(paste0("In '", deparse1(term), "': response (left side of ~) of 'formula' must not be specified."))
      }
      args <- names(formals(dnn))
      allowed_args <- c("formula", "data", "hidden", "activation", "bias", "dropout", "X")
      for(arg in setdiff(args, allowed_args)) {
        if(!is.null(call[[arg]])) warning(paste0("In '", deparse1(term), "': Specifying '", arg, "' here has no effect. When using dnn() within mmn() only the following arguments are important: ", paste0(allowed_args, collapse = ", ")))
      }
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      if(!is.null(call$Y)) stop(paste0("In '", deparse1(term), "': 'Y' must not be specified."))
      if(is.null(call$X)) stop(paste0("In '", deparse1(term), "': 'X' must be specified."))
      if(is.null(call$architecture)) stop(paste0("In '", deparse1(term), "': 'architecture' must be specified."))
      args <- names(formals(cnn))
      allowed_args <- c("X", "architecture")
      for(arg in setdiff(args, allowed_args)) {
        if(!is.null(call[[arg]])) warning(paste0("In '", deparse1(term), "': Specifying '", arg, "' here has no effect. When using cnn() within mmn() only the following arguments are important: ", paste0(allowed_args, collapse = ", ")))
      }
    } else {
      stop(paste0("In '", deparse1(formula), "': Symbol not supported: ", term[[1]]))
    }
  }

  if(formula[[1]] != "~") stop(paste0("Incorrect formula '", deparse1(formula), "': ~ missing."))
  if(length(formula) == 2) stop(paste0("Incorrect formula '", deparse1(formula), "': response (left side of ~) missing."))
  inner(formula[[3]])
}

