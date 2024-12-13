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
#' @return An S3 object of class \code{"citommn"} is returned. It is a list containing everything there is to know about the model and its training process.
#' The list consists of the following attributes:
#' \item{net}{An object of class "nn_module", originates from the torch package and represents the core object of this workflow.}
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
  checkmate::assert(checkmate::checkList(dataList), checkmate::checkNull(dataList))

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

  loss_obj <- get_loss(loss, device = device)
  if(!is.null(loss_obj$parameter)) loss_obj$parameter <- list(parameter = loss_obj$parameter)
  if(!is.null(custom_parameters)){
    if(!inherits(custom_parameters,"list")){
      warning("custom_parameters has to be list")
    } else {
      custom_parameters <- lapply(custom_parameters, function(x) torch::torch_tensor(x, requires_grad = TRUE, device = device))
      loss_obj$parameter <- append(loss_obj$parameter, unlist(custom_parameters))
    }
  }

  formula <- stats::terms(formula)
  check_mmn_formula(formula)

  Y <- eval(formula[[2]], envir = dataList)

  from_folder = FALSE

  X <- format_input_data(formula[[3]], dataList)
  X <- lapply(X, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    else x = torch::torch_tensor(x, dtype=torch::torch_float32() )
    return(x)
    })
  if(any(sapply(X, is.character))) from_folder = TRUE

  targets <- format_targets(Y, loss_obj)
  Y_torch <- targets$Y
  Y_base <- targets$Y_base
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls

  loss.fkt <- loss_obj$loss
  if(!is.null(loss_obj$parameter)) list2env(loss_obj$parameter,envir = environment(fun= loss.fkt))
  base_loss = as.numeric(loss.fkt(torch::torch_tensor(loss_obj$link(Y_base$cpu()), dtype = Y_base$dtype)$to(device = device), Y_torch$to(device = device))$mean()$cpu())

  if(validation != 0) {
    n_samples <- dim(Y_torch)[1]
    valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
    train <- c(1:n_samples)[-valid]
    train_dl <- do.call(get_data_loader, append(lapply(append(X, Y_torch), function(x) x[train, drop=F]), list(batch_size = batchsize, shuffle = shuffle, from_folder = from_folder)))
    valid_dl <- do.call(get_data_loader, append(lapply(append(X, Y_torch), function(x) x[valid, drop=F]), list(batch_size = batchsize, shuffle = shuffle, from_folder = from_folder)))
  } else {
    train_dl <- do.call(get_data_loader, append(X, list(Y_torch, batch_size = batchsize, shuffle = shuffle, from_folder = from_folder)))
    valid_dl <- NULL
  }

  model_properties <- list()
  model_properties$subModules <- get_model_properties(formula[[3]], dataList)
  model_properties$fusion <- list(output = y_dim,
                                  hidden = fusion_hidden,
                                  activation = fusion_activation,
                                  bias = fusion_bias,
                                  dropout = fusion_dropout)
  class(model_properties) <- "citommn_properties"

  net <- build_mmn(model_properties)

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
  class(out) <- "citommn"
  out$net <- net
  out$call <- match.call()
  out$call$formula <- formula #necessary?
  out$loss <- loss_obj
  out$data <- list(dataList=dataList)
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
  out <- train_model(model = out, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)

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
#' @example /inst/examples/predict.citommn-example.R
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

  if(is.null(device)) device <- object$device
  device <- check_device(device)

  if(is.null(batchsize)) batchsize <- object$training_properties$batchsize


  if(type %in% c("response", "class")) {
    link <- object$loss$invlink
  }else{
    link = function(a) a
  }

  object$net$to(device = device)

  if(is.null(newdata)){
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = object$data$dataList)
  } else {
    #TODO check whether elements in newdata have the same dimensions as they had in object$data$dataList
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = newdata)
  }
  from_folder = FALSE
  newdata <- lapply(newdata, function(x) {
    if(is.character(x)) x = list.files(x, full.names = TRUE)
    else x = torch::torch_tensor(x, dtype=torch::torch_float32() )
    return(x)
  })
  if(any(sapply(newdata, is.character))) from_folder = TRUE

  dl <- do.call(get_data_loader, append(newdata, list(batch_size = batchsize, shuffle = FALSE, from_folder=from_folder)))

  pred <- NULL
  coro::loop(for(b in dl) {
    b <- lapply(b, function(x) x$to(device=device, non_blocking= TRUE))
    if(is.null(pred)) pred <- torch::as_array(link(object$net(b))$to(device="cpu"))
    else pred <- rbind(pred, torch::as_array(link(object$net(b))$to(device="cpu")))
  })

  if(!is.null(sample_names)) rownames(pred) <- sample_names

  if(!is.null(object$data$ylvls)) {
    colnames(pred) <- object$data$ylvls
    if(type == "class") pred <- factor(apply(pred,1, function(x) object$data$ylvls[which.max(x)]), levels = object$data$ylvls)
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
#' @example /inst/examples/print.citommn-example.R
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
#'   \item \code{parameters}: A list of the model's weights and biases for the currently used model epoch.
#'   \item \code{buffers}: A list of buffers (e.g., running statistics) for the currently used model epoch.
#' }
#' @example /inst/examples/coef.citommn-example.R
#' @export
coef.citommn <- function(object,...){
  coefs <- list()
  coefs$parameters <- object$weights[object$use_model_epoch]
  coefs$buffers <- object$buffers[object$use_model_epoch]
  return(coefs)
}

format_input_data <- function(formula, dataList) {

  inner <- function(term, input_data) {
    if(term[[1]] == "+") {
      input_data <- inner(term[[2]], input_data)
      input_data <- inner(term[[3]], input_data)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)

      if(!is.null(call$X)) {
        tmp <- stats::model.matrix(stats::formula("~ ."), data = data.frame(eval(call$X, envir = dataList)))
        X <- tmp[, -1, drop=FALSE]
        attr(X, "assign") <- attr(tmp, "assign")[-1]

      } else if(!is.null(call$formula)) {
        if(!is.null(call$data)) {
          data <- data.frame(eval(call$data, envir = dataList))
          formula <- stats::formula(stats::terms(stats::formula(call$formula), data = data))
          formula <- stats::update.formula(formula, ~ . + 1)
          tmp <- stats::model.matrix(formula, data = data)
          X <- tmp[, -1, drop=FALSE]
          attr(X, "assign") <- attr(tmp, "assign")[-1]
        } else {
          formula <- stats::update.formula(stats::formula(call$formula), ~ . + 1)
          tmp <- stats::model.matrix(formula, data = dataList)
          X <- tmp[, -1, drop=FALSE]
          attr(X, "assign") <- attr(tmp, "assign")[-1]
        }
      }
      input_data <- append(input_data, list(X))
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      input_data <- append(input_data, list(eval(call$X, envir = dataList)))
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
        check_for_embeddings(call$formula[[2]])
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

check_for_embeddings <- function(term) {
  if(length(term) > 1) {
    if(term[[1]] == "+") {
      check_for_embeddings(term[[2]])
      check_for_embeddings(term[[3]])
    } else if(term[[1]] == "e") {
      stop(paste0("'", deparse1(term), "': Embeddings not implemented in mmn() yet."))
    }
  }
}


