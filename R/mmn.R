#' @export
mmn <- function(formula,
                dataList = NULL,
                fusion_hidden = c(50L, 50L),
                fusion_activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
                fusion_bias = TRUE,
                fusion_dropout = 0.0,
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
  device_old <- device
  device <- check_device(device)

  if(identical (fusion_activation, c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                              "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                              "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"))) fusion_activation<- "relu"



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

  formula <- stats::terms(formula)
  if(formula[[1]] != "~") stop(paste0("Incorrect formula '", deparse(formula), "': ~ missing"))
  if(length(formula) == 2) stop("Incorrect formula '", deparse(formula), "': response (left side of ~) missing")

  Y <- eval(formula[[2]], envir = dataList)
  X <- format_input_data(formula[[3]], dataList)

  targets <- format_targets(Y, loss_obj)
  Y_torch <- targets$Y
  Y_base <- targets$Y_base
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls

  loss.fkt <- loss_obj$loss
  if(!is.null(loss_obj$parameter)) list2env(loss_obj$parameter,envir = environment(fun= loss.fkt))
  base_loss = as.numeric(loss.fkt(loss_obj$link(Y_base), Y_torch)$mean())

  if(validation != 0) {
    n_samples <- dim(Y_torch)[1]
    valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
    train <- c(1:n_samples)[-valid]
    train_dl <- do.call(get_data_loader, append(lapply(append(X, Y_torch), function(x) x[train, drop=F]), list(batch_size = batchsize, shuffle = shuffle)))
    valid_dl <- do.call(get_data_loader, append(lapply(append(X, Y_torch), function(x) x[valid, drop=F]), list(batch_size = batchsize, shuffle = shuffle)))
  } else {
    train_dl <- do.call(get_data_loader, append(X, list(Y_torch, batch_size = batchsize, shuffle = shuffle)))
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
  out$call$formula <- formula
  out$loss <- loss_obj
  out$data <- list(dataList=dataList)
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
  out <- train_model(model = out, epochs = epochs, device = device, train_dl = train_dl, valid_dl = valid_dl, verbose = verbose)

  return(out)
}

#' Predict from a fitted mmn model
#'
#' @param object a model created by \code{\link{mmn}}
#' @param newdata new data for predictions
#' @param type which value should be calculated, either raw response, output of link function or predicted class (in case of classification)
#' @param device device on which network should be trained on.
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @example /inst/examples/predict.citommn-example.R
#' @export
predict.citommn <- function(object, newdata = NULL, type=c("link", "response", "class"), device = c("cpu","cuda", "mps"), ...) {

  checkmate::assert(checkmate::checkNull(newdata),
                    checkmate::checkList(newdata))

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
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = object$data$dataList)
  } else {
    newdata <- format_input_data(formula = object$call$formula[[3]], dataList = newdata)
  }

  pred <- torch::as_array(link(object$net(newdata))$to(device="cpu"))

  if(!is.null(object$data$ylvls)) {
    colnames(pred) <- object$data$ylvls
    if(type == "class") pred <- factor(apply(pred,1, function(x) object$data$ylvls[which.max(x)]), levels = object$data$ylvls)
  }

  return(pred)
}

format_input_data <- function(formula, dataList) {

  inner <- function(term, input_data) {
    if(term[[1]] == "+") {
      input_data <- inner(term[[2]], input_data)
      input_data <- inner(term[[3]], input_data)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)

      if(!is.null(call$X)) {
        input_data <- append(input_data, torch::torch_tensor(stats::model.matrix(stats::formula("~ . - 1"), data = data.frame(eval(call$X, envir = dataList)))))
      } else if(!is.null(call$formula)) {
        if(!is.null(call$data)) {
          data <- data.frame(eval(call$data, envir = dataList))
          formula <- stats::formula(stats::terms(stats::formula(call$formula), data = data))
          formula <- stats::update.formula(formula, ~ . - 1)
          input_data <- append(input_data, torch::torch_tensor(stats::model.matrix(formula, data = data)))
        } else {
          formula <- stats::update.formula(stats::formula(call$formula), ~ . - 1)
          input_data <- append(input_data, torch::torch_tensor(stats::model.matrix(formula, data = dataList)))
        }
      } else {
        stop(paste0("In '", deparse(term), "' either 'formula' or 'X' must be specified."))
      }
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      input_data <- append(input_data, torch::torch_tensor(eval(call$X, envir = dataList), dtype = torch::torch_float32()))
    } else {
      stop(paste0("Symbol not supported: ", term[[1]]))
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
    } else {
      stop(paste0("Symbol not supported: ", term[[1]]))
    }
    return(model_properties)
  }

  return(inner(formula, list()))
}






