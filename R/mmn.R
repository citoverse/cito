#' @export
mmn <- function(formula,
                dataList,
                loss = c("mse", "mae", "softmax", "cross-entropy", "gaussian", "binomial", "poisson"),
                hidden = c(50L, 50L),
                activation = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                               "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh",
                               "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"),
                bias = TRUE,
                dropout = 0.0,
                custom_parameters = NULL) {

  build_mmn <- torch::nn_module(
    initialize = function(subModules, dnn) {
      self$subModules <- torch::nn_module_list(subModules)
      self$dnn <- dnn
    },
    forward = function(input) {
      for(i in 1:length(self$subModules)) {
        input[[i]] <- self$subModules[[i]](input[[i]])
      }
      input <- self$dnn(torch::torch_cat(input[1:length(self$subModules)], dim = 2L))
      input
    }
  )

  formula <- terms(formula)
  if(formula[[1]] != "~") stop(paste0("Incorrect formula '", deparse(formula), "': ~ missing"))
  if(length(formula) == 2) stop("Incorrect formula '", deparse(formula), "': response (left side of ~) missing")

  Y <- eval(formula[[2]], envir = dataList)
  X <- format_input_data(formula[[3]], dataList)
  models <- get_models(formula[[3]], dataList)

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

  targets <- format_targets(Y, loss_obj)
  Y_torch <- targets$Y
  Y_base <- targets$Y_base
  y_dim <- targets$y_dim
  ylvls <- targets$ylvls


  dnn_input <- 0
  for(i in 1:length(models)) {
    dnn_input <- dnn_input + dim(models[[i]]$net(X[[i]][1,drop=F]))[2]
  }

  dnn <- build_dnn(dnn_input, y_dim, hidden, activation, bias, dropout)

  net <- build_mmn(lapply(models, function(x) x$net), dnn)

  return(list(net = net, X=X, Y=Y_torch))
}

format_input_data <- function(formula, dataList) {

  inner <- function(term, input_data) {
    if(term[[1]] == "+") {
      input_data <- inner(term[[2]], input_data)
      input_data <- inner(term[[3]], input_data)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)

      if(!is.null(call$X)) {
        input_data <- append(input_data, torch::torch_tensor(model.matrix(as.formula("~ . - 1"), data = data.frame(eval(call$X, envir = dataList)))))
      } else if(!is.null(call$formula)) {
        if(!is.null(call$data)) {
          data <- data.frame(eval(call$data, envir = dataList))
          formula <- formula(stats::terms.formula(formula(call$formula), data = data))
          formula <- stats::update.formula(formula, ~ . - 1)
          input_data <- append(input_data, torch::torch_tensor(model.matrix(formula, data = data)))
        } else {
          formula <- stats::update.formula(formula(call$formula), ~ . - 1)
          input_data <- append(input_data, torch::torch_tensor(model.matrix(formula, data = dataList)))
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

get_models <- function(formula, dataList) {

  inner <- function(term, models) {
    if(term[[1]] == "+") {
      models <- inner(term[[2]], models)
      models <- inner(term[[3]], models)
    } else if(term[[1]] == "dnn") {
      call <- match.call(dnn, term)
      models <- append(models, list(eval(call, envir = dataList)))
    } else if(term[[1]] == "cnn") {
      call <- match.call(cnn, term)
      models <- append(models, list(eval(call, envir = dataList)))
    } else {
      stop(paste0("Symbol not supported: ", term[[1]]))
    }
    return(models)
  }

  return(inner(formula, list()))
}






