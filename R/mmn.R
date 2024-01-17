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

  formula <- terms(formula)
  if(formula[[1]] != "~") stop(paste0("Incorrect formula '", deparse(formula), "': ~ missing"))
  if(length(formula) == 2) stop("Incorrect formula '", deparse(formula), "': response (left side of ~) missing")

  Y <- eval(formula[[2]], envir = dataList)
  X <- format_input_data(formula[[3]], dataList)

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

  model_properties <- list()
  model_properties$subModules <- get_model_properties(formula[[3]], dataList)
  model_properties$fusion <- list(output = y_dim,
                                  hidden = fusion_hidden,
                                  activation = fusion_activation,
                                  bias = fusion_bias,
                                  dropout = fusion_dropout)
  class(model_properties) <- "citomm_properties"



  out <- list(net = build_mmn(model_properties),
              model_properties=model_properties,
              X=X,
              Y=Y_torch)
  class(out) <- "citommn"
  return(out)
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






