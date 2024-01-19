# check if model is loaded and if current parameters are the desired ones
check_model <- function(object) {

  if(!inherits(object, c("citodnn", "citocnn"))) stop("model not of class citodnn or citocnn")

  pointer_check <- tryCatch(torch::as_array(object$net$parameters[[1]]), error = function(e) e)
  if(inherits(pointer_check,"error")){
    object$net <- build_model(object)
    object$loaded_model_epoch <- 0
    object$loss<- get_loss(object$loss$call)
    }

  if(object$loaded_model_epoch!= object$use_model_epoch){

    module_params<- names(object$weights[[object$use_model_epoch]])
    module_name<- sapply(module_params, function(x) {
      period_indices <- which(strsplit(x,"")[[1]]==".")
      last_period_index <- period_indices[length(period_indices)]
      substr(x,1,last_period_index-1)
    })
    module_type<- sapply(module_params, function(x) {
      period_indices <- which(strsplit(x,"")[[1]]==".")
      last_period_index <- period_indices[length(period_indices)]
      substring(x,last_period_index+1)
    })

    for ( i in names(object$net$modules)){
      if(i %in% module_name){
          k<- which(i == module_name)
          sapply(k, function(x) eval(parse(text=paste0("object$net$modules$`",i,"`$parameters$",module_type[k],"$set_data(object$weights[[object$use_model_epoch]]$`",module_params[k],"`)"))))

      }
    }
    object$loaded_model_epoch <-  object$use_model_epoch
  }

  if(!is.null(object$parameter)) object$loss$parameter <- lapply(object$parameter, torch::torch_tensor)

  return(object)
}

check_call_config <- function(mc, variable ,standards, dim = 1, check_var = FALSE, verbose = FALSE){
  value <- NULL
  if(variable %in% names(mc)){
    if(dim ==1){
      eval(parse(text = paste0("value  <- mc$",variable)))
    }else{
      eval(parse(text= paste0("value <- tryCatch(as.numeric(eval(mc$",variable,")), error = function(err)
              print(\"must be numeric input\")) ")))
    }

    if(!isFALSE(check_var)) checkmate::qassert(value,check_var)

  } else{
    value <- unlist(standards[which(names(standards) == variable)])
  }

  if(verbose) cat( paste0(variable,": [", paste(value, collapse = ", "),"] \n"))
  return(value)
}


check_listable_parameter <- function(parameter, check, vname = checkmate::vname(parameter)) {
  checkmate::qassert(parameter, c(check, "l+"), vname)
  if(inherits(parameter, "list")) {
    for (i in names(parameter)) {
      checkmate::qassert(parameter[[i]], check, paste0(vname, "$", i))
    }
  }
}

check_device = function(device) {
  if(device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }

  } else if(device == "mps") {
    if (torch::backends_mps_is_available()) {
      device <- torch::torch_device("mps")}
    else{
      warning("No mps device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }
  } else {
    if(device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }
  return(device)
}


# taken and adopted from lme4:::RHSForm
LHSForm = function (form, as.form = FALSE)
{
  rhsf <- form[[2]]
  if (as.form)
    stats::reformulate(deparse(rhsf))
  else rhsf
}


cast_to_r_keep_dim = function(x) {
  d = dim(x)
  if(length(d) == 1) return(as.numeric(x$cpu()))
  else return(as.matrix(x$cpu()))
}


get_X_Y = function(formula, X, Y, data) {
  if(!is.null(X)) {
    if(!is.null(Y)) {
      if(!is.matrix(Y)) Y <- data.frame(Y)
      if(ncol(Y) == 1) {
        if(is.null(colnames(Y))) colnames(Y) <- "Y"
        formula <- stats::as.formula(paste0(colnames(Y), " ~ . - 1"))
      } else {
        if(is.null(colnames(Y))) colnames(Y) <- paste0("Y", 1:ncol(Y))
        formula <- stats::as.formula(paste0("cbind(", paste(colnames(Y), collapse=","), ") ~ . - 1"))
      }
      data <- cbind(data.frame(Y), data.frame(X))
    } else {
      formula <- stats::as.formula("~ . - 1")
      data <- data.frame(X)
    }
    formula <- formula(stats::terms.formula(formula, data = data))
  } else if(!is.null(formula)) {
    if(!is.null(data)) {
      data <- data.frame(data)
    }
    formula <- formula(stats::terms.formula(formula, data = data))
    formula <- stats::update.formula(formula, ~ . - 1)
  } else {
    stop("Either formula (and data) or X (and Y) have to be specified.")
  }

  if(!is.null(data)) {
    char_cols <- sapply(data, is.character)
    data[,char_cols] <- lapply(data[,char_cols,drop=F], as.factor)
  }

  X <- stats::model.matrix(formula, data)
  Y <- stats::model.response(stats::model.frame(formula, data))
  return(list(X = X, Y = Y, formula = formula, data = data))
}



check_hyperparameters = function(hidden  ,
                              bias ,
                              lambda ,
                              alpha ,
                              dropout,
                              lr ,
                              activation,
                              batchsize,
                              epochs ) {

  out = list()
  if(inherits(hidden, "tune")) {

    if(is.null(hidden$lower)) hidden$lower = c(5, 1)
    if(is.null(hidden$upper)) hidden$upper = c(100, 10)
    if(is.null(hidden$fixed)) hidden$fixed = "both"

    if(hidden$fixed == "depth") {
      out$hidden$sampler = function() {
        return(c(sample(hidden$lower[1]:hidden$upper[1], 1), hidden$additional))
      }
    } else if(hidden$fixed == "width") {
      out$hidden$sampler = function() {
        return(c(hidden$additional, sample(hidden$lower[1]:hidden$upper[1], 1)))
      }
    } else {
      out$hidden$sampler = function() {
        return(c(sample(hidden$lower[1]:hidden$upper[1], 1), sample(hidden$lower[2]:hidden$upper[2], 1)))
      }
    }
  }

  if(inherits(bias, "tune")) {
    out$bias$sampler = function() {
      return(sample(c(TRUE, FALSE), 1))
    }
  } else {
    checkmate::qassert(bias, "B+")
  }

  if(inherits(lambda, "tune")) {
    if(is.null(lambda$lower)) lambda$lower = 0.0
    if(is.null(lambda$upper)) lambda$upper = 0.5
    out$lambda$sampler = function() {
      return(runif(1, lambda$lower, lambda$upper))
    }
  } else {
    checkmate::qassert(lambda, "R1[0,)")
  }

  if(inherits(alpha, "tune")) {
    if(is.null(alpha$lower)) alpha$lower = 0.0
    if(is.null(alpha$upper)) alpha$upper = 1.0
    out$alpha$sampler = function() {
      return(runif(1, alpha$lower, alpha$upper))
    }
  } else {
    checkmate::qassert(alpha, "R1[0,1]")
  }

  if(inherits(activation, "tune")) {
    if(is.null(activation$lower)) activation$lower = c("relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus",
                                             "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink",
                                             "softshrink", "hardshrink", "log_sigmoid")
    out$activation$sampler = function() {
      return(sample(activation$lower, 1))
    }
  } else {
    checkmate::qassert(activation, "S+[1,)")
  }


  if(inherits(dropout, "tune")) {
    if(is.null(dropout$lower)) dropout$lower = 0.0
    if(is.null(dropout$upper)) dropout$upper = 1.0
    out$dropout$sampler = function() {
      return(runif(1, dropout$lower, dropout$upper))
    }
  } else {
    checkmate::qassert(dropout, "R1[0,1]")
  }

  if(inherits(lr, "tune")) {
    if(is.null(lr$lower)) lr$lower = 0.0
    if(is.null(lr$upper)) lr$upper = 1.0
    out$lr$sampler = function() {
      return(runif(1, lr$lower, lr$upper))
    }
  } else {
    checkmate::qassert(lr, "R1[0,1]")
  }


  if(inherits(batchsize, "tune")) {
    if(is.null(batchsize$lower)) batchsize$lower = 1
    if(is.null(batchsize$upper)) batchsize$upper = 100
    out$batchsize$sampler = function() {
      return(sample(batchsize$lower:batchsize$upper, 1))
    }
  }

  if(inherits(epochs, "tune")) {
    if(is.null(epochs$lower)) epochs$lower = 1
    if(is.null(epochs$upper)) epochs$upper = 300
    out$epochs$sampler = function() {
      return(sample(epochs$lower:epochs$upper, 1))
    }
  }
  return(out)
}



