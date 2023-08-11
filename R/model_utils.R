get_data_loader = function(X, Y, batch_size=25L, shuffle=TRUE, y_dtype) {

  ds <- torch::tensor_dataset(X = torch::torch_tensor(as.array(X)),
                              Y = torch::torch_tensor(as.matrix(Y), dtype = y_dtype))

  dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = TRUE)

  return(dl)
}


get_loss <- function(loss) {
  out <- list()
  out$parameter <- NULL

  if(is.character(loss)) loss <- tolower(loss)
  if(!inherits(loss, "family")& is.character(loss)) {
    loss <- switch(loss,
                   "gaussian" = stats::gaussian(),
                   "binomial" = stats::binomial(),
                   "poisson" = stats::poisson(),
                   loss
    )
  }

  if(inherits(loss, "family")) {
    if(loss$family == "gaussian") {
      out$parameter <- torch::torch_tensor(0.1, requires_grad = TRUE)
      out$invlink <- function(a) a
      out$loss <- function(pred, true) {
        return(torch::distr_normal(pred, torch::torch_clamp(torch::torch_tensor(unlist(out$parameter),requires_grad = TRUE), 0.0001, 20))$log_prob(true)$negative())
      }
    } else if(loss$family == "binomial") {
      if(loss$link == "logit") {
        out$invlink <- function(a) torch::torch_sigmoid(a)
      } else if(loss$link == "probit")  {
        out$invlink <- function(a) torch::torch_sigmoid(a*1.7012)
      } else {
        out$invlink <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_bernoulli(probs = out$invlink(pred))$log_prob(true)$negative())
      }
    } else if(loss$family == "poisson") {
      if(loss$link == "log") {
        out$invlink <- function(a) torch::torch_exp(a)
      } else {
        out$invlink <- function(a) a
      }
      out$loss <- function(pred, true) {
        return(torch::distr_poisson( out$invlink(pred) )$log_prob(true)$negative())
      }
    } else { stop("family not supported")}
  } else  if (is.function(loss)){
    if(is.null(formals(loss)$pred) | is.null(formals(loss)$true)){
      stop("loss function has to take two arguments, \"pred\" and \"true\"")
    }
    out$loss <- loss
    out$invlink <- function(a) a
  } else {
    if(loss == "mae"){
      out$invlink <- function(a) a
      out$loss <- function(pred, true) return(torch::nnf_l1_loss(input = pred, target = true))
    }else if(loss == "mse"){
      out$invlink <- function(a) a
      out$loss <- function(pred,true) return(torch::nnf_mse_loss(input= pred, target = true))
    }else if(loss == "softmax" | loss == "cross-entropy") {
      out$invlink <- function(a) torch::nnf_softmax(a, dim = 2)
      out$loss <- function(pred, true) {
        return(torch::nnf_cross_entropy(pred, true$squeeze(), reduction = "none"))
      }
    }
    else{
      cat( "unidentified loss \n")
    }

  }
  out$call <- loss

  return(out)
}

get_activation_layer <- function(activation) {
  return(switch(tolower(activation),
         "relu" = torch::nn_relu(),
         "leaky_relu" = torch::nn_leaky_relu(),
         "tanh" = torch::nn_tanh(),
         "elu" = torch::nn_elu(),
         "rrelu" = torch::nn_rrelu(),
         "prelu" = torch::nn_prelu(),
         "softplus" = torch::nn_softplus(),
         "celu" = torch::nn_celu(),
         "selu" = torch::nn_selu(),
         "gelu" = torch::nn_gelu(),
         "relu6" = torch:: nn_relu6(),
         "sigmoid" = torch::nn_sigmoid(),
         "softsign" = torch::nn_softsign(),
         "hardtanh" = torch::nn_hardtanh(),
         "tanhshrink" = torch::nn_tanhshrink(),
         "softshrink" = torch::nn_softshrink(),
         "hardshrink" = torch::nn_hardshrink(),
         "log_sigmoid" = torch::nn_log_sigmoid(),
         stop(paste0(activation, " as an activation function is not supported"))
  ))
}


fill_layer_parameters <- function(layers, input_dim, n_kernels, kernel_size, stride, padding, dilation,
                                  n_neurons, bias, activation, normalization, dropout, lambda, alpha) {

  default_conv_layer <- list(n_kernels=n_kernels,
                             kernel_size=get_default_parameter(kernel_size, "conv", 3, input_dim),
                             stride=get_default_parameter(stride, "conv", 1, input_dim),
                             padding=get_default_parameter(padding, "conv", 0, input_dim),
                             dilation=get_default_parameter(dilation, "conv", 1, input_dim),
                             bias=get_default_parameter(bias, "conv", TRUE),
                             normalization=get_default_parameter(normalization, "conv", FALSE),
                             activation=get_default_parameter(activation, "conv", "relu"),
                             dropout=get_default_parameter(dropout, "conv", 0.0),
                             lambda=get_default_parameter(lambda, "conv", 0.0),
                             alpha=get_default_parameter(alpha, "conv", 0.5))

  default_linear_layer <- list(n_neurons=n_neurons,
                               bias=get_default_parameter(bias, "linear", TRUE),
                               normalization=get_default_parameter(normalization, "linear", FALSE),
                               activation=get_default_parameter(activation, "linear", "relu"),
                               dropout=get_default_parameter(dropout, "linear", 0.0),
                               lambda=get_default_parameter(lambda, "linear", 0.0),
                               alpha=get_default_parameter(alpha, "linear", 0.5))

  default_pool_layer <- list(kernel_size=get_default_parameter(kernel_size, "pool", 2, input_dim),
                             stride=get_default_parameter(stride, "pool", get_default_parameter(kernel_size, "pool", 2, input_dim), input_dim),
                             padding=get_default_parameter(padding, "pool", 0, input_dim),
                             dilation=get_default_parameter(dilation, "pool", 1, input_dim))

  filled_layers <- list()
  counter <- 1
  for(layer in layers) {
    layer_type <- layer[[1]]
    if(layer_type == "conv") {
      filled_layers[[counter]] <- fill_with_defaults(layer, default_conv_layer)
    } else if(layer_type == "linear") {
      filled_layers[[counter]] <- fill_with_defaults(layer, default_linear_layer)
    } else if(layer_type == "maxPool" || layer_type == "avgPool") {
      filled_layers[[counter]] <- fill_with_defaults(layer, default_pool_layer)
    } else {
      stop(paste0("Unsupported layer type: ", layer_type))
    }
    counter <- counter+1
  }
  return(filled_layers)
}


get_default_parameter <- function(parameter, key, default, input_dim = NA) {
  if(inherits(parameter, "list")) {
    out <- ifelse(is.null(parameter[[key]]), default, parameter[[key]])
  } else {
    out <- parameter
  }
  if(!is.na(input_dim)) {
    if(length(out) == input_dim) {
      return(out)
    } else if(length(out) == 1) {
      return(rep(out, input_dim))
    } else {
      stop(paste0("Parameter length (", length(out), ") must be either 1 or equal to input dimension (", input_dim, ")"))
    }
  }
  return(out)
}

fill_with_defaults <- function(layer, default_layer) {
  for (parameter in names(default_layer)) {
    if(is.null(layer[[parameter]])) {
      layer[[parameter]] <- default_layer[[parameter]]
    }
  }
  return(layer)
}


get_regularization_parameters <- function(layers) {
  lambda <- c()
  alpha <- c()
  for(layer in layers) {
    if(layer[[1]] %in% c("conv", "linear")) {
      lamda <- c(lambda, layer[["lambda"]])
      alpha <- c(alpha, layer[["alpha"]])
    }
  }
  return(list(lambda=lambda, alpha=alpha))
}
