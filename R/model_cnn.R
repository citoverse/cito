build_cnn <- function(input_shape, output, layers) {

  input_dim <- length(input_shape) - 1
  net_layers = list()
  counter <- 1
  flattened <- F
  for(layer in layers) {
    layer_type <- layer[[1]]
    if(layer_type == "conv") {
      if(flattened) stop("Using a convolutional layer after a linear layer is not allowed")

      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_conv1d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]),
                                      torch::nn_conv2d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]),
                                      torch::nn_conv3d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]))
      counter <- counter+1

      input_shape[1] <- layer[["n_kernels"]]
      for(i in 2:length(input_shape)) {
        l <- input_shape[i] + 2*layer[["padding"]][i-1]
        k <- layer[["kernel_size"]][i-1] + (layer[["kernel_size"]][i-1]-1)*(layer[["dilation"]][i-1]-1)
        s <- layer[["stride"]][i-1]
        input_shape[i] <- floor((l-k)/s)+1
      }

      if(layer[["normalization"]]) {
        net_layers[[counter]] <- switch(input_dim,
                                        torch::nn_batch_norm1d(input_shape[1]),
                                        torch::nn_batch_norm2d(input_shape[1]),
                                        torch::nn_batch_norm3d(input_shape[1]))
        counter <- counter+1
      }

      net_layers[[counter]] <- switch(tolower(layer[["activation"]]),
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
                                      stop(paste0(layer[["activation"]], " as an activation function is not supported")))
      counter <- counter+1

      if(layer[["dropout"]] > 0) {
        net_layers[[counter]] <- torch::nn_dropout(layer[["dropout"]])
        counter <- counter+1
      }

    } else if(layer_type == "maxPool") {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_max_pool1d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]),
                                      torch::nn_max_pool2d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]),
                                      torch::nn_max_pool3d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]))
      counter <- counter+1

      for(i in 2:length(input_shape)) {
        l <- input_shape[i] + 2*layer[["padding"]][i-1]
        k <- layer[["kernel_size"]][i-1] + (layer[["kernel_size"]][i-1]-1)*(layer[["dilation"]][i-1]-1)
        s <- layer[["stride"]][i-1]
        input_shape[i] <- floor((l-k)/s)+1
      }

    } else if(layer_type == "avgPool") {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_avg_pool1d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]),
                                      torch::nn_avg_pool2d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]),
                                      torch::nn_avg_pool3d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]))
      counter <- counter+1

      for(i in 2:length(input_shape)) {
        l <- input_shape[i] + 2*layer[["padding"]][i-1]
        k <- layer[["kernel_size"]][i-1]
        s <- layer[["stride"]][i-1]
        input_shape[i] <- floor((l-k)/s)+1
      }

    } else if(layer_type == "linear") {
      if(!flattened) {
        net_layers[[counter]] <- torch::nn_flatten()
        counter <- counter+1
        input_shape <- prod(input_shape)
        flattened <- T
      }

      net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = layer[["n_neurons"]], bias = layer[["bias"]])
      input_shape <- layer[["n_neurons"]]
      counter <- counter+1

      net_layers[[counter]] <- switch(tolower(layer[["activation"]]),
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
                                      stop(paste0(layer[["activation"]], " as an activation function is not supported")))
      counter <- counter+1

      if(layer[["dropout"]] > 0) {
        net_layers[[counter]] <- torch::nn_dropout(layer[["dropout"]])
        counter <- counter+1
      }
    }
  }

  if(!flattened) {
    net_layers[[counter]] <- torch::nn_flatten()
    counter <- counter+1
    input_shape <- prod(input_shape)
  }
  net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = output)

  net = do.call(torch::nn_sequential, net_layers)
  return(net)
}




fill_layer_parameters <- function(layers,
                                  input_dim,
                                  n_kernels,
                                  kernel_size,
                                  stride,
                                  padding,
                                  dilation,
                                  n_neurons,
                                  bias,
                                  activation,
                                  normalization,
                                  dropout) {

  default_conv_layer <- list(n_kernels=n_kernels,
                             kernel_size=get_default_parameter(kernel_size, "conv", 3, input_dim),
                             stride=get_default_parameter(stride, "conv", 1, input_dim),
                             padding=get_default_parameter(padding, "conv", 0, input_dim),
                             dilation=get_default_parameter(dilation, "conv", 1, input_dim),
                             bias=get_default_parameter(bias, "conv", TRUE),
                             normalization=get_default_parameter(normalization, "conv", FALSE),
                             activation=get_default_parameter(activation, "conv", "relu"),
                             dropout=get_default_parameter(dropout, "conv", 0.0))

  default_linear_layer <- list(n_neurons=n_neurons,
                               bias=get_default_parameter(bias, "linear", TRUE),
                               normalization=get_default_parameter(normalization, "linear", FALSE),
                               activation=get_default_parameter(activation, "linear", "relu"),
                               dropout=get_default_parameter(dropout, "linear", 0.0))

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
