build_model <- function(object) {
  if(inherits(object, "citodnn")) {
    net <- build_dnn(input = object$model_properties$input,
                     output = object$model_properties$output,
                     hidden = object$model_properties$hidden,
                     activation = object$model_properties$activation,
                     bias = object$model_properties$bias,
                     dropout = object$model_properties$dropout)
  } else if(inherits(object, "citocnn")) {
    net <- build_cnn(input_shape = object$model_properties$input,
                     output_shape = object$model_properties$output,
                     architecture = object$model_properties$architecture)
  } else {
    stop("model not of class citodnn or citocnn")
  }
  return(net)
}


build_dnn = function(input, output, hidden, activation, bias, dropout) {
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = torch::nn_linear(input, out_features = output, bias = bias)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden)+1 != length(bias)) bias = rep(bias, (length(hidden)+1))
    if(length(hidden) != length(dropout)) dropout = rep(dropout,length(hidden))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = torch::nn_linear(input, out_features = hidden[1], bias = bias[1])
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i])
      }
      counter = counter+1
      layers[[counter]]<- get_activation_layer(activation[i])

      counter = counter+1
      if(dropout[i]>0) {
        layers[[counter]] = torch::nn_dropout(dropout[i])
        counter = counter+1
      }
    }

    if(!is.null(output)) layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = output, bias = bias[i+1])
  }
  net = do.call(torch::nn_sequential, layers)
  return(net)
}


build_cnn <- function(input_shape, output_shape, architecture) {

  input_dim <- length(input_shape) - 1
  net_layers = list()
  counter <- 1
  flattened <- FALSE
  transfer <- FALSE
  for(layer in architecture) {
    if(inherits(layer, "transfer")) {
      if(!(input_dim == 2 && input_shape[1] == 3)) stop("The pretrained models only work on RGB images: [n, 3, x, y]")
      transfer_model <- get_pretrained_model(layer$name, layer$pretrained)

      if(layer$freeze) transfer_model <- freeze_weights(transfer_model)

      if(!layer$replace_classifier) {
        transfer_model <- replace_output_layer(transfer_model, output_shape)
        return(transfer_model)
      }
      transfer <- TRUE
      input_shape <- get_transfer_output_shape(layer$name)
    } else if(inherits(layer, "conv")) {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_conv1d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]),
                                      torch::nn_conv2d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]),
                                      torch::nn_conv3d(input_shape[1], layer[["n_kernels"]], layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]], bias = layer[["bias"]]))
      counter <- counter+1

      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = layer[["n_kernels"]],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])

      if(layer[["normalization"]]) {
        net_layers[[counter]] <- switch(input_dim,
                                        torch::nn_batch_norm1d(input_shape[1]),
                                        torch::nn_batch_norm2d(input_shape[1]),
                                        torch::nn_batch_norm3d(input_shape[1]))
        counter <- counter+1
      }

      net_layers[[counter]] <- get_activation_layer(layer[["activation"]])
      counter <- counter+1

      if(layer[["dropout"]] > 0) {
        net_layers[[counter]] <- switch(input_dim,
                                        torch::nn_dropout(layer[["dropout"]]),
                                        torch::nn_dropout2d(layer[["dropout"]]),
                                        torch::nn_dropout3d(layer[["dropout"]]))
        counter <- counter+1
      }

    } else if(inherits(layer, "maxPool")) {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_max_pool1d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]),
                                      torch::nn_max_pool2d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]),
                                      torch::nn_max_pool3d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]], dilation = layer[["dilation"]]))
      counter <- counter+1

      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])

    } else if(inherits(layer, "avgPool")) {
      net_layers[[counter]] <- switch(input_dim,
                                      torch::nn_avg_pool1d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]),
                                      torch::nn_avg_pool2d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]),
                                      torch::nn_avg_pool3d(layer[["kernel_size"]], padding = layer[["padding"]], stride = layer[["stride"]]))
      counter <- counter+1

      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = rep(1, input_dim))

    } else if(inherits(layer, "linear")) {
      if(!flattened) {
        net_layers[[counter]] <- torch::nn_flatten()
        counter <- counter+1
        input_shape <- prod(input_shape)
        flattened <- T
      }

      net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = layer[["n_neurons"]], bias = layer[["bias"]])
      input_shape <- layer[["n_neurons"]]
      counter <- counter+1

      if(layer[["normalization"]]) {
        net_layers[[counter]] <- torch::nn_batch_norm1d(layer[["n_neurons"]])

        counter <- counter+1
      }

      net_layers[[counter]] <- get_activation_layer(layer[["activation"]])
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

  if(!is.null(output_shape)) net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = output_shape)

  net <- do.call(torch::nn_sequential, net_layers)

  if(transfer) {
    net <- replace_classifier(transfer_model, net)
  }

  return(net)
}



