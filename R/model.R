build_model <- function(object) {
  if(inherits(object, "citodnn")) {
    net <- build_dnn(input = object$model_properties$input,
                     output = object$model_properties$output,
                     hidden = object$model_properties$hidden,
                     activation = object$model_properties$activation,
                     bias = object$model_properties$bias,
                     dropout = object$model_properties$dropout)
  } else if(inherits(object, "citocnn")) {
    net <- build_cnn(input_shape = object$model_properties$input_shape,
                     output_shape = object$model_properties$output_shape,
                     layers = object$model_properties$layers)
  } else {
    stop("model not of class citodnn or citocnn")
  }
  return(net)
}


build_dnn = function(input, output, hidden, activation, bias, dropout) {
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = torch::nn_linear(input, out_features = output,bias = FALSE)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden)+1 != length(bias)) bias = rep(bias, (length(hidden)+1))
    if(length(hidden) != length(dropout)) dropout = rep(dropout,length(hidden))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = torch::nn_linear(input, out_features = hidden[1], bias = FALSE)
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i-1])
      }
      counter = counter+1
      layers[[counter]]<- get_activation_layer(activation[i])

      counter = counter+1
      if(dropout[i]>0) {
        layers[[counter]] = torch::nn_dropout(dropout[i])
        counter = counter+1
      }
    }
    layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = output, bias = bias[i+1])
  }
  net = do.call(torch::nn_sequential, layers)
  return(net)
}


build_cnn <- function(input_shape, output_shape, layers) {

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

      net_layers[[counter]] <- get_activation_layer(layer[["activation"]])
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
  net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, out_features = output_shape)

  net = do.call(torch::nn_sequential, net_layers)
  return(net)
}



