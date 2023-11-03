#' CNN architecture
#'
#' @description
#'
#' creates a 'citoarchitecture' object that is used by \code{\link{cnn}}.
#'
#' @param ... objects of class 'citolayer' created by \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}} or \code{\link{avgPool}}
#' @param default_n_neurons (int) default value: amount of neurons in a linear layer
#' @param default_n_kernels (int) default value: amount of kernels in a convolutional layer
#' @param default_kernel_size (int or tuple) default value: size of the kernels in convolutional and pooling layers. Use a tuple if the kernel size isn't equal in all dimensions
#' @param default_stride (int or tuple) default value: stride of the kernels in convolutional and pooling layers. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param default_padding (int or tuple) default value: zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#' @param default_dilation (int or tuple) default value: dilation of the kernels in convolutional and maxPooling layers. Use a tuple if the dilation isn't equal in all dimensions
#' @param default_bias (boolean) default value: if TRUE, adds a learnable bias to neurons of linear and kernels of convolutional layers
#' @param default_activation (string) default value: activation function that is used after linear and convolutional layers. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#' @param default_normalization (boolean) default value: if TRUE, batch normalization is used after linear and convolutional layers
#' @param default_dropout (float) default value: dropout rate of linear and convolutional layers. Set to 0 for no dropout
#'
#' @details
#' This function creates a 'citoarchitecture' object that provides the \code{\link{cnn}} function with all information about the architecture of the CNN that will be created and trained.
#' The final architecture consists of the layers in the sequence they were passed to this function.
#' All parameters of the 'citolayer' objects, that are still NULL because they haven't been specified at the creation of the layer, are filled with the given default parameters for their specific layer type (linear, conv, maxPool, avgPool).
#' The default values can be changed by either passing a list with the values for specific layer types (in which case the defaults of layer types which aren't in the list remain the same)
#' or by passing a single value (in which case the defaults for all layer types is set to that value).

#' @return S3 object of class \code{"citoarchitecture"}
#' @import checkmate
#' @example /inst/examples/cnnarchitecture-example.R
#' @author Armin Schenk
#' @seealso \code{\link{cnn}}, \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}}
#' @export
create_architecture <- function(...,
                                default_n_neurons = 10,
                                default_n_kernels = 10,
                                default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
                                default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
                                default_padding = list(conv = 0, maxPool = 0, avgPool = 0),
                                default_dilation = list(conv = 1, maxPool = 1),
                                default_bias = list(conv = TRUE, linear = TRUE),
                                default_activation = list(conv = "relu", linear = "relu"),
                                default_normalization = list(conv = FALSE, linear = FALSE),
                                default_dropout = list(conv = 0.0, linear = 0.0)) {

  checkmate::qassert(default_n_neurons, "X1")
  checkmate::qassert(default_n_kernels, "X1")
  check_listable_parameter(default_kernel_size, "X<=3[1,)")
  check_listable_parameter(default_stride, c("X<=3[1,)","0"))
  check_listable_parameter(default_padding, "X<=3[0,)")
  check_listable_parameter(default_dilation, "X<=3[1,)")
  check_listable_parameter(default_bias, "B1")
  check_listable_parameter(default_activation, "S1[1,)")
  check_listable_parameter(default_normalization, "B1")
  check_listable_parameter(default_dropout, "R1[0,1)")

  default_n_neurons <- list(linear = default_n_neurons)
  default_n_kernels <- list(conv = default_n_kernels)
  default_kernel_size <- fill_defaults(default_kernel_size, list(conv = 3, maxPool = 2, avgPool = 2))
  default_stride <- fill_defaults(default_stride, list(conv = 1, maxPool = NULL, avgPool = NULL))
  default_padding <- fill_defaults(default_padding, list(conv = 0, maxPool = 0, avgPool = 0))
  default_dilation <- fill_defaults(default_dilation, list(conv = 1, maxPool = 1))
  default_bias <- fill_defaults(default_bias, list(conv = TRUE, linear = TRUE))
  default_activation <- fill_defaults(default_activation, list(conv = "relu", linear = "relu"))
  default_normalization <- fill_defaults(default_normalization, list(conv = FALSE, linear = FALSE))
  default_dropout <- fill_defaults(default_dropout, list(conv = 0.0, linear = 0.0))

  fill_layer <- function(layer) {
    type <- class(layer)[1]
    for(parameter in names(layer)) {
      if(is.null(layer[[parameter]])) {
        eval(parse(text=paste0("if(!is.null(default_", parameter,"[[type]])) layer[[parameter]] <- default_", parameter, "[[type]]")))
      }
    }
    return(layer)
  }

  architecture <- list()

  transfer <- FALSE
  flattened <- FALSE
  for(layer in list(...)) {
    if(!inherits(layer, "citolayer")) stop("Objects must be of class citolayer")
    if(inherits(layer, "transfer")) {
      if(length(architecture) != 0) stop("There mustn't be any layers before a transfer layer")
      layer$replace_classifier <- length(list(...)) > 1
      architecture <- append(architecture, list(layer))
      transfer <- TRUE
    } else {
      if(transfer && inherits(layer, c("conv", "maxPool", "avgPool"))) stop("Only linear layers are allowed after a transfer layer")
      if(inherits(layer, "linear")) {
        flattened <- TRUE
      } else {
        if(flattened) stop("Only linear layers are allowed after a linear layer")
      }
      architecture <- append(architecture, list(fill_layer(layer)))
    }
  }

  class(architecture) <- "citoarchitecture"

  return(architecture)
}

fill_defaults <- function(passed_defaults, default_defaults) {
  if(inherits(passed_defaults, "list")) {
    for(entry in names(default_defaults)) {
      if(!entry %in% names(passed_defaults)) {
        passed_defaults[[entry]] <- default_defaults[[entry]]
      }
    }
    return(passed_defaults)

  } else {
    for(entry in names(default_defaults)) {
      default_defaults[[entry]] <- passed_defaults
    }
    return(default_defaults)
  }
}


#' Linear layer
#'
#' @description
#'
#' creates a 'linear' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param n_neurons (int) amount of hidden neurons in this layer
#' @param bias (boolean) if TRUE, adds a learnable bias to the neurons of this layer
#' @param activation (string) activation function that is used after this layer. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#' @param normalization (boolean) if TRUE, batch normalization is used after this layer
#' @param dropout (float) dropout rate of this layer. Set to 0 for no dropout
#'
#' @details
#' This function creates a 'linear' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"linear" "citolayer"}
#' @import checkmate
#' @example /inst/examples/linear-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
linear <- function(n_neurons = NULL,
                   bias = NULL,
                   activation = NULL,
                   normalization = NULL,
                   dropout = NULL) {

  layer <- list(n_neurons = n_neurons,
                bias = bias,
                activation = activation,
                normalization = normalization,
                dropout = dropout)
  class(layer) <- c("linear", "citolayer")
  return(layer)
}

#' Convolutional layer
#'
#' @description
#'
#' creates a 'conv' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param n_kernels (int) amount of kernels in this layer
#' @param kernel_size (int or tuple) size of the kernels in this layer. Use a tuple if the kernel size isn't equal in all dimensions
#' @param stride (int or tuple) stride of the kernels in this layer. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param padding (int or tuple) zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#' @param dilation (int or tuple) dilation of the kernels in this layer. Use a tuple if the dilation isn't equal in all dimensions
#' @param bias (boolean) if TRUE, adds a learnable bias to the kernels of this layer
#' @param activation (string) activation function that is used after this layer. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"
#' @param normalization (boolean) if TRUE, batch normalization is used after this layer
#' @param dropout (float) dropout rate of this layer. Set to 0 for no dropout
#'
#' @details
#' This function creates a 'conv' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"conv" "citolayer"}
#' @import checkmate
#' @example /inst/examples/conv-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
conv <- function(n_kernels = NULL,
                 kernel_size = NULL,
                 stride = NULL,
                 padding = NULL,
                 dilation = NULL,
                 bias = NULL,
                 activation = NULL,
                 normalization = NULL,
                 dropout = NULL) {

  layer <- list(n_kernels = n_kernels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = bias,
                activation = activation,
                normalization = normalization,
                dropout = dropout)
  class(layer) <- c("conv", "citolayer")
  return(layer)
}

#' Average pooling layer
#'
#' @description
#'
#' creates a 'avgPool' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param kernel_size (int or tuple) size of the kernel in this layer. Use a tuple if the kernel size isn't equal in all dimensions
#' @param stride (int or tuple) stride of the kernel in this layer. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param padding (int or tuple) zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#'
#' @details
#' This function creates a 'avgPool' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"avgPool" "citolayer"}
#' @import checkmate
#' @example /inst/examples/avgPool-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
avgPool <- function(kernel_size = NULL,
                    stride = NULL,
                    padding = NULL) {

  layer <- list(kernel_size = kernel_size,
                stride = stride,
                padding = padding)
  class(layer) <- c("avgPool", "citolayer")
  return(layer)
}

#' Maximum pooling layer
#'
#' @description
#'
#' creates a 'maxPool' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param kernel_size (int or tuple) size of the kernel in this layer. Use a tuple if the kernel size isn't equal in all dimensions
#' @param stride (int or tuple) stride of the kernel in this layer. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions
#' @param padding (int or tuple) zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions
#' @param dilation (int or tuple) dilation of the kernel in this layer. Use a tuple if the dilation isn't equal in all dimensions
#'
#' @details
#' This function creates a 'maxPool' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' The parameters that aren't assigned here (and are therefore still NULL) are filled with the default values passed to \code{\link{create_architecture}}.

#' @return S3 object of class \code{"maxPool" "citolayer"}
#' @import checkmate
#' @example /inst/examples/maxPool-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
maxPool <- function(kernel_size = NULL,
                    stride = NULL,
                    padding = NULL,
                    dilation = NULL) {

  layer <- list(kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation)
  class(layer) <- c("maxPool", "citolayer")
  return(layer)
}

#' Transfer learning
#'
#' @description
#'
#' creates a 'transfer' 'citolayer' object that is used by \code{\link{create_architecture}}.
#'
#' @param name The name of the pretrained model
#' @param pretrained if FALSE, random weights are used instead of the pretrained weigths
#' @param freeze if TRUE, the weights of the pretrained model (except the "classifier" part at the end) aren't changed in the training anymore. Only works if pretrained=TRUE
#'
#' @details
#' This function creates a 'transfer' 'citolayer' object that is passed to the \code{\link{create_architecture}} function.
#' With this object the pretrained models that are available in the 'torchvision' package can be used in cito.
#' When 'freeze' is set to TRUE, only the weights of the last part of the network (consisting of one or more linear layers) are adjusted in the training.
#' There mustn't be any other citolayers before the transfer citolayer object when calling \code{\link{create_architecture}}.
#' If there are any citolayers after the transfer citolayer, the linear classifier part of the pretrained model is replaced with the specified citolayers.

#' @return S3 object of class \code{"transfer" "citolayer"}
#' @import checkmate
#' @example /inst/examples/transfer-example.R
#' @author Armin Schenk
#' @seealso \code{\link{create_architecture}}
#' @export
transfer <- function(name = c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"),
                     pretrained = TRUE,
                     freeze = TRUE) {

  if(identical(name, c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"))) {
    name <- "alexnet"
  }

  name <- match.arg(name)

  layer <- list(name = name,
                pretrained = pretrained,
                freeze = pretrained & freeze,
                replace_classifier = FALSE)
  class(layer) <- c("transfer", "citolayer")
  return(layer)
}

#' Print class citoarchitecture
#'
#' @param x an object created by \code{\link{create_architecture}}
#' @param input_shape a vector with the dimensions of a single sample (e.g. c(3,28,28))
#' @param output_shape the number of nodes in the output layer
#' @param ... additional arguments
#' @return original object
#'
#' @example /inst/examples/print.citoarchitecture-example.R
#' @export
print.citoarchitecture <- function(x, input_shape, output_shape, ...) {
  x <- adjust_architecture(x, length(input_shape)-1)
  need_output_layer <- TRUE

  for(layer in x) {
    if(inherits(layer, "transfer")) need_output_layer <- layer$replace_classifier
    input_shape <- print(layer, input_shape, output_shape)
  }

  if(need_output_layer) {
    output_layer <- linear(n_neurons=output_shape, bias = TRUE,
                           activation="Depends on loss", normalization=FALSE, dropout=0)
    print(output_layer, input_shape)
  }
  cat("-------------------------------------------------------------------------------\n")
}

print.linear <- function(layer, input_shape, ...) {

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Linear     |Input: ", prod(input_shape), "\n"))
  cat(paste0("           |Output: ", layer[["n_neurons"]], "\n"))
  cat(paste0("           |Bias: ", layer[["bias"]], "\n"))
  if(layer[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  cat(paste0("           |Activation: ", layer[["activation"]], "\n"))
  if(layer[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", layer[["dropout"]], "\n"))
  }

  return(invisible(layer[["n_neurons"]]))
}

print.conv <- function(layer, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = layer[["n_kernels"]],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = layer[["dilation"]])

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")
  dilation <- paste(layer[["dilation"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Convolution|Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")\n"))
  cat(paste0("           |Bias: ", layer[["bias"]], "\n"))
  if(layer[["normalization"]]) {
    cat(paste0("           |Batch normalization\n"))
  }
  cat(paste0("           |Activation: ", layer[["activation"]], "\n"))
  if(layer[["dropout"]]>0) {
    cat(paste0("           |Dropout: rate=", layer[["dropout"]], "\n"))
  }

  return(invisible(output_shape))
}

print.avgPool <- function(layer, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = rep(1,length(input_shape)-1))

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("AvgPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ")\n"))

  return(invisible(output_shape))
}

print.maxPool <- function(layer, input_shape, ...) {

  output_shape <- get_output_shape(input_shape = input_shape,
                                   n_kernels = input_shape[1],
                                   kernel_size = layer[["kernel_size"]],
                                   stride = layer[["stride"]],
                                   padding = layer[["padding"]],
                                   dilation = layer[["dilation"]])

  kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
  stride <- paste(layer[["stride"]], collapse = "x")
  padding <- paste(layer[["padding"]], collapse = "x")
  dilation <- paste(layer[["dilation"]], collapse = "x")

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("MaxPool    |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Kernel: ", kernel_size, " (stride=", stride, ", padding=", padding, ", dilation=", dilation, ")\n"))

  return(invisible(output_shape))
}

print.transfer <- function(layer, input_shape, output_shape, ...) {

  if(layer$replace_classifier) {
    output_shape <- get_transfer_output_shape(layer$name)
  }

  cat("-------------------------------------------------------------------------------\n")
  cat(paste0("Transfer   |Input: ", paste(input_shape, collapse = "x"), "\n"))
  cat(paste0("           |Output: ", paste(output_shape, collapse = "x"), "\n"))
  cat(paste0("           |Network: ", layer[["name"]] , "\n"))
  cat(paste0("           |Pretrained: ", layer[["pretrained"]] , "\n"))
  if(layer[["pretrained"]]) cat(paste0("           |Weights frozen: ", layer[["freeze"]] , "\n"))

  return(invisible(output_shape))
}


#' Plot the CNN architecture
#'
#' @param x an object of class citoarchitecture created by \code{\link{create_architecture}}
#' @param input_shape a vector with the dimensions of a single sample (e.g. c(3,28,28))
#' @param output_shape the number of nodes in the output layer
#' @param ... additional arguments
#' @return nothing
#'
#' @example /inst/examples/plot.citoarchitecture-example.R
#' @export
plot.citoarchitecture <- function(x, input_shape, output_shape, ...) {
  x <- adjust_architecture(x, length(input_shape)-1)

  transfer_only <- length(x) == 1 && inherits(x[[1]], "transfer")

  text <- c(paste0("Input size: ", paste(input_shape, collapse = "x")), "")
  type <- c("data", "arrow")

  for(layer in x) {
    if(inherits(layer, "transfer")) {
      tmp <- paste0("Transfer network: ", layer[["name"]], " (pretrained weights: ", layer[["pretrained"]])
      if(layer[["pretrained"]]) {
        tmp <- paste0(tmp, ", frozen weights: ", layer[["freeze"]], ")")
      } else {
        tmp <- paste0(tmp, ")")
      }
      text <- c(text, tmp)
      type <- c(type, "transfer")
      if(!transfer_only) {
        input_shape <- get_transfer_output_shape(layer[["name"]])
        text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
        type <- c(type, "arrow")
      } else {
        text <- c(text, "")
        type <- c(type, "arrow")
      }
    } else if(inherits(layer, "linear")) {
      text <- c(text, paste0("Linear layer with ", layer[["n_neurons"]], " neurons"))
      type <- c(type, "linear")
      if(layer[["normalization"]]) {
        text <- c(text, "Batch normalization")
        type <- c(type, "arrow")
      }
      text <- c(text, paste0("Activation: ", layer[["activation"]]))
      type <- c(type, "arrow")
      if(layer[["dropout"]] > 0) {
        text <- c(text, paste0("Dropout rate: ", layer[["dropout"]]))
        type <- c(type, "arrow")
      }
    } else if(inherits(layer, "conv")) {
      kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
      stride <- paste(layer[["stride"]], collapse = "x")
      padding <- paste(layer[["padding"]], collapse = "x")
      dilation <- paste(layer[["dilation"]], collapse = "x")
      text <- c(text, paste0("Convolutional layer with ", layer[["n_kernels"]], " kernels (size=", kernel_size, ", stride=", stride, ", padding=", padding, ", dilation=", dilation, ")"))
      type <- c(type, "conv")
      input_shape <- get_output_shape(input_shape = input_shape,
                                       n_kernels = layer[["n_kernels"]],
                                       kernel_size = layer[["kernel_size"]],
                                       stride = layer[["stride"]],
                                       padding = layer[["padding"]],
                                       dilation = layer[["dilation"]])
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")
      if(layer[["normalization"]]) {
        text <- c(text, "Batch normalization")
        type <- c(type, "arrow")
      }
      text <- c(text, paste0("Activation: ", layer[["activation"]]))
      type <- c(type, "arrow")
      if(layer[["dropout"]] > 0) {
        text <- c(text, paste0("Dropout rate: ", layer[["dropout"]]))
        type <- c(type, "arrow")
      }
    } else if(inherits(layer, "maxPool")) {
      kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
      stride <- paste(layer[["stride"]], collapse = "x")
      padding <- paste(layer[["padding"]], collapse = "x")
      dilation <- paste(layer[["dilation"]], collapse = "x")
      text <- c(text, paste0("Maximum pooling layer (kernel size=", kernel_size, ", stride=", stride, ", padding=", padding, ", dilation=", dilation, ")"))
      type <- c(type, "pool")
      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = layer[["dilation"]])
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")
    } else if(inherits(layer, "avgPool")) {
      kernel_size <- paste(layer[["kernel_size"]], collapse = "x")
      stride <- paste(layer[["stride"]], collapse = "x")
      padding <- paste(layer[["padding"]], collapse = "x")
      text <- c(text, paste0("Average pooling layer (kernel size=", kernel_size, ", stride=", stride, ", padding=", padding, ")"))
      type <- c(type, "pool")
      input_shape <- get_output_shape(input_shape = input_shape,
                                      n_kernels = input_shape[1],
                                      kernel_size = layer[["kernel_size"]],
                                      stride = layer[["stride"]],
                                      padding = layer[["padding"]],
                                      dilation = rep(1,length(input_shape)-1))
      text <- c(text, paste0("Output size: ", paste(input_shape, collapse = "x")))
      type <- c(type, "arrow")
    }
  }

  if(!transfer_only) {
    text <- c(text, paste0("Linear layer with ", output_shape, " neurons"), "")
    type <- c(type, "linear", "arrow")
  }

  text <- c(text, paste0("Output size: ", paste(output_shape, collapse = "x")))
  type <- c(type, "data")


  n <- length(text)
  height <- 1/n
  ybottom <- 1-height
  ytop <- 1
  graphics::plot.new()
  for(i in 1:n) {
    if(type[i] == "arrow") {
      graphics::arrows(0.5, ytop, 0.5, ybottom, length = 0.5*graphics::par("pin")[2]/(2*n-1))
      graphics::text(0.5, ytop-height/2, text[i], pos = 4)
    } else {
      color <- switch(type[i],
                      "transfer" = "lightyellow",
                      "conv" = "lightblue",
                      "linear" = "lightgreen",
                      "pool" = "pink",
                      "data" = "orange")
      graphics::rect(0, ybottom, 1, ytop, col = color)
      graphics::text(0.5, ytop-height/2, text[i])
    }
    ybottom <- ybottom-height
    ytop <- ytop-height
  }
}
