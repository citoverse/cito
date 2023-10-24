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
        eval(parse(text=paste0("layer[[parameter]] <- default_", parameter, "[[type]]")))
      }
    }
    return(layer)
  }

  architecture <- list()

  for(layer in list(...)) {
    if(!inherits(layer, "citolayer")) stop("Objects must be of class citolayer")
    architecture <- append(architecture, list(fill_layer(layer)))
  }

  class(architecture) <- "citoarchitecture"

  return(architecture)
}

fill_defaults <- function(passed_defaults, default_defaults) {
  if(inherits(passed_defaults, "list")) {
    for(entry in names(default_defaults)) {
      if(!entry %in% names(passed_defaults)) {
        passed_defaults <- append(passed_defaults, default_defaults[[entry]])
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
                                   dilation = 1)

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
