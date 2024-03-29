% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cnn.R
\name{create_architecture}
\alias{create_architecture}
\title{CNN architecture}
\usage{
create_architecture(
  ...,
  default_n_neurons = 10,
  default_n_kernels = 10,
  default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
  default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
  default_padding = list(conv = 0, maxPool = 0, avgPool = 0),
  default_dilation = list(conv = 1, maxPool = 1),
  default_bias = list(conv = TRUE, linear = TRUE),
  default_activation = list(conv = "relu", linear = "relu"),
  default_normalization = list(conv = FALSE, linear = FALSE),
  default_dropout = list(conv = 0, linear = 0)
)
}
\arguments{
\item{...}{objects of class 'citolayer' created by \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}} or \code{\link{transfer}}}

\item{default_n_neurons}{(int) default value: amount of neurons in a linear layer}

\item{default_n_kernels}{(int) default value: amount of kernels in a convolutional layer}

\item{default_kernel_size}{(int or tuple) default value: size of the kernels in convolutional and pooling layers. Use a tuple if the kernel size isn't equal in all dimensions}

\item{default_stride}{(int or tuple) default value: stride of the kernels in convolutional and pooling layers. NULL sets the stride equal to the kernel size. Use a tuple if the stride isn't equal in all dimensions}

\item{default_padding}{(int or tuple) default value: zero-padding added to both sides of the input. Use a tuple if the padding isn't equal in all dimensions}

\item{default_dilation}{(int or tuple) default value: dilation of the kernels in convolutional and maxPooling layers. Use a tuple if the dilation isn't equal in all dimensions}

\item{default_bias}{(boolean) default value: if TRUE, adds a learnable bias to neurons of linear and kernels of convolutional layers}

\item{default_activation}{(string) default value: activation function that is used after linear and convolutional layers. The following activation functions are supported: "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid"}

\item{default_normalization}{(boolean) default value: if TRUE, batch normalization is used after linear and convolutional layers}

\item{default_dropout}{(float) default value: dropout rate of linear and convolutional layers. Set to 0 for no dropout}
}
\value{
S3 object of class \code{"citoarchitecture"}
}
\description{
creates a 'citoarchitecture' object that is used by \code{\link{cnn}}.
}
\details{
This function creates a 'citoarchitecture' object that provides the \code{\link{cnn}} function with all information about the architecture of the CNN that will be created and trained.
The final architecture consists of the layers in the sequence they were passed to this function.
All parameters of the 'citolayer' objects, that are still NULL because they haven't been specified at the creation of the layer, are filled with the given default parameters for their specific layer type (linear, conv, maxPool, avgPool).
The default values can be changed by either passing a list with the values for specific layer types (in which case the defaults of layer types which aren't in the list remain the same)
or by passing a single value (in which case the defaults for all layer types is set to that value).
}
\seealso{
\code{\link{cnn}}, \code{\link{linear}}, \code{\link{conv}}, \code{\link{maxPool}}, \code{\link{avgPool}}, \code{\link{transfer}}, \code{\link{print.citoarchitecture}}, \code{\link{plot.citoarchitecture}}
}
\author{
Armin Schenk
}
