% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cnn.R
\name{linear}
\alias{linear}
\title{Create a Linear Layer for a CNN Architecture}
\usage{
linear(
  n_neurons = NULL,
  bias = NULL,
  activation = NULL,
  normalization = NULL,
  dropout = NULL
)
}
\arguments{
\item{n_neurons}{(integer) The number of hidden neurons in this layer.}

\item{bias}{(boolean) If \code{TRUE}, a learnable bias is added to the neurons of this layer.}

\item{activation}{(character) The activation function applied after this layer. Supported activation functions include "relu", "leaky_relu", "tanh", "elu", "rrelu", "prelu", "softplus", "celu", "selu", "gelu", "relu6", "sigmoid", "softsign", "hardtanh", "tanhshrink", "softshrink", "hardshrink", "log_sigmoid".}

\item{normalization}{(boolean) If \code{TRUE}, batch normalization is applied after this layer.}

\item{dropout}{(numeric) The dropout rate for this layer. Set to 0 to disable dropout.}
}
\value{
An S3 object of class \code{"linear" "citolayer"}, representing a linear layer in the CNN architecture.
}
\description{
This function creates a \code{linear} layer object of class \code{citolayer} for use in constructing a Convolutional Neural Network (CNN) architecture. The resulting layer object can be passed to the \code{\link{create_architecture}} function to define the structure of the network.
}
\details{
This function creates a \code{linear} layer object, which is used to define a linear layer in a CNN architecture. Parameters not specified (and thus set to \code{NULL}) will be filled with default values provided to the \code{\link{create_architecture}} function.
}
\examples{
\donttest{
if(torch::torch_is_installed()){
library(cito)

# A linear layer where all available parameters are assigned
# No value will be overwritten by 'create_architecture()'
layer1 <- linear(100, TRUE, "relu", FALSE, 0.5)

# A linear layer where only the activation function is assigned
# n_neurons, bias, normalization and dropout are filled with the defaults
# passed to the 'create_architecture()' function
layer2 <- linear(activation="selu")
}
}
}
\seealso{
\code{\link{create_architecture}}
}
\author{
Armin Schenk
}
