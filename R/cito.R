#' 'cito': Building and training neural networks
#'
#' Building and training custom neural networks in the typical R syntax. The 'torch' package is used for numerical calculations, which allows for training on CPU as well as on a graphics card.
#' The main function is \code{\link{dnn}} which trains a custom deep neural network.
#'
#' @section Installation:
#'
#' in order to install cito please follow these steps:
#'
#' \code{install.packages("cito")}
#'
#' \code{library(torch)}
#'
#' \code{install_torch(reinstall = TRUE)}
#'
#' \code{library(cito)}
#'
#' @section cito functions:
#' - \code{\link{dnn}}: train deep neural network
#' - \code{\link{continue_training}}: continues training of an existing cito dnn model for additional epochs
#' - \code{\link{PDP}}: plot the partial dependency plot for a specific feature
#' - \code{\link{ALE}}: plot the accumulated local effect plot for a specific feature
#'
#' @example /inst/examples/dnn-example.R
#'
#'
#' @docType package
#' @name cito
NULL
#> NULL
