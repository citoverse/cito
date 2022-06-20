#' cito: A package for training neural networks
#'
#' cito is based on the torch Machine Learning framework and allows to train a deep neural network with one line and also analyze it afterwards.
#' The main function is \code{\link{dnn}} which trains a custom deep neural network.
#'
#' @section Installation:
#'
#' in order to install cito please follow these steps:
#'
#' \code{install.packges("cito")}
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
#' - \code{\link{PDP}}: plot the partial dependecy plot for a specific feature
#' - \code{\link{ALE}}: plot the accumulated local effect plot for a specific feature
#'
#' @examples
#' vignette("cito", package="cito")
#'
#' @docType package
#' @name cito
NULL
#> NULL
