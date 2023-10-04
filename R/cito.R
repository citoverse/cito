#' 'cito': Building and training neural networks
#'
#' 'cito' simplifies the building and training of (deep) neural networks by relying on standard R syntax and familiar methods from statistical packages. Model creation and training can be done with a single line of code. Furthermore, all generic R methods such as print or plot can be used on the fitted model. At the same time, 'cito' is computationally efficient because it is based on the deep learning framework 'torch' (with optional GPU support). The 'torch' package is native to R, so no Python installation or other API is required for this package.
#'
#' Cito is built around its main function \code{\link{dnn}}, which creates and trains a deep neural network. Various tools for analyzing the trained neural network are available.
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
#'
#' @section cito functions and typical workflow:
#' - \code{\link{dnn}}: train deep neural network
#' - \code{\link{analyze_training}}: check for convergence by comparing training loss with baseline loss
#' - \code{\link{continue_training}}: continues training of an existing cito dnn model for additional epochs
#' - \code{\link{summary.citodnn}}: extract xAI metrics/effects to understand how predictions are made
#' - \code{\link{PDP}}: plot the partial dependency plot for a specific feature
#' - \code{\link{ALE}}: plot the accumulated local effect plot for a specific feature
#'
#' Check out the vignettes for more details on training NN and how a typical workflow with 'cito' could look like.
#'
#' @example /inst/examples/dnn-example.R
#'
#' @aliases cito-package
#' @docType package
#' @name cito
NULL
#> NULL
