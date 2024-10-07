#' 'cito': Building and training neural networks
#'
#' The 'cito' package provides a user-friendly interface for training and interpreting deep neural networks (DNN). 'cito' simplifies the fitting of DNNs by supporting the familiar formula syntax, hyperparameter tuning under cross-validation, and helps to detect and handle convergence problems.  DNNs can be trained on CPU, GPU and MacOS GPUs. In addition, 'cito' has many downstream functionalities such as various explainable AI (xAI) metrics (e.g. variable importance, partial dependence plots, accumulated local effect plots, and effect estimates) to interpret trained DNNs. 'cito' optionally provides confidence intervals (and p-values) for all xAI metrics and predictions. At the same time, 'cito' is computationally efficient because it is based on the deep learning framework 'torch'. The 'torch' package is native to R, so no Python installation or other API is required for this package.
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
#' @name cito
"_PACKAGE"
#> NULL
