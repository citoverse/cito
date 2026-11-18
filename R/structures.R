#' Embeddings
#'
#' Can be used to create an embedding structure for categorical variables in the function interface
#'
#' @param dim integer, embedding dimension
#' @param weights matrix, to use custom embedding matrices
#' @param train logical, should the embeddings be trained or not
#' @param lambda regularization strength on the embeddings
#' @param alpha mix between L1 and L2 regularization
#'
#' @details The e() structure must be used in the function interface. Although not document in the function signature, the first argument to the e() structure is the categorical variable that codes a group in the data, as in
#'
#' predictors + e(group, ...)
#'
#' For more details, see the example below
#'
#'
#' @example /inst/examples/e-example.R
#' @export

e = function(dim = 1L, weights = NULL, train = TRUE, lambda = 0.0, alpha = 1.0) {
  if(!is.null(weights)) {
    dim = ncol(weights)
  }
  return(list(list(dim = dim, weights = weights,  train = train, lambda = lambda, alpha = alpha)))
}


#' Fixed effects
#'
#' Adds predictors with linear effects to the DNN in the formula interface: \code{dnn(Y ~ X1 + fe(Temp + I(Temp^2)))}.
#'
#' @param lambda regularization strength on the fixed effects
#' @param alpha mix between L1 and L2 regularization
#'
#' @details The first argument of \code{fe()}, not part of its signature, is the right-hand side of a formula, so
#' \code{I()}, \code{poly()} and interactions work as in \code{lm()}. Its terms enter the linear predictor (the link
#' scale) as \eqn{\eta = f(X) + \beta^T x_{fe}}, without an intercept, since the DNN \eqn{f} has one. The variables in
#' \code{fe()} are not passed to the DNN, and \code{.} in the formula expands to all other variables. The global
#' \code{lambda} and \code{alpha} of \code{dnn()} do not apply to the fixed effects; they are regularized only by the
#' \code{lambda} and \code{alpha} given to \code{fe()} (default: not at all). \eqn{\beta} is returned by \code{coef()}
#' as \code{fixed_effects} (one row per column of the model matrix). \code{fe()} can be combined with \code{e()}, but is not supported in \code{mmn()} or with
#' \code{custom}.
#'
#' @example /inst/examples/fe-example.R
#' @export

fe = function(lambda = 0.0, alpha = 1.0) {
  return(list(lambda = lambda, alpha = alpha))
}
