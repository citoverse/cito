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
#' @return A list with the settings of the embedding; it is only meaningful inside the formula of \code{\link{dnn}}.
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
#' @param lambda regularization strength on the fixed effects (not tunable with \code{tune()})
#' @param alpha mix between L1 (\code{alpha = 0}) and L2 (\code{alpha = 1}) regularization
#'
#' @details The first argument of \code{fe()}, not part of its signature, is the right-hand side of a formula, so
#' \code{I()}, interactions and factors work as in \code{lm()} (factors with treatment contrasts; \code{0 +} or
#' \code{- 1} inside \code{fe()} are ignored). Its terms enter the linear predictor (the link scale) as
#' \eqn{\eta = f(X) + \beta^T x_{fe}}, without an intercept, which the output layer of the DNN \eqn{f} provides
#' (so do not combine \code{fe()} with \code{bias = FALSE}). The variables in \code{fe()} are not passed to the DNN,
#' and \code{.} in the formula expands to all other variables; a variable in \code{fe()} and in the DNN or in
#' \code{e()} is an error. Use one \code{fe()} term. Transformations that depend on the data, such as \code{poly()}
#' or \code{scale()}, are re-evaluated on \code{newdata} in \code{predict()}; compute them beforehand.
#'
#' The global \code{lambda} and \code{alpha} of \code{dnn()} do not apply to the fixed effects; they are regularized only
#' by the \code{lambda} and \code{alpha} given to \code{fe()} (default: not at all), on the scale of the model matrix,
#' so center and scale the fixed-effect variables when regularizing them.
#'
#' \eqn{\beta} is returned by \code{coef()} as \code{fixed_effects}, one row per model-matrix column and one column per
#' output of the network. It is estimated like a parametric term in a GAM, and is close to the GLM coefficient only if
#' the training converged and the fixed-effect variables are not functions of the DNN inputs. If they are correlated
#' with the DNN inputs, regularization of the DNN (\code{lambda}, dropout, early stopping) can bias \eqn{\beta}. For
#' losses with a softmax (\code{"cross-entropy"}, \code{"multinomial"}, \code{"clogit"}) only the differences
#' between the columns of \code{fixed_effects} are identified. \code{coef()} gives no standard errors; fit with
#' \code{bootstrap =} for their spread. In \code{summary()} and \code{conditionalEffects()}, the effect of a fixed
#' effect is the effect of its model-matrix column, e.g. separate rows for \code{Temp} and \code{I(Temp^2)}.
#'
#' \code{fe()} can be combined with \code{e()}, but is not supported in \code{mmn()} or with \code{custom}.
#'
#' @return A list with \code{lambda} and \code{alpha}; it is only meaningful inside the formula of \code{\link{dnn}}.
#'
#' @example /inst/examples/fe-example.R
#' @export

fe = function(lambda = 0.0, alpha = 1.0) {
  checkmate::qassert(lambda, "N1[0,)")
  checkmate::qassert(alpha, "N1[0,1]")
  return(list(lambda = lambda, alpha = alpha))
}
