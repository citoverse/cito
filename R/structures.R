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
