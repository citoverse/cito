#' Embeddings
#'
#' Can be used for categorical variables, a more efficient alternative to one-hot encoding
#'
#' @param dim integer, embedding dimension
#' @param weights matrix, to use custom embedding matrices
#' @param train logical, should the embeddings be trained or not
#' @param lambda regularization strength on the embeddings
#' @param alpha mix between L1 and L2 regularization
#'
#' @export

e = function(dim = 1L, weights = NULL, train = TRUE, lambda = 0.0, alpha = 1.0) {
  if(!is.null(weights)) {
    dim = ncol(weights)
  }
  return(list(list(dim = dim, weights = weights,  train = train, lambda = lambda, alpha = alpha)))
}
