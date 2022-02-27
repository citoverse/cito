#' DNN
#'
#' @param formula formula object
#' @param data matrix or data.frame
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of activation functions for each layer. Currently supported: tanh, relu, leakyrelu, selu, or sigmoid
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' @param dropout probability of dropout rate
#'
#' @import checkmate
#' @export
dnn = function(formula,
               data = NULL,
               hidden = c(10L, 10L, 10L),
               activation = "relu",
               bias = TRUE,
               lambda = 0.0,
               alpha = 0.5,
               dropout = 0.0) {
  assert(checkMatrix(data), checkDataFrame(data))
  qassert(activation, "S+[1,)")
  qassert(bias, "B+")
  qassert(lambda, "R1[0,)")
  qassert(alpha, "R1[0,)")
  qassert(dropout, "R1[0,)")

  if(is.data.frame(data)) {

    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
      X = stats::model.matrix(formula, data)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula, data)
    }

  } else {

    if(!is.null(formula)) {
      mf = match.call()
      m = match("formula", names(mf))
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
    } else {
      formula = stats::as.formula("~.")
    }
    data = data.frame(data)
    X = stats::model.matrix(formula, data)
    Y = stats::model.response(stats::model.frame(formula, data))
    if(!inherits(Y, "matrix")) Y = as.matrix(Y)
  }


  ### build model ###
  # bias in first layer is set by formula intercept
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = nn_linear(ncol(X), out_features = ncol(Y),bias = FALSE)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden) != length(bias)) bias = rep(bias, length(bias))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = nn_linear(ncol(X), out_features = hidden[1], bias = FALSE)
      } else {
        layers[[counter]] = nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i-1])
      }
      counter = counter+1
      if(activation[i] == "relu") layers[[counter]] = nn_relu()
      if(activation[i] == "leaky_relu") layers[[counter]] = nn_leaky_relu()
      if(activation[i] == "tanh") layers[[counter]] = nn_tanh()
      counter = counter+1
    }
    layers[[length(layers)+1]] = nn_linear(hidden[i], out_features = ncol(Y), bias = bias[i])
  }
  net = do.call(nn_sequential, layers)
  return(net)
}

