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
#' @param optimizer which optimizer used for training the network,
#' @param lr learning rate given to optimizer
#' @param batchsize how many samples data loader loads per batch
#' @param shuffle TRUE if data should be reshuffled every epoch (default: FALSE)
#' @param epochs epochs for training loop
#' @param ... additional arguments to be passed to optimizer
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
               dropout = 0.0,
               optimizer = "adam",
               lr = 0.01,
               batchsize = 32L,
               shuffle = FALSE,
               epochs = 64,
               ...) {
  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(activation, "S+[1,)")
  checkmate::qassert(bias, "B+")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(alpha, "R1[0,)")
  checkmate::qassert(dropout, "R1[0,)")

  if(is.data.frame(data)) {

    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
      X = stats::model.matrix(formula, data)
      Y = stats::model.response(stats::model.frame(formula, data))
      if(!inherits(Y, "matrix")) Y = as.matrix(Y)
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

  ### dataset  ###
  torch.dataset <- torch::dataset(

    name = "dataset",

    initialize = function(X,Y) {
      self$X <- torch::torch_tensor(as.matrix(X))
      self$Y <- torch::torch_tensor(as.matrix(Y))
    },

    .getitem = function(index) {

      x <- self$X[index,]
      y <- self$Y[index,]
      list(x, y)
    },

    .length = function() {
      self$Y$size()[[1]]
    }
  )

  train_ds<- torch.dataset(X,Y)
  train_dl<- torch::dataloader(train_ds, batch_size= batchsize, shuffle= shuffle)

  ### build model ###
  # bias in first layer is set by formula intercept
  layers = list()
  if(is.null(hidden)) {
    layers[[1]] = torch::nn_linear(ncol(X), out_features = ncol(Y),bias = FALSE)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden) != length(bias)) bias = rep(bias, (length(hidden)+1))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        layers[[1]] = torch::nn_linear(ncol(X), out_features = hidden[1], bias = FALSE)
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i-1])
      }
      counter = counter+1
      if(activation[i] == "relu") layers[[counter]] = torch::nn_relu()
      if(activation[i] == "leaky_relu") layers[[counter]] = torch::nn_leaky_relu()
      if(activation[i] == "tanh") layers[[counter]] = torch::nn_tanh()
      counter = counter+1
    }
    layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = ncol(Y), bias = bias[i])
  }
  net = do.call(torch::nn_sequential, layers)


  ### set optimizer ###
  optim<- switch(tolower(optimizer),
                 "adam"= torch::optim_adam(net$parameters, lr=lr,...),
                 "adadelta" = torch::optim_adadelta(net$parameters, lr=lr,...),
                 "adagrad" =  torch::optim_adagrad(net$parameters, lr=lr,...),
                 "rmsprop"  = torch::optim_rmsprop(net$parameters, lr=lr,...),
                 "rprop" = torch::optim_rprop(net$parameters, lr=lr,...),
                 "sgd" = torch::optim_sgd(net$parameters, lr=lr,...),
                 "lbfgs" = torch::optim_lbfgs(net$parameters, lr=lr,...)

  )

  ### training loop ###
  #training loop without validation dl
  for (epoch in 1:epochs) {
    l <- c()

    coro::loop(for (b in train_dl) {
      optim$zero_grad()
      output <- net(b[[1]])
      loss <- torch::nnf_mse_loss(output, b[[2]])
      loss$backward()
      optim$step()
      l <- c(l, loss$item())
    })

    cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
  }




  return(net)
}

