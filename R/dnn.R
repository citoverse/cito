#' DNN
#'
#' @param formula formula object
#' @param data matrix or data.frame
#' @param family error distribution with link function, see details for supported family functions
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of activation functions for each layer. Currently supported: tanh, relu, leakyrelu, selu, or sigmoid
#' @param validation percantage of data set that should be taken as validation set (chosen randomly)
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' @param dropout probability of dropout rate
#' @param optimizer which optimizer used for training the network,
#' @param lr learning rate given to optimizer
#' @param batchsize how many samples data loader loads per batch
#' @param shuffle TRUE if data should be reshuffled every epoch (default: FALSE)
#' @param epochs epochs for training loop
#' @param plot plot training loss
#' @param ... additional arguments to be passed to optimizer
#'
#' @import checkmate
#' @export
dnn = function(formula,
               data = NULL,
               family = stats::gaussian(),
               hidden = c(10L, 10L, 10L),
               activation = "relu",
               validation = 0,
               bias = TRUE,
               lambda = 0.0,
               alpha = 0.5,
               dropout = 0.0,
               optimizer = "adam",
               lr = 0.01,
               batchsize = 32L,
               shuffle = FALSE,
               epochs = 64,
               plot = TRUE,
               ...) {
  checkmate::assert(checkmate::checkMatrix(data), checkmate::checkDataFrame(data))
  checkmate::qassert(activation, "S+[1,)")
  checkmate::qassert(bias, "B+")
  checkmate::qassert(lambda, "R1[0,)")
  checkmate::qassert(validation, "R1[0,1)")
  checkmate::qassert(alpha, "R1[0,)")
  checkmate::qassert(dropout, "R+[0,)")

  self = NULL

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
    Y = as.matrix(Y)
  }

  fam = get_family(family)

  y_dim = ncol(Y)
  x_dtype = torch::torch_float32()
  y_dtype = torch::torch_float32()
  if(is.character(Y)) {
    y_dim = length(unique(as.integer(as.factor(Y[,1]))))
    Y = matrix(as.integer(as.factor(Y[,1])), ncol = 1L)
    if(fam$family$family == "binomial") {
      Y = torch::as_array(torch::nnf_one_hot( torch::torch_tensor(Y, dtype=torch::torch_long() ))$squeeze() )
    }
  }
  if(fam$family$family == "softmax") y_dtype = torch::torch_long()


  ### dataset  ###
  if(validation!= 0){
    train <- sort(sample(c(1:nrow(X)),replace=FALSE,size = round(validation*nrow(X))))
    valid <- c(1:nrow(X))[-train]
    train_dl <- get_data_loader(X[train,],Y[train,], batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
    valid_dl <- get_data_loader(X[valid,],Y[valid,], batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)

  }else{
    train_dl <- get_data_loader(X,Y, batch_size = batchsize, shuffle = shuffle, x_dtype=x_dtype, y_dtype=y_dtype)
  }


  ### build model ###
  # bias in first layer is set by formula intercept
  net = build_model(input = ncol(X), output = y_dim,
                    hidden = hidden, activation = activation,
                    bias = bias, dropout = dropout)

  parameters = c(net$parameters, fam$parameter)

  ### set optimizer ###
  optim<- switch(tolower(optimizer),
                 "adam"= torch::optim_adam(parameters, lr=lr,...),
                 "adadelta" = torch::optim_adadelta(parameters, lr=lr,...),
                 "adagrad" =  torch::optim_adagrad(parameters, lr=lr,...),
                 "rmsprop"  = torch::optim_rmsprop(parameters, lr=lr,...),
                 "rprop" = torch::optim_rprop(parameters, lr=lr,...),
                 "sgd" = torch::optim_sgd(parameters, lr=lr,...),
                 "lbfgs" = torch::optim_lbfgs(parameters, lr=lr,...)

  )

  ### training loop ###

  loss.fkt<- fam$loss
  losses<- data.frame(epoch=c(1:epochs),train_l=NA,valid_l= NA)
  for (epoch in 1:epochs) {
    train_l <- c()

    coro::loop(for (b in train_dl) {
      optim$zero_grad()
      output <- net(b[[1]])
      loss <- loss.fkt(output, b[[2]])$mean()
      loss$backward()
      optim$step()
      train_l <- c(train_l, loss$item())
      losses$train_l[epoch]<- mean(train_l)
    })

    if(validation!= 0){

      valid_l <- c()

      coro::loop(for (b in valid_dl) {
        output <- net(b[[1]])
        loss <- loss.fkt(output, b[[2]])$mean()
        valid_l <- c(valid_l, loss$item())
        losses$valid_l[epoch]<- mean(valid_l)
      })
      cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f\n",
                  epoch, losses$train_l[epoch], losses$valid_l[epoch]))
    }else{
      cat(sprintf("Loss at epoch %d: %3f\n", epoch, losses$train_l[epoch]))
    }

    ### create plot ###
    if(plot) visualize.training(losses,epoch)

   }

  z<- list()
  class(z)<- "citodnn"
  z$net<- net
  z$call <- match.call()
  z$family = fam
  z$data = list(X = X, Y = Y, data = data)

  return(z)
}

#' Print class citodnn
#'
#' @param x a model created by \code{\link{dnn}}
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @import checkmate
#' @export
print.citodnn<- function(x,...){
  print(x$call)
  print(x$net)
}

#' Predict from a fitted dnn model
#'
#' @param object a model created by \code{\link{dnn}}
#' @param newdata newdata for predictions
#' @param type link or response
#' @param ... additional arguments
#' @return prediction matrix
#'
#' @export
predict.citodnn = function(object, newdata = NULL,type=c("link", "response"),...) {

  checkmate::assert( checkmate::checkNull(newdata),
                     checkmate::checkMatrix(newdata),
                     checkmate::checkDataFrame(newdata))

  type = match.arg(type)

  if(type == "link") link = function(a) object$family$invlink(a)
  else link = function(a) a

  ### TO DO: use dataloaders via get_data_loader function
  if(is.null(newdata)) newdata = torch::torch_tensor(object$data$X)
  else {
    if(is.data.frame(newdata)) {
        newdata <- stats::model.matrix(stats::as.formula(object$call$formula[-2]), newdata)
      } else {
        newdata <- stats::model.matrix(stats::as.formula(object$call$formula[-2]), data.frame(newdata))
      }
    newdata<- torch::torch_tensor(newdata)
  }


  pred = torch::as_array(link(object$net(newdata,...)))
  return(pred)
}





# res = dnn(Species~Sepal.Width+Petal.Length, hidden=rep(10,5), data = iris, family = "softmax",validation= 0.3,epochs =24)
# predict(res,iris[1:5,])
# res = dnn(Sepal.Width~Species +Petal.Length+ I(Petal.Length^2), hidden=rep(10,5), data = iris ,validation= 0.3,epochs =100)
#predict(res,iris[1:5,])
#summary(lm(scale(Sepal.Length)~scale(Sepal.Width)+scale(Petal.Length), data = iris))

